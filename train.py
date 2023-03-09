import gc
import os
import torch
import argparse
import numpy as np


import torch.backends.cudnn as cudnn
from datasets.messytable import Depth_IR, DataLoader
from configs.config import cfg
from utils.reduce import (AverageMeterDict, reduce_scalar_outputs, set_random_seed,
                          synchronize, tensor2float, tensor2numpy)
from utils.util import (adjust_learning_rate, disp_error_img, save_images,
                        save_images_grid, save_scalars, setup_logger)
from utils.losses import AllLosses
from nets.adapter import Adapter

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-file", type=str, default="./configs/temp.yaml")
    parser.add_argument("--local_rank", type=int, default=0)
    parser.add_argument("--device", type=str, default="cuda:0")
    args = parser.parse_args()
    return args


def init_model(backbone, cfg):
    if backbone=="psmnet" and cfg.MODEL.ADAPTER:
        from nets.psmnet.psmnet import PSMNet
        model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    elif backbone=="psmnet":
        from nets.psmnet.psmnet_3 import PSMNet
        model = PSMNet(maxdisp=cfg.MODEL.MAX_DISP).to(cuda_device)
    elif backbone=="dispnet":
        from nets.dispnet.dispnet import DispNet
        model = DispNet().to(cuda_device)
        model.weight_bias_init()
    elif backbone=="raft":
        from nets.raft.raft_stereo import RAFTStereo
        model = RAFTStereo().to(cuda_device)
    else:
        print("Model not implemented!")
    return model

def train(model, model_optimizer, extra, loss_class, TrainImgLoader, ValImgLoader):
    cur_err = np.inf
    adapter_model, adapter_optimizer = extra
    for epoch_idx in range(cfg.SOLVER.EPOCHS):
        # One epoch training loop
        avg_train_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(TrainImgLoader):
            global_step = ((len(TrainImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE * num_gpus)
            if global_step > cfg.SOLVER.STEPS:
                break
            # Adjust learning rate
            if cfg.MODEL.ADAPTER:
                adjust_learning_rate(adapter_optimizer, global_step, cfg.SOLVER.LR, cfg.SOLVER.LR_STEPS)
            if cfg.MODEL.BACKBONE != "raft":
                adjust_learning_rate(model_optimizer, global_step, cfg.SOLVER.LR, cfg.SOLVER.LR_STEPS)

            do_summary = global_step % cfg.SOLVER.SUMMARY_FREQ == 0
            # Train one sample
            # additional output contains all per metric outputs
            item, real_loss_vals = train_sample_onreal(sample, model, model_optimizer, extra, loss_class, isTrain=True)
            print("train liss: {}".format(real_loss_vals))

            # Save checkpoints
            if (global_step) % cfg.SOLVER.SAVE_FREQ == 0:
                checkpoint_data = {
                    "epoch": epoch_idx,
                    "model": model.state_dict(),
                    "model_optimizer": model_optimizer.state_dict(),
                    "adapter": adapter_model.state_dict(),
                    "adapter_optimizer": adapter_optimizer.state_dict()
                }

                save_filename = os.path.join(cfg.SOLVER.LOGDIR, "models", f"model_{global_step}.pth")
                torch.save(checkpoint_data, save_filename)

                # Get average results among all batches
                
        gc.collect()

        avg_val_scalars = AverageMeterDict()
        for batch_idx, sample in enumerate(ValImgLoader):
            global_step = (len(ValImgLoader) * epoch_idx + batch_idx) * cfg.SOLVER.BATCH_SIZE

            do_summary = global_step % cfg.SOLVER.SUMMARY_FREQ == 0
            item, real_loss_vals = train_sample_onreal(sample, model, model_optimizer, extra, loss_class, isTrain=False)
            print("val loss: {}".format(real_loss_vals))


        gc.collect()






def train_sample_onreal(sample, model, model_optimizer, extra, loss_class, isTrain=True):
    item = {}
    adapter_model, adapter_optimizer = extra
    if isTrain and cfg.LOSSES.ONREAL:
        adapter_model.train()
        model.train()
    else:
        adapter_model.eval()
        model.eval()

    # Get reprojection loss on real
    img_real_L = sample["img_L"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_R = sample["img_R"].to(cuda_device)  # [bs, 3, 2H, 2W]
    img_real_L_transformed, img_real_R_transformed = adapter_model(img_real_L, img_real_R)  # [bs, 3, H, W]
    item['img_real_L'] = img_real_L
    item['img_real_R'] = img_real_R
    item["img_L_no_norm"] = sample["img_L_no_norm"].to(cuda_device)
    item["img_R_no_norm"] = sample["img_R_no_norm"].to(cuda_device)
    item['img_real_L_transformed'] = img_real_L_transformed
    item['img_real_R_transformed'] = img_real_R_transformed
    item["depth"] = sample["depth"].to(cuda_device)
    item["disp"] = sample["disp"].to(cuda_device)
    adapter_optimizer.zero_grad()
    model_optimizer.zero_grad()

    real_loss, item, real_loss_vals = loss_class.compute_loss(item, onSim=True, train=(isTrain & cfg.LOSSES.ONREAL))
    real_loss = cfg.LOSSES.REALRATIO * real_loss


    real_loss.backward()
    model_optimizer.step()
    adapter_optimizer.step()


    return item, real_loss_vals

if __name__ == "__main__":
    args = get_args()
    cudnn.benchmark = True
    #  set up device and seed
    set_random_seed(cfg.SOLVER.SEED)
    cuda_device, num_gpus, is_distributed = torch.device("cuda:{}".format(args.device)), 1, False

    os.makedirs(cfg.SOLVER.LOGDIR, exist_ok=True)
    os.makedirs(os.path.join(cfg.SOLVER.LOGDIR, "models"), exist_ok=True)

    #  TODO set up logging
    logger = setup_logger(cfg.NAME, distributed_rank=args.local_rank, save_dir=cfg.SOLVER.LOGDIR)
    logger.info(f"Input args:\n{args}")
    logger.info(f"Running with configs:\n{cfg}")
    logger.info(f"Running with {num_gpus} GPUs")


    # TODO dataloader
    train_dataset = Depth_IR(cfg.SIM.TRAIN, cfg)
    val_dataset = Depth_IR(cfg.SIM.VAL, cfg)
    TrainImgLoader = DataLoader(train_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=True, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=True)
    ValImgLoader = DataLoader(val_dataset, batch_size=cfg.SOLVER.BATCH_SIZE, shuffle=False, num_workers=cfg.SOLVER.NUM_WORKER, drop_last=False)

    # TODO Create Adapter model
    adapter_model = Adapter().to(cuda_device)
    adapter_optimizer = torch.optim.Adam(adapter_model.parameters(), lr=cfg.SOLVER.LR, betas=(0.9, 0.999))

    # TODO load backbone
    backbone = cfg.MODEL.BACKBONE
    model = init_model(backbone, cfg)
    model_optimizer = torch.optim.Adam(model.parameters(), lr=cfg.SOLVER.LR, betas= cfg.SOLVER.BETAS)

    #  TODO  loss function
    loss_class = AllLosses(model, cfg.MODEL.BACKBONE, cfg.MODEL.ADAPTER)

    # Start training
    train(model, model_optimizer, [adapter_model, adapter_optimizer], loss_class, TrainImgLoader, ValImgLoader)

