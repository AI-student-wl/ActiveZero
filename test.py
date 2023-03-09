import os
import cv2
import math
import torch
import logging
import torch.nn as nn
import numpy as np
from tqdm import tqdm
from torch.utils.data import DataLoader

from Config.ActiveZero_config import cfg
from ActiveZero_DataSet import Depth_IR
from nets.adapter import Adapter
from nets.psmnet.psmnet import PSMNet
from utils.Tool import *
from transform_cv2 import Compose, Resize, Random_Center_rotate, Random_Crop


val_test_transform = Compose([
    Resize(size=cfg.dataset.image_resize)
])


def psnr(target, ref):
    diff = ref - target
    diff = diff.flatten('C')
    rmse = math.sqrt(np.mean(diff ** 2.))
    return 20 * math.log10(1.0 / rmse)

def make_file(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

class Active_model(nn.Module):
    def __init__(self, cfg):
        super(Active_model, self).__init__()
        self.model = PSMNet(maxdisp=cfg.model.max_disp)
        self.adapter = Adapter()

    def init_weight_bias(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.Conv3d):
                n = (m.kernel_size[0] * m.kernel_size[1] * m.kernel_size[2] * m.out_channels)
                m.weight.data.normal_(0, math.sqrt(2.0 / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()
    def forward(self, img_L, img_R, args):
        img_L_transform, img_R_transform = self.adapter(img_L, img_R)
        output = self.model(img_L, img_R, img_L_transform, img_R_transform, args)
        return output

if __name__ == "__main__":
    args = get_args()
    make_file(cfg.dataset.test_img)

    #  TODO dataloader
    test_dataset = Depth_IR(cfg.dataset.test_txt, "test", val_test_transform)
    test_loader = DataLoader(test_dataset, batch_size=cfg.dataset.val_test_batch_size,
                             shuffle=True, drop_last=True, num_workers=cfg.dataset.val_test_num_worker)
    #  TODO model
    test_model = Active_model(cfg).to(args.device)
    checkpoint = torch.load("./checkpoints/{}/best_model.pth".format(args.part))
    test_model.load_state_dict(checkpoint["model"])
    
    test_tool = computer_utils()
    with torch.no_grad():
        test_model.eval()
        test_tool.init_zeros()
        for i, data in tqdm(enumerate(test_loader), total=len(test_loader), leave=True):
            img_L = data["left_ir"].to(args.device)
            img_R = data["right_ir"].to(args.device)
            
            img_focal_length = data["ir_left_intrin"][0, 0, 0] / 2
            depth = data["depth"][0, 0, :, :]

            output = test_model(img_L, img_R, args)
            real_pred_disp = output[0]

            pred_depth = (img_focal_length * cfg.baseline) / real_pred_disp  # [1, 256, 256]
            pred_depth = pred_depth.detach().cpu().numpy()[0, :, :]
            depth = depth.detach().cpu().numpy() / 1000
            depth = cv2.resize(depth, (256, 256))

            test_tool.psnr = test_tool.psnr + psnr(depth, pred_depth)
            H, W = pred_depth.shape
            image = np.zeros((H, W*2+10)).astype(np.float64)
            image[:, 0: W] = depth
            image[:, W+10: 2*W+10] = pred_depth
            np.save(os.path.join(cfg.dataset.test_img, str(i)+".npy"), image)

    print("test ave psnr is: {}".format(test_tool.psnr / len(test_loader)))


