import os
import time
import torch
import random
import argparse
import numpy as np
import logging
import matplotlib.pyplot as plt

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, default="cuda:0")
    parser.add_argument("--local_rank", type=int, default=0, help="Rank of device in distributed training")
    parser.add_argument("--part", type=int, default=0)
    args = parser.parse_args()
    return args

def ResNet_warm_up(optimizer, epoch, cfg):
    if epoch <= 20:
        optimizer.param_groups[0]["lr"] = cfg.solver.start_lr
    if epoch > 20:
        temp = (1 - ((epoch - 20) / cfg.solver.end_epoch)) ** 2
        optimizer.param_groups[0]["lr"] = cfg.solver.init_lr * temp

def facebook_warm_np(optimizer, num, train_step, star_lr, end_lr):
    nn = int(train_step * 0.2)
    k = (end_lr - star_lr) / nn
    b = star_lr
    if num < nn:
        optimizer.param_groups[0]["lr"] = k * num + b
    else:
        temp = (1 - ((num - nn) / train_step)) ** 2  # 4
        optimizer.param_groups[0]["lr"] = temp * end_lr

def set_random_seed(cfg, deterministic=False, benchmark=False):
    random.seed(cfg.solver.seed)
    np.random.seed(cfg.solver.seed)
    torch.manual_seed(cfg.solver.seed)
    torch.cuda.manual_seed_all(cfg.solver.seed)
    if deterministic == False:
        torch.backends.cudnn.deterministic = True
    if benchmark == False:
        torch.backends.cudnn.benchmark = True


def Print_Start_Train_Information(cfg):
    logging.info("This is information of train model!")
    logging.info("recover train is {}".format(cfg.recover_train))
    logging.info("epoch {} to {}".format(cfg.solver.strat_epoch, cfg.solver.end_epoch))
    logging.info("init learning rate {}".format(cfg.solver.init_lr))
    logging.info("train batch_size {}".format(cfg.dataset.train_batch_size))
    logging.info("val test batch_size {}".format(cfg.dataset.val_test_batch_size))



class computer_utils:
    def __init__(self):
        self.train_loss = 0
        self.val_loss = 0
        self.psnr = 0

        self.train = []
        self.val = []
        self.lr = []
    def init_zeros(self):
        self.train_loss = 0
        self.val_loss = 0
        self.psnr = 0


def make_file(path):
    if os.path.exists(path) == False:
        os.makedirs(path)

def plot(data, name):
    fig, ax = plt.subplots(nrows=3, ncols=1)
    ax[0].plot(data.train, marker='o', markersize=3)
    ax[0].set_title('train loss')

    ax[1].plot(data.val, marker='o', markersize=3)
    ax[1].set_title('val loss')

    ax[2].plot(data.lr, marker='o', markersize=3)
    ax[2].set_title('train lr')

    plt.savefig('./checkpoints/{}/loss.png'.format(name))
    #plt.show()


