import os
import random

import cv2
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader

from configs.config import cfg
from utils.reprojection import *
from datasets.dataset_utils import *
from utils.test_util import *

class Depth_IR(Dataset):
    def __init__(self, root, cfg):
        self.index_root = self.read_txt(root)
        self.crop_size = (cfg.MODEL.CROP_HEIGHT, cfg.MODEL.CROP_WIDTH)

    def __len__(self):
        return len(self.index_root)

    def __getitem__(self, item):
        # TODO flocal_length and baseline
        path = self.index_root[item].replace("color", "intrin")
        t = path.split("/")
        path = path.replace(t[-1], "ir_left_intrin.txt")
        flocal_length = self.read_intrin(path)[0, 0] / 2
        baseline = 0.05

        #  TODO load image
        img_L = np.array(Image.open(self.index_root[item].replace("color", "left_ir")).convert(mode="L")) / 255
        img_R = np.array(Image.open(self.index_root[item].replace("color", "right_ir")).convert(mode="L")) / 255
        depth = np.array(Image.open(self.index_root[item].replace("color", "depth"))) / 1000
        color = np.array(Image.open(self.index_root[item]))
        mask = depth > 0
        disp = np.zeros_like(depth)
        disp[mask] = (flocal_length * baseline) / depth[mask]
        img_L = np.repeat(img_L[:, :, None], 3, axis=-1)    # [480, 848, 3]
        img_R = np.repeat(img_R[:, :, None], 3, axis=-1)  # [480, 848, 3]

        #  TODO random crop size
        h, w = img_L.shape[:2]
        th, tw = cfg.MODEL.CROP_HEIGHT, cfg.MODEL.CROP_WIDTH
        x = random.randint(0, h - th)
        y = random.randint(0, w - tw)
        img_L = img_L[x: (x + th), y: (y + tw)]
        img_R = img_R[x: (x + th), y: (y + tw)]
        depth = depth[x: (x + th), y: (y + tw)]
        disp = disp[x: (x + th), y: (y + tw)]


        normalization1 = data_augmentation(gaussian_blur=True, color_jitter=True)
        normalization2 = data_augmentation(gaussian_blur=False, color_jitter=False)
        img_L_rgb = normalization1(img_L).type(torch.FloatTensor)
        img_R_rgb = normalization1(img_R).type(torch.FloatTensor)
        img_L = normalization2(img_L).type(torch.FloatTensor)
        img_R = normalization2(img_R).type(torch.FloatTensor)
        item = {}
        item["img_L"] = torch.as_tensor(img_L_rgb, dtype=torch.float32)
        item["img_R"] = torch.as_tensor(img_R_rgb, dtype=torch.float32)
        item["img_L_no_norm"] = torch.as_tensor(img_L, dtype=torch.float32)
        item["img_R_no_norm"] = torch.as_tensor(img_R, dtype=torch.float32)
        item["rgb"] = color
        item["depth"] = torch.as_tensor(depth, dtype=torch.float32).unsqueeze(0)
        item["disp"] = torch.as_tensor(disp, dtype=torch.float32).unsqueeze(0)
        item["baseline"] = torch.as_tensor(baseline, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)
        item["flocal_length"] = torch.as_tensor(flocal_length, dtype=torch.float32).unsqueeze(0).unsqueeze(0).unsqueeze(0)

        return item


    def read_txt(self, root):
        data = []
        with open(root, 'r') as f:
            for file in f.readlines():
                file = file.strip()
                data.append(file)
        return data

    def read_intrin(self, root):
        data = []
        with open(root, 'r') as f:
            for file in f.readlines():
                file = file.strip()
                data.append(file)
        in_cam = np.zeros((3, 3)).astype(np.float32)
        f = data[1][2:-1].split(" ")
        c = data[0][2:-1].split(" ")
        in_cam[0, 0] = eval(f[0])
        in_cam[1, 1] = eval(f[1])
        in_cam[0, 2] = eval(c[0])
        in_cam[1, 2] = eval(c[1])
        in_cam[2, 2] = 1
        return in_cam




