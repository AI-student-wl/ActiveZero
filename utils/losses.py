from configs.config import cfg
from utils.reprojection import get_reproj_error_patch
import torch.nn.functional as F
import torch
import numpy as np

def psmnet_disp(pred_disp, disp_gt, mask):
    pred_disp3, pred_disp2, pred_disp1 = pred_disp
    loss_disp = (0.5 * F.smooth_l1_loss(pred_disp1[mask], disp_gt[mask], reduction="mean")
                 + 0.7 * F.smooth_l1_loss(pred_disp2[mask], disp_gt[mask], reduction="mean")
                 + F.smooth_l1_loss(pred_disp3[mask], disp_gt[mask], reduction="mean"))
    return loss_disp

class computer_loss():
    def __init__(self, model, adapter):
        self.model = model
        self.adapter = adapter
        self.onsuper = None
        self.isTrain = None

    def computer(self, item, onsuper=True, isTrain=True):
        self.onsuper = onsuper
        self.isTrain = isTrain
        loss = 0
        loss_disp, item = self.compute_disp_loss(item)
        loss = loss + loss_disp
        if self.onsuper:
            loss_reproj, item = self.compute_reprojection_loss(item)
            loss += cfg.LOSSES.REPROJECTION.SIMRATIO * loss_reproj
        else:
            loss_reproj, item = self.compute_reprojection_loss(item)
            loss += cfg.LOSSES.REPROJECTION.REALRATIO * loss_reproj
        return loss, item

    def compute_reprojection_loss(self, item):
        if self.onsuper:
            reproj_loss = F.smooth_l1_loss(item["disp"], item["pred_disp"])
        else:
            real_ir_reproj_loss, real_ir_warped, real_ir_reproj_mask = get_reproj_error_patch(
                input_L=item['img_L'],
                input_R=item['img_R'],
                pred_disp_l=item['pred_disp'],
                ps=cfg.LOSSES.REPROJECTION.PATCH_SIZE,
            )
            item['ir_warped'] = real_ir_warped
            item['ir_reproj_mask'] = real_ir_reproj_mask
            reproj_loss = real_ir_reproj_loss
        return reproj_loss, item
    def forward(self, item):
        if self.isTrain:
            output = self.model(item['img_L'], item['img_R'], item['img_L_transformed'], item['img_R_transformed'])
            pred_disp = output[0]
        else:
            with torch.no_grad():
                pred_disp = self.model(item['img_L'], item['img_R'], item['img_L_transformed'],item['img_R_transformed'])
                output = pred_disp

        return output, pred_disp

    def compute_disp_loss(self, item):
        mask = item["depth"] > 0
        func = psmnet_disp
        loss_disp = 0
        values = {'img_L': item['img_L'], 'img_R': item['img_R'],
                  'img_L_transformed': item['img_L_transformed'],
                  'img_R_transformed': item['img_R_transformed']}
        if self.onsuper:
            disp_gt = item['disp']
            output, pred_disp = self.forward(values)
            loss_disp = func(output, disp_gt, mask)
            item['pred_disp'] = pred_disp
        else:
            output, pred_disp = self.forward(values)
            item['pred_disp'] = pred_disp
        return loss_disp, item


