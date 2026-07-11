import torch
import torch.nn as nn
from models.common import Conv


class zxMaskLoss(nn.Module):
    def __init__(self):
        super(zxMaskLoss, self).__init__()

    def forward(self, gt, pred, smooth=1):
        ''' gt: b x c x H x W,
            pred: b x c x h x w'''
        b, c, h, w = pred.shape
        gt = nn.functional.interpolate(gt, (h, w), mode='nearest')
        bce = nn.functional.binary_cross_entropy(pred, gt, reduction='mean')
        intersection = (pred * gt).sum([2, 3])
        dice_loss = (1 - (2. * intersection + smooth) / (pred.sum([2, 3]) + gt.sum([2, 3]) + smooth)).mean()
        return dice_loss + bce


class MASKLOSS(nn.Module):
    def __init__(self):
        super(MASKLOSS, self).__init__()
        self.zxFuseMaskLoss = zxMaskLoss()

    def forward(self, pred_mask, mask):
        return self.zxFuseMaskLoss(mask, pred_mask)