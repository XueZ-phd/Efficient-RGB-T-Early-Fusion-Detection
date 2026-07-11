from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F
# from kornia.losses import ssim_loss
from .ssim import SSIM
'''
Structural Knowledge Distillation for Object Detection
http://arxiv.org/abs/2211.13133
'''

class SSIM_KD(nn.Module):
    def __init__(self, t_inc, s_inc, k=1):
        super(SSIM_KD, self).__init__()
        self.align = nn.Sequential(*[nn.Conv2d(s_inc, t_inc, k, bias=False),
                                     nn.BatchNorm2d(t_inc),
                                     nn.SiLU(True)])
        self.ssim = SSIM(11)

    def forward(self, fm_s, fm_t, eps=1e-6):
        fm_s = self.align(fm_s)
        kd_feat_loss = (1.0 - self.ssim(self.norm_feat(fm_s, eps), self.norm_feat(fm_t, eps))) / 2.0
        return kd_feat_loss * 0.1

    def norm_feat(self, feat, eps=1e-6):
        norm = torch.norm(feat, dim=[2, 3], keepdim=True)
        feat = torch.div(feat, norm + eps)
        return feat

