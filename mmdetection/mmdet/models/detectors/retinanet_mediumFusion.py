# Copyright (c) OpenMMLab. All rights reserved.
import torch

from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .retinanet import RetinaNet
import torch.nn as nn
from torch import Tensor
from typing import Tuple


@MODELS.register_module()
class RetinaNetMediumFusion(RetinaNet):

    def __init__(self,
                 backbone: ConfigType,
                 neck: ConfigType,
                 bbox_head: ConfigType,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor,
            init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)  # rgb
        # self.thermal_backbone = MODELS.build(backbone)  # thermal
        self.fuse_convs = nn.ModuleList([
            nn.Sequential(nn.Conv2d(2*c, c, 1, bias=False),
                                nn.BatchNorm2d(c),
                                nn.ReLU(True))
            for c in neck.get('in_channels')])


    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor]:
        rgb, thermal = torch.chunk(batch_inputs, 2, 1)
        x_rgbs = self.backbone(rgb)
        x_thermals = self.backbone(thermal)
        assert len(x_rgbs) == len(x_thermals) == len(self.fuse_convs)
        x = []
        for x_rgb, x_thermal, fuse_conv in zip(x_rgbs, x_thermals, self.fuse_convs):
            x.append(fuse_conv(torch.cat([x_rgb, x_thermal], 1)))
        if self.with_neck:
            x = self.neck(x)
        return x