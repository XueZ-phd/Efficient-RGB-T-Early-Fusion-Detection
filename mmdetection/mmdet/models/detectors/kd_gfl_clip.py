# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, InstanceList, reduce_mean, OptInstanceList
from .gfl_clip import GFLCLIP
from typing import Any, Optional, Union
from pathlib import Path
from torch import Tensor
from mmdet.structures import SampleList
import torch
import torch.nn as nn
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from ..utils import multi_apply, unpack_gt_instances
import torch.nn.functional as F
from typing import List, Tuple


@MODELS.register_module()
class KDGFLCLIP(GFLCLIP):
    """Implementation of `GFL <https://arxiv.org/abs/2006.04388>`_

    Args:
        backbone (:obj:`ConfigDict` or dict): The backbone module.
        neck (:obj:`ConfigDict` or dict): The neck module.
        bbox_head (:obj:`ConfigDict` or dict): The bbox head module.
        train_cfg (:obj:`ConfigDict` or dict, optional): The training config
            of GFL. Defaults to None.
        test_cfg (:obj:`ConfigDict` or dict, optional): The testing config
            of GFL. Defaults to None.
        data_preprocessor (:obj:`ConfigDict` or dict, optional): Config of
            :class:`DetDataPreprocessor` to process the input data.
            Defaults to None.
        init_cfg (:obj:`ConfigDict` or dict, optional): the config to control
            the initialization. Defaults to None.
    """

    def __init__(self,
                 teacher_config: Union[ConfigType, str, Path],
                 teacher_ckpt: Optional[str] = None,
                 eval_teacher: bool = True,
                 corekd_cfg: ConfigType = None,
                 **kwargs
                 ):
        super().__init__(**kwargs)
        self.eval_teacher = eval_teacher
        # Build teacher model
        if isinstance(teacher_config, (str, Path)):
            teacher_config = Config.fromfile(teacher_config)
        self.teacher_model = MODELS.build(teacher_config['model'])
        if teacher_ckpt is not None:
            load_checkpoint(
                self.teacher_model, teacher_ckpt, map_location='cpu')

        # freeze teacher
        self.freeze(self.teacher_model)

        if corekd_cfg is not None:
            self.corekd_loss = MODELS.build(corekd_cfg)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:

        losses = super().loss(batch_inputs, batch_data_samples)  # CLIP weakly supervised learning

        # knowledge distillation
        x = self.extract_feat(batch_inputs)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(batch_inputs)
            out_teacher = self.teacher_model.bbox_head(teacher_x)

        # student head classification stack convs forward
        pseudo_out_teacher_cls = []
        for i in range(len(x)):
            cls_feat = x[i]
            for cls_conv in self.bbox_head.cls_convs:
                cls_feat = cls_conv(cls_feat)
            cls_score = self.teacher_model.bbox_head.gfl_cls(cls_feat)
            pseudo_out_teacher_cls.append(cls_score)
        # for i in range(len(x)):
        #     reg_feat = x[i]
        #     for reg_conv in self.bbox_head.reg_convs:
        #         reg_feat = reg_conv(reg_feat)
        #     bbox_pred = self.teacher_model.bbox_head.scales[i](self.teacher_model.bbox_head.gfl_reg(reg_feat)).float()
        #     pseudo_out_teacher_reg.append(bbox_pred)
        kd_losses = []
        for pcls, tcls in zip(pseudo_out_teacher_cls, out_teacher[0]):
            kd_losses.append(self.corekd_loss(pcls, tcls))
        kd_losses = dict(kd_losses=kd_losses)

        losses.update(kd_losses)
        return losses


    def cuda(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to cuda when calling ``cuda`` function."""
        self.teacher_model.cuda(device=device)
        return super().cuda(device=device)

    def to(self, device: Optional[str] = None) -> nn.Module:
        """Since teacher_model is registered as a plain object, it is necessary
        to put the teacher model to other device when calling ``to``
        function."""
        self.teacher_model.to(device=device)
        return super().to(device=device)

    def train(self, mode: bool = True) -> None:
        """Set the same train mode for teacher and student model."""
        if self.eval_teacher:
            self.teacher_model.train(False)
        else:
            self.teacher_model.train(mode)
        super().train(mode)

    def __setattr__(self, name: str, value: Any) -> None:
        """Set attribute, i.e. self.name = value

        This reloading prevent the teacher model from being registered as a
        nn.Module. The teacher module is registered as a plain object, so that
        the teacher parameters will not show up when calling
        ``self.parameters``, ``self.modules``, ``self.children`` methods.
        """
        if name == 'teacher_model':
            object.__setattr__(self, name, value)
        else:
            super().__setattr__(name, value)
