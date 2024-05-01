from mmdet.registry import MODELS
from .retinanet_clip import RetinaNetCLIP
from mmdet.utils import ConfigType
from pathlib import Path
from mmengine.config import Config
from mmengine.runner import load_checkpoint
from typing import Any, Optional, Union
import torch.nn as nn
import torch
from torch import Tensor
from mmdet.structures import SampleList


@MODELS.register_module()
class KDRetinaNetCLIP(RetinaNetCLIP):
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
            cls_score = self.teacher_model.bbox_head.retina_cls(cls_feat)
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




