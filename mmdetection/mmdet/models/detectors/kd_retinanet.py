# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.registry import MODELS
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .kd_one_stage import KnowledgeDistillationSingleStageDetector
from typing import Any, Optional, Union
from pathlib import Path
from torch import Tensor
from mmdet.structures import SampleList
import torch
import torch.nn as nn
from ..utils import multi_apply
import torch.nn.functional as F


@MODELS.register_module()
class KDRetinaNet(KnowledgeDistillationSingleStageDetector):
    """Implementation of `RetinaNet <https://arxiv.org/abs/1708.02002>`_"""

    def __init__(
        self,
        backbone: ConfigType,
        neck: ConfigType,
        bbox_head: ConfigType,
        teacher_config: Union[ConfigType, str, Path],
        teacher_ckpt: Optional[str] = None,
        eval_teacher: bool = True,
        train_cfg: OptConfigType = None,
        test_cfg: OptConfigType = None,
        data_preprocessor: OptConfigType = None,
    ) -> None:
        super().__init__(
            backbone=backbone,
            neck=neck,
            bbox_head=bbox_head,
            teacher_config=teacher_config,
            teacher_ckpt=teacher_ckpt,
            eval_teacher=eval_teacher,
            train_cfg=train_cfg,
            test_cfg=test_cfg,
            data_preprocessor=data_preprocessor)
        # In order to reforward teacher model,
        # set requires_grad of teacher model to False
        self.freeze(self.teacher_model)

    @staticmethod
    def freeze(model: nn.Module):
        """Freeze the model."""
        model.eval()
        for param in model.parameters():
            param.requires_grad = False

    # def single_kd(self, feat_s: Tensor, feat_t: Tensor):
    #     feat_s = feat_s.flatten(2)
    #     feat_t = feat_t.flatten(2)
    #     loss = F.mse_loss(F.normalize(feat_s.pow(2), 2, -1), F.normalize(feat_t.pow(2), 2, -1)) * 10.
    #     return loss,
    #
    # def get_anchor_importance(self, logits):
    #     with torch.no_grad():
    #         score = logits.sigmoid()    # [b, na*nc, h, w]
    #         scores_each_anchor = score.split(self.bbox_head.cls_out_channels, 1)
    #         max_score = torch.cat([x.max(1, keepdim=True)[0] for x in scores_each_anchor], 1)
    #         weight = F.normalize(max_score, 1.0, 1) * max_score
    #     return weight
    #
    # def single_cls_kd(self, s, t):
    #     weight = self.get_anchor_importance(t).reshape(-1)
    #     s = s.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
    #     t = t.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.cls_out_channels)
    #     loss = self.cls_kd_loss(pred=s, soft_label=t, weight=weight)    # softmax kd loss
    #     return loss,
    #
    # def single_box_kd(self, s, t):
    #     s = s.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.bbox_coder.encode_size)
    #     t = t.permute(0, 2, 3, 1).reshape(-1, self.bbox_head.bbox_coder.encode_size)
    #     loss = self.box_kd_loss(s, t)
    #     return loss,
    #

    # def loss(self, batch_inputs: Tensor,
    #          batch_data_samples: SampleList) -> dict:
    #     """
    #     Args:
    #         batch_inputs (Tensor): Input images of shape (N, C, H, W).
    #             These should usually be mean centered and std scaled.
    #         batch_data_samples (list[:obj:`DetDataSample`]): The batch
    #             data samples. It usually includes information such
    #             as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.
    #
    #     Returns:
    #         dict[str, Tensor]: A dictionary of loss components.
    #     """
    #     x = self.extract_feat(batch_inputs)
    #     losses = self.bbox_head.loss(x, batch_data_samples)
    #     with torch.no_grad():
    #         teacher_x = self.teacher_model.extract_feat(batch_inputs)
    #         out_teacher = self.teacher_model.bbox_head(teacher_x)   # (cls_score: list, bbox_pred: list)
    #
    #     # student
    #     student_x = self.bbox_head(x)   # (cls_score: list, bbox_pred: list)
    #     # generate pseudo teacher
    #     pseudo_teacher = self.teacher_model.bbox_head(x)    # (cls_score: list, bbox_pred: list)
    #     kd_losses = 0.0
    #     for branch, s, pst, t in (zip(('cls', 'box'), student_x, pseudo_teacher, out_teacher)):
    #         if branch == 'cls': # class branch distillation
    #             kd_loss, = multi_apply(self.single_cls_kd, s, t)
    #             kd_loss.extend(*multi_apply(self.single_cls_kd, pst, t))
    #         elif branch == 'box':
    #             kd_loss, = multi_apply(self.single_box_kd, s, t)
    #             kd_loss.extend(*multi_apply(self.single_box_kd, pst, t))
    #         kd_losses += torch.stack(kd_loss).sum()
    #
    #     losses.update({'loss_feat_kd': kd_losses})
    #     return losses

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> dict:
        """
        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict[str, Tensor]: A dictionary of loss components.
        """
        x = self.extract_feat(batch_inputs)
        with torch.no_grad():
            teacher_x = self.teacher_model.extract_feat(batch_inputs)
            out_teacher = self.teacher_model.bbox_head(teacher_x)
        losses = self.bbox_head.loss(x, out_teacher, self.teacher_model.bbox_head, batch_data_samples)
        return losses




