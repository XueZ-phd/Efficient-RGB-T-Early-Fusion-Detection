import random

import numpy as np
import torch

from mmdet.registry import MODELS
from .retinanet import RetinaNet
import torch.nn as nn
import clip
from typing import Union, Tuple, List
import torch.nn.functional as F
from torch import Tensor
from mmdet.structures import SampleList


@MODELS.register_module()
class RetinaNetCLIP(RetinaNet):
    classes = {6: ['person', 'car', 'bus', 'motorcycle', 'traffic light', 'truck'], 3: ['person', 'bicycle', 'car']}
    extra_classes = ['building', 'plant', 'sky', 'streetlamp', 'streetlight', 'others']
    def __init__(self, **kwargs):
        super(RetinaNetCLIP, self).__init__(**kwargs)
        neck_in_channels = kwargs.get('neck').get('in_channels')
        neck_out_channel = kwargs.get('neck').get('out_channels')
        self.weaklysup_start_level = self.neck.start_level
        self.match_convs = nn.ModuleList([
            nn.Conv2d(neck_inc, neck_out_channel, 1, bias=False)
            for neck_inc in neck_in_channels[self.weaklysup_start_level:]])

        self.num_class = kwargs.get('bbox_head').get('num_classes')
        self.resnet_fc = nn.Linear(neck_out_channel, self.num_class, bias=False)
        self.resnet_mask_conv = nn.Sequential(
                                                    nn.Conv2d(neck_out_channel, self.num_class, 1, bias=False),
                                                    nn.BatchNorm2d(self.num_class),
                                                    nn.ReLU(True),
                                                    nn.Conv2d(self.num_class, self.num_class, 3, 1, 1, bias=False),
                                                    nn.BatchNorm2d(self.num_class),
        )
        # self.clip_model, _ = clip.load("RN50", download_root='/home/zx/rgbx-distillation/code/mmdetection-main/myCodeZoo')
        self.clip_model, self.preprocess = clip.load("ViT-B/32", download_root='/home/zx/rgbx-distillation/code/mmdetection-main/myCodeZoo')
        self.device = self.clip_model.visual.conv1.weight.device
        self.clip_preprocess_mean = torch.tensor(self.preprocess.transforms[-1].mean).view(1, 3, 1, 1).to(self.device)
        self.clip_preprocess_std = torch.tensor(self.preprocess.transforms[-1].std).view(1, 3, 1, 1).to(self.device)
        self.clip_fc = nn.Sequential(nn.Linear(self.num_class, self.num_class//2, bias=False), nn.LayerNorm(self.num_class//2), nn.SiLU(True),
                                     nn.Linear(self.num_class//2, self.num_class, bias=False), nn.LayerNorm(self.num_class), nn.SiLU(True),
                                     nn.Dropout(),
                                     nn.Linear(self.num_class, self.num_class, bias=False)).float()
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.text_inputs = torch.cat([clip.tokenize(f"a photo of the {c}") for c in self.classes[self.num_class] + self.extra_classes]).to(self.device)

        self.loss_mask = MODELS.build(dict(type='DiceLoss', use_sigmoid=True, activate=True, loss_weight=1.0))

    def normalize_batch_inputs(self, batch_inputs):
        rgb_batch, t_batch = torch.chunk(batch_inputs, 2, dim=1)
        rgb_batch = (rgb_batch * self.data_preprocessor.rgb_std + self.data_preprocessor.rgb_mean) / 255.
        t_batch = (t_batch * self.data_preprocessor.thermal_std + self.data_preprocessor.thermal_mean) / 255.
        assert rgb_batch.min() >= 0.0 and rgb_batch.max() <= 1.0
        assert t_batch.min() >= 0.0 and t_batch.max() <= 1.0
        rgb_batch = (rgb_batch - self.clip_preprocess_mean) / self.clip_preprocess_std.clamp_min(1e-8)
        t_batch = (t_batch - self.clip_preprocess_mean.mean()) / self.clip_preprocess_std.mean().clamp_min(1e-8)
        return torch.cat([rgb_batch, t_batch], 1)

    def clip_pred(self, batch_inputs, img_labels):
        (batch_patches_rgb_inputs, batch_patches_t_inputs), patch_shape, valid_patches = unfold_batch(
            batch_inputs, self.clip_model.visual.input_resolution)
        with torch.no_grad():
            local_rgb_feats, local_t_feats = self.clip_model.encode_image(
                batch_patches_rgb_inputs), self.clip_model.encode_image(batch_patches_t_inputs)
            text_features = self.clip_model.encode_text(self.text_inputs)  # [c+1, d]
        local_rgb_feats /= local_rgb_feats.norm(dim=-1, keepdim=True)
        local_t_feats /= local_t_feats.norm(dim=-1, keepdim=True)
        text_features /= text_features.norm(dim=-1, keepdim=True)

        local_rgb_similarity = (100.0 * local_rgb_feats @ text_features.T).softmax(dim=-1)  # [b*nBlock, c+1]
        local_t_similarity = (100.0 * local_t_feats @ text_features.T).softmax(dim=-1)  # [b*nBlock, c+1]
        rgb_similarity = local_rgb_similarity.view(patch_shape[0], patch_shape[1],
                                                   self.num_class + len(self.extra_classes))  # [b, nBlock, c+1]
        t_similarity = local_t_similarity.view(patch_shape[0], patch_shape[1],
                                               self.num_class + len(self.extra_classes))  # [b, nBlock, c+1]
        rgb_similarity = rgb_similarity * valid_patches.unsqueeze(-1)
        t_similarity = t_similarity * valid_patches.unsqueeze(-1)
        local_similarity = torch.maximum(rgb_similarity, t_similarity)  # [b, nBlock, c+1]
        local_similarity = torch.max(local_similarity, 1)[0]  # [b, c+1]
        # clip_scores = torch.maximum(local_similarity, global_similarity)       # [b, c+1]
        clip_scores = local_similarity[:, :self.num_class]  # [b, c]
        clip_scores = clip_scores + img_labels * (1.0 - clip_scores)
        clip_logits = self.clip_fc(clip_scores.float())
        return clip_logits

    def loss(self, batch_inputs: Tensor,
             batch_data_samples: SampleList) -> Union[dict, list]:
        """Calculate losses from a batch of inputs and data samples.

        Args:
            batch_inputs (Tensor): Input images of shape (N, C, H, W).
                These should usually be mean centered and std scaled.
            batch_data_samples (list[:obj:`DetDataSample`]): The batch
                data samples. It usually includes information such
                as `gt_instance` or `gt_panoptic_seg` or `gt_sem_seg`.

        Returns:
            dict: A dictionary of loss components.
        """
        losses = super().loss(batch_inputs, batch_data_samples)

        backbone_feats = self.backbone(batch_inputs)
        # Backbone Auxiliary Branch
        backbone_logits = []
        backbone_masks = []
        for lidx, feat in enumerate(backbone_feats[self.weaklysup_start_level:]):
            feat = self.match_convs[lidx](feat)
            # image level prediction
            logits = self.avgpool(feat).flatten(1)
            logits = self.resnet_fc(logits)
            backbone_logits.append(logits)
            # box level prediction
            mask = self.resnet_mask_conv(feat)
            backbone_masks.append(mask)

        # CLIP
        ## global
        gt_masks = get_gtmasks(batch_data_samples, 1)
        gt_masks = torch.cat([_.unsqueeze(0) for _ in gt_masks], 0).to(self.device)
        batch_inputs = self.normalize_batch_inputs(batch_inputs * gt_masks)
        # masked_batch_inputs = batch_inputs * gt_masks
        # batch_global_inputs = F.interpolate(batch_inputs, self.clip_model.visual.input_resolution, mode='bicubic')
        # with torch.no_grad():
        #     batch_global_rgb_inputs, batch_global_t_inputs = torch.chunk(batch_global_inputs, 2, 1)
        #     global_rgb_feats = self.clip_model.encode_image(batch_global_rgb_inputs)    # [b, d]
        #     global_t_feats = self.clip_model.encode_image(batch_global_t_inputs)    # [b, d]
        #     text_features = self.clip_model.encode_text(self.text_inputs)   # [c+1, d]
        # global_rgb_feats /= global_rgb_feats.norm(dim=-1, keepdim=True)
        # global_t_feats /= global_t_feats.norm(dim=-1, keepdim=True)
        # text_features /= text_features.norm(dim=-1, keepdim=True)
        # global_rgb_similarity = (100.0 * global_rgb_feats @ text_features.T).softmax(dim=-1)    # [b, c+1]
        # global_t_similarity = (100.0 * global_t_feats @ text_features.T).softmax(dim=-1)    # [b, c+1]
        # global_similarity = torch.maximum(global_rgb_similarity, global_t_similarity) # [b, c+1]
        ## local
        img_labels = get_imgLabel(batch_data_samples, self.num_class)
        clip_logits = self.clip_pred(batch_inputs, img_labels)

        clip_loss = clip_train_loss(self.num_class, backbone_logits, backbone_masks,
                                    clip_logits, batch_data_samples, self.loss_mask)

        losses.update(clip_loss)
        return losses

def clip_train_loss(num_class, backbone_logits, backbone_masks, clip_logits,
                    batch_data_samples, loss_cls_mask):
    device = backbone_logits[0].device
    # image level label
    all_labels = get_imgLabel(batch_data_samples, num_class)

    # box level label
    all_mask_labels = []
    batch_masks = get_gtmasks(batch_data_samples, num_class)
    for pred_mask in backbone_masks:
        feat_h, feat_w = pred_mask.shape[2:]
        gt_mask = []
        for _mask in batch_masks:
            gt_mask.append(F.interpolate(_mask.unsqueeze(0), (feat_h, feat_w), mode='nearest')) # [b, nc, feat_h, feat_w]
        all_mask_labels.append(torch.cat(gt_mask, 0).to(device))    # [num_level, [b, nc, feat_h, feat_w]]

    # loss1, loss2, loss3 = 0.0, 0.0, 0.0
    loss_cls = []
    loss_mask = []
    # loss2 += F.binary_cross_entropy_with_logits(clip_logits, all_labels.float())
    assert len(backbone_masks) == len(backbone_logits) == len(all_mask_labels)
    for idx, (logits, pred_mask, gt_mask) in enumerate(zip(backbone_logits, backbone_masks, all_mask_labels)):
        loss_mask.append(1.0 / len(backbone_masks) * 0.5 *
                         (F.binary_cross_entropy_with_logits(pred_mask, gt_mask.float()) +
                          loss_cls_mask(pred_mask, gt_mask.float())))
        alpha = 0.1
        soft_target = (1 - alpha) * all_labels + alpha * clip_logits.sigmoid().detach()
        soft_target1 = (1 - alpha) * all_labels + alpha * logits.sigmoid().detach()
        loss_cls.append(1.0 / len(backbone_masks) * 0.5 *
                        (F.binary_cross_entropy_with_logits(logits, soft_target) +
                         F.binary_cross_entropy_with_logits(clip_logits, soft_target1)))

        # # # accuracy
        # mask_acc = (gt_mask.float() == (pred_mask.sigmoid()>=0.3).float()).sum() / gt_mask.numel()
        # bk_acc, clip_acc= get_accuracy(logits, all_labels), get_accuracy(clip_logits, all_labels)
        # print(f'{"-"*10}\nBackbone Acc: {bk_acc.detach().cpu().numpy()*100.:.2f}% | '
        #       f'CLIP Acc {clip_acc.detach().cpu().numpy()*100.:.2f}% | '
        #       f'Mask Acc {mask_acc.detach().cpu().numpy()*100.:.2f}%\n{"-"*10}')
    loss = {'backbone_loss_mask': loss_mask, 'clip_transfer_loss': loss_cls}
    return loss


def unfold_batch(batch_inputs, patch_size=224):
    rgb_imgs, t_imgs = torch.chunk(batch_inputs, 2, 1)
    b, c, h, w = rgb_imgs.shape
    assert rgb_imgs.shape == t_imgs.shape
    batch_patch_inputs = []
    for img in [rgb_imgs, t_imgs]:
        patches = F.unfold(img, patch_size, stride=patch_size//2)   # [bs, c*h*w, np]
        valid_patches = torch.where(patches.max(1)[0]>0, 1.0, 0.0) # [bs, np]
        num_patches = patches.shape[-1]
        patches = patches.transpose(2, 1).reshape(b * num_patches, c, patch_size, patch_size).contiguous()
        batch_patch_inputs.append(patches)
    return batch_patch_inputs, (b, num_patches, c, h, w), valid_patches


def get_gtmasks(batch_data_samples, num_class):
    gt_masks = []
    for sample in batch_data_samples:
        h, w = sample.batch_input_shape
        mask = torch.zeros((num_class, h, w))
        boxes, labels = sample.gt_instances.bboxes, sample.gt_instances.labels
        for box, lbidx in zip(boxes, labels):
            x0, y0, x1, y1 = list(map(int, box.cpu().numpy().round()))
            if num_class > 1:
                mask[lbidx, y0:y1, x0:x1] = 1.0
            else:
                mask[0, y0:y1, x0:x1] = 1.0
        gt_masks.append(mask)
    return gt_masks

def get_imgLabel(batch_data_samples, num_class):
    all_labels = []
    for sample in batch_data_samples:
        labels = sample.gt_instances.labels.unique()
        if labels.shape[0]:
            labels = F.one_hot(labels, num_classes=num_class)
            labels = labels.sum(0).clamp(min=0, max=1)
        else:
            labels = torch.zeros(num_class, dtype=torch.int64)
        all_labels.append(labels.unsqueeze(0))
    all_labels = torch.cat(all_labels, 0)   # [b, nc]
    return all_labels


def get_accuracy(predict, target):
    assert predict.dim()==target.dim()
    acc = []
    with torch.no_grad():
        tops = target.sum(1)
        for top, pred, tgt in zip(tops, predict.sigmoid(), target):
            idx = pred.topk(top)[1].sort()[0]
            gt = tgt.topk(top)[1].sort()[0]
            acc.extend((idx==gt).float())
    acc = torch.cat(list(map(lambda x: x.unsqueeze(0), acc)), 0)
    return acc.mean()

def layerscale01(x):
    x_min = x.flatten(1).min(1)[0].detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x_max = x.flatten(1).max(1)[0].detach().unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
    x = (x - x_min) / (x_max - x_min).clamp_min(1e-8)
    return x
