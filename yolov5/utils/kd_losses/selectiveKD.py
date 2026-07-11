import torch
import torch.nn as nn
import torch.nn.functional as F
from .focusnet_loss import SmoothL1BCELoss


class SelectiveKD(nn.Module):
    def __init__(self, student_in_channels, loss_weights, nc, na, teacher_head, device, p=2.0):
        super(SelectiveKD, self).__init__()
        assert len(student_in_channels) == len(loss_weights)
        self.in_channels = student_in_channels
        self.loss_weights = loss_weights
        self.nc = nc
        self.na = na
        self.teacher_head = teacher_head
        # self.criterion = nn.BCELoss()
        self.criterion = nn.MSELoss(reduction='none')
        # self.criterion = SmoothL1BCELoss(nn.BCEWithLogitsLoss())
        self.device = device
        assert isinstance(self.teacher_head, nn.ModuleList)
        self.eps = 1e-10
        self.p = p

    def norm(self, feat):
        am = torch.pow(torch.abs(feat), self.p)
        norm = torch.norm(am, dim=1, keepdim=True)
        am = torch.div(am, norm + self.eps)
        return am

    def forward(self, student_feats, teacher_feats, rgbs, thermals, teacher_pred, student_pred, balance):
        loss = torch.zeros(1, device=self.device)
        for lidx, (in_c, lw) in enumerate(zip(self.in_channels, self.loss_weights)):
            tb, tc, th, tw = teacher_feats[lidx].shape  # [b, c, h, w]
            sb, sc, sh, sw = student_feats[lidx].shape  # [b, c, h, w]
            assert ([tb, th, tw] == [sb, sh, sw]) and sc == in_c
            t_head_weights = self.teacher_head[lidx].weight.data    # [oc, c, 1, 1]
            t_head_bias = self.teacher_head[lidx].bias.data    # [oc]
            if student_feats[lidx].shape[1] == teacher_feats[lidx].shape[1]:
                selective_weights = t_head_weights
            else:
                # sort and select weights
                _, indices = t_head_weights.norm(2, (2, 3), True).topk(in_c, dim=1)  # [oc, in_c, 1, 1]
                selective_weights = t_head_weights.gather(1, indices)    # [oc, in_c, 1, 1]
            # compute adaptive anchor loss weight
            teacher_object_prob = teacher_pred[lidx][..., 4].sigmoid()  # [bs, 3, h, w]
            anchor_loss_weights = teacher_object_prob.pow(2) / (teacher_object_prob.pow(2).sum(dim=1, keepdim=True) + self.eps)   # [b, 3, h, w]
            assert not anchor_loss_weights.requires_grad

            pseudo_teacher = F.conv2d(student_feats[lidx], selective_weights, t_head_bias)   # [b, oc, h, w]
            pseudo_teacher = pseudo_teacher.view(sb, self.na, 5 + self.nc, sh, sw).permute(0, 1, 3, 4, 2).contiguous()

            pseudo_t_pconf = self.norm(pseudo_teacher[..., 4]) # [b, 3, h, w]
            t_pconf = self.norm(teacher_pred[lidx][..., 4]) # [b, 3, h, w]
            # teacher prediction
            loss_conf = self.criterion(pseudo_t_pconf, t_pconf) * anchor_loss_weights
            loss += loss_conf.mean() * lw * balance[lidx]
        return loss * tb, loss








