from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
from .at import AT


class GKD(nn.Module):
    def __init__(self, num_group, p):
        super(GKD, self).__init__()
        self.num_group = num_group
        self.at = AT(p)

    def forward(self, student_feat, teacher_feat):
        split_stu_feats = torch.chunk(student_feat, self.num_group, 1)
        split_tea_feats = torch.chunk(teacher_feat, self.num_group, 1)
        for idx, (stu_f, tea_f) in enumerate(zip(split_stu_feats, split_tea_feats)):
            if idx == 0:
                loss = self.at(stu_f, tea_f)
            else:
                loss = loss + self.at(stu_f, tea_f)
        return loss / self.num_group

