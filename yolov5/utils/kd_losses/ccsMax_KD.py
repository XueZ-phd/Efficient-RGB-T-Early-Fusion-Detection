import torch
import torch.nn as nn
import torch.nn.functional as F


class CCSMAX_KD(nn.Module):
    def __init__(self, in_channels, loss_weights, device, reduction=16):
        super(CCSMAX_KD, self).__init__()
        self.device = device
        if not isinstance(loss_weights, list):
            loss_weights = list(loss_weights)
        assert len(loss_weights) == 1 or len(loss_weights) == len(in_channels)
        if len(loss_weights) == 1:
            loss_weights *= len(in_channels)
        self.loss_weights = loss_weights

        for i, in_c in enumerate(in_channels):
            self.add_module(f'proj_layer_{i}',
                            nn.Sequential(
                                nn.Conv2d(in_c, in_c, 1, bias=False),
                                # nn.Conv2d(in_c, in_c // reduction, 1, bias=False),
                                # nn.ReLU(),
                                # nn.Conv2d(in_c // reduction, in_c, 1, bias=False),
                                nn.Sigmoid()
                            ))

    def neg_corr(self, prob):
        return F.binary_cross_entropy(prob, torch.ones_like(prob, requires_grad=False, device=prob.device),
                                      reduction='mean')

    def forward(self, feats_student, feats_teacher, rgb_images, thermal_images):
        assert len(feats_student) == len(feats_teacher) == len(self.loss_weights)
        losses = torch.zeros(len(feats_student), device=self.device)
        for i, (f_s, f_t, l_w) in enumerate(zip(feats_student, feats_teacher, self.loss_weights)):
            att_teacher = torch.sum(F.softmax(f_t, 1) * f_t, dim=1, keepdim=True)   # [b, 1, h, w]
            ccs = F.cosine_similarity(f_s.unsqueeze(2).flatten(2),  # [b, c, 1, h, w] -> [b, c, h * w]
                                      att_teacher.unsqueeze(1).flatten(2),  # [b, 1, 1, h, w] -> [b, 1, h * w]
                                      dim=2).unsqueeze(-1).unsqueeze(-1)    # [b, c, 1, 1]
            l = getattr(self, f'proj_layer_{i}')
            prob = l(ccs)
            losses[i] = self.neg_corr(prob) * l_w
        return losses


