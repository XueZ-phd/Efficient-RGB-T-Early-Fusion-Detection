import torch
import torch.nn as nn
import torch.nn.functional as F


class SmoothL1BCELoss(nn.Module):
    # Wraps Quality focal loss around existing loss_fcn(), i.e. criteria = FocalLoss(nn.BCEWithLogitsLoss(), gamma=1.5)
    def __init__(self, loss_fcn):
        super().__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.L1 = nn.L1Loss(reduction='none')
        # self.L1 = nn.SmoothL1Loss(reduction='none')
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, logits, probs):
        bce_loss = self.loss_fcn(logits, probs)
        sl1_loss = self.L1(logits.sigmoid(), probs)
        loss = sl1_loss * bce_loss

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


if __name__ == '__main__':
    import numpy as np
    import os
    os.environ['DISPLAY'] = 'localhost:12.0'
    import matplotlib
    matplotlib.use('TkAgg')
    import matplotlib.pyplot as plt

    compute_loss = SmoothL1BCELoss(nn.BCEWithLogitsLoss(reduction='none'))

    probs = torch.sigmoid(torch.arange(-1.0, 1.0, 0.1)).unsqueeze(-1).unsqueeze(-1)
    logits = torch.arange(-1.0, 1.0, 0.1).unsqueeze(-1).unsqueeze(-1)
    losses = torch.zeros([probs.shape[0], logits.shape[0]])
    for i, prob in enumerate(probs):
        prob.unsqueeze_(0).unsqueeze_(0)
        for j, logit in enumerate(logits):
            logit.unsqueeze_(0).unsqueeze_(0)
            losses[i, j] = compute_loss(logit, prob)

    # losses = compute_loss(logits, probs).mean(dim=(2, 3)).squeeze()
    plt.figure()
    ax = plt.axes(projection='3d')
    [X, Y] = np.meshgrid(probs.detach().cpu().numpy(), logits.sigmoid().detach().cpu().numpy())
    ax.plot_surface(X, Y, losses.detach().cpu().numpy())
    plt.show()