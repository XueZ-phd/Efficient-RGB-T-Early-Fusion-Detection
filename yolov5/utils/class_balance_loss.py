import torch
import torch.nn as nn

class CBLoss(nn.Module):
    def __init__(self, loss_fcn, gamma=1.0, beta=5, alpha=0.25):
        super(CBLoss, self).__init__()
        self.loss_fcn = loss_fcn  # must be nn.BCEWithLogitsLoss()
        self.gamma = gamma
        self.alpha = alpha
        self.beta = beta
        self.reduction = loss_fcn.reduction
        self.loss_fcn.reduction = 'none'  # required to apply FL to each element

    def forward(self, pred, true):
        loss = self.loss_fcn(pred, true)

        # class balance weight
        n, c = pred.shape
        assert true.unique().cpu().numpy().tolist() == [0.0, 1.0], print(true.unique())
        assert true.sum() == n
        cls_dis = true.sum(0, keepdim=True) # [1, c]
        # mean_dis = cls_dis.mean()
        # std_dis = cls_dis.std()
        # cls_weight = torch.softmax((mean_dis - cls_dis) / std_dis, -1) # [1, c]
        # cls_weight = n / cls_dis.clamp(0.01)
        cls_prob = 1 - cls_dis / n  # [1, c]
        cls_weight = torch.exp(self.gamma * cls_prob) # [1, c]
        cls_weight = cls_weight / cls_weight.norm()
        weights = torch.sum(true * cls_weight, 1, keepdim=True) # [n, 1]

        loss = weights * loss #.sum(1, keepdim=True) / (weights * loss).sum(1, keepdim=True) * loss

        pred_prob = torch.sigmoid(pred)  # prob from logits
        p_t = true * pred_prob + (1 - true) * (1 - pred_prob)
        modulating_factor = torch.exp(self.beta * (1.0 - p_t))
        loss *= modulating_factor

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:  # 'none'
            return loss


