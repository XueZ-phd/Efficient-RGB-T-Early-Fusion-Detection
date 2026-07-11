import torch
from utils.general import xywhn2xyxy

def generateMask(imgs, targets):
    b, c, h, w = imgs.shape
    masks = torch.zeros([b, 1, h, w], device=imgs.device) + 0.1
    bs_ids = torch.unique(targets[:, 0])
    for i in bs_ids:
        labels = targets[targets[:, 0] == i]
        labels = xywhn2xyxy(labels[:, 2:6], w, h)
        for l in labels:
            x0, y0, x1, y1 = l
            masks[int(i), 0, int(y0):int(y1), int(x0):int(x1)] = 1.0
    return masks