import torch.nn as nn
import torch.nn.functional as F

class SIMILARITY(nn.Module):
    def __init__(self):
        super(SIMILARITY, self).__init__()

    def forward(self, fms, fmt):
        sc = fms.shape[1]
        tc = fmt.shape[1]

