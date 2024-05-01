from .resnet import ResNet
from mmdet.registry import MODELS
import torch
import torch.nn as nn
import torch.nn.functional as F
from kornia.filters import Sobel
from kornia.metrics import SSIM

class RGBTStem(nn.Module):
    def __init__(self, c2):
        super(RGBTStem, self).__init__()
        self.sobel = Sobel()
        self.max_edge = nn.MaxPool3d((2, 3, 3), stride=(1, 1, 1), padding=(0, 1, 1))
        self.ssim = SSIM(7)
        self.softmax = nn.Softmax(dim=1)
        self.conv = nn.Conv2d(6, 3, 7, 1, 3, bias=False)
    def forward(self, x):
        rgb, t = torch.chunk(x, 2, 1)
        rgb_edge, t_edge = self.sobel(rgb.mean(1, keepdim=True)), self.sobel(t.mean(1, keepdim=True))
        edge = torch.cat([rgb_edge.unsqueeze(2), t_edge.unsqueeze(2)], 2)   # [b, 1, 2, h, w]
        edge = self.max_edge(edge).squeeze(2)
        rgb_ssim_map = self.ssim(F.normalize(edge, dim=(2, 3)), F.normalize(rgb_edge, dim=(2, 3)))
        t_ssim_map = self.ssim(F.normalize(edge, dim=(2, 3)), F.normalize(t_edge, dim=(2, 3)))
        ssim_map = self.softmax(torch.cat([rgb_ssim_map, t_ssim_map], 1))
        x = torch.cat([rgb * ssim_map[:, 0:1], t * ssim_map[:, 1:2]], 1)
        x = self.conv(x)    # [b, c, h, w]
        return x


@MODELS.register_module()
class ResNetRGBTEarlyModifiedStem(ResNet):
    def __init__(self, **kwargs):
        c1 = kwargs['in_channels']
        assert c1 == 3
        super(ResNetRGBTEarlyModifiedStem, self).__init__(**kwargs)
        c2 = self.stem_channels
        assert not hasattr(self, 'stem') and hasattr(self, 'conv1')
        # assert self.frozen_stages < 0
        self.stem = RGBTStem(c2)

    def forward(self, x):
        """Forward function."""
        h, w = x.shape[2:]
        x = self.stem(x)
        assert x.shape[2:] == (h, w)

        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)
        outs = []
        for i, layer_name in enumerate(self.res_layers):
            res_layer = getattr(self, layer_name)
            x = res_layer(x)
            if i in self.out_indices:
                outs.append(x)
        return tuple(outs)
