# import random
#
# import numpy as np
# import torch

from mmdet.registry import MODELS

from .retinanet_clip import RetinaNetCLIP
# import torch.nn as nn
# import clip
# from typing import Union, Tuple, List
# import torch.nn.functional as F
# from torch import Tensor
# from mmdet.structures import SampleList
# from .retinanet_clip import clip_train_loss
# from .gfl import GFL

@MODELS.register_module()
class GFLCLIP(RetinaNetCLIP):
    def __init__(self, **kwargs):
        super(GFLCLIP, self).__init__(**kwargs)

