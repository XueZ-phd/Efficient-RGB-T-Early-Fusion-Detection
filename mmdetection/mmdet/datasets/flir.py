# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO

@DATASETS.register_module()
class FLIRDataset(CocoDataset):
    """Dataset for FLIR."""

    METAINFO = {
        'classes': ('Person', 'Bicycle', 'Car'),
        'palette': [
            (220, 20, 60), (255, 77, 255), (0, 0, 142), ],
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

