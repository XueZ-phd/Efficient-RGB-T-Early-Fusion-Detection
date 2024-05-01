# Copyright (c) OpenMMLab. All rights reserved.

from mmdet.registry import DATASETS
from .coco import CocoDataset
from .api_wrappers import COCO

@DATASETS.register_module()
class M3FDDataset(CocoDataset):
    """Dataset for M3FD."""

    METAINFO = {
        'classes': ('People', 'Car', 'Bus', 'Motorcycle', 'Lamp', 'Truck'),
        'palette': [
            (220, 20, 60), (255, 77, 255), (0, 0, 142), (0, 0, 230), (106, 0, 228), (0, 60, 100)],
    }
    COCOAPI = COCO
    # ann_id is unique in coco dataset.
    ANN_ID_UNIQUE = True

