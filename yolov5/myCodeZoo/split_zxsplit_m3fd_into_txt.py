import os
import os.path as osp
import shutil

import numpy as np
from glob import glob
import random
from tqdm import tqdm

m3fd_zxsplit_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/infrared'

train_files = sorted(glob(osp.join(m3fd_zxsplit_path, 'train', '*.png')))
val_files = sorted(glob(osp.join(m3fd_zxsplit_path, 'test', '*.png')))
assert len(train_files) + len(val_files) == 4200

sota_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_Comparisons'

sota_approaches = sorted(os.listdir(sota_path))
sota_approaches.remove('earlyFusion')
print(f'sota approaches: {len(sota_approaches)} | {sota_approaches}')

# sota approaches
for app in sota_approaches:
    out_path = osp.join(sota_path, app)
    with open(osp.join(out_path, 'zxSplit_train.txt'), 'w') as f:
        for file in train_files:
            f.write(f'{osp.join(out_path, "images", osp.basename(file))}\n')
    with open(osp.join(out_path, 'zxSplit_val.txt'), 'w') as f:
        for file in val_files:
            f.write(f'{osp.join(out_path, "images", osp.basename(file))}\n')

# early fusion
out_path = osp.join(sota_path, 'earlyFusion')
for mo in ['infrared', 'visible']:
    with open(osp.join(out_path, f'zxSplit_{mo}_train.txt'), 'w') as f:
        for file in train_files:
            f.write(f'{osp.join(out_path, "images", mo, osp.basename(file))}\n')
    with open(osp.join(out_path, f'zxSplit_{mo}_val.txt'), 'w') as f:
        for file in val_files:
            f.write(f'{osp.join(out_path, "images", mo, osp.basename(file))}\n')