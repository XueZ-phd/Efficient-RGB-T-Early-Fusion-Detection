import os
import os.path as osp
import shutil
from glob import glob

import tqdm

sota_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_Comparisons'
sota_approaches = sorted(glob(osp.join(sota_path, '*')))
sota_approaches.remove(osp.join(sota_path, 'earlyFusion'))
assert len(sota_approaches) == 10

for sota_app in sota_approaches:
    detect_val_img_path = osp.join(sota_app, 'detect_val_images')
    if osp.exists(detect_val_img_path):
        shutil.rmtree(detect_val_img_path)
        os.makedirs(detect_val_img_path, exist_ok=True)
    else:
        os.makedirs(detect_val_img_path, exist_ok=True)
    with open(osp.join(sota_app, 'zxSplit_val.txt'), 'r') as f:
        lines = f.readlines()
        for line in tqdm.tqdm(lines, total=len(lines)):
            shutil.copy(line.strip(), detect_val_img_path)
