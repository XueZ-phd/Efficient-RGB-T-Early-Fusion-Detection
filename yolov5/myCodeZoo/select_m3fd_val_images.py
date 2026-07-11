import os
import os.path as osp
import shutil
from glob import glob

import numpy as np

mode = 'test'
root = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format'
label_path = f'labels/visible/{mode}'

vis_demo_root = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/demo'
ther_demo_root = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/infrared/demo'
if osp.exists(vis_demo_root):
    shutil.rmtree(vis_demo_root)
    shutil.rmtree(ther_demo_root)
    shutil.rmtree(ther_demo_root.replace('images', 'labels'))
    shutil.rmtree(vis_demo_root.replace('images', 'labels'))
os.makedirs(vis_demo_root)
os.makedirs(ther_demo_root)
os.makedirs(ther_demo_root.replace('images', 'labels'))
os.makedirs(vis_demo_root.replace('images', 'labels'))

label_files = sorted(glob(osp.join(root, label_path, '*')))

for lf in label_files:
    labels = np.unique(np.loadtxt(lf, int, delimiter=' ', usecols=0, ndmin=1))
    # if not (2 in labels or 3 in labels or 4 in labels or 5 in labels):
    if not 4 in labels:
        continue
    vis_img_file = osp.join(root, 'images', 'visible', mode, osp.basename(lf).replace('txt', 'png'))
    thermal_img_file = osp.join(root, 'images', 'infrared', mode, osp.basename(lf).replace('txt', 'png'))
    assert osp.isfile(vis_img_file) and osp.isfile(thermal_img_file)
    shutil.copy(vis_img_file, vis_demo_root)
    shutil.copy(thermal_img_file, ther_demo_root)
    shutil.copy(lf, vis_demo_root.replace('images', 'labels'))
    shutil.copy(lf.replace('visible', 'infrared'), ther_demo_root.replace('images', 'labels'))
