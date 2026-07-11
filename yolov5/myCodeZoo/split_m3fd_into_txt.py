import os
import os.path as osp
import shutil

import numpy as np
from glob import glob
import random
from tqdm import tqdm

random.seed(0)
state = random.getstate()

m3fd_fusion_data_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_Comparisons'
sota_approaches = os.listdir(m3fd_fusion_data_path)
print(f'sota approaches: total number {len(sota_approaches)} | {sota_approaches}')

# inlabels_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/labels'
# label_files = sorted(glob(osp.join(inlabels_path, "infrared", 'train', '*.txt'))) + sorted(glob(osp.join(inlabels_path, "infrared", 'test', '*.txt')))
# assert len(set([osp.basename(x) for x in label_files])) == 4200
#
# for sota_app in sota_approaches:
#     label_path = osp.join(m3fd_fusion_data_path, sota_app, 'labels')
#     shutil.rmtree(label_path)
#     os.makedirs(label_path, exist_ok=True)
#     for la in tqdm(label_files, total=len(label_files)):
#         shutil.copy(la, label_path)

train_fraction = 0.8

modality = 'visible'
image_files = sorted(glob(osp.join(m3fd_fusion_data_path, 'earlyFusion', 'images', modality, '*.png')))
assert len(image_files) == 4200
random.setstate(state)
random.shuffle(image_files)
#
num_train = 0
num_val = 0
train_txt = open(osp.join(m3fd_fusion_data_path, 'earlyFusion', f'{modality}_train.txt'), "w+")
val_txt = open(osp.join(m3fd_fusion_data_path, 'earlyFusion', f'{modality}_val.txt'), "w+")
for ii, file in tqdm(enumerate(image_files), total=len(image_files), desc=f'processing {"earlyFusion"+"...":10}'):
    if ii <= int(len(image_files) * train_fraction):
        train_txt.write(f'{file}\n')
        num_train += 1
    else:
        val_txt.write(f'{file}\n')
        num_val += 1
train_txt.close()
val_txt.close()
print(f'train: {num_train} | val: {num_val}')



# for idx, sota in enumerate(sota_approaches):
#     image_files = sorted(glob(osp.join(m3fd_fusion_data_path, sota, 'images', '*.png')))
#     assert len(image_files) == 4200
#     random.setstate(state)
#     random.shuffle(image_files)
#     #
#     if idx == 0:
#         temp_files = image_files
#     else:
#         assert len(temp_files) == len(image_files)
#         assert all([os.path.basename(x)==os.path.basename(y) for x, y in zip(temp_files, image_files)])
#     #
#     num_train = 0
#     num_val = 0
#     train_txt = open(osp.join(m3fd_fusion_data_path, sota, 'train.txt'), "w+")
#     val_txt = open(osp.join(m3fd_fusion_data_path, sota, 'val.txt'), "w+")
#     for ii, file in tqdm(enumerate(image_files), total=len(image_files), desc=f'processing {sota+"...":10}'):
#         if ii <= int(len(image_files) * train_fraction):
#             train_txt.write(f'{file}\n')
#             num_train += 1
#         else:
#             val_txt.write(f'{file}\n')
#             num_val += 1
#     train_txt.close()
#     val_txt.close()
#     print(f'train: {num_train} | val: {num_val}')