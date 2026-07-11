import os
import os.path as osp
import shutil
from glob import glob
import numpy as np
from tqdm import tqdm

data_root_dir = '/home/zx/rgbx-distillation/datasets/detection'
out_dir = osp.join(data_root_dir, 'm3fd_rgbxfusion_split')

# out path
out_img_path = osp.join(out_dir, 'images')
out_label_path = osp.join(out_dir, 'labels')
if osp.exists(out_img_path):
    shutil.rmtree(out_img_path)
    shutil.rmtree(out_label_path)
for modality in ['visible', 'infrared']:
    for mode in ['train', 'val', 'test']:
        os.makedirs(osp.join(out_label_path, modality, mode), exist_ok=True)
        os.makedirs(osp.join(out_img_path, modality, mode), exist_ok=True)


train_split_path = osp.join(out_dir, 'splits', 'challenge_train.txt') # overcast_train | train | challenge_train
val_split_path = osp.join(out_dir, 'splits', 'challenge_val.txt')    # val | overcast_val | challenge_val
test_split_path = osp.join(out_dir, 'splits', 'challenge_test.txt')   # test | overcast_test | challenge_test
train_names, val_names, test_names = np.loadtxt(train_split_path, str), np.loadtxt(val_split_path, str), np.loadtxt(test_split_path, str)
print(f'Train: {len(train_names)} | Val: {len(val_names)} | Test: {len(test_names)} | Total: {len(train_names) + len(val_names) + len(test_names)}')

all_files = set(np.concatenate([train_names, val_names, test_names]))
assert len(all_files) == len(train_names) + len(val_names) + len(test_names)

for mode in ['train', 'val', 'test']:
    for file in tqdm(eval(f'{mode}_names'), desc=f'processing {mode} ...'):
        query_dir = osp.join(data_root_dir, 'M3FD_zxSceneSplit_yolov5Format')
        ir_img = glob(osp.join(query_dir, 'images', 'infrared', 'train', file.strip() + '.png')) +\
                 glob(osp.join(query_dir, 'images', 'infrared', 'test', file.strip() + '.png'))
        assert len(ir_img) == 1
        # source image & label
        ir_img = ir_img[0]
        vis_img = ir_img.replace('infrared', 'visible')
        label = ir_img.replace('images', 'labels').replace('png', 'txt')
        assert osp.isfile(ir_img) and osp.isfile(vis_img) and osp.isfile(label)
        # out image & label
        shutil.copy(ir_img, osp.join(out_img_path, 'infrared', mode))
        shutil.copy(vis_img, osp.join(out_img_path, 'visible', mode))
        shutil.copy(label, osp.join(out_label_path, 'infrared', mode))
        shutil.copy(label, osp.join(out_label_path, 'visible', mode))


