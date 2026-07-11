import shutil
import os
from tqdm import tqdm
# 读取文本文件中的路径
method = 'earlyFusion'
with open(f'/home/zx/rgbx-distillation/datasets/detection/M3FD_Comparisons/{method}/zxSplit_infrared_val.txt', 'r') as f:
    paths = f.readlines()

# 创建目标文件夹（如果不存在）
target_folder = f'/home/zx/rgbx-distillation/datasets/detection/M3FD_Comparisons/{method}/zxSplit_valImages'
os.makedirs(os.path.join(target_folder, 'infrared'), exist_ok=True)
os.makedirs(os.path.join(target_folder, 'visible'), exist_ok=True)

# 将每个图片复制到目标文件夹
for path in tqdm(paths, total=len(paths)):
    path = path.strip()  # 去除路径两端的空白字符
    if os.path.exists(path):
        shutil.copy(path, os.path.join(target_folder, 'infrared'))  # 复制文件到目标文件夹
        shutil.copy(path.replace('infrared', 'visible'), os.path.join(target_folder, 'visible'))  # 复制文件到目标文件夹
        print(f"已复制: {path}")
    else:
        print(f"文件不存在: {path}")
