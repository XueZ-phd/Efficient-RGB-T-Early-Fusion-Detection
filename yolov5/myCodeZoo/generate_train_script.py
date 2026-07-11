from glob import glob
import os.path as osp

import tqdm

data_root = '/home/zx/rgbx-distillation/code/yolov5-master/data/sota_fusions'
files = sorted(glob(osp.join(data_root, "*.yaml")))
files = [x for x in files if 'earlyFusion' not in x]
tasks_per_gpu = 2
epochs = 36
batchsize = 32

for idx, file in tqdm.tqdm(enumerate(files), total=len(files)):
    sota_approach = osp.basename(file).rsplit('-')[-1].split('.yaml')[0]
    if idx % tasks_per_gpu == 0:
        sh_file = open(f'./train_sh{idx // tasks_per_gpu}.sh', 'w')
        ii = 0
    sh_file.write(f'# train {sota_approach}\n')
    sh_file.write(f'python train.py '
                  f'--device {idx // tasks_per_gpu} '
                  f'--weights yolov5s.pt '
                  f'--data data/sota_fusions/{osp.basename(file)} '
                  f'--hyp data/hyps/hyp.scratch-low.yaml '
                  f'--epochs {epochs} '
                  f'--batch-size {batchsize} '
                  f'--img 640 '
                  f'--name m3fd_zxSplit_sotafusion_v5small_e{epochs}_bs{batchsize}_{sota_approach} '
                  f'--exist-ok\n')
    ii += 1
    if ii == tasks_per_gpu:
        sh_file.close()



