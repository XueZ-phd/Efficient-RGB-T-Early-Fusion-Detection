import os
import os.path as osp
import shutil
from glob import glob
import random
import xml.etree.ElementTree as ET

import numpy as np

input_root = './M3FD/M3FD_Detection'
out_root = './M3FD_zxSceneSplit_yolov5Format'
classes = {'People': 0, 'Car': 1, 'Bus': 2, 'Motorcycle': 3, 'Lamp': 4, 'Truck': 5}
if osp.exists(out_root):
    shutil.rmtree(out_root)
os.makedirs(out_root, exist_ok=True)
os.makedirs(osp.join(out_root, 'images', 'visible', 'train'), exist_ok=True)
os.makedirs(osp.join(out_root, 'labels', 'visible', 'train'), exist_ok=True)
os.makedirs(osp.join(out_root, 'images', 'infrared', 'train'), exist_ok=True)
os.makedirs(osp.join(out_root, 'labels', 'infrared', 'train'), exist_ok=True)
os.makedirs(osp.join(out_root, 'images', 'visible', 'test'), exist_ok=True)
os.makedirs(osp.join(out_root, 'labels', 'visible', 'test'), exist_ok=True)
os.makedirs(osp.join(out_root, 'images', 'infrared', 'test'), exist_ok=True)
os.makedirs(osp.join(out_root, 'labels', 'infrared', 'test'), exist_ok=True)

train = np.loadtxt(osp.join('./m3fd-zxSplit/train.txt'), str)
test = np.loadtxt(osp.join('./m3fd-zxSplit/test.txt'), str)

train = [x.rsplit('.png')[0] for x in train]
test = [x.rsplit('.png')[0] for x in test]
for mode in ['train', 'test']:
    for idx in eval(mode):
        # images
        vis_img = osp.join(input_root, 'vi', f'{idx}.png')
        ir_img = osp.join(input_root, 'ir', f'{idx}.png')
        shutil.copy(vis_img, osp.join(out_root, 'images', 'visible', mode))
        shutil.copy(ir_img, osp.join(out_root, 'images', 'infrared', mode))
        # labels
        xml_root = ET.parse(osp.join(input_root, 'Annotation', f'{idx}.xml')).getroot()
        height = float(xml_root.find('size').find('height').text)
        width = float(xml_root.find('size').find('width').text)

        objs = xml_root.findall('object')
        with open(osp.join(out_root, 'labels', 'visible', mode, f'{idx}.txt'), 'a') as f:
            for obj in objs:
                xmin = float(obj.find('bndbox').find('xmin').text)
                xmax = float(obj.find('bndbox').find('xmax').text)
                ymin = float(obj.find('bndbox').find('ymin').text)
                ymax = float(obj.find('bndbox').find('ymax').text)
                xc = (xmin + xmax) / 2.0 / width
                yc = (ymin + ymax) / 2.0 / height
                w = (xmax - xmin) / width
                h = (ymax - ymin) / height
                if obj.find('name').text not in classes.keys(): continue
                cls = classes[obj.find('name').text]
                f.write(f'{cls} {xc} {yc} {w} {h}\n')
        shutil.copy(osp.join(out_root, 'labels', 'visible', mode, f'{idx}.txt'), osp.join(out_root, 'labels', 'infrared', mode))





