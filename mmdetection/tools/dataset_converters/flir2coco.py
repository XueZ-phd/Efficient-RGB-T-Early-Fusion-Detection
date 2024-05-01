import argparse
import copy
import os.path as osp
from glob import glob
import numpy as np
import os
import warnings

import mmcv
import mmengine
from PIL import Image


''''
LWIR和VISIBLE共享同一套Annotations
该代码根据Annotations从文件中索引LWIR的图片，并根据路径索引相匹配的VISIBLE图片
最后只保存LWIR的json文件，即coco['file_name'] = path/to/lwir/image
在mmdetection/mmdet/datasets/pipelines/my_load_rgbt_pipeline.py中，我根据file_name索引lwir image，并匹配相应的visible图片
LLVIP中两个模态原始的文件夹名称为infrared和visible，为了和my_load_rgbt_pipeline.py中的代码匹配，我将infrared重命名为lwir。
'''

label_ids = dict([(k, v) for (v, k) in enumerate(['Person', 'Bicycle', 'Car'])])

def xywhn2xyxy(x, w=640, h=640, padw=0, padh=0):
    # Convert nx4 boxes from [x, y, w, h] normalized to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right
    y = np.copy(x)
    y[..., 0] = w * (x[..., 0] - x[..., 2] / 2) + padw  # top left x
    y[..., 1] = h * (x[..., 1] - x[..., 3] / 2) + padh  # top left y
    y[..., 2] = w * (x[..., 0] + x[..., 2] / 2) + padw  # bottom right x
    y[..., 3] = h * (x[..., 1] + x[..., 3] / 2) + padh  # bottom right y
    return y

def parse(args):
    label_path, img_path = args
    img = mmcv.imread(img_path)
    h, w = img.shape[:2]
    bboxes = np.loadtxt(label_path, float, delimiter=' ', usecols=[1, 2, 3, 4], ndmin=2)
    labels = np.loadtxt(label_path, int, delimiter=' ', usecols=0, ndmin=1)
    bboxes = xywhn2xyxy(bboxes, w, h)
    bboxes = np.ceil(bboxes)
    bboxes_ignore = np.zeros((0, 4))
    labels_ignore = np.zeros((0, ))
    annotation = {
        'filename': img_path,
        'width': w,
        'height': h,
        'ann': {
            'bboxes': bboxes.astype(np.float32),
            'labels': labels.astype(np.int64),
            'bboxes_ignore': bboxes_ignore.astype(np.float32),
            'labels_ignore': labels_ignore.astype(np.int64)
        }
    }
    return annotation


def cvt_annotations(in_path, dataset_name, split, out_file):
    assert split in dataset_name
    img_paths = sorted(glob(osp.join(in_path, dataset_name, '*.png')))
    if 'train' in dataset_name:
        val_img_paths = sorted(glob(osp.join(in_path, dataset_name.replace('train', 'val'), '*.png')))
        assert len(val_img_paths)
        img_paths = img_paths + val_img_paths
    labels_paths = [
        img_name.replace('images', 'labels').replace('png', 'txt')
        for img_name in img_paths
    ]

    if split == 'train':
        assert len(img_paths) == len(labels_paths)
    elif split == 'test':
        assert len(img_paths) == len(labels_paths)

    annotations = mmengine.utils.progressbar.track_progress(parse,
                                           list(zip(labels_paths, img_paths)))
    if out_file.endswith('json'):
        annotations = cvt_to_coco_json(annotations)
    mmengine.fileio.io.dump(annotations, out_file)
    return annotations


def cvt_to_coco_json(annotations):
    image_id = 0
    annotation_id = 0
    coco = dict()
    coco['images'] = []
    coco['type'] = 'instance'
    coco['categories'] = []
    coco['annotations'] = []
    image_set = set()

    def addAnnItem(annotation_id, image_id, category_id, bbox, difficult_flag):
        annotation_item = dict()
        annotation_item['segmentation'] = []

        seg = []
        # bbox[] is x1,y1,x2,y2
        # left_top
        seg.append(int(bbox[0]))
        seg.append(int(bbox[1]))
        # left_bottom
        seg.append(int(bbox[0]))
        seg.append(int(bbox[3]))
        # right_bottom
        seg.append(int(bbox[2]))
        seg.append(int(bbox[3]))
        # right_top
        seg.append(int(bbox[2]))
        seg.append(int(bbox[1]))

        annotation_item['segmentation'].append(seg)

        xywh = np.array(
            [bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]])
        annotation_item['area'] = int(xywh[2] * xywh[3])
        if difficult_flag == 1:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 1
        else:
            annotation_item['ignore'] = 0
            annotation_item['iscrowd'] = 0
        annotation_item['image_id'] = int(image_id)
        annotation_item['bbox'] = xywh.astype(int).tolist()
        annotation_item['category_id'] = int(category_id)
        annotation_item['id'] = int(annotation_id)
        coco['annotations'].append(annotation_item)
        return annotation_id + 1

    for category_id, name in enumerate(list(label_ids.keys())):
        category_item = dict()
        category_item['supercategory'] = str('none')
        category_item['id'] = int(category_id)
        category_item['name'] = str(name)
        coco['categories'].append(category_item)

    for ann_dict in annotations:
        file_name = ann_dict['filename']
        ann = ann_dict['ann']
        assert file_name not in image_set
        image_item = dict()
        image_item['id'] = int(image_id)
        image_item['file_name'] = str(file_name)
        image_item['height'] = int(ann_dict['height'])
        image_item['width'] = int(ann_dict['width'])
        coco['images'].append(image_item)
        image_set.add(file_name)

        bboxes = ann['bboxes'][:, :4]
        labels = ann['labels']
        for bbox_id in range(len(bboxes)):
            bbox = bboxes[bbox_id]
            label = labels[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=0)

        bboxes_ignore = ann['bboxes_ignore'][:, :4]
        labels_ignore = ann['labels_ignore']
        for bbox_id in range(len(bboxes_ignore)):
            bbox = bboxes_ignore[bbox_id]
            label = labels_ignore[bbox_id]
            annotation_id = addAnnItem(
                annotation_id, image_id, label, bbox, difficult_flag=1)

        image_id += 1

    return coco


def parse_args():
    parser = argparse.ArgumentParser(
        description='Convert FLIR annotations to mmdetection format')
    parser.add_argument('--in_path', default='/home/zx/rgbx-distillation/datasets/detection/FLIR_yolo',
                        help='FLIR path')
    parser.add_argument('--out-dir', default='/home/zx/rgbx-distillation/datasets/detection/FLIR_coco',
                        help='output path')
    args = parser.parse_args()
    return args


def main():
    args = parse_args()
    in_path = args.in_path
    out_dir = args.out_dir
    mmengine.utils.path.mkdir_or_exist(out_dir)

    for split in ['train', 'test']:
        dataset_name = osp.join('images', 'infrared', split)
        print(f'processing {dataset_name} ...')
        cvt_annotations(in_path, dataset_name, split,
                        osp.join(out_dir, f'infrared_{split}'+ '.json'))
    print('Done!')

if __name__ == '__main__':
    main()