import glob
import os

import torch

os.environ['DISPLAY'] = 'localhost:10.0'
import matplotlib
matplotlib.use('TkAgg')
import matplotlib.pyplot as plt
import torch.nn.functional as F
import cv2
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import manifold


def plot_attention(feat, img_id, reduction='softmax', images=None):
    if reduction is 'softmax':
        weight = F.softmax(feat, dim=1)
        att = torch.sum(weight * feat, dim=1)
    elif reduction is 'mean':
        att = torch.mean(feat, dim=1)
    elif reduction is 'sum':
        att = torch.sum(feat, dim=1)
    plt.figure()
    if images is not None:
        rgbs, thermals = images
        h, w = rgbs.shape[2:]
        att = cv2.resize(att[img_id].detach().cpu().numpy(), (w, h))
        plt.imshow(np.hstack([rgbs[img_id].permute((1, 2, 0)).detach().cpu().numpy(),
                              thermals[img_id].permute((1, 2, 0)).detach().cpu().numpy()]))
        plt.imshow(np.hstack([att, att]), alpha=0.3)
    else:
        plt.imshow(att[img_id].detach().cpu().numpy())
    plt.show()


def plotTeacherStudentPred(teacher_preds, student_preds, part, scale, img_id, anchor_id, rgbs, thermals, save_path=None):
    assert part in ['obj', 'cls']
    rgb = rgbs[img_id].permute((1, 2, 0)).detach().cpu().numpy()
    thermal = thermals[img_id].permute((1, 2, 0)).detach().cpu().numpy()
    h, w = rgb.shape[:2]
    teacher_preds = teacher_preds
    student_preds = student_preds
    tpred = teacher_preds[-1][scale][img_id].sigmoid()
    spred = student_preds[-1][scale][img_id].sigmoid()
    if part is 'obj':
        tp = tpred[anchor_id, :, :, 4].detach().cpu().numpy()
        sp = spred[anchor_id, :, :, 4].detach().cpu().numpy()
        tp = cv2.resize(tp, (w, h))
        sp = cv2.resize(sp, (w, h))
    elif part is 'cls':
        tmask = torch.zeros_like(tpred[anchor_id, :, :, 4])
        smask = torch.zeros_like(spred[anchor_id, :, :, 4])
        tmask[tpred[anchor_id, :, :, 4] >= 0.1] = 1.0
        smask[spred[anchor_id, :, :, 4] >= 0.1] = 1.0
        np.random.seed(0)
        palette = {k: v for k, v in zip([x for x in range(7)],
                                        np.vstack([np.zeros((1, 3)),
                                                   np.array([1.0, 0.0, 0.0]),
                                                   np.array([0.0, 1.0, 0.0]),
                                                   np.array([0.0, 0.0, 1.0]),
                                                   np.array([1.0, 1.0, 0.0]),
                                                   np.array([0.3, 0.7, 0.5]),
                                                   np.array([0.1, 0.6, 0.9]),
                                                   ]))}
        tp = (torch.argmax(tpred[anchor_id, :, :, 5:] * tpred[anchor_id, :, :, 4:5], -1) + 1) * tmask
        sp = (torch.argmax(spred[anchor_id, :, :, 5:] * spred[anchor_id, :, :, 4:5], -1) + 1) * smask
        # numpy
        tp, sp = tp.detach().cpu().numpy().astype('int'), sp.detach().cpu().numpy().astype('int')
        fh, fw = tp.shape
        t_color_map, s_color_map = np.zeros((fh, fw, 3)), np.zeros((fh, fw, 3))
        # print(t_color_map.shape)
        for j in np.unique(tp):
            t_color_map[tp == j] = palette[j]
        for j in np.unique(sp):
            s_color_map[sp == j] = palette[j]
        tp, sp = cv2.resize(t_color_map, (w, h)), cv2.resize(s_color_map, (w, h))
    plt.figure()
    plt.subplot(2, 1, 1)
    plt.imshow(np.hstack([rgb, thermal]))
    plt.imshow(np.hstack([tp, tp]), alpha=0.5, cmap='Reds', vmax=1.0, vmin=0.0)
    plt.axis('off')
    plt.title(f'teacher {part} scale {scale} anchor {anchor_id}')
    plt.subplot(2, 1, 2)
    plt.imshow(np.hstack([rgb, thermal]))
    plt.imshow(np.hstack([sp, sp]), alpha=0.5, cmap='Reds', vmax=1.0, vmin=0.0)
    plt.axis('off')
    plt.title(f'student {part} scale {scale} anchor {anchor_id}')
    if save_path is None:
        plt.show()
    else:
        img_n = len(glob.glob(os.path.join(save_path, '*.png')))
        plt.savefig(os.path.join(save_path, f'{img_n:04d}_scale{str(scale)}_anchor{str(anchor_id)}_{part}.png'),
                    bbox_inches='tight', pad_inches=0.0)
        plt.close()


def plotTeacherStudentBoxeSNE(teacher_preds, student_preds, afterNMS=True, save_path=None):
    assert afterNMS
    t_preds = torch.cat(teacher_preds, 0)
    t_boxes = t_preds[:, :4]
    t_cls = t_preds[:, 5:6]
    s_preds = torch.cat(student_preds, 0)
    s_boxes = s_preds[:, :4]
    s_cls = s_preds[:, 5:6]
    print(f'# teacher boxes: {len(t_boxes)}')
    print(f'# student boxes: {len(s_boxes)}')
    # numpy
    t_boxes = t_boxes.detach().cpu().numpy()
    t_cls = t_cls.detach().cpu().numpy()
    s_boxes = s_boxes.detach().cpu().numpy()
    s_cls = s_cls.detach().cpu().numpy()
    boxes = np.row_stack((t_boxes, s_boxes))
    classes = np.row_stack((t_cls, s_cls))
    models = np.row_stack((np.zeros((len(t_cls), 1)), np.ones((len(s_cls), 1))))
    # TSNE
    tsne = manifold.TSNE(n_components=2, random_state=42, init='pca')
    transformed_data = tsne.fit_transform(boxes)
    transformed_data[:, 0] = (transformed_data[:, 0] - np.min(transformed_data[:, 0])) / (np.max(transformed_data[:, 0]) - np.min(transformed_data[:, 0]))
    transformed_data[:, 1] = (transformed_data[:, 1] - np.min(transformed_data[:, 1])) / (np.max(transformed_data[:, 1]) - np.min(transformed_data[:, 1]))
    tsne_df = pd.DataFrame(
        np.column_stack((transformed_data, models, classes)),
        columns=["x", "y", "models", "classes"]
    )
    tsne_df.loc[:, "classes"] = tsne_df.classes.astype(int)
    tsne_df.loc[:, "models"] = tsne_df.models.astype(int)
    grid = sns.FacetGrid(tsne_df, hue="models", height=8, palette=['red', 'blue'], hue_kws={'marker': ['o', 'X'],
                                                                                            'alpha': [1.0, 0.3]})
    grid.map(sns.scatterplot, "x", "y")
    grid.add_legend()
    plt.show()














