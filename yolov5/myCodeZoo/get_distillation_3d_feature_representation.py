import os

import cv2
import numpy as np

from models.experimental import attempt_load
from models.yolo_distillation import Model
from models.yolo_mediumFuse_notSharing import Model as MidFusionModel
import torch
from utils.torch_utils import select_device
from utils.general import intersect_dicts
from utils.general import LOGGER
from utils.kd_losses import *
import matplotlib
import matplotlib.pyplot as plt
import torch.nn.functional as F

os.environ['DISPLAY'] = 'localhost:10.0'
matplotlib.use('TkAgg')

device = select_device(device='7', batch_size=1)
weights = '../runs/train/debug1/weights/best.pt'
ckpt = torch.load(weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
model = Model(ckpt['model'].yaml, ch=6, nc=6, anchors=None).to(device)  # create
exclude = []  # exclude keys
csd = ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
csd = intersect_dicts(csd, model.state_dict(), exclude=exclude)  # intersect
model.load_state_dict(csd, strict=False)  # load
LOGGER.info(f'Transferred {len(csd)}/{len(model.state_dict())} items from {weights}')  # report

# ------------------------------------------------------------------------------
# Teacher YOLOv5-Large Medium Fusion
teacher_weights = '../runs/train/M3FDsceneSplit_RGBTMediumNotSharingCustomLR_large_bs32_epoch36_cocoPretrain_convWeightConv3dBNSiluDwBNSilu_t4xLR/weights/best.pt'
teacher_ckpt = torch.load(teacher_weights, map_location='cpu')  # load checkpoint to CPU to avoid CUDA memory leak
teacher_model = MidFusionModel(teacher_ckpt['model'].yaml, ch=3, nc=6, anchors=None).to(device)  # create
teacher_csd = teacher_ckpt['model'].float().state_dict()  # checkpoint state_dict as FP32
teacher_csd = intersect_dicts(teacher_csd, teacher_model.state_dict())  # intersect
teacher_model.load_state_dict(teacher_csd, strict=True)  # load
LOGGER.info(
    f'Teacher Transferred {len(teacher_csd)}/{len(teacher_model.state_dict())} items from {teacher_weights}')  # report
# Freeze Teacher
for teak, teav in teacher_model.named_parameters():
    teav.requires_grad = False  # freeze all layers of teacher model
teacher_head = teacher_model.model[-1].m
model.eval()
teacher_model.eval()


def norm(feat):
    am = torch.pow(torch.abs(feat), 2.0)
    norm = torch.norm(am, dim=(2, 3), keepdim=True)
    am = torch.div(am, norm + 1e-8)
    return am


#----------------------
file_path = '/home/zx/rgbx-distillation/datasets/detection/M3FD_zxSceneSplit_yolov5Format/images/visible/train'
img_name = '00001.png'
rgb_file = os.path.join(file_path, img_name)
thermal_file = os.path.join(file_path.replace('visible', 'infrared'), img_name)
rgbs = torch.from_numpy(cv2.cvtColor(cv2.imread(rgb_file), cv2.COLOR_BGR2RGB))
thermals = torch.from_numpy(cv2.cvtColor(cv2.imread(thermal_file), cv2.COLOR_BGR2RGB))
rgbs = torch.unsqueeze(rgbs, 0).permute(0, 3, 1, 2)
thermals = torch.unsqueeze(thermals, 0).permute(0, 3, 1, 2)
rgbs = rgbs.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
thermals = thermals.to(device, non_blocking=True).float() / 255  # uint8 to float32, 0-255 to 0.0-1.0
assert rgbs.shape == thermals.shape
h, w = rgbs.shape[2:]
# Forward
# teacher
teacher_pred, teacher_dis_out = teacher_model(torch.cat([rgbs, thermals], 1), distillation=True)  # teacher forward
# student
pred, student_dis_out = model(torch.cat([rgbs, thermals], 1), distillation=True)  # forward

lidx = 1
channels = {0: 128, 1: 256, 2: 512}
tb, tc, th, tw = teacher_dis_out[lidx].shape
sb, sc, sh, sw = student_dis_out[lidx].shape
assert ([tb, th, tw] == [sb, sh, sw]) and sc == channels[lidx]
t_head_weights = teacher_head[lidx].weight.data
t_head_bias = teacher_head[lidx].bias.data
# distill all features
t_obj_weights = t_head_weights
t_obj_bias = t_head_bias
# sort and select weights
_, indices = t_obj_weights.norm(2, (2, 3), True).topk(channels[lidx], dim=1)  # [33, in_c, 1, 1]
selective_weights = t_obj_weights.gather(1, indices)    # [33, in_c, 1, 1]
# compute adaptive anchor loss weight
plt.figure()
for anchor_id in range(3):
    teacher_object_prob = teacher_pred[1][lidx][..., 4][:, anchor_id:anchor_id+1].sigmoid()  # [bs, 3, h, w]
    student_object_prob = pred[1][lidx][..., 4][:, anchor_id:anchor_id+1].sigmoid()  # [bs, 3, h, w]
    pseudo_teacher = F.conv2d(teacher_dis_out[lidx][:, indices[anchor_id * 11 + 4].squeeze()],
                              selective_weights[anchor_id * 11 + 4: anchor_id * 11 + 4 + 1],
                              t_obj_bias[anchor_id * 11 + 4: anchor_id * 11 + 4 + 1])
    pseudo_teacher = pseudo_teacher.sigmoid()
    pseudo_student = F.conv2d(student_dis_out[lidx],
                              selective_weights[anchor_id * 11 + 4: anchor_id * 11 + 4 + 1],
                              t_obj_bias[anchor_id * 11 + 4: anchor_id * 11 + 4 + 1],)
    pseudo_student = pseudo_student.sigmoid()

    plt.subplot(3, 4, anchor_id * 4 + 1)
    plt.imshow(rgbs[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(cv2.resize(teacher_object_prob.squeeze().detach().cpu().numpy(), (w, h)), alpha=0.5)
    plt.axis('off')
    plt.subplot(3, 4, anchor_id * 4 + 2)
    plt.imshow(rgbs[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(cv2.resize(pseudo_teacher.squeeze().detach().cpu().numpy(), (w, h)), alpha=0.5)
    plt.axis('off')
    plt.subplot(3, 4, anchor_id * 4 + 3)
    plt.imshow(rgbs[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(cv2.resize(pseudo_student.squeeze().detach().cpu().numpy(), (w, h)), alpha=0.5)
    plt.axis('off')
    plt.subplot(3, 4, anchor_id * 4 + 4)
    plt.imshow(rgbs[0].permute(1, 2, 0).detach().cpu().numpy())
    plt.imshow(cv2.resize(student_object_prob.squeeze().detach().cpu().numpy(), (w, h)), alpha=0.5)
    plt.axis('off')
plt.show()
# for i, (t, p, s) in enumerate(zip(teacher_object_prob, pseudo_teacher, student_object_prob)):
#     plt.subplot(3, 3, i * 3 + 1)
#     plt.imshow(t.detach().cpu().numpy())
#     plt.subplot(3, 3, i * 3 + 2)
#     plt.imshow(p.detach().cpu().numpy())
#     plt.subplot(3, 3, i * 3 + 3)
#     plt.imshow(s.detach().cpu().numpy())
# plt.show()
#
