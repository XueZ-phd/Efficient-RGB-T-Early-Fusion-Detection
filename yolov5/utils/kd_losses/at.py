from __future__ import absolute_import
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.nn.functional as F


'''
AT with sum of absolute values with power p
'''
class AT(nn.Module):
	'''
	Paying More Attention to Attention: Improving the Performance of Convolutional
	Neural Netkworks wia Attention Transfer
	https://arxiv.org/pdf/1612.03928.pdf
	'''
	def __init__(self, p, loss_weights, device):
		super(AT, self).__init__()
		self.p = p
		self.device = device
		self.loss_weights = loss_weights

	def forward_once(self, fm_s, fm_t):
		loss = F.mse_loss(self.attention_map(fm_s), self.attention_map(fm_t))

		return loss

	def forward(self, student_feats, teacher_feats, *args):
		losses = torch.zeros(len(student_feats), device=self.device)
		for idx, (fm_s, fm_t) in enumerate(zip(student_feats, teacher_feats)):
			losses[idx] += self.forward_once(fm_s, fm_t) * self.loss_weights[idx]
		return losses * fm_s.shape[0]

	def attention_map(self, fm, eps=1e-6):
		am = torch.pow(torch.abs(fm), self.p)
		am = torch.sum(am, dim=1, keepdim=True)
		norm = torch.norm(am, dim=(2,3), keepdim=True)
		am = torch.div(am, norm+eps)

		return am