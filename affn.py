# Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
#
# NVIDIA CORPORATION and its licensors retain all intellectual property
# and proprietary rights in and to this software, related documentation
# and any modifications thereto.  Any use, reproduction, disclosure or
# distribution of this software and related documentation without an express
# license agreement from NVIDIA CORPORATION is strictly prohibited.


import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_utils import WeightedFilter

from unet import *
from utils import *

EPSILON = 0.000001 # Small epsilon to avoid division by zero

###############################################################################
# Sample network definition
#
# A scaled-down version of Sample-based Monte Carlo Denoising using a Kernel-Splatting Network
#
# https://groups.csail.mit.edu/graphics/rendernet/
#
###############################################################################

class AffinityNet(nn.Module):
	def __init__(self, n_in, splat=True, use_sample_info=True, kernel_size=13, feat_size=8):

		super(AffinityNet, self).__init__() 
		self.output_channels = (feat_size + 2)*3
		self.embed_channels  = 32 
		self.kernel_size     = kernel_size
		self.splat           = splat
		self.input_channels = n_in
		self.use_sample_info = use_sample_info

		# Sample Reducer: Maps from input channels to sample embeddings, uses 1x1 convolutions
		if self.use_sample_info:
			self._sample_reducer = nn.Sequential(
				nn.Conv2d(self.input_channels, self.embed_channels, 1, padding=0),
				Activation,
				nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
				Activation,
				nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
				Activation,
			)

		else:
		# Pixel reducer: Used instead of sample reducer for the per-pixel network, uses 1x1 convolutions
			self._pixel_reducer = nn.Sequential(
				nn.Conv2d(self.input_channels*2, self.embed_channels, 1, padding=0),
				Activation,
				nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
				Activation,
				nn.Conv2d(self.embed_channels, self.embed_channels, 1, padding=0),
				Activation,
			)

		# Kernel generator: Combines UNet per-pixel output with per-sample or per-pixel embeddings, uses 1x1 convolutions
		# self._kernel_generator = nn.Sequential(
		# 	nn.Conv2d(self.output_channels+self.embed_channels, 128, 1, padding=0),
		# 	Activation,
		# 	nn.Conv2d(128, 128, 1, padding=0),
		# 	Activation,
		# 	nn.Conv2d(128, self.kernel_size*self.kernel_size, 1, padding=0), # output kernel weights
		# )

		# U-Net: Generates context features
		self._unet = UNet(self.embed_channels, self.output_channels, encoder_features=[[64, 64], [128], [256], [512], [512]], bottleneck_features=[512], decoder_features=[[512, 512], [256, 256], [128, 128], [128, 128], [128, 128]])

		# Filter for applying predicted kernels
		kernels = []
		for i in range(3):
			kernels.append(WeightedFilter(channels=3, kernel_size=self.kernel_size, bias=False, splat=self.splat))
		self.kernels = kernels

		# Initialize network weights
		for m in self.modules():
			if isinstance(m, nn.Conv2d):
				nn.init.xavier_uniform_(m.weight.data)
				if m.bias is not None:
					m.bias.data.zero_()

	def _kernel_generator(self, affn_feat):
		B, C, H, W = affn_feat.shape
		kernel_weights = []
		for k in range(1, 4):
			cc = C//3
			dilation = 2**(k-1)
			padding = (self.kernel_size-1) * dilation // 2

			f, c, a = affn_feat[:, cc*(k-1):cc*k-2], F.sigmoid(affn_feat[:, cc*k-2:cc*k-1]), torch.pow(affn_feat[:, cc*k-1:cc*k], 2)
			f_pad = F.pad(f, (padding, padding, padding, padding), 'reflect')
			i = torch.arange(H, dtype=torch.long, device=f_pad.device)
			j = torch.arange(W, dtype=torch.long, device=f_pad.device)
			ii, jj = torch.meshgrid(i, j, indexing='ij')
			offsets = torch.arange(-self.kernel_size/2, self.kernel_size/2 + 1, dtype=torch.long, device=f_pad.device) * dilation
			offset_ii, offset_jj = torch.meshgrid(offsets, offsets, indexing='ij')
			sh = [1, 1, 1, 1, -1]
			offset_ii = offset_ii.reshape(sh)
			offset_jj = offset_jj.reshape(sh)
			ii = ii.unsqueeze(-1) + offset_ii
			jj = jj.unsqueeze(-1) + offset_jj
			ii = ii % H
			jj = jj % W
			kernel_weight = f_pad[:, :, ii, jj]
			kernel_weight = kernel_weight.reshape(B, -1, H, W, self.kernel_size, self.kernel_size)

			kernel_weight = torch.exp(-a.unsqueeze(-1).unsqueeze(-1).repeat(1, 1, 1, 1, self.kernel_size, self.kernel_size)*torch.norm(kernel_weight, dim=1, keepdim=True).pow(2))
			kernel_weight[..., self.kernel_size//2, self.kernel_size//2] = c
			kernel_weights.append(kernel_weight.squeeze(1).permute(0, 3, 4, 1, 2).reshape(B, -1, H, W))
		return kernel_weights


	def forward(self, samples):
		num_weights = self.kernel_size*self.kernel_size

		radiance = samples["radiance"]
		features = samples["features"]
		
		# loop over samples to create embeddings
		sh = features.shape
		embedding = torch.cuda.FloatTensor(sh[0], sh[1], self.embed_channels, sh[3], sh[4]).fill_(0)


		if self.use_sample_info:
			# loop over samples to create embeddings
			for i in range(sh[1]):
				embedding[:,i,...] = self._sample_reducer(features[:,i,...])
			avg_embeddings = embedding.mean(dim=1) # average over embeddings dimension
		else:
			# average per-sample info 
			xc_mean = torch.mean(features, dim=1)
			xc_variance = torch.var(features, dim=1, unbiased=False)
			embedding[:,0,...] = self._pixel_reducer(torch.cat((xc_mean,xc_variance), dim=1))
			avg_embeddings = embedding[:,0,...]

		affn = self._unet(avg_embeddings)
		ones  = torch.cuda.FloatTensor(sh[0], 1, sh[3], sh[4]).fill_(1.0).contiguous()
		kernel_weights = self._kernel_generator(affn)

		filtered = torch.mean(radiance, dim=1)
		for i in range(len(kernel_weights)):
			pixel_weights = kernel_weights[i]
			kpn = self.kernels[i]
			filtered = kpn(filtered.contiguous(), pixel_weights.contiguous())
			w   = kpn(ones, pixel_weights.contiguous())
			filtered = filtered/(w+EPSILON)

		return filtered

	def inference(self, sequenceData):
		return self.forward(sequenceData)
