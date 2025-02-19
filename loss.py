# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# This work is licensed under a Creative Commons
# Attribution-NonCommercial-ShareAlike 4.0 International License.
# You should have received a copy of the license along with this
# work. If not, see http://creativecommons.org/licenses/by-nc-sa/4.0/

"""Loss functions used in the paper
"Elucidating the Design Space of Diffusion-Based Generative Models"."""

from typing import Any
import torch
from torch_utils import persistence
import numpy as np

#----------------------------------------------------------------------------
# Loss function corresponding to the variance preserving (VP) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

class LossArgs:
    def __init__(self, opt):
        self.frame_cond_strategy = opt.frame_cond_strategy if hasattr(opt, 'frame_cond_strategy') else None
        
    def _frame_cond(self, x):
        # x: B*C*T*dim_1*dim_2*...
        if self.frame_cond_strategy is None or 'none' in self.frame_cond_strategy:
            return None
        if 'fixed' in self.frame_cond_strategy:
            num_frame = int(self.frame_cond_strategy.split('_')[-1])
            B, _, T = x.shape[:3]
            mask = torch.ones([B, 1, T], device=x.device, dtype=torch.bool)
            mask[:, :, :num_frame] = False
            return mask
        elif 'cross_attn' in self.frame_cond_strategy:
            num_frame_cond = int(self.frame_cond_strategy.split('_')[-1])
            if 'ol' in self.frame_cond_strategy:
                return lambda xx: (xx, xx[:, :, :num_frame_cond])
            else:
                return lambda xx: (xx[:, :, num_frame_cond:], xx[:, :, :num_frame_cond])
        else:
            raise NotImplementedError('Unknown types of frame_cond_strategy!')
    
    def __call__(self, x, **kwargs):
        args = dict()
        if 'fixed' in self.frame_cond_strategy:
            args['mask_indices'] = self._frame_cond(x)
        if 'cross_attn' in self.frame_cond_strategy:
            args['augment_pipe'] = self._frame_cond(x)
        return args


@persistence.persistent_class
class VPLoss:
    def __init__(self, beta_d=19.9, beta_min=0.1, epsilon_t=1e-5):
        self.beta_d = beta_d
        self.beta_min = beta_min
        self.epsilon_t = epsilon_t

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma(1 + rnd_uniform * (self.epsilon_t - 1))
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

    def sigma(self, t):
        t = torch.as_tensor(t)
        return ((0.5 * self.beta_d * (t ** 2) + self.beta_min * t).exp() - 1).sqrt()

#----------------------------------------------------------------------------
# Loss function corresponding to the variance exploding (VE) formulation
# from the paper "Score-Based Generative Modeling through Stochastic
# Differential Equations".

@persistence.persistent_class
class VELoss:
    def __init__(self, sigma_min=0.02, sigma_max=100):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, net, images, labels, augment_pipe=None):
        rnd_uniform = torch.rand([images.shape[0], 1, 1, 1], device=images.device)
        sigma = self.sigma_min * ((self.sigma_max / self.sigma_min) ** rnd_uniform)
        weight = 1 / sigma ** 2
        y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
        n = torch.randn_like(y) * sigma
        D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
        loss = weight * ((D_yn - y) ** 2)
        return loss

#----------------------------------------------------------------------------
# Improved loss function proposed in the paper "Elucidating the Design Space
# of Diffusion-Based Generative Models" (EDM).

@persistence.persistent_class
class EDMLoss:
    def __init__(self, P_mean=-1.2, P_std=1.2, sigma_data=0.5):
        self.P_mean = P_mean
        self.P_std = P_std
        self.sigma_data = sigma_data

    def __call__(self, net, images, labels=None, augment_pipe=None, square=True, mask_indices=None):
        # mask_indices: shape B*C*T bool tensor (1 for the frames that should be noised)
        if mask_indices is not None:
            B, C, T = mask_indices.shape
            assert T == images.shape[2]
            n_shape = [B, C, T] + [1]*(len(images.shape)-3)
            mask_indices = mask_indices.reshape(*n_shape)
            rnd_normal = torch.randn(B, device=images.device, dtype=images.dtype)
            rnd_normal = rnd_normal[:, None, None].repeat(1, C, T)
            rnd_normal = rnd_normal.reshape(n_shape)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma * mask_indices
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
            if square:
                loss = mask_indices * weight * ((D_yn - y) ** 2)
            else:
                loss = mask_indices * weight * (D_yn - y)
        else:
            rnd_normal = torch.randn([images.shape[0]] + [1]*(len(images.shape)-1), device=images.device, dtype=images.dtype)
            sigma = (rnd_normal * self.P_std + self.P_mean).exp()
            weight = (sigma ** 2 + self.sigma_data ** 2) / (sigma * self.sigma_data) ** 2
            y, augment_labels = augment_pipe(images) if augment_pipe is not None else (images, None)
            n = torch.randn_like(y) * sigma
            D_yn = net(y + n, sigma, labels, augment_labels=augment_labels)
            if square:
                loss = weight * ((D_yn - y) ** 2)
            else:
                loss = weight * (D_yn - y)
        return loss
