# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import math
import torch
from einops import rearrange

from model.base import BaseModule
from model.diffusion import *
import torch.nn as nn

def zero_conv(in_channels, out_channels):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class GradLogPEstimator2dWithControlNet(GradLogPEstimator2d):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2dWithControlNet, self).__init__(dim, dim_mults, groups, n_spks, spk_emb_dim, n_feats, pe_scale)

        self.z_input = zero_conv(self.n_feats, self.n_feats)
        self.z_middle = zero_conv(self.n_feats, self.n_feats)

        self.z_downs = torch.nn.ModuleList()

        for i in range(len(self.downs)):
            self.z_downs.append(zero_conv(self.n_feats, self.n_feats))


    def forward(self, x, mask, mu, t, c, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
        else:
            s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        # for now assume c is the same size as x, enforce this later
        assert c.shape[-1] == x.shape[-1]

        c = self.z_input(c)
        c = c + x

        hiddens = []
        masks = [mask]

        # x forward
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        # c forward  - TODO critical -> understand the mask part, it seems it's not needed to save for c
        hiddens_c = []
        mask_down_c = mask
        for resnet1, resnet2, attn, downsample in self.downs:
            # mask_down = masks[-1]
            c = resnet1(c, mask_down_c, t)
            c = resnet2(c, mask_down_c, t)
            c = attn(c)
            hiddens_c.append(c)
            c = downsample(c * mask_down_c)
            # masks.append(mask_down[:, :, :, ::2])
            mask_down_c = mask_down_c[:, :, :, ::2]

        masks = masks[:-1]
        mask_mid = masks[-1]

        # x middle
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        # x middle
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)

        for resnet1, resnet2, attn, upsample in self.ups:
            mask_up = masks.pop()
            x = torch.cat((x, hiddens.pop()), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        return (output * mask).squeeze(1)


class DiffusionWithControlNet(BaseModule):
    def __init__(self, n_feats, dim,
                 n_spks=1, spk_emb_dim=64,
                 beta_min=0.05, beta_max=20, pe_scale=1000):
        super(DiffusionWithControlNet, self).__init__()
        self.n_feats = n_feats
        self.dim = dim
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.pe_scale = pe_scale

        self.estimator = GradLogPEstimator2dWithControlNet(dim, n_spks=n_spks,
                                                           spk_emb_dim=spk_emb_dim,
                                                           pe_scale=pe_scale)

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)
        z = torch.randn(x0.shape, dtype=x0.dtype, device=x0.device,
                        requires_grad=False)
        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, c, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask
        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(z.shape[0], dtype=z.dtype,
                                                   device=z.device)
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max,
                                cumulative=False)
            if stoc:  # adds stochastic term
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, c, spk)
                dxt_det = dxt_det * noise_t * h
                dxt_stoc = torch.randn(z.shape, dtype=z.dtype, device=z.device,
                                       requires_grad=False)
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)
                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, c, spk))
                dxt = dxt * noise_t * h
            xt = (xt - dxt) * mask
        return xt

    @torch.no_grad()
    def forward(self, z, mask, mu, c, n_timesteps, stoc=False, spk=None):
        return self.reverse_diffusion(z, mask, mu, c, n_timesteps, stoc, spk)

    def loss_t(self, x0, mask, mu, t, c, spk=None):
        xt, z = self.forward_diffusion(x0, mask, mu, t)
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)
        noise_estimation = self.estimator(xt, mask, mu, t, c, spk)
        noise_estimation *= torch.sqrt(1.0 - torch.exp(-cum_noise))
        loss = torch.sum((noise_estimation + z) ** 2) / (torch.sum(mask) * self.n_feats)
        return loss, xt

    def compute_loss(self, x0, mask, mu, c, spk=None, offset=1e-5):
        t = torch.rand(x0.shape[0], dtype=x0.dtype, device=x0.device,
                       requires_grad=False)
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, c, spk)
