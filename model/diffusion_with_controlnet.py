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
import torch
from typing import Iterable, Dict


def is_control_layer(name: str) -> bool:
    # control branches and zero-conv taps
    return name.startswith("control_")

def _is_zero_conv_layer(name: str) -> bool:
    # control branches and zero-conv taps
    return name.startswith("z_input") \
    or name.startswith("z_middle") \
    or name.startswith("z_downs") \
    or name.startswith("z_ups")

def is_base_layer(name: str) -> bool:
    return not (is_control_layer(name) or _is_zero_conv_layer(name))

def zero_conv(in_channels, out_channels):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv

class GradLogPEstimator2dWithControlNet(GradLogPEstimator2d):
    def __init__(self, dim, dim_mults=(1, 2, 4), groups=8,
                 n_spks=None, spk_emb_dim=64, n_feats=80, pe_scale=1000):
        super(GradLogPEstimator2dWithControlNet, self).__init__(dim, dim_mults, groups, n_spks, spk_emb_dim, n_feats, pe_scale)

        self.z_ups = torch.nn.ModuleList()

        # parameters needed for controlnet init loop
        dims = [2 + (1 if n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        mid_dim = dims[-1]


        self.z_input = zero_conv(dims[0], dims[0])
        self.z_middle = zero_conv(mid_dim, mid_dim)
        self.control_downs = torch.nn.ModuleList()

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.control_downs.append(torch.nn.ModuleList([
                       ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                       ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                       Residual(Rezero(LinearAttention(dim_out))),
                       Downsample(dim_out) if not is_last else torch.nn.Identity()]))

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.z_ups.append(zero_conv(dim_out, dim_out))

        self.control_mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.control_mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.control_mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)


        for name, p in self.named_parameters():
            if is_base_layer(name):
                p.requires_grad = False

    @torch.no_grad()
    def init_weights_from_base(self, state_dict, prefix_to_ignore=None):
        """
        Initialize from a *base* (non-control) checkpoint dict.

        Args:
            state_dict: mapping {param_name: tensor}
            prefix_to_ignore: str or iterable[str]; any of these prefixes (with or without trailing '.')
                              will be stripped from keys before matching (e.g., 'decoder.estimator.')

        Steps:
          1) Normalize keys by stripping prefixes.
          2) Load all non-control, non-z_* weights into the base.
          3) Copy corresponding base weights into control mirrors.
          4) Zero z_input / z_middle / z_downs[*].
          5) Verify completeness; raise ValueError on missing keys or shape mismatches.
        """
        # -------- normalize prefixes & strip keys --------
        # user-provided prefixes

        if prefix_to_ignore is None:
            pref_list = []
        elif isinstance(prefix_to_ignore, str):
            pref_list = [prefix_to_ignore]
        else:
            pref_list = list(prefix_to_ignore)

        # allow both with/without trailing dot; add common wrappers too
        norm_pfx = []
        for p in pref_list:
            if not p:
                continue
            norm_pfx.append(p if p.endswith('.') else p + '.')
            norm_pfx.append(p)  # also allow exact given form
        norm_pfx += ['module.', 'model.', 'generator.']  # common wrappers

        def strip_prefixes(k):
            changed = True
            while changed:
                changed = False
                for pf in norm_pfx:
                    if k.startswith(pf):
                        k = k[len(pf):]
                        changed = True
            return k

        # build normalized state_dict
        norm_sd = {}
        for k, v in state_dict.items():
            nk = strip_prefixes(k)
            if nk not in norm_sd:  # keep first if collision
                norm_sd[nk] = v

        # -------- identify expected base keys --------
        target_sd = self.state_dict()

        def is_control_key(k):
            return k.startswith('control_')

        def is_z_key(k):
            root = k.split('.', 1)[0]
            return root in {'z_input', 'z_middle', 'z_ups'}

        base_keys_expected = {k for k in target_sd.keys()
                              if not is_control_key(k) and not is_z_key(k)}

        # coverage check
        missing_base = sorted(k for k in base_keys_expected if k not in norm_sd)
        if missing_base:
            preview = ", ".join(missing_base[:10])
            raise ValueError(
                f"init_weights_from_base: missing {len(missing_base)} required base keys "
                f"after prefix stripping. First few: {preview}"
            )

        # -------- load base subset --------
        base_subset = {k: norm_sd[k] for k in base_keys_expected}
        self.load_state_dict(base_subset, strict=False)

        # -------- map base -> control and copy --------
        def map_to_control(k):
            if k.startswith("downs."):
                return "control_" + k  # downs.N.* -> control_downs.N.*
            if k.startswith("mid_block1."):
                return "control_mid_block1." + k[len("mid_block1."):]
            if k.startswith("mid_attn."):
                return "control_mid_attn." + k[len("mid_attn."):]
            if k.startswith("mid_block2."):
                return "control_mid_block2." + k[len("mid_block2."):]
            return None  # ups.*, final_block, final_conv not mirrored

        copied_to_control = 0
        control_shape_mismatch = []

        # refresh view after base load
        target_sd = self.state_dict()

        for k in base_keys_expected:
            dst = map_to_control(k)
            if dst is None:
                continue
            if dst in target_sd:
                src_tensor = target_sd[k]
                dst_tensor = target_sd[dst]
                if dst_tensor.shape != src_tensor.shape:
                    control_shape_mismatch.append(
                        (dst, tuple(dst_tensor.shape), tuple(src_tensor.shape))
                    )
                else:
                    dst_tensor.copy_(src_tensor)
                    copied_to_control += 1

        if control_shape_mismatch:
            examples = "; ".join(f"{k}: got {sd}, want {ss}"
                                 for k, sd, ss in control_shape_mismatch[:10])
            raise ValueError(
                f"init_weights_from_base: {len(control_shape_mismatch)} control targets have shape "
                f"mismatches after base load. Examples: {examples}"
            )

        # -------- zero all z_* taps explicitly --------
        zeroed = 0
        for m in [self.z_input, self.z_middle, *list(self.ups)]:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.zeros_(m.weight)
                zeroed += 1
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

        return {
            "loaded_base_params": len(base_subset),
            "copied_to_control": copied_to_control,
            "zero_convs_zeroed": zeroed,
            "prefixes_stripped": list(dict.fromkeys(norm_pfx)),  # unique, preserve order
        }


    def forward(self, x, mask, mu, t, c, spk=None):
        if not isinstance(spk, type(None)):
            s = self.spk_mlp(spk)

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
            c = torch.stack([mu, c], 1)
        else:
            raise NotImplementedError("Controlled Diffusion with multiple speakers not implemented")
            # s = s.unsqueeze(-1).repeat(1, 1, x.shape[-1])
            # x = torch.stack([mu, x, s], 1)
        mask = mask.unsqueeze(1)

        # for now assume c is the same size as x
        assert c.shape[-1] == x.shape[-1]

        c = self.z_input(c)
        c = c + x

        hiddens = []
        masks = [mask]

        # x downs
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        # c downs  - TODO critical -> understand the mask part, it seems it's not needed to save for c

        hiddens_c = []
        mask_down_c = mask
        for resnet1, resnet2, attn, downsample in self.control_downs:
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

        # c middle
        c = self.control_mid_block1(c, mask_mid, t)
        c = self.control_mid_attn(c)
        c = self.control_mid_block2(c, mask_mid, t)
        c = self.z_middle(c)

        # x middle
        x = self.mid_block1(x, mask_mid, t)
        x = self.mid_attn(x)
        x = self.mid_block2(x, mask_mid, t)
        x = x + c

        # Ups
        for (resnet1, resnet2, attn, upsample), z_up in zip(self.ups, self.z_ups):
            mask_up = masks.pop()
            skip = hiddens.pop()
            c_skip = hiddens_c.pop()
            x = torch.cat((x, skip + z_up(c_skip)), dim=1)
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
