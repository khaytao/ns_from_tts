# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import torch
import torch.nn as nn

from model.diffusion import *
from model.controlnet_helpers import is_base_layer, zero_conv, is_control_layer


class GradLogPEstimator2dWithControlNet(GradLogPEstimator2d):
    def __init__(
        self,
        dim,
        dim_mults=(1, 2, 4),
        groups=8,
        n_spks=None,
        spk_emb_dim=64,
        n_feats=80,
        pe_scale=1000,
    ):
        super().__init__(dim, dim_mults, groups, n_spks, spk_emb_dim, n_feats, pe_scale)

        self.z_ups = nn.ModuleList()

        # parameters needed for ControlNet init
        dims = [2 + (1 if n_spks and n_spks > 1 else 0), *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))
        num_resolutions = len(in_out)
        mid_dim = dims[-1]

        self.z_input = zero_conv(dims[0], dims[0])
        self.z_middle = zero_conv(mid_dim, mid_dim)
        self.control_downs = nn.ModuleList()

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)
            self.control_downs.append(
                nn.ModuleList(
                    [
                        ResnetBlock(dim_in, dim_out, time_emb_dim=dim),
                        ResnetBlock(dim_out, dim_out, time_emb_dim=dim),
                        Residual(Rezero(LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        for _, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            self.z_ups.append(zero_conv(dim_out, dim_out))

        self.control_mid_block1 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)
        self.control_mid_attn = Residual(Rezero(LinearAttention(mid_dim)))
        self.control_mid_block2 = ResnetBlock(mid_dim, mid_dim, time_emb_dim=dim)

        # freeze base network; train only control branches and zero-convs
        for name, p in self.named_parameters():
            if is_base_layer(name):
                p.requires_grad = False

        self.is_initialized = False

    @torch.no_grad()
    def init_weights_from_base(self, state_dict, prefix_to_ignore=None):
        """
        Initialize from a base (non-control) diffusion checkpoint.

        Args:
            state_dict: mapping {param_name: tensor} or checkpoint state dict.
            prefix_to_ignore: str or iterable[str]; prefixes to strip from keys
                              before matching (e.g. 'decoder.estimator.').
        """
        if prefix_to_ignore is None:
            pref_list = []
        elif isinstance(prefix_to_ignore, str):
            pref_list = [prefix_to_ignore]
        else:
            pref_list = list(prefix_to_ignore)

        # normalize prefixes; allow with/without trailing dot + common wrappers
        norm_pfx = []
        for p in pref_list:
            if not p:
                continue
            norm_pfx.append(p if p.endswith(".") else p + ".")
            norm_pfx.append(p)

        norm_pfx += [
            "module.",
            "model.",
            "generator.",
            "decoder.",
            "estimator.",
        ]

        def strip_prefixes(k: str) -> str:
            changed = True
            while changed:
                changed = False
                for pf in norm_pfx:
                    if k.startswith(pf):
                        k = k[len(pf) :]
                        changed = True
            return k

        # build normalized state_dict
        norm_sd = {}
        for k, v in state_dict.items():
            nk = strip_prefixes(k)
            if nk not in norm_sd:  # keep first on collision
                norm_sd[nk] = v

        target_sd = self.state_dict()

        is_control_key = is_control_layer

        def is_z_key(k: str) -> bool:
            root = k.split(".", 1)[0]
            return root in {"z_input", "z_middle", "z_ups"}

        base_keys_expected = {
            k for k in target_sd.keys() if not is_control_key(k) and not is_z_key(k)
        }

        missing_base = sorted(k for k in base_keys_expected if k not in norm_sd)
        if missing_base:
            preview = ", ".join(missing_base[:10])
            raise ValueError(
                "init_weights_from_base: missing "
                f"{len(missing_base)} required base keys after prefix stripping. "
                f"First few: {preview}"
            )

        base_subset = {k: norm_sd[k] for k in base_keys_expected}
        self.load_state_dict(base_subset, strict=False)

        def map_to_control(k: str):
            if k.startswith("downs."):
                return "control_" + k  # downs.N.* -> control_downs.N.*
            if k.startswith("mid_block1."):
                return "control_mid_block1." + k[len("mid_block1.") :]
            if k.startswith("mid_attn."):
                return "control_mid_attn." + k[len("mid_attn.") :]
            if k.startswith("mid_block2."):
                return "control_mid_block2." + k[len("mid_block2.") :]
            # ups.*, final_block, final_conv are not mirrored
            return None

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
            examples = "; ".join(
                f"{k}: got {sd}, want {ss}"
                for k, sd, ss in control_shape_mismatch[:10]
            )
            raise ValueError(
                "init_weights_from_base: "
                f"{len(control_shape_mismatch)} control targets have shape mismatches "
                f"after base load. Examples: {examples}"
            )

        # zero some modules (kept as in original implementation)
        zeroed = 0
        for m in [self.z_input, self.z_middle, *list(self.ups)]:
            if hasattr(m, "weight") and m.weight is not None:
                nn.init.zeros_(m.weight)
                zeroed += 1
            if hasattr(m, "bias") and m.bias is not None:
                nn.init.zeros_(m.bias)

        self.is_initialized = True
        return {
            "loaded_base_params": len(base_subset),
            "copied_to_control": copied_to_control,
            "zero_convs_zeroed": zeroed,
            "prefixes_stripped": list(dict.fromkeys(norm_pfx)),  # unique, preserve order
        }

    def load_from_state_dict(self, state_dict):
        self.load_state_dict(state_dict)
        self.is_initialized = True

    def forward(self, x, mask, mu, t, c, spk=None):
        if not self.is_initialized:
            raise ValueError("DiffusionWithControlNet is not initialized.")

        t = self.time_pos_emb(t, scale=self.pe_scale)
        t = self.mlp(t)

        if self.n_spks < 2:
            x = torch.stack([mu, x], 1)
            c = torch.stack([mu, c], 1)
        else:
            raise NotImplementedError(
                "Controlled diffusion with multiple speakers is not implemented."
            )

        mask = mask.unsqueeze(1)

        # assume c is the same size as x
        assert c.shape[-1] == x.shape[-1]

        c = self.z_input(c)
        c = c + x

        hiddens = []
        masks = [mask]

        assert torch.isfinite(x).all()
        assert torch.isfinite(c).all()

        # x downs
        for resnet1, resnet2, attn, downsample in self.downs:
            mask_down = masks[-1]
            x = resnet1(x, mask_down, t)
            x = resnet2(x, mask_down, t)
            x = attn(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, :, ::2])

        # c downs
        hiddens_c = []
        mask_down_c = mask
        for resnet1, resnet2, attn, downsample in self.control_downs:
            c = resnet1(c, mask_down_c, t)
            c = resnet2(c, mask_down_c, t)
            c = attn(c)
            hiddens_c.append(c)
            c = downsample(c * mask_down_c)
            mask_down_c = mask_down_c[:, :, :, ::2]

        masks = masks[:-1]
        mask_mid = masks[-1]

        assert torch.isfinite(x).all()
        assert torch.isfinite(c).all()

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

        assert torch.isfinite(x).all()
        assert torch.isfinite(c).all()

        # ups with control injection
        for (resnet1, resnet2, attn, upsample), z_up in zip(self.ups, self.z_ups):
            mask_up = masks.pop().to(x.dtype).clamp(0, 1)
            skip = hiddens.pop()
            c_skip = hiddens_c.pop()

            x = torch.cat((x, skip + z_up(c_skip)), dim=1)
            x = resnet1(x, mask_up, t)
            x = resnet2(x, mask_up, t)
            x = attn(x)
            x = upsample(x * mask_up)

        assert torch.isfinite(x).all()
        assert torch.isfinite(c).all()

        x = self.final_block(x, mask)
        output = self.final_conv(x * mask)

        assert torch.isfinite(x).all()
        assert torch.isfinite(c).all()
        return (output * mask).squeeze(1)


class DiffusionWithControlNet(Diffusion):
    def __init__(
        self,
        n_feats,
        dim,
        n_spks=1,
        spk_emb_dim=64,
        beta_min=0.05,
        beta_max=20,
        pe_scale=1000,
    ):
        super().__init__(n_feats, dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.estimator = GradLogPEstimator2dWithControlNet(
            dim,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            pe_scale=pe_scale,
        )

    def forward_diffusion(self, x0, mask, mu, t):
        time = t.unsqueeze(-1).unsqueeze(-1)
        cum_noise = get_noise(time, self.beta_min, self.beta_max, cumulative=True)

        mean = x0 * torch.exp(-0.5 * cum_noise) + mu * (1.0 - torch.exp(-0.5 * cum_noise))
        variance = 1.0 - torch.exp(-cum_noise)

        z = torch.randn(
            x0.shape,
            dtype=x0.dtype,
            device=x0.device,
            requires_grad=False,
        )

        xt = mean + z * torch.sqrt(variance)
        return xt * mask, z * mask

    @torch.no_grad()
    def reverse_diffusion(self, z, mask, mu, c, n_timesteps, stoc=False, spk=None):
        h = 1.0 / n_timesteps
        xt = z * mask

        for i in range(n_timesteps):
            t = (1.0 - (i + 0.5) * h) * torch.ones(
                z.shape[0], dtype=z.dtype, device=z.device
            )
            time = t.unsqueeze(-1).unsqueeze(-1)
            noise_t = get_noise(time, self.beta_min, self.beta_max, cumulative=False)

            if stoc:
                dxt_det = 0.5 * (mu - xt) - self.estimator(xt, mask, mu, t, c, spk)
                dxt_det = dxt_det * noise_t * h

                dxt_stoc = torch.randn(
                    z.shape,
                    dtype=z.dtype,
                    device=z.device,
                    requires_grad=False,
                )
                dxt_stoc = dxt_stoc * torch.sqrt(noise_t * h)

                dxt = dxt_det + dxt_stoc
            else:
                dxt = 0.5 * (mu - xt - self.estimator(xt, mask, mu, t, c, spk))
                if not torch.isfinite(dxt).all():
                    # keep the check, but avoid noisy prints in library code
                    raise RuntimeError("reverse_diffusion: non-finite update dxt")
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

        loss = torch.sum((noise_estimation + z) ** 2) / (
            torch.sum(mask) * self.n_feats
        )
        return loss, xt

    def compute_loss(self, x0, mask, mu, c, spk=None, offset=1e-5):
        t = torch.rand(
            x0.shape[0],
            dtype=x0.dtype,
            device=x0.device,
            requires_grad=False,
        )
        t = torch.clamp(t, offset, 1.0 - offset)
        return self.loss_t(x0, mask, mu, t, c, spk)
