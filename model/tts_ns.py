import math
import random

import torch
import torch.nn.functional as F

from model import monotonic_align
from model.diffusion_with_controlnet import DiffusionWithControlNet
from model.utils import (
    sequence_mask,
    generate_path,
    duration_loss,
    fix_len_compatibility,
)
from model.tts import GradTTS
from model.controlnet_helpers import renormalize_durations


class GradTTS_NS(GradTTS):
    def __init__(
            self,
            n_vocab,
            n_spks,
            spk_emb_dim,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
            n_feats,
            dec_dim,
            beta_min,
            beta_max,
            pe_scale,
    ):
        super().__init__(
            n_vocab,
            n_spks,
            spk_emb_dim,
            n_enc_channels,
            filter_channels,
            filter_channels_dp,
            n_heads,
            n_enc_layers,
            enc_kernel,
            enc_dropout,
            window_size,
            n_feats,
            dec_dim,
            beta_min,
            beta_max,
            pe_scale,
        )

        # Replace vanilla decoder with ControlNet-based diffusion decoder
        self.decoder = DiffusionWithControlNet(
            n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale
        )

        self.freeze_encoder()

    def init_controlnet(self, base_weight_path: str) -> None:
        """Initialize ControlNet estimator from a base diffusion checkpoint."""
        self.decoder.estimator.init_weights_from_base(base_weight_path)

    def freeze_encoder(self) -> None:
        """Freeze text encoder parameters."""
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(
            self,
            x,
            x_lengths,
            c,
            c_lengths,
            n_timesteps,
            temperature: float = 1.0,
            stoc: bool = False,
            spk=None,
            length_scale: float = 1.0,
            use_mas: bool = False,
            clean=None,
    ):
        """
        Generate a mel-spectrogram from text and control features.

        Returns:
            encoder_outputs: aligned encoder features, [B, n_feats, T]
            decoder_outputs: generated mel, [B, n_feats, T]
            attn: alignment map, [B, 1, L_text, T]
        """
        x, x_lengths, c, c_lengths = self.relocate_input(
            [x, x_lengths, c, c_lengths]
        )

        if self.n_spks > 1 and spk is not None:
            spk = self.spk_emb(spk)

        # Encoder: text posterior mean and log-durations
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        T_control = c.shape[-1]
        T_pad = fix_len_compatibility(T_control)
        pad = T_pad - T_control

        c_pad = F.pad(c, (0, pad)) if pad > 0 else c  # [B, n_feats, T_pad]

        if clean is not None and isinstance(clean, torch.Tensor):
            clean = clean.to(c.device)
            clean_lengths = c_lengths
            clean_pad = F.pad(clean, (0, pad)) if pad > 0 else clean
        else:
            clean_pad = None
            clean_lengths = None

        # Alignment: duration-based or MAS
        if not use_mas and clean is None:
            # Duration-based path: renormalize log-durations to match T_control
            w_ceil = renormalize_durations(logw, x_mask, T_control, length_scale)
            y_lengths = w_ceil.new_full(
                (w_ceil.size(0),), T_control, dtype=torch.long
            )
            y_max_length = int(y_lengths.max())
            y_max_length_ = fix_len_compatibility(y_max_length)

            y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(
                x_mask.dtype
            )
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(
                w_ceil.squeeze(1), attn_mask.squeeze(1)
            ).unsqueeze(1)
        else:
            # MAS-based alignment; use either clean or noisy control as reference
            mas_ref = clean_pad if clean_pad is not None else c_pad
            c_max_length = mas_ref.shape[-1]
            c_mask = sequence_mask(
                c_lengths if clean_lengths is None else clean_lengths,
                c_max_length,
            ).unsqueeze(1).to(x_mask)

            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(
                mu_x.shape, dtype=mu_x.dtype, device=mu_x.device
            )

            y_square = torch.matmul(factor.transpose(1, 2), mas_ref ** 2)
            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2), mas_ref
            )
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn_mask = x_mask.unsqueeze(-1) * c_mask.unsqueeze(2)
            attn = monotonic_align.maximum_path(
                log_prior, attn_mask.squeeze(1)
            )
            attn = attn.detach()

            y_max_length = c_max_length
            y_mask = c_mask

            # Sanity checks on alignment
            assert (attn.sum(dim=1)[y_mask.squeeze(1).bool()] == 1).all()
            assert (attn * (1 - attn_mask.squeeze(1))).sum() == 0

        # Align encoder outputs with time axis
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample terminal latent and run reverse diffusion
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        decoder_outputs = self.decoder(
            z, y_mask, mu_y, c_pad, n_timesteps, stoc, spk
        )
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def compute_loss(self, x, x_lengths, y,
                     y_lengths,
                     c,
                     c_lengths=None,
                     spk=None, out_size=None):
        x, x_lengths, y, y_lengths, c, c_lengths = self.relocate_input(
            [x, x_lengths, y, y_lengths, c, c_lengths]
        )

        if self.n_spks > 1 and spk is not None:
            spk = self.spk_emb(spk)

        # Encoder: text posterior and log-durations
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # MAS alignment between text and target mel
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(
                mu_x.shape, dtype=mu_x.dtype, device=mu_x.device
            )

            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(
                2.0 * (factor * mu_x).transpose(1, 2), y
            )
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(
                log_prior, attn_mask.squeeze(1)
            )
            attn = attn.detach()

        # Duration prediction loss
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Optionally cut a random segment to increase effective batch size
        if out_size is not None:
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(
                zip([0] * max_offset.shape[0], max_offset.cpu().numpy())
            )
            out_offset = torch.LongTensor(
                [
                    random.choice(range(start, end)) if end > start else 0
                    for start, end in offset_ranges
                ]
            ).to(y_lengths)

            B = y.shape[0]
            attn_cut = torch.zeros(
                B, attn.shape[1], out_size, dtype=attn.dtype, device=attn.device
            )
            y_cut = torch.zeros(
                B, self.n_feats, out_size, dtype=y.dtype, device=y.device
            )
            c_cut = torch.zeros(
                B, self.n_feats, out_size, dtype=y.dtype, device=y.device
            )

            y_cut_lengths = []
            for i, (y_, c_, out_offset_) in enumerate(
                    zip(y, c, out_offset)
            ):
                y_cut_length = out_size + (
                        y_lengths[i] - out_size
                ).clamp(max=0)
                y_cut_lengths.append(y_cut_length)
                cut_lower = out_offset_
                cut_upper = out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                c_cut[i, :, :y_cut_length] = c_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[
                                                i, :, cut_lower:cut_upper
                                                ]

            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            c = c_cut
            y_mask = y_cut_mask

        # Prior matching: project encoder features with alignment
        mu_y = torch.matmul(
            attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2)
        )
        mu_y = mu_y.transpose(1, 2)

        # Diffusion loss
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, c, spk)

        # Prior (reconstruction) loss
        prior_loss = torch.sum(
            0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask
        )
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    def load_weights(self, weights_path: str) -> None:
        """Load model and initialize ControlNet estimator from a single checkpoint."""
        state_dict = torch.load(
            weights_path, map_location=lambda loc, storage: loc
        )
        self.load_state_dict(state_dict, strict=False)
        self.decoder.estimator.init_weights_from_base(state_dict)
