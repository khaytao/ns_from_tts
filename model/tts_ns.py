import math
import random
import torch
import torch.nn.functional as F

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion_with_controlnet import DiffusionWithControlNet
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from model.tts import GradTTS


def renormalize_durations(
    logw: torch.Tensor,          # [B, 1, T_text]
    x_mask: torch.Tensor,        # [B, 1, T_text], 1 for valid tokens
    T_target: int,      # [B] int or long (e.g., control length per sample)
    length_scale: float = 1.0,
    eps: float = 1e-8,
) -> torch.Tensor:
    """
    Returns integer durations per token that sum exactly to T_target per sample.
    Shape: [B, 1, T_text] (dtype long)
    """
    B, _, T_txt = logw.shape

    # 1) raw positive durations (masked)
    w = torch.exp(logw) * x_mask            # [B,1,T_text]
    w = w.squeeze(1)                        # [B,T_text]
    xm = x_mask.squeeze(1).bool()           # [B,T_text]

    # 2) normalize to sum 1 over valid tokens
    sums = w.sum(dim=1, keepdim=True)       # [B,1]
    w_hat = w / (sums + eps)                # [B,T_text]
    w_hat = torch.where(xm, w_hat, torch.zeros_like(w_hat))

    # 3) scale to target frames (per sample)
    # If you MUST exactly match the control length, set length_scale=1.0
    T_float = T_target * float(length_scale)   # [B,1]
    w_star = w_hat * T_float                                        # [B,T_text]

    # 4) largest remainder to get exact integer totals
    w_floor = torch.floor(w_star)                                    # [B,T_text]
    frac    = w_star - w_floor                                       # [B,T_text]
    # ensure masked tokens remain zero
    w_floor = torch.where(xm, w_floor, torch.zeros_like(w_floor))
    frac    = torch.where(xm, frac,   torch.full_like(frac, -1.0))   # masked tokens never chosen

    # frames left to distribute per sample
    remainder = (T_float - w_floor.sum(dim=1)).long()     # [B]

    # start from floor as integers
    w_int = w_floor.long()                                           # [B,T_text]

    # distribute remainder frames to largest fractional parts
    for b in range(B):
        r = int(remainder[b].item())
        if r <= 0:
            continue
        # pick top-r tokens among valid ones by fractional part
        # NOTE: if r > #valid tokens, topk will raise; clamp r
        r = min(r, int(xm[b].sum().item()))
        if r > 0:
            vals, idx = torch.topk(frac[b], k=r)
            w_int[b, idx] += 1

    # final safety: zero out masked tokens and assert sums
    w_int = torch.where(xm, w_int, torch.zeros_like(w_int))
    # (optional) sanity check during dev
    # assert torch.all(w_int.sum(dim=1) == T_target), "Durations do not sum to target!"

    return w_int.unsqueeze(1)  # [B,1,T_text]

class GradTTS_NS(GradTTS):

    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS_NS, self).__init__(n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                                         n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                                         n_feats, dec_dim, beta_min, beta_max, pe_scale)

        self.decoder = DiffusionWithControlNet(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.freeze_encoder()

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x, x_lengths, c, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0, use_mas=True):
        """
        Generates mel-spectrogram from text. Returns:
            1. encoder outputs
            2. decoder outputs
            3. generated alignment

        Args:
            x (torch.Tensor): batch of texts, converted to a tensor with phoneme embedding ids.
            x_lengths (torch.Tensor): lengths of texts in batch.
            n_timesteps (int): number of steps to use for reverse diffusion in decoder.
            c (torch.Tensor): batch of control mel spectrograms. shape [Batch, n_mels, Length]
            temperature (float, optional): controls variance of terminal distribution.
            stoc (bool, optional): flag that adds stochastic term to the decoder sampler.
                Usually, does not provide synthesis improvements.
            length_scale (float, optional): controls speech pace.
                Increase value to slow down generated speech and vice versa.
        """
        x, x_lengths = self.relocate_input([x, x_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        T_control = c.shape[-1]
        T_pad = fix_len_compatibility(T_control)
        pad = T_pad - T_control

        c_pad = F.pad(c, (0, pad)) if pad > 0 else c  # [B, n_feats, T_pad]

        if not use_mas:  # assume based on durations
            # w = torch.exp(logw) * x_mask
            w_ceil = renormalize_durations(logw, x_mask, T_control, length_scale)
            # w_ceil = torch.ceil(w) * length_scale
            # y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
            y_lengths = w_ceil.new_full((w_ceil.size(0),), T_control, dtype=torch.long)
            y_max_length = int(y_lengths.max())
            y_max_length_ = fix_len_compatibility(y_max_length)

            # Using obtained durations `w` construct alignment map `attn`
            y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
            attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
            attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)

        else:  # use MAS
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const
                attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
            # Compute loss between predicted log-scaled durations and those obtained from MAS
            logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
            w = torch.exp(logw_) * x_mask
            w_ceil = torch.ceil(w) * length_scale
        # Align encoded text and get mu_y
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]

        # Sample latent representation from terminal distribution N(mu_y, I)
        z = mu_y + torch.randn_like(mu_y, device=mu_y.device) / temperature
        # Generate sample by performing reverse dynamics
        decoder_outputs = self.decoder(z, y_mask, mu_y, c_pad, n_timesteps, stoc, spk)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]

        return encoder_outputs, decoder_outputs, attn[:, :, :y_max_length]

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location=lambda loc, storage: loc)
        self.load_state_dict(state_dict, strict=False)
        self.decoder.estimator.init_weights_from_base(state_dict)