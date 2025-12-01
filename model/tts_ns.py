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
from model.controlnet_helpers import renormalize_durations


class GradTTS_NS(GradTTS):

    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(GradTTS_NS, self).__init__(n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                                         n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                                         n_feats, dec_dim, beta_min, beta_max, pe_scale)

        self.decoder = DiffusionWithControlNet(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)

        self.freeze_encoder()

    def init_controlnet(self, base_weight_path):
        self.decoder.estimator.init_weights_from_base(base_weight_path)

    def freeze_encoder(self):
        for p in self.encoder.parameters():
            p.requires_grad = False
        self.encoder.eval()

    @torch.no_grad()
    def forward(self, x, x_lengths, c, c_lengths, n_timesteps, temperature=1.0, stoc=False, spk=None, length_scale=1.0, use_mas=False, clean=None):
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
        x, x_lengths, c, c_lengths = self.relocate_input([x, x_lengths, c, c_lengths])



        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)

        T_control = c.shape[-1]
        T_pad = fix_len_compatibility(T_control)
        pad = T_pad - T_control

        c_pad = F.pad(c, (0, pad)) if pad > 0 else c  # [B, n_feats, T_pad]

        if clean is not None and isinstance(clean, torch.Tensor):
            clean = clean.to(c.device)
            clean_lengths = c_lengths
            clean_pad = F.pad(clean, (0, pad)) if pad > 0 else clean  # [B, n_feats, T_pad]
        # c_smooth = F.avg_pool1d(c_pad, kernel_size=5, stride=1, padding=2)  # [B, n_feats, T_pad]

        if not use_mas and clean is None:  # assume based on durations
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

            if clean is not None and isinstance(clean, torch.Tensor):
                mas_ref = clean_pad
            else:
                mas_ref = c_pad
            c_max_length = mas_ref.shape[-1]
            c_mask = sequence_mask(c_lengths, c_max_length).unsqueeze(1).to(x_mask)
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), mas_ref ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), mas_ref)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const
            attn_mask = x_mask.unsqueeze(-1) * c_mask.unsqueeze(2)
            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

            y_max_length = c_max_length
            y_mask = c_mask
            # each valid frame should be assigned to exactly one token
            assert (attn.sum(dim=1)[y_mask.squeeze(1).bool()] == 1).all()
            # nothing assigned outside mask
            assert (attn * (1 - attn_mask.squeeze(1))).sum() == 0

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

    def compute_loss(self, x, x_lengths, y, y_lengths, c, c_lengths=None, spk=None, out_size=None):
        x, x_lengths, y, y_lengths, c, c_lengths = self.relocate_input([x, x_lengths, y, y_lengths, c, c_lengths])

        if self.n_spks > 1:
            # Get speaker embedding
            spk = self.spk_emb(spk)

        # Get encoder_outputs `mu_x` and log-scaled token durations `logw`
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spk)
        y_max_length = y.shape[-1]

        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)

        # Use MAS to find most likely alignment `attn` between text and mel-spectrogram
        with torch.no_grad():
            const = -0.5 * math.log(2 * math.pi) * self.n_feats
            factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
            y_square = torch.matmul(factor.transpose(1, 2), y ** 2)
            y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
            mu_square = torch.sum(factor * (mu_x ** 2), 1).unsqueeze(-1)
            log_prior = y_square - y_mu_double + mu_square + const

            attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
            attn = attn.detach()

        # Compute loss between predicted log-scaled durations and those obtained from MAS
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)

        # Cut a small segment of mel-spectrogram in order to increase batch size
        if not isinstance(out_size, type(None)):
            max_offset = (y_lengths - out_size).clamp(0)
            offset_ranges = list(zip([0] * max_offset.shape[0], max_offset.cpu().numpy()))
            out_offset = torch.LongTensor([
                torch.tensor(random.choice(range(start, end)) if end > start else 0)
                for start, end in offset_ranges
            ]).to(y_lengths)

            attn_cut = torch.zeros(attn.shape[0], attn.shape[1], out_size, dtype=attn.dtype, device=attn.device)
            y_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)
            c_cut = torch.zeros(y.shape[0], self.n_feats, out_size, dtype=y.dtype, device=y.device)

            y_cut_lengths = []
            for i, (y_, c_, out_offset_) in enumerate(zip(y, c, out_offset)):
                y_cut_length = out_size + (y_lengths[i] - out_size).clamp(None, 0)
                y_cut_lengths.append(y_cut_length)
                cut_lower, cut_upper = out_offset_, out_offset_ + y_cut_length
                y_cut[i, :, :y_cut_length] = y_[:, cut_lower:cut_upper]
                c_cut[i, :, :y_cut_length] = c_[:, cut_lower:cut_upper]
                attn_cut[i, :, :y_cut_length] = attn[i, :, cut_lower:cut_upper]
            y_cut_lengths = torch.LongTensor(y_cut_lengths)
            y_cut_mask = sequence_mask(y_cut_lengths).unsqueeze(1).to(y_mask)

            attn = attn_cut
            y = y_cut
            c = c_cut
            y_mask = y_cut_mask

        # Align encoded text with mel-spectrogram and get mu_y segment
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)

        # Compute loss of score-based decoder
        diff_loss, xt = self.decoder.compute_loss(y, y_mask, mu_y, c, spk)

        # Compute loss between aligned encoder outputs and mel-spectrogram
        prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
        prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)

        return dur_loss, prior_loss, diff_loss

    def load_weights(self, weights_path):
        state_dict = torch.load(weights_path, map_location=lambda loc, storage: loc)
        self.load_state_dict(state_dict, strict=False)
        self.decoder.estimator.init_weights_from_base(state_dict)