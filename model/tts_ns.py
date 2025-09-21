import math
import random
import torch

from model import monotonic_align
from model.base import BaseModule
from model.text_encoder import TextEncoder
from model.diffusion_with_controlnet import DiffusionWithControlNet
from model.utils import sequence_mask, generate_path, duration_loss, fix_len_compatibility
from model.tts import GradTTS


class TTSNS(GradTTS):

    def __init__(self, n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                 n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                 n_feats, dec_dim, beta_min, beta_max, pe_scale):
        super(TTSNS, self).__init__(n_vocab, n_spks, spk_emb_dim, n_enc_channels, filter_channels, filter_channels_dp,
                                    n_heads, n_enc_layers, enc_kernel, enc_dropout, window_size,
                                    n_feats, dec_dim, beta_min, beta_max, pe_scale)

        self.decoder = DiffusionWithControlNet(n_feats, dec_dim, n_spks, spk_emb_dim, beta_min, beta_max, pe_scale)