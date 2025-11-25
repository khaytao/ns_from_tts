import argparse
import os
import json
import datetime as dt
import shutil
from pathlib import Path

import numpy as np
from scipy.io.wavfile import write

import torch
import torchaudio as ta

import controlnet_params as params
from model import GradTTS, GradTTS_NS  # baseline and control
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from data import TextMelNoisyMelDataset

import sys

sys.path.append('./hifi-gan/')
from env import AttrDict
from models import Generator as HiFiGAN

import random

torch.manual_seed(1234)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
np.random.seed(1234)
random.seed(1234)


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

USE_MAS = True  # kept for interface compatibility, not used here


def count_lines(path, encoding="utf-8", skip_blank=False):
    """Count lines in a text file."""
    count = 0
    with open(path, "r", encoding=encoding, errors="ignore") as f:
        if skip_blank:
            for line in f:
                if line.strip():
                    count += 1
        else:
            for _ in f:
                count += 1
    return count


def save_wav(audio_tensor: torch.Tensor, dest_path: str, sr: int = 22050):
    """Save float tensor in [-1,1] to 16-bit PCM wav."""
    audio_int16 = (audio_tensor.clamp(-1, 1).numpy() * 32768).astype(np.int16)
    write(dest_path, sr, audio_int16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare clean reference audio with baseline TTS synthesis (no ControlNet).')
    parser.add_argument('-f', '--file', type=str, required=True, help='Path to filelist (wav|text) to analyse')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='Path to GradTTS_NS checkpoint (with ControlNet)')
    parser.add_argument('-gradc', '--grad_checkpoint', type=str, required=True, help='Path to GradTTS checkpoint (baseline, no control)')
    parser.add_argument('-n', '--num_samples', type=int, default=10, help='Number of random samples to analyse')
    parser.add_argument('-t', '--timesteps', type=int, default=10, help='Number of diffusion reverse steps')
    parser.add_argument('-s', '--speaker_id', type=int, default=None, help='Speaker id for multi-speaker model')
    parser.add_argument('-o', '--outdir', type=str, default='./analysis', help='Directory to write results')
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Speaker conditioning
    if args.speaker_id is not None:
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    # Dataset (provides wav paths + text)
    CMU_PATH = './resources/cmu_dictionary'
    cmu = cmudict.CMUDict(CMU_PATH)
    add_blank = params.add_blank

    dataset = TextMelNoisyMelDataset(args.file, CMU_PATH, add_blank,
                                     n_fft, n_feats, sample_rate, hop_length,
                                     win_length, f_min, f_max, params.snr_db)

    # Random subset of indices
    total_available = len(dataset)
    n_select = min(args.num_samples, total_available)
    indices = random.sample(range(total_available), n_select)

    # Initialize baseline Grad-TTS (no control)
    print('Initializing Grad-TTS (baseline)...')
    base_generator = GradTTS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                             params.n_enc_channels, params.filter_channels,
                             params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                             params.enc_kernel, params.enc_dropout, params.window_size,
                             params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    base_generator.load_state_dict(torch.load(args.grad_checkpoint, map_location=lambda loc, storage: loc), strict=True)
    _ = base_generator.cuda().eval()
    print(f'Baseline parameters: {base_generator.nparams}')

    # Initialize control Grad-TTS_NS
    print('Initializing Grad-TTS-NS (with ControlNet)...')
    control_generator = GradTTS_NS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                                   params.n_enc_channels, params.filter_channels,
                                   params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                                   params.enc_kernel, params.enc_dropout, params.window_size,
                                   params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    control_state = torch.load(args.checkpoint, map_location=lambda loc, storage: loc)
    control_generator.load_state_dict(control_state, strict=True)
    # mark estimator initialized if attribute present
    if hasattr(control_generator.decoder, 'estimator'):
        control_generator.decoder.estimator.is_initialized = True
    _ = control_generator.cuda().eval()
    print(f'Control parameters: {control_generator.nparams}')

    # Initialize HiFi-GAN vocoder
    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    print(f'Processing {n_select} samples...')
    with torch.no_grad():
        for out_idx, dataset_idx in enumerate(indices):
            item = dataset[dataset_idx]
            wav_path = dataset.filepaths_and_text[dataset_idx][0]
            text = dataset.filepaths_and_text[dataset_idx][1]

            # Prepare text tokens
            x = item['x'].to(torch.long).unsqueeze(0).cuda()
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()

            t0 = dt.datetime.now()
            y_enc_base, y_dec_base, _ = base_generator.forward(x, x_lengths,
                                                               n_timesteps=args.timesteps,
                                                               temperature=1.5, stoc=False,
                                                               spk=spk, length_scale=1.0)
            gen_time = (dt.datetime.now() - t0).total_seconds()
            rtf_base = gen_time * sample_rate / (y_dec_base.shape[-1] * hop_length)

            # Control synthesis
            mel_control = item['c'].cuda()[None, :, :]
            mel_clean = item['y'].cuda()
            c_lengths = torch.LongTensor([mel_control.shape[-1]]).cuda()

            t1 = dt.datetime.now()
            y_enc_ctrl, y_dec_ctrl, _ = control_generator.forward(x, x_lengths,
                                                                  mel_control, c_lengths,
                                                                  n_timesteps=args.timesteps,
                                                                  temperature=1.5, stoc=False,
                                                                  spk=spk, length_scale=1.0,
                                                                  use_mas=USE_MAS, clean=mel_clean)
            gen_time_ctrl = (dt.datetime.now() - t1).total_seconds()
            rtf_ctrl = gen_time_ctrl * sample_rate / (y_dec_ctrl.shape[-1] * hop_length)

            print(f'Sample {out_idx}: RTF base={rtf_base:.3f} | RTF control={rtf_ctrl:.3f}')

            # Vocoder
            audio_base = vocoder.forward(y_dec_base).cpu().squeeze()
            audio_control = vocoder.forward(y_dec_ctrl).cpu().squeeze()

            # Load reference clean wav
            audio_clean, sr = ta.load(wav_path)
            if sr != sample_rate:
                raise ValueError(f'Sample rate mismatch, expected {sample_rate}, got {sr}')
            if audio_clean.size(0) > 1:
                audio_clean = audio_clean.mean(dim=0, keepdim=False)

            # Prepare output directory
            sample_dir = Path(args.outdir) / f'sample_{out_idx}'
            sample_dir.mkdir(parents=True, exist_ok=True)

            # Save files
            save_wav(audio_clean.squeeze(), sample_dir / 'clean.wav', sr)
            save_wav(audio_base, sample_dir / 'synth_base.wav', sr)
            save_wav(audio_control, sample_dir / 'synth_control.wav', sr)

            # Also store metadata for convenience
            with open(sample_dir / 'meta.txt', 'w', encoding='utf-8') as meta_f:
                meta_f.write(f'text: {text}\n')
                meta_f.write(f'rtf_base: {rtf_base:.4f}\n')
                meta_f.write(f'rtf_control: {rtf_ctrl:.4f}\n')

    print(f'Done. Check "{Path(args.outdir).resolve()}" for results.')