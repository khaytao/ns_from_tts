# Copyright (C) 2021. Huawei Technologies Co., Ltd. All rights reserved.
# This program is free software; you can redistribute it and/or modify
# it under the terms of the MIT License.
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the
# MIT License for more details.

import argparse
import json
import datetime as dt
import numpy as np
from scipy.io.wavfile import write

import torch

import params
from model import GradTTS
from model import GradTTS_NS
from text import text_to_sequence, cmudict
from text.symbols import symbols
from utils import intersperse

from torch.utils.data import DataLoader
from data import TextMelDataset, TextMelBatchCollate

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

def count_lines(path, encoding="utf-8", skip_blank=False):
    """
    Count lines in a text file.

    Args:
        path (str | Path): File path.
        encoding (str): Text encoding.
        skip_blank (bool): If True, ignore empty/whitespace-only lines.

    Returns:
        int: Number of lines.
    """
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


HIFIGAN_CONFIG = './checkpts/hifigan-config.json'
HIFIGAN_CHECKPT = './checkpts/hifigan.pt'

n_feats = params.n_feats
n_fft = params.n_fft
sample_rate = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min = params.f_min
f_max = params.f_max

USE_MAS = False

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', type=str, required=True, help='path to a file with texts to synthesize')
    parser.add_argument('-c', '--checkpoint', type=str, required=True, help='path to a checkpoint of Grad-TTS')
    parser.add_argument('-t', '--timesteps', type=int, required=False, default=10,
                        help='number of timesteps of reverse diffusion')
    parser.add_argument('-s', '--speaker_id', type=int, required=False, default=None,
                        help='speaker id for multispeaker model')
    args = parser.parse_args()

    if not isinstance(args.speaker_id, type(None)):
        assert params.n_spks > 1, "Ensure you set right number of speakers in `params.py`."
        spk = torch.LongTensor([args.speaker_id]).cuda()
    else:
        spk = None

    num_samples = count_lines(args.file, skip_blank=True)


    weight_name = args.checkpoint.split('/')[-1]
    weight_name = weight_name.split('.')[0]
    print('Initializing Grad-TTS...')
    generator = GradTTS_NS(len(symbols) + 1, params.n_spks, params.spk_emb_dim,
                           params.n_enc_channels, params.filter_channels,
                           params.filter_channels_dp, params.n_heads, params.n_enc_layers,
                           params.enc_kernel, params.enc_dropout, params.window_size,
                           params.n_feats, params.dec_dim, params.beta_min, params.beta_max, params.pe_scale)

    generator.load_state_dict(torch.load(args.checkpoint, map_location=lambda loc, storage: loc), strict=True)
    generator.decoder.estimator.is_initialized = True
    _ = generator.cuda().eval()
    print(f'Number of parameters: {generator.nparams}')

    print('Initializing HiFi-GAN...')
    with open(HIFIGAN_CONFIG) as f:
        h = AttrDict(json.load(f))
    vocoder = HiFiGAN(h)
    vocoder.load_state_dict(torch.load(HIFIGAN_CHECKPT, map_location=lambda loc, storage: loc)['generator'])
    _ = vocoder.cuda().eval()
    vocoder.remove_weight_norm()

    # with open(args.file, 'r', encoding='utf-8') as f:
    #     texts = [line.strip() for line in f.readlines()]

    CMU_PATH = './resources/cmu_dictionary'
    cmu = cmudict.CMUDict(CMU_PATH)
    add_blank = params.add_blank

    test_dataset = TextMelDataset(args.file, CMU_PATH, add_blank,
                                  n_fft, n_feats, sample_rate, hop_length,
                                  win_length, f_min, f_max)
    # batch_collate = TextMelBatchCollate()
    # loader = DataLoader(dataset=test_dataset, batch_size=1,
    #                     collate_fn=batch_collate, drop_last=True,
    #                     num_workers=1, shuffle=False)
    test_batch = test_dataset.sample_test_batch(num_samples)

    with torch.no_grad():
        for i, item in enumerate(test_batch):
            mel = item['y'].cuda()
            mel = mel[None, :, :]

            print(f'Synthesizing {i} text...', end=' ')
            # x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
            x = item['x'].to(torch.long).unsqueeze(0).cuda()
            x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
            c_lengths = torch.LongTensor([mel.shape[-1]]).cuda()
            t = dt.datetime.now()
            y_enc, y_dec, attn = generator.forward(x, x_lengths, mel, c_lengths, n_timesteps=args.timesteps,
                                                   temperature=1.5,
                                                   stoc=False, spk=spk, length_scale=1, use_mas=USE_MAS)
            t = (dt.datetime.now() - t).total_seconds()
            print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')

            audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)

            write(f'./out/sample_{i}_{weight_name}.wav', 22050, audio)

    print('Done. Check out `out` folder for samples.')

    # with torch.no_grad():
    #     for i, text in enumerate(test_batch):
    #         print(f'Synthesizing {i} text...', end=' ')
    #         x = torch.LongTensor(intersperse(text_to_sequence(text, dictionary=cmu), len(symbols))).cuda()[None]
    #         x_lengths = torch.LongTensor([x.shape[-1]]).cuda()
    #
    #         t = dt.datetime.now()
    #         y_enc, y_dec, attn = generator.forward(x, x_lengths, n_timesteps=args.timesteps, temperature=1.5,
    #                                                stoc=False, spk=spk, length_scale=0.91)
    #         t = (dt.datetime.now() - t).total_seconds()
    #         print(f'Grad-TTS RTF: {t * 22050 / (y_dec.shape[-1] * 256)}')
    #
    #         audio = (vocoder.forward(y_dec).cpu().squeeze().clamp(-1, 1).numpy() * 32768).astype(np.int16)
    #
    #         write(f'./out/sample_{i}.wav', 22050, audio)
    #
    # print('Done. Check out `out` folder for samples.')
