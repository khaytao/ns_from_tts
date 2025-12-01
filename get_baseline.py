import torch
import random
from data import TextMelNoisyMelDataset
from mel_comparison import baseline_comparison

FILELIST = r"resources/filelists/ljspeech/test_fixed.txt"
CMU_PATH = "./resources/cmu_dictionary"

import controlnet_params as params

n_fft      = params.n_fft
n_feats    = params.n_feats
sr         = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min      = params.f_min
f_max      = params.f_max

dataset = TextMelNoisyMelDataset(
    FILELIST,
    CMU_PATH,
    params.add_blank,
    n_fft, n_feats, sr,
    hop_length, win_length,
    f_min, f_max,
    params.snr_db
)

idx = random.randint(0, len(dataset) - 1)
item = dataset[idx]

mel_ref = item["y"]
mel_ref = mel_ref.cuda()

print("Loaded reference mel shape:", mel_ref.shape)

result = baseline_comparison(mel_ref, n=50)

print("\nBaseline DTW comparison:")
print("  Mean:", result["mean"])
print("  Std :", result["std"])
