import torch
import random
from data import TextMelNoisyMelDataset
from mel_comparison import baseline_comparison   # the function we wrote

# ---------------------------
# Minimal configuration
# ---------------------------
FILELIST = r".\resources\filelists\ljspeech\metadata_test_fixed copy.txt"
CMU_PATH = "./resources/cmu_dictionary"

import controlnet_params as params

n_fft      = params.n_fft
n_feats    = params.n_feats
sr         = params.sample_rate
hop_length = params.hop_length
win_length = params.win_length
f_min      = params.f_min
f_max      = params.f_max

# ---------------------------
# Load dataset
# ---------------------------
dataset = TextMelNoisyMelDataset(
    FILELIST,
    CMU_PATH,
    params.add_blank,
    n_fft, n_feats, sr,
    hop_length, win_length,
    f_min, f_max,
    params.snr_db
)

# ---------------------------
# Pick *one* random reference item
# ---------------------------
idx = random.randint(0, len(dataset) - 1)
item = dataset[idx]

mel_ref = item["y"]    # shape [n_mels, T]
mel_ref = mel_ref.cuda()  # optional

print("Loaded reference mel shape:", mel_ref.shape)

# ---------------------------
# Run baseline comparison
# ---------------------------
result = baseline_comparison(mel_ref, n=50)  # 50 random Gaussian signals

print("\nBaseline DTW comparison:")
print("  Mean:", result["mean"])
print("  Std :", result["std"])
