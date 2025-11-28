import torch
import numpy as np
from librosa.sequence import dtw

# -------------------------------------------------------
# Normalize mel spectrograms (per frequency bin)
# -------------------------------------------------------
def normalize_mel(mel: torch.Tensor):
    """
    mel: [n_mels, T]
    Returns normalized mel of same shape.
    """
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + 1e-9
    return (mel - mean) / std

# -------------------------------------------------------
# DTW distance between two mel spectrograms
# -------------------------------------------------------
def dtw_mel_distance(mel_ref: torch.Tensor, mel_test: torch.Tensor):
    """
    mel_ref, mel_test: torch tensors [n_mels, T]
    Returns scalar distance (lower = more similar)
    """
    mel_ref = normalize_mel(mel_ref)
    mel_test = normalize_mel(mel_test)

    # Move to CPU + numpy for librosa
    A = mel_ref.detach().cpu().numpy()
    B = mel_test.detach().cpu().numpy()

    # Compute DTW
    D, wp = dtw(A, B, metric='euclidean')

    # Normalize by path length
    total_cost = D[-1, -1]
    path_len = len(wp) + 1e-9
    return float(total_cost / path_len)

# -------------------------------------------------------
# Main comparison helper
# -------------------------------------------------------
def compare_to_reference(mel_ref, mel_1, mel_2, name1="test1", name2="test2"):
    """
    Returns distances and which test is closer.
    """
    d1 = dtw_mel_distance(mel_ref, mel_1)
    d2 = dtw_mel_distance(mel_ref, mel_2)

    if d1 < d2:
        winner = name1
    elif d2 < d1:
        winner = name2
    else:
        winner = "equal"

    return {
        f"distance_{name1}": d1,
        f"distance_{name2}": d2,
        "closer": winner
    }
