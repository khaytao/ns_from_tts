import torch
import numpy as np
from librosa.sequence import dtw
import torch


def normalize_mel(mel: torch.Tensor):
    """
    mel: [n_mels, T]
    Returns normalized mel of same shape.
    """
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + 1e-9
    return (mel - mean) / std


def dtw_mel_distance(mel_ref: torch.Tensor, mel_test: torch.Tensor):
    """
    mel_ref, mel_test: torch tensors [n_mels, T]
    Returns scalar distance (lower = more similar)
    """
    mel_ref = normalize_mel(mel_ref)
    mel_test = normalize_mel(mel_test)


    A = mel_ref.detach().cpu().numpy()
    B = mel_test.detach().cpu().numpy()


    D, wp = dtw(A, B, metric='euclidean')


    total_cost = D[-1, -1]
    path_len = len(wp) + 1e-9
    return float(total_cost / path_len)


def compare_to_reference(mel_ref, mel_1, mel_2, name1="test1", name2="test2"):
    """
    Compute DTW distances between a reference mel and two test mels.

    Supports:
      - single example:  mel_* shape [n_feats, T]  -> returns dict
      - batched input:   mel_* shape [B, n_feats, T]  -> returns list[dict]
    """

    def _single(m_ref, m_a, m_b):
        d1 = dtw_mel_distance(m_ref, m_a)
        d2 = dtw_mel_distance(m_ref, m_b)

        if d1 < d2:
            winner = name1
        elif d2 < d1:
            winner = name2
        else:
            winner = "equal"

        return {
            f"distance_{name1}": d1,
            f"distance_{name2}": d2,
            "closer": winner,
        }


    ndim = getattr(mel_ref, "ndim", len(mel_ref.shape))

    if ndim == 2:
        # [n_feats, T] – Allow compatabilty with 2d array
        return _single(mel_ref, mel_1, mel_2)

    elif ndim == 3:
        # [B, n_feats, T] – batch mode
        if not (mel_1.shape[0] == mel_ref.shape[0] == mel_2.shape[0]):
            raise ValueError("Batched compare_to_reference: batch sizes must match.")

        B = mel_ref.shape[0]
        results = []
        for i in range(B):
            results.append(_single(mel_ref[i], mel_1[i], mel_2[i]))
        return results

    else:
        raise ValueError(
            f"Expected mel tensors of shape [n_feats, T] or [B, n_feats, T], "
            f"got ndim={ndim} with shape {getattr(mel_ref, 'shape', None)}"
        )

def baseline_comparison(mel_ref: torch.Tensor, n: int, mean: float = 0.0, std: float = 1.0):
    """
    Compare n random Gaussian mel spectrograms to a reference and
    return the mean and std of the DTW distances.

    Parameters
    ----------
    mel_ref : torch.Tensor
        Reference mel, shape [n_mels, T].
    n : int
        Number of random Gaussian samples to compare.
    mean : float, optional
        Mean of the Gaussian noise (default: 0.0).
    std : float, optional
        Std of the Gaussian noise (default: 1.0).

    Returns
    -------
    dict
        {
            "mean": float,
            "std": float,
            "all_distances": list of float
        }
    """
    if mel_ref.ndim != 2:
        raise ValueError(f"baseline_comparison expects mel_ref with shape [n_mels, T], got {mel_ref.shape}")

    distances = []
    print("mel-distance to self ",dtw_mel_distance(mel_ref, mel_ref) )
    for _ in range(n):
        rand_mel = torch.randn_like(mel_ref) * std + mean
        d = dtw_mel_distance(mel_ref, rand_mel)
        distances.append(d)

    distances = np.array(distances)
    return {
        "mean": float(distances.mean()),
        "std": float(distances.std()),
        "all_distances": distances.tolist()
    }
