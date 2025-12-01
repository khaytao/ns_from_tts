import numpy as np
import torch
from librosa.sequence import dtw


def normalize_mel(mel: torch.Tensor) -> torch.Tensor:
    """Normalize mel spectrogram per frequency bin: (x - mean) / std."""
    # mel: [n_mels, T]
    mean = mel.mean(dim=1, keepdim=True)
    std = mel.std(dim=1, keepdim=True) + 1e-9
    return (mel - mean) / std


def dtw_mel_distance(mel_ref: torch.Tensor, mel_test: torch.Tensor) -> float:
    """
    Compute DTW distance between two mel spectrograms.

    Args:
        mel_ref: [n_mels, T_ref]
        mel_test: [n_mels, T_test]

    Returns:
        Scalar distance (lower = more similar).
    """
    mel_ref = normalize_mel(mel_ref)
    mel_test = normalize_mel(mel_test)

    A = mel_ref.detach().cpu().numpy()
    B = mel_test.detach().cpu().numpy()

    D, wp = dtw(A, B, metric="euclidean")
    total_cost = D[-1, -1]
    path_len = len(wp) + 1e-9

    return float(total_cost / path_len)


def compare_to_reference(
    mel_ref: torch.Tensor,
    mel_1: torch.Tensor,
    mel_2: torch.Tensor,
    name1: str = "test1",
    name2: str = "test2",
):
    """
    Compare two mel spectrograms to a reference using DTW.

    Supports:
        - Single example: [n_feats, T] -> returns dict.
        - Batch: [B, n_feats, T] -> returns list[dict].
    """

    def _single(m_ref: torch.Tensor, m_a: torch.Tensor, m_b: torch.Tensor):
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

    ndim = mel_ref.ndim

    if ndim == 2:
        # [n_feats, T] – compatibility with 2D tensors
        return _single(mel_ref, mel_1, mel_2)

    if ndim == 3:
        # [B, n_feats, T] – batch mode
        if not (mel_1.shape[0] == mel_ref.shape[0] == mel_2.shape[0]):
            raise ValueError(
                "Batched compare_to_reference: batch sizes must match."
            )

        B = mel_ref.shape[0]
        results = []
        for i in range(B):
            results.append(_single(mel_ref[i], mel_1[i], mel_2[i]))
        return results

    raise ValueError(
        "Expected mel tensors of shape [n_feats, T] or [B, n_feats, T], "
        f"got ndim={ndim} with shape {getattr(mel_ref, 'shape', None)}"
    )


def baseline_comparison(
    mel_ref: torch.Tensor,
    n: int,
    mean: float = 0.0,
    std: float = 1.0,
):
    """
    Compare random Gaussian mels to a reference using DTW.

    Args:
        mel_ref: reference mel, [n_mels, T].
        n: number of random samples.
        mean: Gaussian mean.
        std: Gaussian std.

    Returns:
        dict with keys:
            "mean": float
            "std": float
            "all_distances": list[float]
    """
    if mel_ref.ndim != 2:
        raise ValueError(
            "baseline_comparison expects mel_ref with shape [n_mels, T], "
            f"got {mel_ref.shape}"
        )

    distances = []
    for _ in range(n):
        rand_mel = torch.randn_like(mel_ref) * std + mean
        d = dtw_mel_distance(mel_ref, rand_mel)
        distances.append(d)

    distances = np.array(distances)
    return {
        "mean": float(distances.mean()),
        "std": float(distances.std()),
        "all_distances": distances.tolist(),
    }
