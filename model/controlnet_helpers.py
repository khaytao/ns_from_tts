import torch
import torch.nn as nn
import torch.nn.functional as F

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

def get_active_times_mask(c_pad, c_mask, smooth_k=5, silence_ratio=0.3):
    B, _, T = c_pad.shape
    energy = (c_pad ** 2).sum(dim=1, keepdim=True)                 # [B,1,T_pad]
    if smooth_k and smooth_k > 1:
        pad = (smooth_k - 1) // 2
        energy = F.avg_pool1d(energy, kernel_size=smooth_k, stride=1, padding=pad)

    # per-sample threshold: median * silence_ratio (0.2-0.4 typical)
    med = energy.median(dim=-1, keepdim=True).values               # [B,1,1]
    thr = med * float(silence_ratio)
    voiced = (energy >= thr).to(c_mask.dtype)                   # [B,1,T_pad]
    # keep only valid frames (respect y_mask)
    voiced = voiced * c_mask
    return voiced

def is_control_layer(name: str) -> bool:
    # control branches and zero-conv taps
    return name.startswith("control_")


def _is_zero_conv_layer(name: str) -> bool:
    # control branches and zero-conv taps
    return name.startswith("z_input") \
        or name.startswith("z_middle") \
        or name.startswith("z_downs") \
        or name.startswith("z_ups")


def is_base_layer(name: str) -> bool:
    return not (is_control_layer(name) or _is_zero_conv_layer(name))


def zero_conv(in_channels, out_channels):
    conv = nn.Conv2d(in_channels, out_channels, kernel_size=1)
    nn.init.zeros_(conv.weight)
    nn.init.zeros_(conv.bias)
    return conv
