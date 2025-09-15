import importlib
import torch
from torch import nn
import pytest
import os
from model.diffusion_with_controlnet import *
# Import zero_conv from your module


from model.diffusion_with_controlnet import GradLogPEstimator2dWithControlNet

pytest.importorskip("model.diffusion_with_controlnet")
from model.diffusion_with_controlnet import DiffusionWithControlNet

CKPT_PATH = "../checkpts/grad-tts.pt"

@pytest.mark.parametrize("in_ch,out_ch", [(1, 1), (3, 5), (80, 80)])
def test_zero_conv_init_and_shapes(in_ch, out_ch):
    z = zero_conv(in_ch, out_ch)
    # Parameter existence & shapes
    assert isinstance(z, nn.Conv2d)
    assert z.kernel_size == (1, 1)
    assert z.in_channels == in_ch
    assert z.out_channels == out_ch
    assert z.weight.shape == (out_ch, in_ch, 1, 1)
    assert z.bias.shape == (out_ch,)

    # Zero initialization
    assert torch.count_nonzero(z.weight) == 0
    assert torch.count_nonzero(z.bias) == 0

    # Trainability by default
    assert z.weight.requires_grad is True
    assert z.bias.requires_grad is True

@pytest.mark.parametrize("in_ch,out_ch,H,W", [(1, 1, 4, 6), (3, 5, 7, 9)])
def test_zero_conv_forward_outputs_zero(in_ch, out_ch, H, W):
    z = zero_conv(in_ch, out_ch)
    x = torch.randn(2, in_ch, H, W)  # B=2
    y = z(x)
    # Shape correct
    assert y.shape == (2, out_ch, H, W)
    # Because weights and bias are zero, output must be exactly zero
    assert torch.allclose(y, torch.zeros_like(y))

def test_zero_conv_grad_flow():
    """Even though init is zero, params should receive gradients when used in a loss."""
    z = zero_conv(2, 3)
    x = torch.randn(4, 2, 8, 8)
    # Target not all zeros so gradient is non-trivial
    target = torch.randn(4, 3, 8, 8)

    y = z(x)
    loss = (y - target).pow(2).mean()
    loss.backward()

    # Both weight and bias should accumulate non-zero (or at least defined) grads
    assert z.weight.grad is not None
    assert z.bias.grad is not None
    # Grad norms should be > 0 for a generic random target
    assert z.weight.grad.norm().item() > 0
    assert z.bias.grad.norm().item() > 0


@pytest.mark.parametrize(
    "n_spks, dim, dim_mults, n_feats",
    [
        (1, 16, (1, 2), 80),     # single-speaker small
        (10, 16, (1, 2, 4), 80), # multi-speaker small
    ],
)
def test_controlnet_init_structure(n_spks, dim, dim_mults, n_feats):
    """Basic structural checks on init (no forward)."""
    # m = GradLogPEstimator2dWithControlNet(
    #     dim=dim, dim_mults=dim_mults, n_spks=n_spks, n_feats=n_feats
    # )
    m = GradLogPEstimator2dWithControlNet(dim, n_spks=n_spks)
    # Base attributes exist (inherited)
    assert isinstance(m.downs, nn.ModuleList)
    assert isinstance(m.ups, nn.ModuleList)
    assert isinstance(m.mid_block1, ResnetBlock)
    assert isinstance(m.mid_attn, Residual)
    assert isinstance(m.mid_block2, ResnetBlock)

    # ControlNet taps (as implemented now): z_input, z_middle, z_downs
    assert isinstance(m.z_input, nn.Conv2d)
    assert isinstance(m.z_middle, nn.Conv2d)
    assert isinstance(m.z_downs, nn.ModuleList)
    assert len(m.z_downs) == len(m.downs)

    # zero_conv properties: 1x1, zeros at init
    for conv in [m.z_input, m.z_middle, *list(m.z_downs)]:
        assert conv.kernel_size == (1, 1)
        assert torch.count_nonzero(conv.weight) == 0
        assert torch.count_nonzero(conv.bias) == 0

    # Control branches exist and mirror the down path count
    assert isinstance(m.control_downs, nn.ModuleList)
    assert len(m.control_downs) == len(m.downs)

    # Each control_downs stage has the same block layout as the base downs:
    # [ResnetBlock, ResnetBlock, Residual(Rezero(LinearAttention)), Downsample|Identity]
    for i, stage in enumerate(m.control_downs):
        assert isinstance(stage, nn.ModuleList)
        assert len(stage) == 4
        assert isinstance(stage[0], ResnetBlock)
        assert isinstance(stage[1], ResnetBlock)
        assert isinstance(stage[2], Residual)
        # Try to peek inside Residual(Rezero(LinearAttention)) if attributes are exposed
        # (This is a soft check: at least the wrapper types exist)
        assert isinstance(stage[3], (Downsample, nn.Identity))

    # Control mid blocks exist
    assert isinstance(m.control_mid_block1, ResnetBlock)
    assert isinstance(m.control_mid_attn, Residual)
    assert isinstance(m.control_mid_block2, ResnetBlock)


def test_zero_conv_factory_behaviour():
    """Directly test zero_conv factory returns a 1x1, zero-initialized Conv2d."""
    z = zero_conv(7, 11)
    assert isinstance(z, nn.Conv2d)
    assert z.kernel_size == (1, 1)
    assert z.in_channels == 7
    assert z.out_channels == 11
    # zero init
    assert torch.count_nonzero(z.weight) == 0
    assert torch.count_nonzero(z.bias) == 0
    # forward returns zeros at init
    x = torch.randn(2, 7, 5, 9)
    y = z(x)
    assert y.shape == (2, 11, 5, 9)
    assert torch.allclose(y, torch.zeros_like(y))


def _unwrap_state_dict(loaded, key_hint=None):
    """
    Accepts whatever torch.load returned and extracts a plain state_dict:
    - if already a dict of tensors, return as-is
    - else try common wrappers: key_hint, 'model', 'state_dict'
    - strip common DDP prefixes (module./model.)
    """
    # 1) Pick the mapping that holds tensors
    sd = None
    if isinstance(loaded, dict):
        # Try the hint first if provided
        if key_hint and isinstance(loaded.get(key_hint), dict):
            sd = loaded[key_hint]
        elif all(isinstance(v, torch.Tensor) for v in loaded.values()):
            sd = loaded
        elif isinstance(loaded.get("model"), dict):
            sd = loaded["model"]
        elif isinstance(loaded.get("state_dict"), dict):
            sd = loaded["state_dict"]

    if sd is None or not isinstance(sd, dict):
        raise AssertionError("Could not find a model state_dict in checkpoint: "
                             "checked key_hint, 'model', and 'state_dict'")

    # 2) Normalize common prefixes
    def strip_prefix(k):
        for pfx in ("module.", "model.", "generator."):
            if k.startswith(pfx):
                return k[len(pfx):]
        return k

    return {strip_prefix(k): v for k, v in sd.items()}


@pytest.mark.skipif(not os.path.exists(CKPT_PATH), reason=f"Missing checkpoint: {CKPT_PATH}")
def test_init_weights_from_base_checkpoint(tmp_path):
    # ---- Load checkpoint ----
    loaded = torch.load(CKPT_PATH, map_location="cpu")
    sd = _unwrap_state_dict(loaded)

    # ---- Instantiate model (adjust if your arch differs) ----
    # Common Grad-TTS defaults: dim=128, dim_mults=(1,2,4), n_spks=1, n_feats=80
    model = GradLogPEstimator2dWithControlNet(
        dim=64, dim_mults=(1, 2, 4), n_spks=1, n_feats=80, pe_scale=1000
    ).cpu()

    # Sanity: ensure expected attributes exist
    assert hasattr(model, "downs") and hasattr(model, "control_downs")
    assert hasattr(model, "z_input") and hasattr(model, "z_middle") and hasattr(model, "z_downs")

    # ---- Initialize from base checkpoint ----
    weight_prefix = "decoder.estimator"
    summary = model.init_weights_from_base(sd, prefix_to_ignore=weight_prefix)

    # Basic summary signals
    assert summary["loaded_base_params"] > 0
    assert summary["copied_to_control"] > 0

    # ---- Zero-conv taps must be exactly zero ----
    with torch.no_grad():
        for z in [model.z_input, model.z_middle, *list(model.z_downs)]:
            assert torch.count_nonzero(z.weight) == 0
            assert torch.count_nonzero(z.bias) == 0

    # ---- Spot-check: a sample of base -> control tensors must match ----
    # Collect a few representative keys that should be mirrored
    model_sd = model.state_dict()
    mirrored_pairs = []
    for k in model_sd.keys():
        # map base -> control name the same way init_weights_from_base does
        if k.startswith("downs."):
            mirrored_pairs.append((k, "control_" + k))
        elif k.startswith("mid_block1."):
            mirrored_pairs.append((k, "control_mid_block1." + k[len("mid_block1."):]))
        elif k.startswith("mid_attn."):
            mirrored_pairs.append((k, "control_mid_attn." + k[len("mid_attn."):]))
        elif k.startswith("mid_block2."):
            mirrored_pairs.append((k, "control_mid_block2." + k[len("mid_block2."):]))
        # Only need a subset
        if len(mirrored_pairs) >= 12:
            break

    assert mirrored_pairs, "No mirrored base/control key pairs found to check"

    # Compare tensors for equality
    for base_k, ctrl_k in mirrored_pairs:
        if ctrl_k in model_sd and model_sd[base_k].shape == model_sd[ctrl_k].shape:
            assert torch.allclose(model_sd[base_k], model_sd[ctrl_k]), \
                f"Mismatch between {base_k} and {ctrl_k}"

    # ---- Extra check: base tensors equal checkpoint where shapes match ----
    # Strip the same prefix we ignored during init so keys line up.

    def strip_prefix(k: str, pfx: str) -> str:
        pfx_dot = pfx + "."
        return k[len(pfx_dot):] if k.startswith(pfx_dot) else k

    checked = 0
    for k, v in sd.items():
        nk = strip_prefix(k, weight_prefix)
        if nk in model_sd and model_sd[nk].shape == v.shape:
            assert torch.allclose(model_sd[nk], v), f"Base weight mismatch for {nk}"
            checked += 1
            if checked >= 10:
                break
    assert checked > 0, "No overlapping base keys found after prefix stripping"




def _randomize_params(model, seed=123):
    g = torch.Generator(device="cpu").manual_seed(seed)
    with torch.no_grad():
        for p in model.parameters():
            if p.is_floating_point():
                p.copy_(torch.randn_like(p, generator=g))
            else:
                # Leave non-float tensors (e.g., buffers or ints) unchanged
                pass

