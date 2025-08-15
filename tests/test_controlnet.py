import importlib
import torch
from torch import nn
import pytest

from model.diffusion_with_controlnet import *
# Import zero_conv from your module


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




# # Import base & controlnet modules
# base_mod = importlib.import_module("model.diffusion")                      # original UNet parts
# ctrl_mod = importlib.import_module("model.diffusion_with_controlnet")      # your subclass
#
# GradLogPEstimator2d = getattr(base_mod, "GradLogPEstimator2d")
# ResnetBlock = getattr(base_mod, "ResnetBlock")
# Residual = getattr(base_mod, "Residual")
# Rezero = getattr(base_mod, "Rezero")
# LinearAttention = getattr(base_mod, "LinearAttention")
# Downsample = getattr(base_mod, "Downsample")
#
# GradLogPEstimator2dWithControlNet = getattr(
#     ctrl_mod, "GradLogPEstimator2dWithControlNet"
# )
# zero_conv = getattr(ctrl_mod, "zero_conv")


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
