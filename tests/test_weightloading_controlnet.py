import torch
from torch import nn
import importlib

mod = importlib.import_module("model.diffusion_with_controlnet")
ControlCls = getattr(mod, "GradLogPEstimator2dWithControlNet")

class Tiny(ControlCls):
    """Tiny test double that avoids calling the heavy Grad-TTS base ctor."""
    def __init__(self):
        nn.Module.__init__(self)
        self.conv = nn.Conv2d(2, 2, kernel_size=3, padding=1)
        self.ctrl_in_ch = 1
        self.control = nn.ModuleDict()
        self.zero_convs = nn.ModuleList()
        self._control_built = False

def test_build_control_from_base_deepcopies_children():
    m = Tiny()
    assert len(m.control) == 0
    m.build_control_from_base()
    assert 'conv' in m.control
    assert isinstance(m.control['conv'], nn.Module)
    assert m.control['conv'].weight is not m.conv.weight

def test_load_weights_from_base_copies_into_control():
    m = Tiny()
    m.build_control_from_base()
    base_sd = {
        'conv.weight': torch.ones_like(m.conv.weight),
        'conv.bias': torch.zeros_like(m.conv.bias),
    }
    summary = m.load_weights_from_base(base_sd)
    assert summary['copied_to_control'] == 2
    assert torch.allclose(m.conv.weight, torch.ones_like(m.conv.weight))
    assert torch.allclose(m.conv.bias, torch.zeros_like(m.conv.bias))
    assert torch.allclose(m.control['conv'].weight, torch.ones_like(m.control['conv'].weight))
    assert torch.allclose(m.control['conv'].bias, torch.zeros_like(m.control['conv'].bias))

def test_freeze_base_trains_only_control():
    m = Tiny()
    m.build_control_from_base()
    m.load_weights_from_base({'conv.weight': m.conv.weight.data.clone(),
                              'conv.bias': m.conv.bias.data.clone()})
    counts = m.freeze_base(train_zero_convs=True)
    assert all(not p.requires_grad for n,p in m.named_parameters() if not n.startswith('control.') and not n.startswith('zero_convs'))
    assert all(p.requires_grad for n,p in m.named_parameters() if n.startswith('control.'))
    assert counts['trainable'] + counts['frozen'] == sum(p.numel() for p in m.parameters())

def test_load_weights_from_control_roundtrip():
    m = Tiny()
    m.build_control_from_base()
    m.load_weights_from_base({'conv.weight': m.conv.weight.data.clone(),
                              'conv.bias': m.conv.bias.data.clone()})
    sd = m.state_dict()
    m2 = Tiny()
    m2.build_control_from_base()
    summary = m2.load_weights_from_control(sd, strict=True)
    assert summary['missing'] == []
    assert summary['unexpected'] == []
    assert torch.allclose(m2.conv.weight, m.conv.weight)
    assert torch.allclose(m2.control['conv'].weight, m.control['conv'].weight)