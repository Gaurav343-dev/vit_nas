import random
import numpy as np
import torch
from torch import nn
import os
import sys

# ensure repo root is on sys.path so `modules` package can be imported when tests run from inside `modules/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.dynamic_modules import DynamicLayerNorm


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_dynamic_layernorm_at_max_size():
    """Simplified test: only validate DynamicLayerNorm at its max feature size."""
    seed_all(1234)

    max_feat = 16
    batch, seq = 4, 3

    full = nn.LayerNorm(max_feat)
    dyn = DynamicLayerNorm(max_feat)

    # copy params
    if full.weight is not None:
        dyn.layer_norm.weight.data.copy_(full.weight.data)
    if full.bias is not None:
        dyn.layer_norm.bias.data.copy_(full.bias.data)

    dyn.active_features = max_feat

    ref = nn.LayerNorm(max_feat)
    ref.weight.data.copy_(full.weight.data)
    ref.bias.data.copy_(full.bias.data)

    # create input with last dim == max_feat
    x = torch.randn(batch, seq, max_feat, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    y_dyn = dyn(x)
    y_ref = ref(x_ref)

    assert torch.allclose(y_dyn, y_ref, atol=1e-6, rtol=1e-6), "Forward outputs differ at max size"

    grad = torch.ones_like(y_dyn)
    y_dyn.backward(grad)
    y_ref.backward(grad)

    assert x.grad is not None and x_ref.grad is not None, "Input gradients missing"
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6), "Input gradients differ"

    assert dyn.layer_norm.weight.grad is not None and ref.weight.grad is not None, "Weight grads missing"
    assert torch.allclose(dyn.layer_norm.weight.grad, ref.weight.grad, atol=1e-6, rtol=1e-6), "Weight gradients differ"

    assert dyn.layer_norm.bias.grad is not None and ref.bias.grad is not None, "Bias grads missing"
    assert torch.allclose(dyn.layer_norm.bias.grad, ref.bias.grad, atol=1e-6, rtol=1e-6), "Bias gradients differ"
