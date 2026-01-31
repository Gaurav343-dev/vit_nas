import random
import numpy as np
import torch
from torch import nn
import os
import sys

# ensure repo root is on sys.path so `modules` package can be imported when tests run from inside `modules/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.dynamic_modules import DynamicLinear


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_dynamic_linear_at_max_size():
    """Simplified test: only validate dynamic linear at its max sizes (forward + backward)."""
    seed_all(1234)

    max_in, max_out = 16, 10
    bias = True
    batch = 4

    # full (max) layer to derive deterministic weights
    full = nn.Linear(max_in, max_out, bias=bias)

    # dynamic layer wraps the full-size linear
    dyn = DynamicLinear(max_in, max_out, bias=bias)
    # copy full weights into dynamic's internal linear for deterministic test
    dyn.linear.weight.data.copy_(full.weight.data)
    if bias and dyn.linear.bias is not None:
        dyn.linear.bias.data.copy_(full.bias.data)

    # ensure dynamic uses full (max) active sizes
    dyn.active_out = max_out

    # reference linear with full sizes
    ref = nn.Linear(max_in, max_out, bias=bias)
    ref.weight.data.copy_(full.weight.data)
    if bias and ref.bias is not None:
        ref.bias.data.copy_(full.bias.data)

    # inputs at full feature size
    x = torch.randn(batch, max_in, requires_grad=True)
    x_ref = x.clone().detach().requires_grad_(True)

    y_dyn = dyn(x)
    y_ref = ref(x_ref)

    assert torch.allclose(y_dyn, y_ref, atol=1e-6, rtol=1e-6), "Forward outputs differ at max size"

    # backward
    grad = torch.ones_like(y_dyn)
    y_dyn.backward(grad)
    y_ref.backward(grad)

    assert x.grad is not None and x_ref.grad is not None, "Input gradients missing"
    assert torch.allclose(x.grad, x_ref.grad, atol=1e-6, rtol=1e-6), "Input gradients differ"

    # parameter grads (compare full tensors)
    assert dyn.linear.weight.grad is not None and ref.weight.grad is not None, "Weight grads missing"
    assert torch.allclose(dyn.linear.weight.grad, ref.weight.grad, atol=1e-6, rtol=1e-6), "Weight gradients differ"

    if bias:
        assert dyn.linear.bias is not None and dyn.linear.bias.grad is not None, "Bias grad missing"
        assert ref.bias is not None and ref.bias.grad is not None, "Ref bias grad missing"
        assert torch.allclose(dyn.linear.bias.grad, ref.bias.grad, atol=1e-6, rtol=1e-6), "Bias gradients differ"
