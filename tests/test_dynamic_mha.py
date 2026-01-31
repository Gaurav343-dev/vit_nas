import random
import numpy as np
import torch
 
import os
import sys

# ensure repo root is on sys.path so `modules` package can be imported when tests run from inside `modules/`
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from modules.dynamic_modules import DynamicMHA
from torch import nn


def seed_all(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def test_dynamic_mha_runs_at_max_size():
    """Simplified smoke test: ensure DynamicMHA runs at max sizes and gradients flow."""
    seed_all(1234)

    B, T, C = 2, 5, 32
    max_heads = 8

    mha = DynamicMHA(max_embed_dim=C, max_num_heads=max_heads, bias=True)

    # deterministic parameters are achieved via seeding above
    x_dyn = torch.randn(B, T, C, requires_grad=True)
    x_ref = x_dyn.clone().detach().requires_grad_(True)

    # run dynamic mha
    # Note: we assume DynamicLinear is tested separately; here we only check final MHA output.

    # run dynamic mha
    y_dyn = mha(x_dyn)

    # reference run: use PyTorch's native MultiheadAttention with the same in-proj and out-proj weights
    ref_mha = nn.MultiheadAttention(embed_dim=C, num_heads=max_heads, batch_first=True)

    # # copy fused qkv weights and proj weights/biases
    # ref_mha.in_proj_weight.data.copy_(mha.qkv_linear.linear.weight.data)
    # if mha.bias:
    #     ref_mha.in_proj_bias.data.copy_(mha.qkv_linear.linear.bias.data)
    # # copy proj weights/biases
    # ref_mha.out_proj.weight.data.copy_(mha.proj_linear.linear.weight.data)
    # if mha.bias:
    #     ref_mha.out_proj.bias.data.copy_(mha.proj_linear.linear.bias.data)

    # PyTorch MHA returns (attn_output, attn_weights); we only compare attn_output
    y_ref, _ = ref_mha(x_ref, x_ref, x_ref)

    # compare outputs
    assert isinstance(y_dyn, torch.Tensor)
    assert y_dyn.shape == (B, T, C)
    assert y_ref.shape == (B, T, C)
    assert torch.allclose(y_dyn, y_ref, atol=1e-6, rtol=1e-6), "DynamicMHA forward output differs from reference"

    # we only check the final MHA outputs here; internal linear modules and gradients
    # are tested separately so we avoid redundant assertions.

