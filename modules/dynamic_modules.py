from torch import nn
import torch.nn.functional as F


class DynamicLinear(nn.Module):
    def __init__(self, max_in, max_out, bias=True):
        super().__init__()

        self.max_in = max_in
        self.max_out = max_out
        # self.active_in = max_in # not needed as input features
        # are set by input tensor shape
        self.active_out = max_out
        self.bias = bias
        self.linear = nn.Linear(self.max_in, self.max_out, bias=self.bias)

    def forward(self, x):
        active_in = x.size(-1)
        weight = self.linear.weight[:self.active_out, :active_in]
        bias = self.linear.bias[: self.active_out] if self.bias else None
        return F.linear(
            x, weight.contiguous(), bias
        )  # contiguous for potential performance


class DynamicLayerNorm(nn.Module):
    def __init__(self, max_in):
        super().__init__()
        self.max_features = max_in
        self.layer_norm = nn.LayerNorm(max_in)
        self.active_features = max_in

    def forward(self, x):
        weight = self.layer_norm.weight[: self.active_features]
        bias = self.layer_norm.bias[: self.active_features]
        return F.layer_norm(x, (self.active_features,), weight, bias)


class DynamicMHA(nn.Module):
    def __init__(self, max_embed_dim, max_num_heads, bias=True):
        super().__init__()
        self.max_embed_dim = max_embed_dim
        self.max_num_heads = max_num_heads
        self.bias = bias
        self.head_dim = max_embed_dim // max_num_heads

        self.qkv_linear = DynamicLinear(
            max_embed_dim, max_embed_dim * 3, bias=bias
        )  # out: heads*head_dim
        self.proj_linear = DynamicLinear(max_embed_dim, max_embed_dim, bias=bias)

        self.active_embed_dim = max_embed_dim
        self.active_num_heads = max_num_heads

    def forward(self, x):
        batch_size, token_size, embed_size = x.shape
        # Extract Q, K, V in one go and reshape
        # (B, T, 3*E) -> (B, T, 3, num_heads, head_dim) -> (3, B, num_heads, T, head_dim)
        # embed_size = num_heads * head_dim
        qkv = (
            self.qkv_linear(x)
            .reshape(batch_size, token_size, 3, self.active_num_heads, -1)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        # scale by per-head dimension (head_dim) as in "Attention is All You Need"
        # which uses sqrt(d_k) where d_k is the key/query dimension for each head.
        attn = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, E)
        # or permute(0, 2, 1, 3) -> reshape
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_size, -1)
        x = self.proj_linear(x)
        return x
