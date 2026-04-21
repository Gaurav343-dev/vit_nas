from torch import nn
import torch.nn.functional as F


class BaseDynamicModule(nn.Module):
    def get_active_submodule(self):
        raise NotImplementedError

    def set_active_submodule(self, *args, **kwargs):
        raise NotImplementedError


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
        weight = self.linear.weight[: self.active_out, :active_in]
        bias = self.linear.bias[: self.active_out] if self.bias else None
        return F.linear(
            x, weight.contiguous(), bias
        )  # contiguous for potential performance

    def get_macs(self, seq_len: int, active_in: int) -> int:
        """MACs = seq_len × active_out × active_in."""
        return seq_len * self.active_out * active_in


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

    def get_macs(self, seq_len: int) -> int:
        """LayerNorm: ~2 ops per element (scale + shift); variance/mean are small constants."""
        return seq_len * self.active_features


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
        # TODO: BUG - self.head_dim is computed once from max dims at init.
        # When active_num_heads != max_num_heads, this scale is incorrect.
        # Fix: replace self.head_dim with (self.active_embed_dim // self.active_num_heads)
        attn = q @ k.transpose(-2, -1) * (self.head_dim**-0.5)
        attn = F.softmax(attn, dim=-1)
        # (B, num_heads, T, head_dim) -> (B, T, num_heads, head_dim) -> (B, T, E)
        # or permute(0, 2, 1, 3) -> reshape
        x = (attn @ v).transpose(1, 2).reshape(batch_size, token_size, -1)
        x = self.proj_linear(x)
        return x

    def get_macs(self, seq_len: int) -> int:
        """
        MACs for multi-head attention:
          - QKV projection:  seq × E × 3E
          - Q·Kᵀ scores:     seq² × E   (all heads combined, head_dim × num_heads = E)
          - scores · V:      seq² × E
          - Output proj:     seq × E × E
        """
        E = self.active_embed_dim
        total = 0
        total += self.qkv_linear.get_macs(seq_len, E)   # seq × E → seq × 3E
        total += seq_len * seq_len * E                         # Q·Kᵀ
        total += seq_len * seq_len * E                         # scores·V
        total += self.proj_linear.get_macs(seq_len, E)  # seq × E → seq × E
        return total


class DynamicMlp(nn.Module):
    def __init__(
        self,
        max_in,
        max_hidden,
        max_out,
        activation=nn.GELU,
        dropout=0.0,
        bias=True,
    ):
        super().__init__()
        self.max_in_features = max_in
        self.max_hidden_features = max_hidden
        self.max_out_features = max_out

        self.active_in_features = max_in
        self.active_hidden_features = max_hidden
        self.active_out_features = max_out

        self.fc1 = DynamicLinear(max_in, max_hidden, bias=bias)
        self.activation = activation()
        self.fc2 = DynamicLinear(max_hidden, max_out, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activation(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x

    def get_macs(self, seq_len: int) -> int:
        """
        MACs for MLP:
          fc1: seq × embed_dim → seq × mlp_dim   (active_in inferred from fc2.active_out)
          fc2: seq × mlp_dim   → seq × embed_dim
        """
        embed_dim = self.fc2.active_out  # set by set_active_subnet
        mlp_dim = self.fc1.active_out    # set by set_active_subnet
        total = self.fc1.get_macs(seq_len, embed_dim)
        total += self.fc2.get_macs(seq_len, mlp_dim)
        return total


class DynamicTransformerBlock(nn.Module):
    def __init__(
        self,
        embed_dim,
        num_heads,
        mlp_hidden_dim,
        dropout=0.0,
        qkv_bias=True,
        mlp_bias=True,
    ):
        super().__init__()
        self.norm1 = DynamicLayerNorm(embed_dim)
        self.mha = DynamicMHA(embed_dim, num_heads, bias=qkv_bias)
        self.norm2 = DynamicLayerNorm(embed_dim)
        self.mlp = DynamicMlp(embed_dim, mlp_hidden_dim, embed_dim, dropout=dropout, bias=mlp_bias)

    def forward(self, x):
        # TODO: which is first norm or add + attn -> norm? 
        # # based on Xiong et al. "On Layer Normalization in the Transformer Architecture" (https://arxiv.org/abs/2002.04745) 
        # we use pre-norm
        x = x + self.mha(self.norm1(x))
        # TODO: pytorch implements dropout after attention; is that needed here?
        x = x + self.mlp(self.norm2(x))
        return x

    def get_macs(self, seq_len: int) -> int:
        total = self.norm1.get_macs(seq_len)
        total += self.mha.get_macs(seq_len)
        total += self.norm2.get_macs(seq_len)
        total += self.mlp.get_macs(seq_len)
        return total
