"""Static subnet extracted from a trained SuperNet.

All layers are standard fixed-size nn.Linear / nn.LayerNorm — no dynamic wrappers.
Parameter count matches exactly the searched config.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class StaticMHA(nn.Module):
    def __init__(self, embed_dim, num_heads, bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads

        self.qkv = nn.Linear(embed_dim, 3 * embed_dim, bias=bias)
        self.proj = nn.Linear(embed_dim, embed_dim, bias=bias)

    def forward(self, x):
        B, T, E = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, T, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = q @ k.transpose(-2, -1) * (self.head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        x = (attn @ v).transpose(1, 2).reshape(B, T, E)
        return self.proj(x)


class StaticMlp(nn.Module):
    def __init__(self, embed_dim, mlp_dim, dropout=0.0, bias=True):
        super().__init__()
        self.fc1 = nn.Linear(embed_dim, mlp_dim, bias=bias)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(mlp_dim, embed_dim, bias=bias)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.fc2(self.act(self.fc1(x))))


class StaticTransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.0):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.mha = StaticMHA(embed_dim, num_heads)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = StaticMlp(embed_dim, mlp_dim, dropout=dropout)

    def forward(self, x):
        x = x + self.mha(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x


class SubNet(nn.Module):
    """Static ViT subnet with fixed architecture dimensions."""

    def __init__(self, img_size, patch_size, embed_dim, num_layers,
                 num_heads, mlp_dim, num_classes, dropout=0.0):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        self.num_layers = num_layers

        num_patches = (img_size // patch_size) ** 2

        self.patch_embed = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_encoding = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.dropout = nn.Dropout(dropout)

        self.blocks = nn.ModuleList([
            StaticTransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x).flatten(2).transpose(1, 2)
        cls = self.cls_token.expand(B, -1, -1)
        x = torch.cat((cls, x), dim=1)
        x = x + self.pos_encoding
        x = self.dropout(x)
        for block in self.blocks:
            x = block(x)
        x = self.norm(x)
        return self.head(x[:, 0])
