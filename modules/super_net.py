"""Create a supernet that can be used for NAS."""
import torch
from torch import nn
from .dynamic_modules import DynamicTransformerBlock
from .sub_net import SubNet

class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, max_embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.max_embed_dim = max_embed_dim
        self.proj = nn.Conv2d(3, max_embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        # x shape: (B, 3, H, W)
        x = self.proj(x)  # (B, max_embed_dim, H/patch_size, W/patch_size)
        x = x.flatten(2).transpose(1, 2)  # (B, num_patches, max_embed_dim)
        # output shape: (B, num_patches, max_embed_dim)
        return x

    def get_macs(self) -> int:
        """Conv2d with kernel=patch_size, stride=patch_size (non-overlapping).
        MACs = C_in × C_out × H_out × W_out × K²
        """
        num_patches = (self.img_size // self.patch_size) ** 2
        return 3 * self.max_embed_dim * num_patches * (self.patch_size ** 2)

class SuperNet(nn.Module):
    def __init__(self, img_size=224, patch_size=16, embed_dim=768, num_layers=12, num_heads=12, mlp_dim=1024, num_classes=10, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbed(img_size, patch_size, embed_dim)
        self.pos_encoding = nn.Parameter(torch.zeros(1, (img_size // patch_size) ** 2 + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.dropout = nn.Dropout(dropout)
        self.transformer_blocks = nn.ModuleList([
            DynamicTransformerBlock(embed_dim, num_heads, mlp_dim, dropout=dropout)
            for _ in range(num_layers)
        ])  
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)

        # search space for NAS
        self.active_embed_dim = embed_dim
        self.active_num_heads = num_heads
        self.active_mlp_dim = mlp_dim
        self.active_num_layers = num_layers

    def forward(self, x):
        B = x.size(0)
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_tokens, x), dim=1)  # (B, num_patches + 1, embed_dim)
        x = x + self.pos_encoding[:, :x.size(1), :]  # Add positional encoding
        x = self.dropout(x)

        for block in self.transformer_blocks:
            x = block(x)

        x = self.norm(x)
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)
        return logits
    
    def set_active_subnet(self, config: dict):
        # This method would set the active subnet configuration for all dynamic modules based on the provided config
        # config includes 
        # - embed_dim
        # - num_heads
        # - mlp_dim
        # - num_layers
        self.active_embed_dim = config.get("embed_dim", self.active_embed_dim)
        self.active_num_heads = config.get("num_heads", self.active_num_heads)
        self.active_mlp_dim = config.get("mlp_dim", self.active_mlp_dim)
        self.active_num_layers = config.get("num_layers", self.active_num_layers)

        for block in self.transformer_blocks:
            block.mha.active_embed_dim = self.active_embed_dim
            block.mha.active_num_heads = self.active_num_heads
            block.mha.qkv_linear.active_out = 3 * self.active_embed_dim
            block.mha.proj_linear.active_out = self.active_embed_dim
            
            block.mlp.fc1.active_out = self.active_mlp_dim
            block.mlp.fc2.active_out = self.active_embed_dim
            block.norm1.active_features = self.active_embed_dim
            block.norm2.active_features = self.active_embed_dim

    def get_active_subnet(self) -> SubNet:
        """Extract the currently active subnet as a standalone static nn.Module.

        Slices weights from the supernet's dynamic layers so the returned SubNet
        has exactly the parameter count dictated by the active config.
        Call set_active_subnet(config) before this method.
        """
        E = self.active_embed_dim
        H = self.active_num_heads
        M = self.active_mlp_dim
        L = self.active_num_layers

        subnet = SubNet(
            img_size=self.patch_embed.img_size,
            patch_size=self.patch_embed.patch_size,
            embed_dim=E,
            num_layers=L,
            num_heads=H,
            mlp_dim=M,
            num_classes=self.head.out_features,
            dropout=self.dropout.p,
        )

        # --- patch embedding (Conv2d) ---
        subnet.patch_embed.weight.data.copy_(self.patch_embed.proj.weight[:E])
        subnet.patch_embed.bias.data.copy_(self.patch_embed.proj.bias[:E])

        # --- cls token and positional encoding ---
        subnet.cls_token.data.copy_(self.cls_token[:, :, :E])
        subnet.pos_encoding.data.copy_(self.pos_encoding[:, :, :E])

        # --- transformer blocks ---
        for i in range(L):
            src = self.transformer_blocks[i]
            dst = subnet.blocks[i]

            # LayerNorm 1
            dst.norm1.weight.data.copy_(src.norm1.layer_norm.weight[:E])
            dst.norm1.bias.data.copy_(src.norm1.layer_norm.bias[:E])

            # MHA — QKV projection
            dst.mha.qkv.weight.data.copy_(src.mha.qkv_linear.linear.weight[:3*E, :E])
            dst.mha.qkv.bias.data.copy_(src.mha.qkv_linear.linear.bias[:3*E])

            # MHA — output projection
            dst.mha.proj.weight.data.copy_(src.mha.proj_linear.linear.weight[:E, :E])
            dst.mha.proj.bias.data.copy_(src.mha.proj_linear.linear.bias[:E])

            # LayerNorm 2
            dst.norm2.weight.data.copy_(src.norm2.layer_norm.weight[:E])
            dst.norm2.bias.data.copy_(src.norm2.layer_norm.bias[:E])

            # MLP fc1: embed_dim → mlp_dim
            dst.mlp.fc1.weight.data.copy_(src.mlp.fc1.linear.weight[:M, :E])
            dst.mlp.fc1.bias.data.copy_(src.mlp.fc1.linear.bias[:M])

            # MLP fc2: mlp_dim → embed_dim
            dst.mlp.fc2.weight.data.copy_(src.mlp.fc2.linear.weight[:E, :M])
            dst.mlp.fc2.bias.data.copy_(src.mlp.fc2.linear.bias[:E])

        # --- final LayerNorm ---
        subnet.norm.weight.data.copy_(self.norm.weight[:E])
        subnet.norm.bias.data.copy_(self.norm.bias[:E])

        # --- classification head ---
        subnet.head.weight.data.copy_(self.head.weight[:, :E])
        subnet.head.bias.data.copy_(self.head.bias)

        return subnet

    def get_macs(self) -> int:
        """Total MACs for the currently active subnet configuration.
        Call set_active_subnet() first to configure the subnet before measuring.
        Returns MACs (multiply-accumulate operations).
        """
        num_patches = (self.patch_embed.img_size // self.patch_embed.patch_size) ** 2
        seq_len = num_patches + 1  # +1 for cls token

        total = self.patch_embed.get_macs()

        for i, block in enumerate(self.transformer_blocks):
            if i >= self.active_num_layers:
                break
            total += block.get_macs(seq_len)

        # Final LayerNorm (operates on all tokens, static nn.LayerNorm)
        total += seq_len * self.active_embed_dim

        # Classification head (CLS token only → 1 token)
        total += self.active_embed_dim * self.head.out_features

        return total
