"""Create a supernet that can be used for NAS."""
import torch
from torch import nn
from .dynamic_modules import DynamicTransformerBlock

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

    # TODO: Complete get active subnet method that returns a nn.Module 
    # with only the active subnet parameters and architecture. 
    # This can be used for evaluation or export after NAS search is done.
    def get_active_subnet(self):
        pass 
        # Placeholder for method to extract the active subnet as a standalone nn.Module
