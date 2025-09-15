"""
Advanced Vision Transformer (ViT) Implementation
Optimized for performance and efficiency with latest research improvements.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple
from functools import partial


class OptimizedViT(nn.Module):
    """
    Advanced Vision Transformer with performance optimizations:
    - 2D sinusoidal positional encoding
    - Global average pooling (no CLS token)
    - Efficient attention mechanism
    - Memory-optimized implementation
    """
    
    def __init__(
        self,
        image_size: int = 224,
        patch_size: int = 16,
        num_classes: int = 1000,
        dim: int = 768,
        depth: int = 12,
        heads: int = 12,
        mlp_dim: int = 3072,
        channels: int = 3,
        dropout: float = 0.0,
        pool: str = 'mean',
        pos_embedding: str = 'sincos2d'
    ):
        super().__init__()
        
        assert image_size % patch_size == 0, f"Image size {image_size} must be divisible by patch size {patch_size}"
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        self.dim = dim
        self.num_classes = num_classes
        
        # Efficient patch embedding using convolution
        self.patch_embed = nn.Conv2d(
            channels, dim, kernel_size=patch_size, stride=patch_size
        )
        
        # Positional encoding
        if pos_embedding == 'sincos2d':
            self.register_buffer("pos_embed", self._get_2d_sincos_pos_embed())
        else:
            self.pos_embed = nn.Parameter(torch.randn(1, self.num_patches, dim) * 0.02)
        
        # Transformer encoder
        self.blocks = nn.ModuleList([
            TransformerBlock(
                dim=dim,
                num_heads=heads,
                mlp_ratio=mlp_dim // dim,
                dropout=dropout,
                drop_path=0.1 * i / depth  # Stochastic depth
            ) for i in range(depth)
        ])
        
        # Classification head
        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes) if num_classes > 0 else nn.Identity()
        
        self.apply(self._init_weights)
    
    def _get_2d_sincos_pos_embed(self) -> torch.Tensor:
        """Generate 2D sinusoidal positional embeddings"""
        grid_size = int(self.num_patches ** 0.5)
        pos_embed = get_2d_sincos_pos_embed(self.dim, grid_size)
        return torch.from_numpy(pos_embed).float().unsqueeze(0)
    
    def _init_weights(self, m):
        """Initialize model weights"""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Conv2d):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass through feature extraction layers"""
        B, C, H, W = x.shape
        
        # Patch embedding
        x = self.patch_embed(x)  # [B, dim, H//patch_size, W//patch_size]
        x = x.flatten(2).transpose(1, 2)  # [B, num_patches, dim]
        
        # Add positional encoding
        x = x + self.pos_embed
        
        # Apply transformer blocks
        for block in self.blocks:
            x = block(x)
        
        return self.norm(x)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        x = self.forward_features(x)
        
        # Global average pooling
        x = x.mean(dim=1)
        
        # Classification head
        x = self.head(x)
        return x


class TransformerBlock(nn.Module):
    """Transformer block with efficient implementation"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
        drop_path: float = 0.0,
        act_layer: nn.Module = nn.GELU,
        norm_layer: nn.Module = nn.LayerNorm
    ):
        super().__init__()
        
        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )
        
        self.drop_path = DropPath(drop_path) if drop_path > 0 else nn.Identity()
        self.norm2 = norm_layer(dim)
        
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            dropout=dropout,
            act_layer=act_layer
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


class Attention(nn.Module):
    """Multi-head self-attention with optimizations"""
    
    def __init__(
        self,
        dim: int,
        num_heads: int = 8,
        dropout: float = 0.0
    ):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divisible by num_heads {num_heads}"
        
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Fused QKV projection
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.attn_dropout = nn.Dropout(dropout)
        self.proj = nn.Linear(dim, dim)
        self.proj_dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # [3, B, num_heads, N, head_dim]
        q, k, v = qkv.unbind(0)
        
        # Attention computation
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_dropout(x)
        
        return x


class MLP(nn.Module):
    """MLP block with GELU activation"""
    
    def __init__(
        self,
        in_features: int,
        hidden_features: Optional[int] = None,
        out_features: Optional[int] = None,
        act_layer: nn.Module = nn.GELU,
        dropout: float = 0.0
    ):
        super().__init__()
        
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class DropPath(nn.Module):
    """Drop paths (Stochastic Depth) per sample for residual blocks"""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape,) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        return output


def get_2d_sincos_pos_embed(embed_dim: int, grid_size: int, cls_token: bool = False):
    """Generate 2D sinusoidal position embedding"""
    import numpy as np
    
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)
    
    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    
    if cls_token:
        pos_embed = np.concatenate([np.zeros([1, embed_dim]), pos_embed], axis=0)
    
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim: int, grid):
    """Generate 2D sinusoidal position embedding from grid"""
    import numpy as np
    
    assert embed_dim % 2 == 0
    
    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid)  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[23])  # (H*W, D/2)
    
    emb = np.concatenate([emb_h, emb_w], axis=1)  # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim: int, pos):
    """Generate 1D sinusoidal position embedding"""
    import numpy as np
    
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=float)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)
    
    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product
    
    emb_sin = np.sin(out)  # (M, D/2)
    emb_cos = np.cos(out)  # (M, D/2)
    
    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb


# Model factory functions
def vit_tiny_patch16_224(**kwargs):
    """ViT-Tiny model"""
    model = OptimizedViT(
        patch_size=16, dim=192, depth=12, heads=3, mlp_dim=768, **kwargs
    )
    return model


def vit_small_patch16_224(**kwargs):
    """ViT-Small model"""
    model = OptimizedViT(
        patch_size=16, dim=384, depth=12, heads=6, mlp_dim=1536, **kwargs
    )
    return model


def vit_base_patch16_224(**kwargs):
    """ViT-Base model"""
    model = OptimizedViT(
        patch_size=16, dim=768, depth=12, heads=12, mlp_dim=3072, **kwargs
    )
    return model


def vit_large_patch16_224(**kwargs):
    """ViT-Large model"""
    model = OptimizedViT(
        patch_size=16, dim=1024, depth=24, heads=16, mlp_dim=4096, **kwargs
    )
    return model


if __name__ == "__main__":
    # Quick test
    model = vit_base_patch16_224(num_classes=1000)
    x = torch.randn(2, 3, 224, 224)
    y = model(x)
    print(f"Input shape: {x.shape}")
    print(f"Output shape: {y.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
    print("✅ Model test passed!")
