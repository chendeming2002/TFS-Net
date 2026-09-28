#!/usr/bin/env python3
"""
Spatial Summary: 保留空间结构的帧级 token 提取

替代 naive 的 AdaptiveAvgPool2d(1) → 1×1 (丢失所有空间信息)
使用 AdaptiveAvgPool2d(4) → 4×4=16 个空间位置 → 线性投影
"""
import torch
import torch.nn as nn


class SpatialSummary(nn.Module):
    """
    空间摘要: [B, T, C, H, W] → [B, T, D]

    流程:
      1. 每帧 AdaptiveAvgPool2d(spatial_size) → [B, C, s, s]
      2. Flatten → [B, C*s*s]
      3. Linear → [B, D]
      4. LayerNorm
    """

    def __init__(self, in_dim: int = 128, out_dim: int = 192, spatial_size: int = 4):
        super().__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim
        self.spatial_size = spatial_size

        self.pool = nn.AdaptiveAvgPool2d(spatial_size)
        self.proj = nn.Linear(in_dim * spatial_size * spatial_size, out_dim)
        self.ln = nn.LayerNorm(out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, T, C, H, W]
        Returns: [B, T, D]
        """
        B, T, C, H, W = x.shape
        tokens = []
        for t in range(T):
            pooled = self.pool(x[:, t])          # [B, C, s, s]
            flat = pooled.flatten(1)              # [B, C*s*s]
            tokens.append(self.proj(flat))        # [B, D]
        tokens = torch.stack(tokens, dim=1)       # [B, T, D]
        return self.ln(tokens)


if __name__ == '__main__':
    B, T, C, H, W = 2, 5, 128, 32, 32
    x = torch.randn(B, T, C, H, W)

    print("=== SpatialSummary ===")
    for s in [1, 2, 4]:
        m = SpatialSummary(in_dim=128, out_dim=192, spatial_size=s)
        out = m(x)
        n = sum(p.numel() for p in m.parameters())
        print(f"  spatial_size={s}: in={x.shape} → out={out.shape}, params={n/1e3:.1f}K")
