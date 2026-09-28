#!/usr/bin/env python3
"""
Context Decomposition + Branch FiLM

替代 Golf R2 的独立 TCA 解耦模块:
  - RWKV 多头输出包含不同退化分量信息
  - 通过线性投影 + 正交约束, 显式解耦为 N/L/M 三个上下文向量
  - 通过 FiLM 注入回空间特征

退化分量:
  - N (Noise):    成像噪声 (i.i.d., 高频)
  - L (Light):    光照衰减 (低频, 帧间强相关)
  - M (Motion):   运动位移 (帧间结构变化)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict


class ContextDecomposition(nn.Module):
    """
    将 RWKV 输出的帧级上下文解耦为 N/L/M 三个分量

    Input:  frame_ctx [B, T, D]
    Output: (ctx_N, ctx_L, ctx_M) each [B, ctx_dim]
    """

    def __init__(self, dim: int = 192, ctx_dim: int = 64):
        super().__init__()
        self.dim = dim
        self.ctx_dim = ctx_dim

        self.proj_N = nn.Linear(dim, ctx_dim)
        self.proj_L = nn.Linear(dim, ctx_dim)
        self.proj_M = nn.Linear(dim, ctx_dim)

        self.ln_N = nn.LayerNorm(ctx_dim)
        self.ln_L = nn.LayerNorm(ctx_dim)
        self.ln_M = nn.LayerNorm(ctx_dim)

    def forward(self, frame_ctx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        frame_ctx: [B, T, D]
        Returns: ctx_N, ctx_L, ctx_M each [B, ctx_dim] (中心帧)
        """
        T = frame_ctx.shape[1]
        center = frame_ctx[:, T // 2]  # [B, D]

        ctx_N = self.ln_N(self.proj_N(center))
        ctx_L = self.ln_L(self.proj_L(center))
        ctx_M = self.ln_M(self.proj_M(center))

        return ctx_N, ctx_L, ctx_M

    @staticmethod
    def ortho_loss(ctx_N: torch.Tensor, ctx_L: torch.Tensor,
                   ctx_M: torch.Tensor) -> torch.Tensor:
        """
        正交约束: 三个上下文应尽量正交 (彼此不相关)
        返回平均绝对余弦相似度, 越接近 0 越好
        """
        cos_NL = F.cosine_similarity(ctx_N, ctx_L, dim=-1).abs().mean()
        cos_NM = F.cosine_similarity(ctx_N, ctx_M, dim=-1).abs().mean()
        cos_LM = F.cosine_similarity(ctx_L, ctx_M, dim=-1).abs().mean()
        return (cos_NL + cos_NM + cos_LM) / 3.0


class BranchFiLM(nn.Module):
    """
    将上下文向量注入空间特征 (FiLM 调制)

    F_out = F_in * (1 + scale(ctx)) + shift(ctx)

    零初始化 → 初始时 FiLM 是恒等变换
    """

    def __init__(self, ctx_dim: int = 64, feat_dim: int = 128):
        super().__init__()
        self.to_scale = nn.Linear(ctx_dim, feat_dim)
        self.to_shift = nn.Linear(ctx_dim, feat_dim)

        nn.init.zeros_(self.to_scale.weight)
        nn.init.zeros_(self.to_scale.bias)
        nn.init.zeros_(self.to_shift.weight)
        nn.init.zeros_(self.to_shift.bias)

    def forward(self, feat: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """
        feat: [B, feat_dim, H, W]
        ctx:  [B, ctx_dim]
        Returns: [B, feat_dim, H, W]
        """
        scale = self.to_scale(ctx)[:, :, None, None]  # [B, feat_dim, 1, 1]
        shift = self.to_shift(ctx)[:, :, None, None]
        return feat * (1 + scale) + shift


if __name__ == '__main__':
    B, T, D = 2, 5, 192
    frame_ctx = torch.randn(B, T, D)

    print("=== ContextDecomposition ===")
    cd = ContextDecomposition(dim=192, ctx_dim=64)
    ctx_N, ctx_L, ctx_M = cd(frame_ctx)
    print(f"  Input:  {frame_ctx.shape}")
    print(f"  ctx_N:  {ctx_N.shape}, ctx_L: {ctx_L.shape}, ctx_M: {ctx_M.shape}")
    print(f"  Params: {sum(p.numel() for p in cd.parameters()) / 1e3:.1f}K")
    print(f"  Ortho loss: {cd.ortho_loss(ctx_N, ctx_L, ctx_M).item():.4f}")

    print("\n=== BranchFiLM ===")
    feat = torch.randn(2, 128, 32, 32)
    film = BranchFiLM(ctx_dim=64, feat_dim=128)
    out = film(feat, ctx_N)
    print(f"  feat:   {feat.shape} → out: {out.shape}")
    print(f"  Params: {sum(p.numel() for p in film.parameters()) / 1e3:.1f}K")
    # 零初始化验证: 初始输出应等于输入
    print(f"  Identity check (should be True): {torch.allclose(out, feat, atol=1e-5)}")
