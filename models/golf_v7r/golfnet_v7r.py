#!/usr/bin/env python3
"""
GolfNet v7r: Matrix RWKV + 多元噪声分割

架构 (详见 docs/v7/02-v7-architecture-design.md):
  Stage 1: SharedEncoder (逐帧, 复用 Golf R2)
  Stage 2: feature_proj + PixelTemporalAttentionSimple
  Stage 3: MatrixRWKV-TCA ⭐
            SpatialSummary → MatrixRWKVBlock ×2 → ContextDecomposition → BranchFiLM
  Stage 4: 简化三分支 BranchN/L/M
  Stage 5: AdaptiveFusion

输入: [B, T=5, 3, H, W]
输出: dict {final, branch_N/L/M, frame_ctx, ortho_loss, fusion_weights}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.golf.encoder import SharedEncoder
from models.golf_v7.pixel_temporal import PixelTemporalAttentionSimple

from models.golf_v7r.spatial_summary import SpatialSummary
from models.golf_v7r.matrix_rwkv import MatrixRWKVBlock
from models.golf_v7r.context_decomp import ContextDecomposition, BranchFiLM
from models.golf_v7r.fusion_v7r import V7RFusion
from models.golf_v7r.branch_n_simple import BranchNSimple
from models.golf_v7r.branch_l_simple import BranchLSimple
from models.golf_v7r.branch_m_simple import BranchMSimple


class GolfNet_v7r(nn.Module):
    """GolfNet v7r: Matrix RWKV + 多元噪声分割"""

    def __init__(self,
                 num_frames: int = 5,
                 encoder_channels: list = [32, 64, 128],
                 feature_dim: int = 128,
                 rwkv_dim: int = 192,
                 rwkv_heads: int = 6,
                 rwkv_head_size: int = 32,
                 rwkv_blocks: int = 2,
                 ctx_dim: int = 64,
                 spatial_size: int = 2,
                 branch_channels: int = 128):
        super().__init__()
        self.num_frames = num_frames
        self.C1, self.C2, self.C3 = encoder_channels
        self.feature_dim = feature_dim
        self.rwkv_dim = rwkv_dim

        # ========== Stage 1: Encoder (逐帧独立) ==========
        self.encoder = SharedEncoder(
            in_channels=3,
            out_channels_list=encoder_channels,
            num_blocks_per_scale=[2, 2, 2],
        )

        # ========== Stage 2: 投影 + 像素级时序对齐 ==========
        self.feature_proj = nn.Conv2d(self.C2, feature_dim, 1)
        self.pixel_temporal = PixelTemporalAttentionSimple(dim=feature_dim)

        # ========== Stage 3: Matrix RWKV-TCA ⭐ ==========
        # 3.1 Spatial Summary (保留空间结构)
        self.spatial_summary = SpatialSummary(
            in_dim=feature_dim, out_dim=rwkv_dim, spatial_size=spatial_size
        )

        # 3.2 Matrix RWKV Blocks
        self.rwkv_blocks = nn.ModuleList([
            MatrixRWKVBlock(dim=rwkv_dim, num_heads=rwkv_heads,
                            head_size=rwkv_head_size)
            for _ in range(rwkv_blocks)
        ])
        self.rwkv_norm = nn.LayerNorm(rwkv_dim)

        # 3.3 Context Decomposition (N/L/M)
        self.ctx_decomp = ContextDecomposition(dim=rwkv_dim, ctx_dim=ctx_dim)

        # 3.4 Branch FiLM (上下文注入)
        self.film_N = BranchFiLM(ctx_dim=ctx_dim, feat_dim=feature_dim)
        self.film_L = BranchFiLM(ctx_dim=ctx_dim, feat_dim=feature_dim)
        self.film_M = BranchFiLM(ctx_dim=ctx_dim, feat_dim=feature_dim)

        # ========== Stage 4: 三分支 ==========
        self.branch_N = BranchNSimple(channels=branch_channels, num_blocks=2)
        self.branch_L = BranchLSimple(channels=branch_channels, num_blocks=2)
        self.branch_M = BranchMSimple(channels=branch_channels,
                                      enc_channels=self.C2,
                                      num_frames=num_frames, num_blocks=2)

        # ========== Stage 5: Fusion ==========
        # v7r 专用融合: 不在 RGB 上跑 NAFBlock (会毁图), 权重均匀初始化, gamma 从 0 起
        self.fusion = V7RFusion(in_channels=3, hidden_channels=32,
                                num_blocks=2)

    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, 3, H, W]
        """
        B, T, C_in, H, W = x.shape
        assert T == self.num_frames, f"expect T={self.num_frames}, got {T}"
        center_idx = T // 2
        X_center = x[:, center_idx]  # [B, 3, H, W]

        # ========== Stage 1: 逐帧编码 ==========
        feats_f2 = []
        for t in range(T):
            _, f2, _ = self.encoder(x[:, t])   # [B, 64, H/2, W/2]
            feats_f2.append(f2)
        F2_seq = torch.stack(feats_f2, dim=1)   # [B, T, 64, H/2, W/2]

        # 投影到 feature_dim
        feats_proj = []
        for t in range(T):
            feats_proj.append(self.feature_proj(F2_seq[:, t]))
        feats_proj = torch.stack(feats_proj, dim=1)  # [B, T, 128, H/2, W/2]

        # ========== Stage 2: 像素级时序对齐 ==========
        feat_aligned = self.pixel_temporal(feats_proj)  # [B, 128, H/2, W/2]

        # ========== Stage 3: Matrix RWKV-TCA ==========
        # 3.1 帧级 token (保留空间结构)
        frame_tokens = self.spatial_summary(feats_proj)  # [B, T, 192]

        # 3.2 矩阵 RWKV 时序建模
        ctx = frame_tokens
        for block in self.rwkv_blocks:
            ctx = block(ctx)
        ctx = self.rwkv_norm(ctx)               # [B, T, 192]

        # 3.3 解耦为 N/L/M 上下文
        ctx_N, ctx_L, ctx_M = self.ctx_decomp(ctx)
        ortho = ContextDecomposition.ortho_loss(ctx_N, ctx_L, ctx_M)

        # 3.4 注入回空间特征
        F_N = self.film_N(feat_aligned, ctx_N)
        F_L = self.film_L(feat_aligned, ctx_L)
        F_M = self.film_M(feat_aligned, ctx_M)

        # ========== Stage 4: 三分支 ==========
        out_N = self.branch_N(F_N)['Y_N']                     # [B,3,H,W]
        out_L = self.branch_L(F_L, X_center)['Y_L']
        out_M_dict = self.branch_M(F_M, F2_seq)
        out_M = out_M_dict['Y_M']

        # ========== Stage 5: Fusion ==========
        fusion_out = self.fusion(out_N, out_L, out_M, X_center)

        return {
            'final': fusion_out['O_t'],
            'branch_N': out_N,
            'branch_L': out_L,
            'branch_M': out_M,
            'frame_ctx': ctx,
            'ortho_loss': ortho,
            'fusion_weights': fusion_out['weights'],
        }


def count_params(model):
    """按模块统计参数量"""
    stats = {}
    stats['encoder'] = sum(p.numel() for p in model.encoder.parameters())
    stats['pixel_temporal'] = sum(p.numel() for p in model.pixel_temporal.parameters())
    stats['spatial_summary'] = sum(p.numel() for p in model.spatial_summary.parameters())
    stats['matrix_rwkv'] = sum(p.numel() for p in model.rwkv_blocks.parameters())
    stats['ctx_decomp'] = sum(p.numel() for p in model.ctx_decomp.parameters())
    stats['film'] = (sum(p.numel() for p in model.film_N.parameters())
                     + sum(p.numel() for p in model.film_L.parameters())
                     + sum(p.numel() for p in model.film_M.parameters()))
    stats['branch_N'] = sum(p.numel() for p in model.branch_N.parameters())
    stats['branch_L'] = sum(p.numel() for p in model.branch_L.parameters())
    stats['branch_M'] = sum(p.numel() for p in model.branch_M.parameters())
    stats['fusion'] = sum(p.numel() for p in model.fusion.parameters())
    stats['feature_proj'] = sum(p.numel() for p in model.feature_proj.parameters())
    return stats


if __name__ == '__main__':
    print("Testing GolfNet_v7r...")
    model = GolfNet_v7r(num_frames=5)
    x = torch.randn(2, 5, 3, 256, 256)
    out = model(x)

    print(f"\nInput:            {x.shape}")
    print(f"Output final:     {out['final'].shape}")
    print(f"Branch N/L/M:     {out['branch_N'].shape}")
    print(f"Frame ctx:        {out['frame_ctx'].shape}")
    print(f"Ortho loss:       {out['ortho_loss'].item():.4f}")

    total = sum(p.numel() for p in model.parameters())
    print(f"\n=== 总参数: {total/1e6:.2f}M ===")
    print("=== 参数分布 ===")
    stats = count_params(model)
    for k, v in stats.items():
        print(f"  {k:20s}: {v/1e6:.3f}M")

    # 梯度检查
    loss = out['final'].sum() + out['branch_N'].sum()
    loss.backward()
    n_grad = sum(1 for p in model.parameters() if p.grad is not None)
    print(f"\nGradient check: {n_grad} params have grads")
