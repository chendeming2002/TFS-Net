#!/usr/bin/env python3
"""
GolfNet v7r-v3: Matrix RWKV + Triple Query TCA + 多元噪声分割

架构差异 (vs v7r-v2):
  v7r-v2: PixelTemporal → MatrixRWKV → ContextDecomposition (事后线性投影解耦) → FiLM → 三分支
  v7r-v3: PixelTemporal → MatrixRWKV → TripleQueryTCA (三路 Q × 共享统计先验 KV) → 三分支

关键改进:
  1. 三路差异化查询 (显式解耦, 对比 v2 的隐式事后分离)
  2. 共享 KV 从统计先验 (均值/低频/差分) 生成, 避免 R4 的全时序噪声放大
  3. 直接输出空间特征 F_N/L/M, 无需额外 FiLM 调制

Stage flow:
  1. SharedEncoder (逐帧)
  2. feature_proj + PixelTemporalAttentionSimple
  3. MatrixRWKV-TCA:
       SpatialSummary → MatrixRWKVBlock ×2 → ctx [B,T,192]
       TripleQueryTCA(feat_aligned, feats_proj, ctx) → F_N/L/M + ortho_loss
         · ctx 经 MatrixRWKVInjector 生成三路门控, 调制共享统计 KV (方案 B)
  4. 简化三分支 BranchN/L/M
  5. V7RFusion
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.golf.encoder import SharedEncoder
from models.golf_v7.pixel_temporal import PixelTemporalAttentionSimple

from models.golf_v7r.spatial_summary import SpatialSummary
from models.golf_v7r.matrix_rwkv import MatrixRWKVBlock
from models.golf_v7r.triple_query_tca import TripleQueryTCA
from models.golf_v7r.fusion_v7r import V7RFusion
from models.golf_v7r.branch_n_simple import BranchNSimple
from models.golf_v7r.branch_l_simple import BranchLSimple
from models.golf_v7r.branch_m_simple import BranchMSimple


class GolfNet_v7r_v3(nn.Module):
    """GolfNet v7r-v3: Matrix RWKV + Triple Query TCA"""

    def __init__(self,
                 num_frames: int = 5,
                 encoder_channels: list = [32, 64, 128],
                 feature_dim: int = 128,
                 rwkv_dim: int = 192,
                 rwkv_heads: int = 6,
                 rwkv_head_size: int = 32,
                 rwkv_blocks: int = 2,
                 spatial_size: int = 2,
                 branch_channels: int = 128,
                 motion_aware_diff: bool = False,
                 diff_smooth_kernel: int = 5):
        super().__init__()
        self.num_frames = num_frames
        self.C1, self.C2, self.C3 = encoder_channels
        self.feature_dim = feature_dim
        self.rwkv_dim = rwkv_dim
        self.motion_aware_diff = motion_aware_diff
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
        # 3.1 Spatial Summary (帧级 token 提取, 辅助用)
        self.spatial_summary = SpatialSummary(
            in_dim=feature_dim, out_dim=rwkv_dim, spatial_size=spatial_size
        )

        # 3.2 Matrix RWKV Blocks (帧级时序建模)
        self.rwkv_blocks = nn.ModuleList([
            MatrixRWKVBlock(dim=rwkv_dim, num_heads=rwkv_heads,
                            head_size=rwkv_head_size)
            for _ in range(rwkv_blocks)
        ])
        self.rwkv_norm = nn.LayerNorm(rwkv_dim)

        # 3.3 Triple Query TCA (三路查询 × 共享统计先验 KV)
        self.triple_query_tca = TripleQueryTCA(
            feat_dim=feature_dim, rwkv_dim=rwkv_dim, num_frames=num_frames,
            motion_aware_diff=motion_aware_diff,
            diff_smooth_kernel=diff_smooth_kernel,
        )

        # ========== Stage 4: 三分支 ==========
        self.branch_N = BranchNSimple(channels=branch_channels, num_blocks=2)
        self.branch_L = BranchLSimple(channels=branch_channels, num_blocks=2)
        self.branch_M = BranchMSimple(channels=branch_channels,
                                      enc_channels=self.C2,
                                      num_frames=num_frames, num_blocks=2)

        # ========== Stage 5: Fusion ==========
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
        # 3.1 帧级 token (辅助, 保留以便未来扩展)
        frame_tokens = self.spatial_summary(feats_proj)  # [B, T, 192]

        # 3.2 矩阵 RWKV 时序建模
        ctx = frame_tokens
        for block in self.rwkv_blocks:
            ctx = block(ctx)
        ctx = self.rwkv_norm(ctx)               # [B, T, 192]

        # 3.3 三路查询 TCA (核心改进)
        # MatrixRWKV 上下文 ctx 作为辅助三路门控注入共享 KV (方案 B)
        F_N, F_L, F_M, ortho, inject_stat = self.triple_query_tca(
            feat_aligned, feats_proj, ctx
        )  # 各 [B, 128, H/2, W/2]

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
            'inject_stat': inject_stat,
        }


def count_params(model):
    """按模块统计参数量"""
    stats = {}
    stats['encoder'] = sum(p.numel() for p in model.encoder.parameters())
    stats['pixel_temporal'] = sum(p.numel() for p in model.pixel_temporal.parameters())
    stats['spatial_summary'] = sum(p.numel() for p in model.spatial_summary.parameters())
    stats['matrix_rwkv'] = sum(p.numel() for p in model.rwkv_blocks.parameters())
    stats['triple_query_tca'] = sum(p.numel() for p in model.triple_query_tca.parameters())
    stats['branch_N'] = sum(p.numel() for p in model.branch_N.parameters())
    stats['branch_L'] = sum(p.numel() for p in model.branch_L.parameters())
    stats['branch_M'] = sum(p.numel() for p in model.branch_M.parameters())
    stats['fusion'] = sum(p.numel() for p in model.fusion.parameters())
    stats['total'] = sum(stats.values())
    return stats


if __name__ == '__main__':
    from torchsummary import summary
    import sys

    B, T, H, W = 1, 5, 256, 256
    x = torch.randn(B, T, 3, H, W)

    print("=== GolfNet_v7r_v3 ===")
    model = GolfNet_v7r_v3(
        num_frames=5,
        encoder_channels=[32, 64, 128],
        feature_dim=128,
        rwkv_dim=192,
        rwkv_heads=6,
        rwkv_head_size=32,
        rwkv_blocks=2,
        spatial_size=2,
        branch_channels=128
    )

    print("\n--- Forward pass ---")
    with torch.no_grad():
        out = model(x)
    print(f"  Input:       {x.shape}")
    print(f"  Output:      {out['final'].shape}")
    print(f"  branch_N:    {out['branch_N'].shape}")
    print(f"  branch_L:    {out['branch_L'].shape}")
    print(f"  branch_M:    {out['branch_M'].shape}")
    print(f"  ortho_loss:  {out['ortho_loss'].item():.4f}")

    print("\n--- Parameter count ---")
    stats = count_params(model)
    for k, v in stats.items():
        print(f"  {k:20s}: {v/1e6:6.3f}M")

    print("\n--- Gradient check ---")
    loss = out['final'].sum() + out['ortho_loss']
    loss.backward()
    print("  Passed!")
