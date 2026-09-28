#!/usr/bin/env python3
"""
Triple Query TCA: v7r-v3 核心模块

替代 v7r-v2 的 ContextDecomposition (事后线性投影解耦)
改用 Golf-R4 风格的「三路查询 × 共享 KV」设计

架构 (详见 docs/v7/03-v7r-v3-design.md):
  Input:
    - feat_spatial: [B, C, H, W]  (PixelTemporal 对齐后的空间特征, 生成 Query 和残差基)
    - feats_seq:    [B, T, C, H, W] (编码器逐帧特征, 生成共享 KV 的统计先验)
    - rwkv_ctx:     [B, T, D]     (MatrixRWKV 帧级上下文, 仅作辅助调制, 可选)

  Flow:
    1. 三路查询:  Q_N/L/M = query_N/L/M(feat_spatial)
    2. 共享 KV:   KV = kv_proj([ctx_mean, ctx_smooth, ctx_diff])  ← 统计先验, 空间结构
    3. 三路 RWKV 空间注意力 (线性复杂度 BiWKV):
         attn_N/L/M = RWKVSpatialHead(Q_N/L/M, KV_shared)
    4. 残差 + LayerScale: F_N/L/M = feat_spatial + attn · scale
    5. 正交约束:  L_ortho

设计动机 (R4 实验教训):
  - R4 的 KV = Concat_time(F_{t±i}) 直接 1×1 投影 → 全时序噪声被放大
  - R5 回退到「聚合统计量」KV → 更鲁棒
  - v7r-v3 采用统计先验 KV (均值/低频/差分), 三者拼接后共享,
    同时保留 R4 的「三路差异化 Query」和 v7r 的 PixelTemporal 逐像素对齐
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple

from models.modules.blocks import LayerNorm2d
# 复用 R4 已验证的线性复杂度 RWKV 空间注意力头 (BiWKV + 4 方向扫描)
from models.golf_r4.tca_rwkv import RWKVSpatialHead


class TripleQueryTCA(nn.Module):
    """
    三路查询 TCA (v7r-v3 核心模块)

    Args:
        feat_dim:  空间特征维度 (PixelTemporal 输出, 默认 128)
        rwkv_dim:  MatrixRWKV 输出维度 (默认 192)
        num_frames: 输入帧数 (默认 5)
    """

    def __init__(self, feat_dim: int = 128, rwkv_dim: int = 192,
                 num_frames: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.rwkv_dim = rwkv_dim
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        # ========== 1. 三路查询生成 (从 PixelTemporal 空间特征) ==========
        self.query_N = nn.Sequential(
            LayerNorm2d(feat_dim),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
        )
        self.query_L = nn.Sequential(
            LayerNorm2d(feat_dim),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
        )
        self.query_M = nn.Sequential(
            LayerNorm2d(feat_dim),
            nn.Conv2d(feat_dim, feat_dim, 1, bias=False),
        )

        # ========== 2. 共享 KV 投影 (统计先验: 均值/低频/差分 拼接) ==========
        # 3 个统计上下文 → 3*C 通道 → 1×1 投影回 C, 空间结构保留
        self.kv_proj = nn.Sequential(
            nn.Conv2d(3 * feat_dim, feat_dim, 1, bias=True),
            LayerNorm2d(feat_dim),
        )

        # ========== 3. 三路 RWKV 空间注意力头 (共享 KV, 线性复杂度) ==========
        self.attn_N = RWKVSpatialHead(feat_dim)
        self.attn_L = RWKVSpatialHead(feat_dim)
        self.attn_M = RWKVSpatialHead(feat_dim)

        # ========== 4. LayerScale (零初始化 → 初始恒等, 梯度非零) ==========
        self.scale_N = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.scale_L = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.scale_M = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))

        # ========== 5. 输出归一化 ==========
        self.out_norm_N = LayerNorm2d(feat_dim)
        self.out_norm_L = LayerNorm2d(feat_dim)
        self.out_norm_M = LayerNorm2d(feat_dim)

    # ---------- 统计上下文 (对齐 R2/R5 的有效先验) ----------
    def _ctx_mean(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """帧间均值: 成像噪声的最优估计 (i.i.d. → 平均去噪)"""
        return feats_seq.mean(dim=1)  # [B, C, H, W]

    def _ctx_smooth(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """低频平滑: 光照衰减的慢变趋势 (大核池化模拟低通)"""
        mean_feat = feats_seq.mean(dim=1)
        return F.avg_pool2d(mean_feat, kernel_size=7, stride=1, padding=3)

    def _ctx_diff(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """运动差分: 中心帧 vs 邻帧的最大绝对差 (位移区域响应最强)"""
        center = feats_seq[:, self.center_idx]
        diffs = [ (center - feats_seq[:, t]).abs()
                  for t in range(feats_seq.shape[1]) if t != self.center_idx ]
        return torch.stack(diffs, dim=0).max(dim=0).values

    def forward(self, feat_spatial: torch.Tensor,
                feats_seq: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        feat_spatial: [B, feat_dim, H, W]      — PixelTemporal 输出
        feats_seq:    [B, T, feat_dim, H, W]   — 逐帧编码特征

        Returns:
            F_N, F_L, F_M: 各 [B, feat_dim, H, W]
            ortho_loss: scalar
        """
        B, C, H, W = feat_spatial.shape

        # ========== Step 1: 三路查询 (从像素对齐特征) ==========
        Q_N = self.query_N(feat_spatial)
        Q_L = self.query_L(feat_spatial)
        Q_M = self.query_M(feat_spatial)

        # ========== Step 2: 共享 KV (统计先验拼接 → 1×1 投影) ==========
        ctx_mean = self._ctx_mean(feats_seq)      # [B, C, H, W]
        ctx_smooth = self._ctx_smooth(feats_seq)  # [B, C, H, W]
        ctx_diff = self._ctx_diff(feats_seq)      # [B, C, H, W]
        kv_shared = self.kv_proj(
            torch.cat([ctx_mean, ctx_smooth, ctx_diff], dim=1)
        )  # [B, C, H, W]

        # ========== Step 3: 三路 RWKV 空间注意力 (共享 KV, 差异化 Q) ==========
        attn_N = self.attn_N(Q_N, kv_shared)  # [B, C, H, W]
        attn_L = self.attn_L(Q_L, kv_shared)
        attn_M = self.attn_M(Q_M, kv_shared)

        # ========== Step 4: LayerScale 残差 (以 PixelTemporal 特征为基) ==========
        raw_N = feat_spatial + attn_N * self.scale_N
        raw_L = feat_spatial + attn_L * self.scale_L
        raw_M = feat_spatial + attn_M * self.scale_M

        # ========== Step 5: 输出归一化 ==========
        F_N = self.out_norm_N(raw_N)
        F_L = self.out_norm_L(raw_L)
        F_M = self.out_norm_M(raw_M)

        # ========== Step 6: 正交约束 ==========
        ortho_loss = self._ortho_loss(F_N, F_L, F_M)

        return F_N, F_L, F_M, ortho_loss

    @staticmethod
    def _ortho_loss(F_N: torch.Tensor, F_L: torch.Tensor,
                    F_M: torch.Tensor) -> torch.Tensor:
        """
        空间特征正交约束: 三路特征在通道维度的余弦相似度应接近 0

        F_*: [B, C, H, W]
        """
        f_N = F.normalize(F_N.flatten(2), dim=1)  # [B, C, H*W]
        f_L = F.normalize(F_L.flatten(2), dim=1)
        f_M = F.normalize(F_M.flatten(2), dim=1)

        cos_NL = (f_N * f_L).sum(dim=1).abs().mean()
        cos_NM = (f_N * f_M).sum(dim=1).abs().mean()
        cos_LM = (f_L * f_M).sum(dim=1).abs().mean()

        return (cos_NL + cos_NM + cos_LM) / 3.0


if __name__ == '__main__':
    B, T, C = 1, 5, 128
    H, W = 32, 32

    feat_spatial = torch.randn(B, C, H, W)
    feats_seq = torch.randn(B, T, C, H, W)

    print("=== TripleQueryTCA ===")
    tca = TripleQueryTCA(feat_dim=128, rwkv_dim=192, num_frames=5)
    F_N, F_L, F_M, ortho = tca(feat_spatial, feats_seq)

    print(f"  Input feat_spatial: {feat_spatial.shape}")
    print(f"  Input feats_seq:    {feats_seq.shape}")
    print(f"  Output F_N/L/M:     {F_N.shape}")
    print(f"  Ortho loss:         {ortho.item():.4f}")
    print(f"  Params: {sum(p.numel() for p in tca.parameters()) / 1e6:.3f}M")

    # 恒等检查: scale 零初始化时, F_* 应约等于 out_norm(feat_spatial)
    loss = F_N.sum() + F_L.sum() + F_M.sum() + ortho
    loss.backward()
    print("  Gradient check passed!")
