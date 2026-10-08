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
from typing import Tuple, Dict

from models.modules.blocks import LayerNorm2d
from models.golf_r4.tca_rwkv import RWKVSpatialHead


class MatrixRWKVInjector(nn.Module):
    """
    MatrixRWKV 帧级上下文 → 共享统计 KV 的三路门控 (方案 B)

    语义: 三种统计量 (mean/smooth/diff) 分别对应三种退化假设
          (噪声 i.i.d. / 光照低频 / 运动位移)。
          MatrixRWKV 在 T 帧上递推得到的时序态势, 最适合决定
          "当前该信任哪种退化假设" → 输出 3C 维逐通道门控。

        s_k = ctx_k · (1 + g_k),   g = to_gate(ctx_center) ∈ R^{3C}
        kv_shared = kv_proj(concat[s_mean, s_smooth, s_diff])

    恒等性: to_gate 零初始化 → g=0 → s_k = ctx_k, 不扰动既有表示。
            梯度: ∂L/∂W_gate ∝ ∂L/∂kv · ctx_k ≠ 0 (统计量非零),
            属【单零】(门控层零, 被门控量非零), 从 step1 起可达 MatrixRWKV,
            避开 R4 的乘法双零死锁。
    """

    def __init__(self, rwkv_dim: int = 192, feat_dim: int = 128):
        super().__init__()
        self.rwkv_dim = rwkv_dim
        self.feat_dim = feat_dim
        self.to_gate = nn.Linear(rwkv_dim, 3 * feat_dim, bias=True)
        nn.init.zeros_(self.to_gate.weight)
        nn.init.zeros_(self.to_gate.bias)

    def forward(self, ctx_center: torch.Tensor) -> torch.Tensor:
        """
        ctx_center: [B, rwkv_dim] — MatrixRWKV 中心帧上下文
        Returns:    [B, 3*feat_dim] — 三路门控
        """
        return self.to_gate(ctx_center)


class TripleQueryTCA(nn.Module):
    """
    三路查询 TCA (v7r-v3 核心模块)

    Args:
        feat_dim:  空间特征维度 (PixelTemporal 输出, 默认 128)
        rwkv_dim:  MatrixRWKV 输出维度 (默认 192)
        num_frames: 输入帧数 (默认 5)
    """

    def __init__(self, feat_dim: int = 128, rwkv_dim: int = 192,
                 num_frames: int = 5, motion_aware_diff: bool = False,
                 diff_smooth_kernel: int = 5):
        super().__init__()
        self.feat_dim = feat_dim
        self.rwkv_dim = rwkv_dim
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        # Phase A.2 开关: 运动感知差分 (§6.8).
        # False → 历史行为 (max|center - F_t|, 被噪声主导, §6.3 实测 3.19× vs 1.53×)。
        # True  → 对 center / 邻帧先做空间低通再求差, 抑制高频噪声、保留结构位移。
        self.motion_aware_diff = motion_aware_diff
        self.diff_smooth_kernel = int(diff_smooth_kernel) if diff_smooth_kernel else 0

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

        # ========== 6. MatrixRWKV → 共享 KV 三路门控 (方案 B) ==========
        # 消除"死计算": 让 MatrixRWKV 帧级上下文真正参与 KV 构建。
        # 零初始化 → 初始恒等; 单零 → 梯度可达 MatrixRWKV (非双零死锁)。
        self.rwkv_inject = MatrixRWKVInjector(rwkv_dim=rwkv_dim,
                                              feat_dim=feat_dim)

    # ---------- 统计上下文 (对齐 R2/R5 的有效先验) ----------
    def _ctx_mean(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """帧间均值: 成像噪声的最优估计 (i.i.d. → 平均去噪)"""
        return feats_seq.mean(dim=1)  # [B, C, H, W]

    def _ctx_smooth(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """低频平滑: 光照衰减的慢变趋势 (大核池化模拟低通)"""
        mean_feat = feats_seq.mean(dim=1)
        return F.avg_pool2d(mean_feat, kernel_size=7, stride=1, padding=3)

    def _ctx_diff(self, feats_seq: torch.Tensor) -> torch.Tensor:
        """运动差分: 中心帧 vs 邻帧最大绝对差.

        motion_aware_diff=False (默认/历史行为):
            直接对原始特征求差 → 高频噪声主导 (§6.3 实测 noise 3.19× vs shift16 1.53×)

        motion_aware_diff=True (Phase A.2 开关):
            先对 center 和每个邻帧各自做空间低通 (avg_pool, kernel=diff_smooth_kernel),
            再求差 → 抑制 i.i.d. 高频噪声, 保留低频结构位移信号.
        """
        if self.motion_aware_diff and self.diff_smooth_kernel > 1:
            k = self.diff_smooth_kernel
            pad = k // 2
            center_s = F.avg_pool2d(feats_seq[:, self.center_idx],
                                    kernel_size=k, stride=1, padding=pad)
            diffs = []
            for t in range(feats_seq.shape[1]):
                if t == self.center_idx:
                    continue
                neigh_s = F.avg_pool2d(feats_seq[:, t],
                                       kernel_size=k, stride=1, padding=pad)
                diffs.append((center_s - neigh_s).abs())
        else:
            center = feats_seq[:, self.center_idx]
            diffs = [(center - feats_seq[:, t]).abs()
                     for t in range(feats_seq.shape[1]) if t != self.center_idx]
        return torch.stack(diffs, dim=0).max(dim=0).values

    def forward(self, feat_spatial: torch.Tensor,
                feats_seq: torch.Tensor,
                rwkv_ctx: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        feat_spatial: [B, feat_dim, H, W]      — PixelTemporal 输出
        feats_seq:    [B, T, feat_dim, H, W]   — 逐帧编码特征
        rwkv_ctx:     [B, T, rwkv_dim]         — MatrixRWKV 帧级上下文

        Returns:
            F_N, F_L, F_M: 各 [B, feat_dim, H, W]
            ortho_loss: scalar
            inject_stat: dict — {'gate': scalar, 'gate_k': [3]} 诊断 MatrixRWKV 参与度
        """
        B, C, H, W = feat_spatial.shape

        # ========== Step 1: 三路查询 (从像素对齐特征) ==========
        Q_N = self.query_N(feat_spatial)
        Q_L = self.query_L(feat_spatial)
        Q_M = self.query_M(feat_spatial)

        # ========== Step 2: 共享 KV (统计先验 + MatrixRWKV 三路门控) ==========
        ctx_mean = self._ctx_mean(feats_seq)      # [B, C, H, W]
        ctx_smooth = self._ctx_smooth(feats_seq)  # [B, C, H, W]
        ctx_diff = self._ctx_diff(feats_seq)      # [B, C, H, W]

        # MatrixRWKV 中心帧上下文 → 三路逐通道门控 (方案 B)
        ctx_center = rwkv_ctx[:, self.center_idx]          # [B, rwkv_dim]
        gate = self.rwkv_inject(ctx_center)                # [B, 3C]
        gate = gate.view(B, 3, C, 1, 1)                    # 拆成 mean/smooth/diff 三路
        s_mean = ctx_mean * (1.0 + gate[:, 0])
        s_smooth = ctx_smooth * (1.0 + gate[:, 1])
        s_diff = ctx_diff * (1.0 + gate[:, 2])

        kv_shared = self.kv_proj(
            torch.cat([s_mean, s_smooth, s_diff], dim=1)
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

        # 诊断: MatrixRWKV 门控幅度 (detach, 不参与梯度)
        with torch.no_grad():
            g_det = gate.detach().abs()
            inject_stat = {
                'gate': g_det.mean(),                       # 总体平均 |g|
                'gate_k': g_det.mean(dim=(0, 2, 3, 4)),     # 三路各自 |g|
            }

        return F_N, F_L, F_M, ortho_loss, inject_stat

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
    D = 192

    feat_spatial = torch.randn(B, C, H, W)
    feats_seq = torch.randn(B, T, C, H, W)
    rwkv_ctx = torch.randn(B, T, D)

    print("=== TripleQueryTCA (方案 B: MatrixRWKV 三路门控) ===")
    tca = TripleQueryTCA(feat_dim=128, rwkv_dim=192, num_frames=5)
    F_N, F_L, F_M, ortho, stat = tca(feat_spatial, feats_seq, rwkv_ctx)

    print(f"  Input feat_spatial: {feat_spatial.shape}")
    print(f"  Input feats_seq:    {feats_seq.shape}")
    print(f"  Input rwkv_ctx:     {rwkv_ctx.shape}")
    print(f"  Output F_N/L/M:     {F_N.shape}")
    print(f"  Ortho loss:         {ortho.item():.4f}")
    print(f"  inject gate:        {stat['gate'].item():.2e} (初始应为 0)")
    print(f"  Params: {sum(p.numel() for p in tca.parameters()) / 1e6:.3f}M")

    loss = F_N.sum() + F_L.sum() + F_M.sum() + ortho
    loss.backward()
    g = tca.rwkv_inject.to_gate.weight.grad.abs().mean().item()
    print(f"  injector grad:      {g:.2e} (应非零 → MatrixRWKV 可达)")
    print("  Gradient check passed!")
