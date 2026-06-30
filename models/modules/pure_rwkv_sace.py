"""
PureRWKVSACE — Charlie-Mark4 (Delta): 空间扫描 + 时序对应矩阵 (2026-06-30)
===========================================================================
核心改动 vs Charlie3:
  1. 扫描轴 T→H×W：RSRWKV 2D-WKV 四方向并行空间扫描
  2. Token shift: Q-Shift→MVC-Shift (多尺度空洞 DWConv)
  3. 输出 C_omega_list + F_t_aligned 替代 F_aligned_list
  4. 删除 edge_weight 死代码

文献依据:
  - RSRWKV (TCSVT 2025): 2D-WKV + MVC-Shift
  - Vision-RWKV (2024): Bi-WKV bidirectional attention 基础
  - C²-STVSR (CVPRW 2026): 4D correlation volume 引导时序对应
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict

from .blocks import LayerNorm2d


# ============================================================
# 1. MVC-Shift (Multi-View Context Token Shift)
# ============================================================
class MVCShift(nn.Module):
    """RSRWKV MVC-Shift: 多尺度空洞 depthwise conv + 1×1 跨通道交互"""

    def __init__(self, channels: int):
        super().__init__()
        self.branches = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels, channels, 3, 1, d, dilation=d, groups=channels, bias=False),
                nn.Conv2d(channels, channels, 1, bias=False),
            )
            for d in [1, 2, 3]
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x
        for branch in self.branches:
            out = out + branch(x)
        return out


# ============================================================
# 2. Bi-WKV (双向 WKV 注意力，单方向)
# ============================================================
class BiWKV(nn.Module):
    """Vision-RWKV Bi-WKV 核心 — 空间序列版，线性复杂度 O(L×C)"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.spatial_decay = nn.Parameter(torch.randn(channels) * 0.1)
        self.spatial_first = nn.Parameter(torch.randn(channels) * 0.1)

    def forward(self, k: torch.Tensor, v: torch.Tensor,
                total_tokens: int) -> torch.Tensor:
        """Bi-WKV 双向扫描 — cumsum 实现 (fwd+bwd)/2"""
        B, L, C = k.shape
        w = self.spatial_decay.clamp(-8, 8)
        u = self.spatial_first.clamp(-5, 5)
        ew = (-w.abs() / total_tokens).exp().view(1, 1, C)
        u_coef = (u / total_tokens).exp().view(1, 1, C)
        ek, ekv = k.exp(), k.exp() * v
        arange_L = torch.arange(L, device=k.device).float().view(1, L, 1)
        ew_pow = ew.pow(arange_L)
        # forward: S_t = Σ_{i≤t} ew^{t-i}·ekv_i
        S_fwd = (ekv / ew_pow).cumsum(dim=1) * ew_pow
        D_fwd = (ek  / ew_pow).cumsum(dim=1) * ew_pow
        wkv_fwd = (u_coef * ekv + S_fwd) / (u_coef * ek + D_fwd + 1e-8)
        # backward: scan from t=L-1 to 0
        S_bwd = (ekv.flip(1) / ew_pow).cumsum(dim=1) * ew_pow
        D_bwd = (ek.flip(1)  / ew_pow).cumsum(dim=1) * ew_pow
        wkv_bwd = (u_coef * ekv + S_bwd) / (u_coef * ek.flip(1) + D_bwd + 1e-8)
        wkv_bwd = wkv_bwd.flip(1)
        return (wkv_fwd + wkv_bwd) * 0.5


# ============================================================
# 3. SpatialWKV2D — 四方向空间扫描
# ============================================================
class SpatialWKV2D(nn.Module):
    """RSRWKV 2D-WKV: 水平/垂直/主对角线/副对角线 四方向 + recep gate"""

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0
        self.channels = channels
        self.head_dim = channels // 4
        self.bi_wkv = BiWKV(self.head_dim)

        self.proj_r = nn.Linear(channels, channels, bias=False)
        self.proj_k = nn.Linear(channels, channels, bias=False)
        self.proj_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=False)
        self.post_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.proj_out.weight)

    @staticmethod
    def _scan_horizontal(x: torch.Tensor) -> torch.Tensor:
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _scan_vertical(x: torch.Tensor) -> torch.Tensor:
        return x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)

    @staticmethod
    def _scan_diag_main(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        coords = []
        for s in range(H + W - 1):
            for i in range(max(0, s - W + 1), min(s + 1, H)):
                coords.append(i * W + (s - i))
        idx = torch.tensor(coords, device=x.device)
        x_flat = x.flatten(2)
        return x_flat[:, :, idx].transpose(1, 2)

    @staticmethod
    def _scan_diag_anti(x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        coords, used = [], set()
        for s in range(H + W - 1):
            for i in range(max(0, s - W + 1), min(s + 1, H)):
                j = W - 1 - (s - i)
                if 0 <= j < W:
                    ij = i * W + j
                    coords.append(ij)
                    used.add(ij)
        for ij in range(H * W):
            if ij not in used:
                coords.append(ij)
        idx = torch.tensor(coords[:H * W], device=x.device)
        x_flat = x.flatten(2)
        return x_flat[:, :, idx].transpose(1, 2)

    @staticmethod
    def _inv_scan(scan_fn, shape, device):
        B, C, H, W = shape
        identity = torch.arange(H * W, device=device).float().view(1, 1, H, W)
        scanned = scan_fn(identity).squeeze(-1).long()
        inv_idx = torch.zeros(1, H * W, dtype=torch.long, device=device)
        for b in range(1):
            inv_idx[b].scatter_(0, scanned[b], torch.arange(H * W, device=device))
        return inv_idx

    def forward(self, x_2d: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x_2d.shape
        N = H * W
        x_tokens = x_2d.flatten(2).transpose(1, 2)
        r = self.proj_r(x_tokens)
        k = self.proj_k(x_tokens)
        v = self.proj_v(x_tokens)

        scan_fns = [self._scan_horizontal, self._scan_vertical,
                    self._scan_diag_main, self._scan_diag_anti]

        heads = []
        for i, scan_fn in enumerate(scan_fns):
            c0, c1 = i * self.head_dim, (i + 1) * self.head_dim
            k_head = k[:, :, c0:c1]
            v_head = v[:, :, c0:c1]
            k_2d = k_head.transpose(1, 2).reshape(B, self.head_dim, H, W)
            v_2d = v_head.transpose(1, 2).reshape(B, self.head_dim, H, W)
            k_seq = scan_fn(k_2d)
            v_seq = scan_fn(v_2d)
            wkv_seq = self.bi_wkv(k_seq, v_seq, total_tokens=N)
            inv_idx = self._inv_scan(scan_fn, (B, self.head_dim, H, W), x_2d.device)
            wkv_restored = wkv_seq[:, inv_idx[0]]
            heads.append(wkv_restored)

        wkv_concat = torch.cat(heads, dim=-1)
        output = torch.sigmoid(r) * wkv_concat
        output = self.proj_out(output)
        output = self.post_norm(output)
        return output.transpose(1, 2).reshape(B, C, H, W)


# ============================================================
# 4. TemporalCorrespondence → C_omega_list
# ============================================================
class TemporalCorrespondence(nn.Module):
    """生成 C_omega_list: 中心帧与每个邻帧的空间 cosine similarity 矩阵"""

    def __init__(self, channels: int, proj_dim: int = 0):
        super().__init__()
        proj_dim = proj_dim if proj_dim > 0 else max(channels // 4, 16)
        self.proj_dim = proj_dim
        self.proj_q = nn.Conv2d(channels, proj_dim, 1, bias=False)
        self.proj_k = nn.Conv2d(channels, proj_dim, 1, bias=False)
        self.tau = nn.Parameter(torch.ones(1) * 0.07)

    def forward(self, center_feat: torch.Tensor,
                neighbor_feats: torch.Tensor) -> list:
        B, C, H, W = center_feat.shape
        ds = max(1, min(H, W) // 4)
        center_ds = F.adaptive_avg_pool2d(center_feat, (ds, ds))
        neighbor_ds = F.adaptive_avg_pool2d(
            neighbor_feats.reshape(-1, C, H, W), (ds, ds)
        ).reshape(B, -1, C, ds, ds)

        N = ds * ds
        T_n = neighbor_ds.shape[1]
        q = self.proj_q(center_ds)
        q_flat = F.normalize(q.flatten(2).transpose(1, 2), dim=-1)
        tau = self.tau.clamp(min=0.01)
        C_omega_list = []
        for t in range(T_n):
            k = self.proj_k(neighbor_ds[:, t])
            k_flat = F.normalize(k.flatten(2).transpose(1, 2), dim=-1)
            sim = torch.bmm(q_flat, k_flat.transpose(1, 2)) / tau
            C_omega_list.append(F.softmax(sim, dim=-1))
        return C_omega_list


# ============================================================
# 5. TemporalAggregation → F_t_aligned
# ============================================================
class TemporalAggregation(nn.Module):
    """用 C_omega_list 对齐邻帧到中心帧坐标系，加权聚合得到 F_t_aligned"""

    def __init__(self, channels: int):
        super().__init__()
        self.frame_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        self.out_norm = LayerNorm2d(channels)

    def forward(self, center_feat: torch.Tensor,
                neighbor_feats: torch.Tensor,
                C_omega_list: list) -> torch.Tensor:
        B, C, H, W = center_feat.shape
        N = C_omega_list[0].shape[-1]
        ds = int(N ** 0.5)  # spatial resolution of C_omega
        T_n = neighbor_feats.shape[1]

        # 降采样特征到与 C_omega 匹配的分辨率
        center_ds = F.adaptive_avg_pool2d(center_feat, (ds, ds))
        neighbor_ds = F.adaptive_avg_pool2d(
            neighbor_feats.reshape(-1, C, H, W), (ds, ds)
        ).reshape(B, T_n, C, ds, ds)

        warped_list, weight_list = [], []
        for t in range(T_n):
            omega = C_omega_list[t]  # (B, N, N)
            f_t = neighbor_ds[:, t].flatten(2)  # (B, C, N)
            warped = torch.bmm(f_t, omega.transpose(1, 2))  # (B, C, N)
            warped_2d = warped.reshape(B, C, ds, ds)
            warped_list.append(warped_2d)
            wt = self.frame_gate(torch.cat([center_ds, warped_2d], dim=1))
            weight_list.append(wt)
        warped_stack = torch.stack(warped_list, dim=1)
        weights = F.softmax(torch.stack(weight_list, dim=1), dim=1)
        agg = (warped_stack * weights).sum(dim=1)  # (B, C, ds, ds)

        # 上采样回原始分辨率 + 残差
        agg_up = F.interpolate(agg, size=(H, W), mode='bilinear', align_corners=False)
        return self.out_norm(agg_up + center_feat)


# ============================================================
# 6. PureRWKVSACE — Charlie-Mark4 完整模块
# ============================================================
class PureRWKVSACE(nn.Module):
    """Delta SACE: 空间扫描 + 时序对应

    Args:
        channels: 特征通道数 (默认 64)
        num_frames: 帧数 T (默认 5)
    """

    def __init__(self, channels: int = 64, num_frames: int = 5,
                 lff_module=None, n_layer: int = 1):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        # --- 帧内空间处理 ---
        self.mvc_shift = MVCShift(channels)
        self.spatial_wkv = SpatialWKV2D(channels)
        self.channel_mix = nn.Sequential(
            LayerNorm2d(channels),
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.spatial_gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        # --- 时序对应 ---
        self.corr_gen = TemporalCorrespondence(channels)
        self.temporal_agg = TemporalAggregation(channels)

    def forward(self, feats: torch.Tensor,
                tfsi_out: Dict = None,
                cached_lff: Dict = None) -> Dict:
        """
        feats: (B, T, C, H, W) Encoder 输出特征
        Returns:
            sace_out: (B, T, C, H, W) 空间增强后的多帧特征
            C_omega_list: list of (T-1) tensors, each (B, N, N)
            F_t_aligned: (B, C, H, W) 中心帧对齐增强特征
            mu_t_clean / sigma_t_clean: 兼容旧接口
        """
        B, T, C, H, W = feats.shape
        device = feats.device

        # Delta: 降采样到 H/2×W/2 控制显存
        feats_ds = F.interpolate(
            feats.reshape(B * T, C, H, W), scale_factor=0.5, mode='bilinear', align_corners=False
        ).reshape(B, T, C, H // 2, W // 2)
        H_ds, W_ds = H // 2, W // 2

        # --- Step 1: 逐帧空间处理 ---
        x_flat = feats_ds.reshape(B * T, C, H_ds, W_ds)
        x_shifted = self.mvc_shift(x_flat)
        x_wkv = self.spatial_wkv(x_shifted)
        x_cm = self.channel_mix(x_wkv)
        sace_out_ds = x_flat + x_cm * self.spatial_gamma

        # 上采样回原始分辨率
        sace_out = F.interpolate(
            sace_out_ds, size=(H, W), mode='bilinear', align_corners=False
        ).reshape(B, T, C, H, W)

        # --- Step 2: 统计量 ---
        mu_t_clean = sace_out[:, self.center_idx]
        sigma_t_clean = sace_out.std(dim=1, unbiased=False)

        # --- Step 3: 时序对应 → C_omega_list (在降采样分辨率下计算) ---
        center_orig = feats_ds[:, self.center_idx]
        neighbor_idx = [t for t in range(T) if t != self.center_idx]
        neighbor_orig = feats_ds[:, neighbor_idx]
        C_omega_list = self.corr_gen(center_orig, neighbor_orig)

        # --- Step 4: 时序聚合 → F_t_aligned (在降采样分辨率下计算，然后上采样) ---
        center_enhanced_ds = sace_out_ds.reshape(B, T, C, H_ds, W_ds)[:, self.center_idx]
        neighbor_enhanced_ds = sace_out_ds.reshape(B, T, C, H_ds, W_ds)[:, neighbor_idx]
        F_t_aligned_ds = self.temporal_agg(
            center_enhanced_ds, neighbor_enhanced_ds, C_omega_list
        )
        # 上采样到原始分辨率 H×W
        F_t_aligned = F.interpolate(
            F_t_aligned_ds, size=(H, W), mode='bilinear', align_corners=False
        )

        return {
            "sace_out":       sace_out,
            "mu_t_clean":     mu_t_clean,
            "sigma_t_clean":  sigma_t_clean,
            "C_omega_list":   C_omega_list,
            "F_t_aligned":    F_t_aligned,
            "lff_feats":      [],
            "attn_maps":      [],
        }
