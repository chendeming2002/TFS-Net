"""
TCA (Temporal Correspondence & Alignment) — Mark1: SWD 输入 (2026-07-01)
========================================================================
核心改动 vs Delta:
  1. 扫描轴 T→H×W：RSRWKV 2D-WKV 四方向并行空间扫描
  2. Token shift: Q-Shift→MVC-Shift
  3. 输出 C_omega_list + F_t_aligned 替代 F_aligned_list
  4. Mark1: 输入来自 SWD feat_tca (已是 H/2×W/2)，不再内部降采样

文献依据:
  - RSRWKV (TCSVT 2025): 2D-WKV + MVC-Shift
  - Vision-RWKV (2024): Bi-WKV bidirectional attention
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
    """Vision-RWKV Bi-WKV — chunk-wise cumsum + forced decay < 1 (Mod1)"""

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.spatial_decay = nn.Parameter(torch.randn(channels) * 0.1)
        self.spatial_first = nn.Parameter(torch.randn(channels) * 0.1)

    @staticmethod
    def _scan_cumsum(ek, ekv, u_coef, ew_pow):
        """chunk-wise cumsum: 每 256 token 递推一次 state.
        T-A1 修复: (1) 跨 chunk 衰减因子方向改正 ew^{j'+1} (原 ew^{cs-1-j'} 反向)
                   (2) state 更新 off-by-one: ew^{cs} (原 ew^{cs-1})
                   (3) 删除死代码 ew_chunk"""
        CHUNK = 256
        B, L, C = ek.shape
        out = torch.zeros(B, L, C, device=ek.device)
        state_num = torch.zeros(B, 1, C, device=ek.device)
        state_den = torch.zeros(B, 1, C, device=ek.device)
        for s in range(0, L, CHUNK):
            e = min(s + CHUNK, L)
            cs = e - s
            ek_c, ekv_c = ek[:, s:e], ekv[:, s:e]
            # local cumsum with chunk-level precision
            S_loc = (ekv_c / ew_pow[:, :cs].clamp(min=1e-12)).cumsum(dim=1) * ew_pow[:, :cs]
            D_loc = (ek_c  / ew_pow[:, :cs].clamp(min=1e-12)).cumsum(dim=1) * ew_pow[:, :cs]
            # combine with state from previous chunks
            # 上一 chunk 末位 (全局 s-1) 到当前位 (全局 s+j') 距离 = j'+1 → 衰减因子 ew^{j'+1}
            decay_state = ew_pow[:, 1:cs+1]
            S = S_loc + state_num * decay_state
            D = D_loc + state_den * decay_state
            out[:, s:e] = (u_coef * ekv_c + S) / (u_coef * ek_c + D + 1e-8)
            # update state: 旧 state 从 prev-last 衰减 cs 步到 new-last (ew^{cs}), 加上新 chunk 局部累积
            state_num = ew_pow[:, cs:cs+1] * state_num + S_loc[:, -1:]
            state_den = ew_pow[:, cs:cs+1] * state_den + D_loc[:, -1:]
        return out

    def forward(self, k: torch.Tensor, v: torch.Tensor,
                total_tokens: int) -> torch.Tensor:
        B, L, C = k.shape
        k = k.clamp(-8, 8)
        v = v.clamp(-8, 8)
        # Mod1: 强制 decay ∈ (0, 1)，用 -softplus 保证 w < 0
        w = -F.softplus(self.spatial_decay)
        ew = (w / total_tokens).exp().view(1, 1, C)
        u = self.spatial_first.clamp(-5, 5)
        u_coef = (u / total_tokens).exp().view(1, 1, C)
        ek, ekv = k.exp(), k.exp() * v
        arange_L = torch.arange(L, device=k.device).float().view(1, L, 1)
        ew_pow = ew.pow(arange_L)
        wkv_fwd = self._scan_cumsum(ek, ekv, u_coef, ew_pow)
        wkv_bwd = self._scan_cumsum(ek.flip(1), ekv.flip(1), u_coef, ew_pow).flip(1)
        return (wkv_fwd + wkv_bwd) * 0.5


# ============================================================
# 3. SpatialWKV2D — 四方向空间扫描
# ============================================================
class SpatialWKV2D(nn.Module):
    """RSRWKV 2D-WKV: 4方向, 每方向独立 BiWKV (per-head LN removed — Vision-RWKV style)"""

    def __init__(self, channels: int):
        super().__init__()
        assert channels % 4 == 0
        self.channels = channels
        self.head_dim = channels // 4
        self.bi_wkv_list = nn.ModuleList([BiWKV(self.head_dim) for _ in range(4)])

        self.proj_r = nn.Linear(channels, channels, bias=False)
        self.proj_k = nn.Linear(channels, channels, bias=False)
        self.proj_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=False)
        self.pre_norm = nn.LayerNorm(channels)
        self.post_norm = nn.LayerNorm(channels)
        nn.init.zeros_(self.proj_out.weight)

        # Mod1: RWKV-7 风格小初始化 (±0.05/√C, ±0.5/√C)
        import math
        nn.init.uniform_(self.proj_k.weight, -0.05/math.sqrt(channels), 0.05/math.sqrt(channels))
        nn.init.uniform_(self.proj_r.weight, -0.5/math.sqrt(channels), 0.5/math.sqrt(channels))
        nn.init.uniform_(self.proj_v.weight, -0.5/math.sqrt(channels), 0.5/math.sqrt(channels))

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
        x_tokens = self.pre_norm(x_tokens)  # pre-norm 防溢出
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
            # Per-direction BiWKV (no per-head LN — post_norm handles fusion distribution)
            wkv_seq = self.bi_wkv_list[i](k_seq, v_seq, total_tokens=N)
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
        self.tau_raw = nn.Parameter(torch.zeros(1))

    @property
    def tau(self):
        return F.softplus(self.tau_raw).clamp(0.02, 0.5) + 0.02

    def forward(self, center_feat: torch.Tensor,
                neighbor_feats: torch.Tensor) -> list:
        B, C, H, W = center_feat.shape
        ds = max(1, min(min(H, W) // 4, 96))  # cap N≤9216 → C_omega≤340MB, safe for 4K
        center_ds = F.adaptive_avg_pool2d(center_feat, (ds, ds))
        neighbor_ds = F.adaptive_avg_pool2d(
            neighbor_feats.reshape(-1, C, H, W), (ds, ds)
        ).reshape(B, -1, C, ds, ds)

        N = ds * ds
        T_n = neighbor_ds.shape[1]
        q = self.proj_q(center_ds)
        q_flat = F.normalize(q.flatten(2).transpose(1, 2), dim=-1)
        tau = self.tau  # softplus-bounded, always > 0.05
        C_omega_list = []
        for t in range(T_n):
            k = self.proj_k(neighbor_ds[:, t])
            k_flat = F.normalize(k.flatten(2).transpose(1, 2), dim=-1)
            sim = torch.bmm(q_flat, k_flat.transpose(1, 2)) / tau
            C_omega_list.append(F.softmax(sim.clamp(-20, 20), dim=-1))
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
class TCA(nn.Module):
    """TCA (Flight9): WKV @ H/2, direct l2_lat input, no internal FPN.

    Flight9 changes:
      - Cancel WFR: input is encoder l2_lat (H/2) directly, no feat_tca residual
      - Remove internal FPN — WKV stays at H/2 (128×128)
      - Remove wfr_lambda parameter
      - C_omega computed on the same l2_lat feature resolution
    """

    def __init__(self, channels: int = 64, num_frames: int = 5):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2

        self.mvc_shift = MVCShift(channels)
        self.spatial_wkv = SpatialWKV2D(channels)
        self.channel_mix = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1),
            nn.GELU(),
            nn.Conv2d(channels * 4, channels, 1),
        )
        self.spatial_gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

        self.corr_gen = TemporalCorrespondence(channels)
        self.temporal_agg = TemporalAggregation(channels)

        # Flight10m1: minimal WFR — HaarDWT LL anchor + HF edge for TCA alignment
        self.dwt_anchor = nn.Sequential(
            nn.InstanceNorm2d(channels, affine=True),   # remove illumination bias from LL
            nn.Conv2d(channels, channels, 1, bias=False),
        )
        self.dwt_hf_proj = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 3, padding=1, groups=4, bias=False),
            LayerNorm2d(channels),
        )
        self.anchor_fuse = nn.Conv2d(channels * 3, channels, 1)

    @staticmethod
    def _haar_dwt(x_4d: torch.Tensor):
        """x: (B, C, H, W) → LL, LH, HL, HH: each (B, C, H/2, W/2)"""
        B, C, H, W = x_4d.shape
        x01, x02 = x_4d[:,:,0::2,:], x_4d[:,:,1::2,:]
        L = (x01 + x02) * 0.5
        H_ = (x01 - x02) * 0.5
        L0, L1 = L[:,:,:,0::2], L[:,:,:,1::2]
        H0, H1 = H_[:,:,:,0::2], H_[:,:,:,1::2]
        LL = (L0 + L1) * 0.5
        LH = (L0 - L1) * 0.5
        HL = (H0 + H1) * 0.5
        HH = (H0 - H1) * 0.5
        return LL, LH, HL, HH

    def forward(self, feats: torch.Tensor) -> Dict:
        """
        feats: (B, T, C, H/2, W/2) — encoder l2_lat directly
        """
        B, T, C, H_ds, W_ds = feats.shape

        # Flight10m1: HaarDWT anchor — LL(IN去光照) + HF(边缘先验) → 增强 TCA 输入
        feats_flat = feats.reshape(B * T, C, H_ds, W_ds)
        LL, LH, HL, HH = self._haar_dwt(feats_flat)
        anchor = self.dwt_anchor(LL)
        anchor_up = F.interpolate(anchor, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
        hf_cat = torch.cat([LH, HL, HH], dim=1)
        hf_up = F.interpolate(hf_cat, size=(H_ds, W_ds), mode='bilinear', align_corners=False)
        hf_feat = self.dwt_hf_proj(hf_up)
        x_enhanced = self.anchor_fuse(torch.cat([feats_flat, anchor_up, hf_feat], dim=1))

        x_shifted = self.mvc_shift(x_enhanced)
        x_wkv = self.spatial_wkv(x_shifted)
        x_cm = self.channel_mix(x_wkv)
        sace_out_ds = x_enhanced + x_cm * self.spatial_gamma
        sace_out = sace_out_ds.reshape(B, T, C, H_ds, W_ds)

        mu_t_clean = sace_out[:, self.center_idx]
        sigma_t_clean = sace_out.std(dim=1, unbiased=False)

        center_orig = feats[:, self.center_idx]
        neighbor_idx = [t for t in range(T) if t != self.center_idx]
        neighbor_orig = feats[:, neighbor_idx]
        C_omega_list = self.corr_gen(center_orig, neighbor_orig)

        sace_4d = sace_out_ds.reshape(B, T, C, H_ds, W_ds)
        center_enhanced = sace_4d[:, self.center_idx]
        neighbor_enhanced = sace_4d[:, neighbor_idx]
        F_t_aligned = self.temporal_agg(center_enhanced, neighbor_enhanced, C_omega_list)

        return {
            "tca_out":        sace_out,
            "mu_t_clean":     mu_t_clean,
            "sigma_t_clean":  sigma_t_clean,
            "C_omega_list":   C_omega_list,
            "F_t_aligned":    F_t_aligned,
            "lff_feats":      [],
            "attn_maps":      [],
        }
