"""
TFSNet — RWKV-Delta 完整模型
=================================

Delta 核心架构改进:
  1. MVC-Shift 替代 Q-Shift (多尺度空洞DWConv)
  2. SpatialWKV2D: 4方向空间扫描 (替代帧间Bi-WKV)
  3. TemporalCorrespondence → C_omega_list (显式时序对应)
  4. TemporalAggregation → F_t_aligned (对齐锚)
  5. A_illu: s_illum → IFPN → IGRF (替代直接注入)
  6. CrossFusionGate: deploy模式 支持重参数化

已移除: DWT-LFF在SACE、VRWKVStyleSpatialMix帧间扫描、
  DeformableCrossAttention、AmpEnhance、LFFFeatureAdapter
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import (
    ConvBlock, ResBlock, NAFBlock, LayerNorm2d,
    window_partition_video, window_reverse_video,
    window_partition_2d, window_reverse_2d,
    pad_to_window, unpad_from_window,
    safe_divide, pairwise_cosine_logits,
)
from .dwt_lff import SpatialDWTLFFAdapter

# ============================================================
# Utility Classes
# ============================================================


class ChannelNorm(nn.Module):
    """可学习的 channel-wise 仿射变换"""
    def __init__(self, channels: int):
        super().__init__()
        self.gamma = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        mean = x.mean(dim=(2, 3), keepdim=True)
        std = x.std(dim=(2, 3), keepdim=True) + 1e-6
        return (x - mean) / std * self.gamma + self.beta


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, act=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, x):
        return self.block(x)



class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=64,
                 num_bottleneck_blocks: int = 0):
        super().__init__()
        if len(level_channels) == 4:
            c1, c2, c3, c4 = level_channels
        else:
            c1, c2, c3 = level_channels
            c4 = None

        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)
        self.has_stage4 = c4 is not None
        if self.has_stage4:
            self.stage4 = EncoderStage(c3, c4, stride=2)
            self.lateral4 = nn.Conv2d(c4, fused_channels, 1, 1, 0)

        # v5.8: 瓶颈块 — 在最粗层 (c3, H/4) 增加深度, 最省显存
        self.num_bottleneck_blocks = num_bottleneck_blocks
        if num_bottleneck_blocks > 0:
            bottleneck_ch = c4 if c4 is not None else c3
            self.bottleneck = nn.Sequential(
                *[ResBlock(bottleneck_ch) for _ in range(num_bottleneck_blocks)]
            )
        else:
            self.bottleneck = None

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        # v5.9.1: fuse 前加 LayerNorm2d 控制值域
        # 根因: lateral 累加导致 p1 值域 ±1800 → fuse conv 输出 ±12000 → GELU(全负)=0 → 特征死亡
        # LayerNorm2d 把 p1 归一化到均值 0 方差 1 → fuse conv 正常工作 → GELU 不饱和
        self.fuse_norm = LayerNorm2d(fused_channels)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x, return_coarse=False):
        """
        Args:
            x             : (B, C, H, W) 单帧图像
            return_coarse : 是否同时返回最粗尺度特征

        Returns:
            return_coarse=False: fused_feat (B, C_f, H, W)
            return_coarse=True : (fused_feat, coarse_feat)
                                 4-stage: coarse = l4 (B, c4, H/8, W/8)
                                 3-stage: coarse = l3 (B, c3, H/4, W/4)
        """
        l1 = self.stage1(x)       # (B, c1, H, W)
        l2 = self.stage2(l1)      # (B, c2, H/2, W/2)
        l3 = self.stage3(l2)      # (B, c3, H/4, W/4)

        if self.has_stage4:
            l4 = self.stage4(l3)  # (B, c4, H/8, W/8)
            # v5.8: 瓶颈块在最粗层
            if self.bottleneck is not None:
                l4 = self.bottleneck(l4)
            p4 = self.lateral4(l4)
            p3 = self.lateral3(l3) + F.interpolate(p4, size=l3.shape[-2:], mode="bilinear", align_corners=False)
        else:
            l4 = None
            # v5.8: 瓶颈块在最粗层 (3级编码器时在 l3)
            if self.bottleneck is not None:
                l3 = self.bottleneck(l3)
            p3 = self.lateral3(l3)

        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        # v5.9.1: fuse 前归一化, 防止 lateral 累加值域爆炸导致 GELU 饱和
        p1_normed = self.fuse_norm(p1)
        fused = self.fuse(p1_normed)     # (B, C_f, H, W)

        if return_coarse:
            coarse = l4 if self.has_stage4 else l3
            return fused, coarse
        return fused

    def forward(self, x, return_coarse=False):
        """
        Args:
            x             : (B, T, C, H, W) 多帧序列
            return_coarse : 是否同时返回最粗尺度特征

        Returns:
            return_coarse=False: feats (B, T, C_f, H_f, W_f)
            return_coarse=True : (feats, coarse_feats)
                                 feats        : (B, T, C_f, H, W)
                                 coarse_feats : (B, T, c3, H/4, W/4)
        """
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)

        if return_coarse:
            fused, coarse = self.forward_single(x, return_coarse=True)
            _, cf, hf, wf = fused.shape
            _, cc, hc, wc = coarse.shape
            fused = fused.view(b, t, cf, hf, wf)
            coarse = coarse.view(b, t, cc, hc, wc)
            return fused, coarse
        else:
            feat = self.forward_single(x, return_coarse=False)
            _, cf, hf, wf = feat.shape
            return feat.view(b, t, cf, hf, wf)


# ============================================================
# TFSI: Time-Frequency Source Indicator
# ============================================================


class SpatialBranch(nn.Module):
    """
    TFSI 空间分支：时域统计量 → Conv → F_s

    v5.6 P1-5: median 改为 soft_median（可导），让 μ_t 梯度能回传到 encoder。
    旧版用 torch.median（不可导），μ_t 不向 encoder 传梯度。

    对多帧特征 {F_i} 沿时间维计算：
        μ_t(x,y)  = soft_median_i{F_i(x,y)}     # v5.6: 可导软中位值
        σ_t²(x,y) = Var_i{F_i(x,y)}             # 时域方差（噪声/运动度量）
        SNR(x,y)  = μ_t / (σ_t + ε)             # 信噪比估计

    拼接后经 3×3 Conv 得到空间特征 F_s ∈ R^{C_f × H × W}。
    """

    def __init__(self, channels: int, fused_channels: int, eps: float = 1e-6,
                 soft_median_tau: float = 0.1, use_soft_median: bool = True):
        super().__init__()
        self.eps = eps
        self.use_soft_median = use_soft_median
        self.soft_median_tau = soft_median_tau
        # 输入：[μ_t, σ_t, SNR]，共 3 通道（每通道为 C 维特征的压缩统计）
        self.conv = nn.Sequential(
            ConvBlock(channels * 3, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    @staticmethod
    def _soft_median(x: torch.Tensor, dim: int = 1, tau: float = 0.1) -> torch.Tensor:
        """梯度友好的软中位值近似（参照 SACE._soft_median）。"""
        with torch.no_grad():
            med = x.median(dim=dim).values.unsqueeze(dim)
        dist = (x - med).abs()
        weights = F.softmax(-dist / tau, dim=dim)
        return (weights * x).sum(dim=dim)

    def forward(self, feats: torch.Tensor) -> dict:
        """
        Args:
            feats: (B, T, C, H, W) 多帧编码器特征

        Returns:
            dict:
                F_s       : (B, C_f, H, W) 空间特征
                mu_t      : (B, C, H, W) 时域中位值（供 SACE/NDPN 使用）
                sigma_t   : (B, C, H, W) 时域标准差（供 SACE/NDPN 使用）
                snr       : (B, C, H, W) 原始 SNR 估计 μ_t/(σ_t+ε)
        """
        if self.use_soft_median:
            # v5.6 P1-5: soft_median 可导，梯度能回传到 encoder
            mu_t = self._soft_median(feats, dim=1, tau=self.soft_median_tau)
        else:
            # 旧版: torch.median 不可导
            mu_t = feats.median(dim=1).values          # (B, C, H, W)

        # 时域方差：对 T 维取 var
        sigma_t_sq = feats.var(dim=1, unbiased=False)  # (B, C, H, W)
        sigma_t = torch.sqrt(sigma_t_sq + self.eps)     # (B, C, H, W) 标准差

        # 信噪比估计（TFSI 空间分支用原始 μ_t，非光照归一化后的 μ_t^clean）
        snr = mu_t / (sigma_t + self.eps)          # (B, C, H, W)

        # 拼接三路统计量 → Conv
        stats = torch.cat([mu_t, sigma_t, snr], dim=1)  # (B, 3C, H, W)
        f_s = self.conv(stats)                     # (B, C_f, H, W)

        return {
            "F_s": f_s,
            "mu_t": mu_t,
            "sigma_t": sigma_t,
            "snr": snr,
        }




class FrequencyBranch(nn.Module):
    """
    TFSI 频域分支 — Charlie P0-1: 多帧邻居融合用于 s_noise 估计。

    v6.4: 仅中心帧 → 噪声/光照解耦物理上不足
    Charlie: 拼接邻帧 LFF 时域均值 → 频域分支获得时序上下文
    """

    def __init__(
        self,
        channels: int,
        fused_channels: int,
        K: int = 10,
        n_ang_freq: int = 1,
        per_channel_rbf: bool = False,
        phase_preserving: bool = True,
        dwt_lff: SpatialDWTLFFAdapter = None,
        use_temporal_fusion: bool = True,
    ):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.dwt_lff = dwt_lff
        self.use_temporal_fusion = use_temporal_fusion

        if channels == fused_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
            lff_channels = channels
        else:
            self.in_proj = nn.Conv2d(channels, fused_channels, kernel_size=1, bias=True)
            self.out_proj = nn.Identity()
            lff_channels = fused_channels

        if dwt_lff is not None:
            self.lff = None
        else:
            self.lff = LFFFeatureAdapter(channels=lff_channels, K=K, n_ang_freq=n_ang_freq,
                                         per_channel_rbf=per_channel_rbf, phase_preserving=phase_preserving)
        # Charlie P0-1: 时域融合 Conv (中心 ∥ 邻帧均值 → 单通道特征)
        self.temporal_fuse = nn.Conv2d(channels, fused_channels, 3, 1, 1) if use_temporal_fusion else None

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        f_center = feats[:, center_idx]
        f_center = self.in_proj(f_center)

        if self.dwt_lff is not None:
            out = self.dwt_lff(f_center)
            f_f_center = out["feat_tfsi"]
        else:
            f_f_center = self.lff(f_center)

        # Charlie P0-1: 邻帧时域均值 → 时序上下文
        if self.temporal_fuse is not None:
            f_neighbors = torch.cat([feats[:, :center_idx], feats[:, center_idx+1:]], dim=1)
            f_neighbor_mean = f_neighbors.mean(dim=1)  # (B, C, H, W) 邻帧均值
            f_fused = self.temporal_fuse(f_f_center + f_neighbor_mean)
        else:
            f_fused = f_f_center

        f_f = self.out_proj(f_fused)
        return f_f




class ConcatFusion(nn.Module):
    """
    TFSI 拼接融合（v5.2: 替代 sigmoid 门控）

    公式：
        F_fused = Conv3x3(GELU(Conv3x3(Concat[F_s, F_f])))

    两分支拼接后经两层 3×3 Conv 学习跨通道交互，
    比 sigmoid 门控（g + (1-g) = 1 互补约束）表达更灵活。
    """

    def __init__(self, fused_channels: int):
        super().__init__()
        # 输入：Concat[F_s, F_f]，共 2*C_f 通道
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels * 2, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_s: (B, C_f, H, W) 空间分支特征
            f_f: (B, C_f, H, W) 频域分支特征

        Returns:
            F_fused: (B, C_f, H, W) 融合特征
        """
        return self.fuse(torch.cat([f_s, f_f], dim=1))




class IntensityHead(nn.Module):
    """
    TFSI 双源独立强度输出头 (v5: 移除 s_motion)

    v6 Bravo: 新增 phase_conf 通道 — 相位置信度估计
      低光视频相位 = 结构 + 运动模糊 + 噪声扭曲 (FDN 2025 + FAN 2025 + FourierDiff 2024)
      相位置信度低 → s_noise 应提高 (不可靠区域增加去噪力度)
    """

    def __init__(self, fused_channels: int, hidden_channels: int = 32):
        super().__init__()
        self.conv = nn.Conv2d(fused_channels + 1, 2, 1, 1, 0)  # +1 for phase_conf

    def forward(self, f_fused: torch.Tensor, phase_conf: torch.Tensor = None) -> dict:
        if phase_conf is not None:
            f_input = torch.cat([f_fused, phase_conf], dim=1)
        else:
            # 向后兼容: phase_conf=None → 用零填充
            f_input = torch.cat([f_fused, torch.zeros_like(f_fused[:, :1])], dim=1)
        raw = self.conv(f_input)
        intensities = torch.sigmoid(raw)
        s_illum = intensities[:, 0:1]
        s_noise = intensities[:, 1:2]
        # Charlie: phase_conf 调制 s_noise — 相位不可靠 → 减小 s_noise → 增大中心帧残差贡献
        if phase_conf is not None:
            s_noise = s_noise * (1.0 - 0.3 * (1.0 - phase_conf))
            s_noise = s_noise.clamp(0.0, 1.0)
        return {"s_illum": s_illum, "s_noise": s_noise}



# ============================================================
# PureRWKVSACE — Delta: 空间扫描 + 时序对应
# ============================================================

# Delta 核心改动 vs Charlie3:
#   1. MVC-Shift 替代 Q-Shift (多尺度空洞DWConv)
#   2. SpatialWKV2D: 4方向空间扫描 (替帧间Bi-WKV)
#   3. TemporalCorrespondence → C_omega_list
#   4. TemporalAggregation → F_t_aligned
#   5. 移除 DWT-LFF 路径, 直接使用 Encoder 原始特征


class TFSI(nn.Module):
    """
    TFSI 时频源指示器：整合空间分支、频域分支、拼接融合、强度输出

    数据流：
        feats (B,T,C,H,W)
            ├── SpatialBranch  → F_s (B,C_f,H,W), μ_t, σ_t, snr
            └── FrequencyBranch → F_f (B,C_f,H,W)
                    ↓
              ConcatFusion(F_s, F_f) → F_fused (B,C_f,H,W)
                    ↓
              IntensityHead(F_fused) → s_illum, s_noise (B,1,H,W)

    freq_branch 属性可为 None（当前状态）或注入外部 LFF 模块（待实现），
    供后续 SACE 共享 LFF 时扩展。
    """

    def __init__(self, channels: int = 64, fused_channels: int = 64, eps: float = 1e-6,
                 use_soft_median: bool = True, dwt_lff: SpatialDWTLFFAdapter = None):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.eps = eps

        self.norm = LayerNorm2d(channels)
        self.spatial_branch = SpatialBranch(channels, fused_channels, eps=eps,
                                            use_soft_median=use_soft_median)
        self.freq_branch = FrequencyBranch(
            channels=channels,
            fused_channels=fused_channels,
            K=10,
            n_ang_freq=1,
            per_channel_rbf=False,
            phase_preserving=True,
            dwt_lff=dwt_lff,
        )
        self.concat_fusion = ConcatFusion(fused_channels)
        self.intensity_head = IntensityHead(fused_channels)
        # v6 Bravo: 相位置信度估计
        self.phase_conf_head = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels // 2, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels // 2, 1, 1),
            nn.Sigmoid(),
        )

    def forward(self, feats: torch.Tensor) -> dict:
        """
        Args:
            feats: (B, T, C, H, W) 多帧编码器特征（所有帧）

        Returns:
            dict:
                F_fused   : (B, C_f, H, W) 融合特征
                F_s       : (B, C_f, H, W) 空间分支特征（供调试/可视化）
                F_f       : (B, C_f, H, W) 频域分支特征（供调试/可视化）
                mu_t      : (B, C, H, W) 时域中位值
                sigma_t   : (B, C, H, W) 时域标准差
                snr       : (B, C, H, W) 原始 SNR 估计
                s_illum   : (B, 1, H, W) 光照强度
                s_noise   : (B, 1, H, W) 噪声强度
        """
        b, t, c, h, w = feats.shape
        center_idx = t // 2

        # 逐帧 LayerNorm（对每帧独立归一化）
        feats_norm = feats.view(b * t, c, h, w)
        feats_norm = self.norm(feats_norm).view(b, t, c, h, w)

        # 空间分支
        spatial_out = self.spatial_branch(feats_norm)
        f_s = spatial_out["F_s"]

        # 频域分支（当前为零张量占位）
        f_f = self.freq_branch(feats_norm, center_idx)

        # 拼接融合
        f_fused = self.concat_fusion(f_s, f_f)

        # v6 Bravo: 相位置信度 — 从频域特征估计相位可靠度
        phase_conf = self.phase_conf_head(f_f)

        # 强度输出 (phase_conf 调制 s_noise)
        intensities = self.intensity_head(f_fused, phase_conf=phase_conf)

        return {
            "F_fused": f_fused,
            "F_s": f_s,
            "F_f": f_f,
            "mu_t": spatial_out["mu_t"],
            "sigma_t": spatial_out["sigma_t"],
            "snr": spatial_out["snr"],
            "s_illum": intensities["s_illum"],
            "s_noise": intensities["s_noise"],
        }



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



# ============================================================
# IFPN: Illumination Feature Pyramid Network
# ============================================================


class IllumExtract(nn.Module):
    """
    Retinexformer 风格的光照图提取器。

    Args:
        img_channels       : 输入图像通道 (默认 3)
        feat_channels      : 输入特征通道 (默认 96)
        feat_proj_channels : 特征压缩后通道 (默认 16)
        n_fea_middle       : 中间通道数 (默认 32, 能被 groups=4 整除)
        n_fea_in           : groups 数 (默认 4)
    """

    def __init__(
        self,
        img_channels: int = 3,
        feat_channels: int = 96,
        feat_proj_channels: int = 16,
        n_fea_middle: int = 32,
        n_fea_in: int = 4,
    ):
        super().__init__()
        assert n_fea_middle % n_fea_in == 0

        n_fea_total = img_channels + 1 + feat_proj_channels

        self.feat_proj = nn.Conv2d(feat_channels, feat_proj_channels, kernel_size=1, bias=True)
        self.conv1 = nn.Conv2d(n_fea_total, n_fea_middle, kernel_size=1, bias=True)
        self.depth_conv = nn.Conv2d(
            n_fea_middle, n_fea_middle,
            kernel_size=5, padding=2, bias=True,
            groups=n_fea_in,
        )
        self.conv2 = nn.Conv2d(n_fea_middle, img_channels, kernel_size=1, bias=True)

    def forward(self, img_down: torch.Tensor, feat_L: torch.Tensor) -> torch.Tensor:
        img_mean = img_down.mean(dim=1, keepdim=True)
        feat_proj = self.feat_proj(feat_L)
        x = torch.cat([img_down, img_mean, feat_proj], dim=1)
        x = self.conv1(x)
        x = self.depth_conv(x)
        L = self.conv2(x)
        return L




class IFPN(nn.Module):
    """
    Illumination-Filtering Pyramid Network.

    v5.3: 用 SACE 对齐特征替代 Encoder 粗特征，通过 coarse_adapter 适配维度。

    Args:
        fused_channels   : 中心帧特征通道数 (默认 64)
        aligned_channels : SACE 对齐特征通道数 (默认 = fused_channels)
        coarse_channels  : 内部粗特征通道 (默认 128, IllumExtract 输入)
        img_channels     : 图像通道 (默认 3)
    """

    def __init__(
        self,
        fused_channels: int = 64,
        aligned_channels: int = None,
        coarse_channels: int = 128,
        img_channels: int = 3,
        feat_proj_channels: int = 16,
        n_fea_middle: int = 32,
        n_fea_in: int = 4,
        sim_temperature: float = 1.0,
        max_bright: float = 4.0,
    ):
        super().__init__()
        self.fused_channels = fused_channels
        self.aligned_channels = aligned_channels or fused_channels
        self.coarse_channels = coarse_channels
        self.img_channels = img_channels
        self.sim_temperature = sim_temperature
        self.max_bright = max_bright

        # v5.3: SACE 对齐特征 → 粗特征适配器 (1×1 Conv + 空间下采样)
        self.coarse_adapter = nn.Sequential(
            nn.Conv2d(self.aligned_channels, coarse_channels, 1, 1, 0),
            nn.GELU(),
        )

        # Charlie3 P0: s_illum 先验注入 (零初始化，渐进学习)
        # s_illum 作为光照先验投影到粗特征空间，与 coarse_adapter 输出相加
        self.s_illum_proj = nn.Sequential(
            nn.Conv2d(1, self.aligned_channels, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(self.aligned_channels, coarse_channels, 3, 1, 1),
        )
        nn.init.zeros_(self.s_illum_proj[0].weight)
        nn.init.zeros_(self.s_illum_proj[2].weight)
        nn.init.zeros_(self.s_illum_proj[2].bias)

        # Delta: A_illu 生成 (从 IGRF 移入)
        self.illu_conv = nn.Sequential(
            nn.Conv2d(fused_channels, fused_channels, 3, 1, 1, groups=fused_channels),
            nn.Conv2d(fused_channels, 1, 1),
            nn.Sigmoid(),
        )

        # Delta: F_t_aligned 锚定光照 (参考 FRESCO spatial correspondence)
        self.illu_anchor = nn.Sequential(
            nn.Conv2d(fused_channels * 2, fused_channels, 1),
            nn.GELU(),
            nn.Conv2d(fused_channels, fused_channels, 1),
        )
        self.illu_anchor_gate = nn.Parameter(torch.zeros(1, fused_channels, 1, 1))

        self.illum_extract = IllumExtract(
            img_channels=img_channels,
            feat_channels=coarse_channels,
            feat_proj_channels=feat_proj_channels,
            n_fea_middle=n_fea_middle,
            n_fea_in=n_fea_in,
        )

        self.img_estimator = nn.Sequential(
            nn.Conv2d(coarse_channels, 32, kernel_size=1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, kernel_size=1, bias=True),
        )

        self.ratio_proj = nn.Conv2d(img_channels, fused_channels, kernel_size=1, bias=True)

        # v5.5: s_illum removed from IFPN (directly injected into IGRF instead)
        # illum_cond_proj (65->64) no longer needed

        self.feat_refine = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=1, stride=1, padding=0, act=False),
        )

        # v4.3: hybrid estimation - feature space delta projected to image space
        self.lit_up_proj = nn.Conv2d(fused_channels, img_channels, kernel_size=1, bias=True)

        # v5.9.2: side_head 用于中间感知监督 (DarkIR EnhanceLoss 风格)
        # 把 f_illum_feat 投影为低分辨率图像, 监督 GT 下采样, 强制光照特征有意义
        self.side_head = nn.Sequential(
            nn.Conv2d(fused_channels, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, img_channels, 3, 1, 1, bias=True),
        )

    def forward(
        self,
        I_t_down: torch.Tensor,
        aligned_feats: torch.Tensor,
        center_idx: int,
        imgs_down: torch.Tensor = None,
        s_illum: torch.Tensor = None,
        F_t_aligned: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            I_t_down      : (B, 3, h, w) 下采样后的中心帧图像
            aligned_feats : (B, T, C_a, H, W) SACE 对齐后的多帧特征
            center_idx    : 中心帧索引
            imgs_down     : (B, T, 3, h, w) 下采样后的多帧图像（可选）
            s_illum       : (B, 1, H, W) TFSI 光照退化强度
            F_t_aligned   : (B, C_coarse, H, W) Delta SACE 对齐增强特征
        """
        B, T, C_a, H, W = aligned_feats.shape

        # Charlie3 P0: s_illum 先验注入 (替换原 aligned_feats 调制)
        # 注入位置: coarse_adapter 之后，作为光照先验加到粗特征上
        if s_illum is not None:
            # s_illum_proj: (B,C_a,H,W) → (B,coarse_c,H,W)
            illum_prior = self.s_illum_proj(s_illum)
            # broadcast 到每一帧: (B,1,C_c,h,w) → (B,T,C_c,h,w)
            BT = B * T
            aligned_flat = aligned_feats.reshape(BT, *aligned_feats.shape[2:])
            projected = self.coarse_adapter(aligned_flat)
            h, w = I_t_down.shape[-2:]
            coarse_flat = F.adaptive_avg_pool2d(projected, (h, w))
            # 注入 s_illum 先验 (每帧共享)
            illum_prior_2d = F.interpolate(illum_prior, size=(h, w), mode='bilinear',
                                           align_corners=False)
            coarse_flat = coarse_flat + illum_prior_2d.repeat(T, 1, 1, 1)
            coarse_feats = coarse_flat.reshape(B, T, self.coarse_channels, h, w)
        else:
            BT = B * T
            aligned_flat = aligned_feats.reshape(BT, *aligned_feats.shape[2:])
            projected = self.coarse_adapter(aligned_flat)
            h, w = I_t_down.shape[-2:]
            coarse_flat = F.adaptive_avg_pool2d(projected, (h, w))
            coarse_feats = coarse_flat.reshape(B, T, self.coarse_channels, h, w)

        # F_t_L: 中心帧粗特征（内部生成）
        F_t_L = coarse_feats[:, center_idx]

        # Step 1: 中心帧光照
        L_t = self.illum_extract(I_t_down, F_t_L)

        # Step 2: 逐帧光照
        L_list = []
        for i in range(T):
            F_i_L = coarse_feats[:, i]
            if imgs_down is not None:
                I_i_down = imgs_down[:, i]
            else:
                I_i_down = self.img_estimator(F_i_L)
            L_i = self.illum_extract(I_i_down, F_i_L)
            L_list.append(L_i)

        # Step 3: 帧间相似度
        f_t_coarse = coarse_feats[:, center_idx]
        neighbor_indices = [i for i in range(T) if i != center_idx]
        neighbors = coarse_feats[:, neighbor_indices]
        sim_logits = pairwise_cosine_logits(f_t_coarse, neighbors)
        weights = F.softmax(sim_logits / self.sim_temperature, dim=-1)
        weights = weights.reshape(B, T - 1, 1, 1, 1)

        # Step 4: 邻帧光照加权
        L_neighbors = torch.stack([L_list[i] for i in neighbor_indices], dim=1)
        L_ref = (weights * L_neighbors).sum(dim=1)

        # Step 5: Illumination ratio estimation
        eps = 1e-3
        L_ratio_lr = (L_ref / (L_t.abs() + eps)).clamp(0.5, 8.0)
        L_ratio = F.interpolate(L_ratio_lr, size=(H, W), mode='bilinear', align_corners=False)
        ratio_feat = self.ratio_proj(L_ratio)

        # v5.5: s_illum no longer concatenated here — directly injected into IGRF
        f_illum_feat = self.feat_refine(ratio_feat)

        # Delta: F_t_aligned 锚定光照特征 (防止帧间闪烁)
        if F_t_aligned is not None:
            F_t_resized = F.interpolate(F_t_aligned, size=f_illum_feat.shape[-2:],
                                        mode='bilinear', align_corners=False)
            anchor_feat = self.illu_anchor(torch.cat([f_illum_feat, F_t_resized], dim=1))
            f_illum_feat = f_illum_feat + anchor_feat * self.illu_anchor_gate.tanh()

        # Delta: A_illu 生成 (从 IGRF 移入)
        A_illu = self.illu_conv(f_illum_feat)  # (B, C_coarse, h, w)

        # v4.3: hybrid estimation - L_ratio anchor + feature-space delta
        lit_up_delta = self.lit_up_proj(f_illum_feat)              # 64ch -> 3ch
        lit_up_feat = L_ratio + lit_up_delta                        # physical anchor + feature correction
        lit_up_map_raw = 1.0 + self.max_bright * torch.sigmoid(lit_up_feat)  # bounded [1, 1+max_bright]

        # v5.9.2: 中间监督输出 — 把 f_illum_feat 投影为图像, 供增强损失用
        ifpn_side = self.side_head(f_illum_feat)

        return {
            "lit_up_map_raw": lit_up_map_raw,
            "f_illum_feat":   f_illum_feat,
            "A_illu":         A_illu,         # Delta: 从 IGRF 移入
            "L_t":            L_t,
            "L_ref":          L_ref,
            "L_ratio":        L_ratio,
            "ifpn_side":      ifpn_side,
        }



# ============================================================
# NDPN: Noise-Denoising Pyramid Network
# ============================================================


class NDPN(nn.Module):
    """
    Args:
        channels       : 特征通道数 (默认 64)
        tau_mid_init   : SNR 归一化中心初始值
        tau_scale_init : SNR 归一化尺度初始值
    """

    def __init__(
        self,
        channels: int = 64,
        tau_mid_init: float = 1.0,
        tau_scale_init: float = 1.0,
        num_frames: int = 5,
    ):
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames

        self.tau_mid = nn.Parameter(torch.tensor(tau_mid_init))
        self.log_tau_scale = nn.Parameter(torch.tensor(float(torch.log(torch.tensor(tau_scale_init)))))

        self.alpha_conv = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, 1, kernel_size=1, bias=True),
        )
        nn.init.zeros_(self.alpha_conv[-1].weight)
        nn.init.zeros_(self.alpha_conv[-1].bias)

        self.refine = nn.Sequential(
            ConvBlock(channels, channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(channels, channels, kernel_size=1, stride=1, padding=0, act=False),
        )
        self.noise_proj = nn.Conv2d(1, channels, 1, 1, 0)
        nn.init.zeros_(self.noise_proj.weight)
        nn.init.zeros_(self.noise_proj.bias)

        # Delta: correspondence confidence projector (from C_omega diagonals)
        self.conf_proj = nn.Sequential(
            nn.Linear(num_frames - 1, channels // 4),
            nn.GELU(),
            nn.Linear(channels // 4, 1),
            nn.Sigmoid(),
        )

    def forward(
        self,
        feats: torch.Tensor,
        F_aligned_list: List[torch.Tensor],
        mu_t_clean: torch.Tensor,
        sigma_t_clean: torch.Tensor,
        s_noise: torch.Tensor,
        center_idx: int,
        C_omega_list: list = None,
        F_t_aligned: torch.Tensor = None,
    ) -> Dict[str, torch.Tensor]:
        B, T, C, H, W = feats.shape
        assert len(F_aligned_list) == T
        assert C == self.channels

        # Step 1: SNR 估计
        eps = 1e-6
        signal = mu_t_clean.abs().mean(dim=1, keepdim=True)
        noise = sigma_t_clean.mean(dim=1, keepdim=True)
        snr_hat = signal / (noise + eps)
        tau_scale = torch.exp(self.log_tau_scale).clamp(min=1e-2)
        s_snr = torch.sigmoid((snr_hat - self.tau_mid) / tau_scale)

        # Delta: correspondence confidence from C_omega_list
        conf_map = None
        if C_omega_list is not None and len(C_omega_list) > 0:
            diag_scores = []
            for C_t in C_omega_list:
                diag = C_t.diagonal(dim1=-2, dim2=-1)
                diag_scores.append(diag)
            diag_stack = torch.stack(diag_scores, dim=-1)  # (B, N, T-1)
            conf_raw = self.conf_proj(diag_stack).squeeze(-1)  # (B, N)
            ds = int(conf_raw.shape[-1] ** 0.5)
            conf_map = conf_raw.reshape(B, 1, ds, ds)
            conf_map = F.interpolate(conf_map, size=(H, W), mode='bilinear',
                                     align_corners=False)

        # Step 2: 各帧权重
        F_t = F_t_aligned if F_t_aligned is not None else feats[:, center_idx]
        alphas: List[torch.Tensor] = []

        for i in range(T):
            F_i_aligned = F_aligned_list[i]
            if i == center_idx:
                alpha_i = s_snr
            else:
                resid = (F_i_aligned - F_t).abs()
                alpha_raw = torch.sigmoid(self.alpha_conv(resid))
                alpha_i = alpha_raw * (1.0 - s_snr)
            alphas.append(alpha_i)

        # Step 3: 归一化加权聚合
        alpha_sum = torch.stack(alphas, dim=1).sum(dim=1) + eps

        F_denoised = torch.zeros_like(F_t)
        for i in range(T):
            w_i = alphas[i] / alpha_sum
            F_denoised = F_denoised + w_i * F_aligned_list[i]

        F_denoised = self.refine(F_denoised)

        # Delta: correspondence confidence modulates denoising
        if conf_map is not None:
            F_denoised = F_denoised * (0.5 + 0.5 * conf_map)

        # s_noise 条件调制
        noise_cond = self.noise_proj(s_noise)
        f_noise_out = F_denoised + noise_cond

        return {
            "f_noise_out": f_noise_out,
            "s_snr":       s_snr,
            "snr_hat":     snr_hat,
        }



# ============================================================
# MRPN: Motion-Refining Pyramid Network
# ============================================================


class MRPN(nn.Module):
    def __init__(self, channels=64, window_size=8, num_frames=5):
        super().__init__()
        self.channels = channels
        self.window_size = window_size
        self.num_frames = num_frames
        self.gate = nn.Conv2d(channels * 2, channels, 1, 1, 0)
        self.refine = ResBlock(channels)

        self.blur_estimator = nn.Sequential(
            nn.Conv2d(channels + 1, channels // 4, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 3, 1, 1),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.blur_estimator[-2].weight)
        nn.init.zeros_(self.blur_estimator[-2].bias)

        # Delta: motion magnitude estimator from C_omega diagonals
        self.motion_estimator = nn.Sequential(
            nn.Conv2d(num_frames - 1, channels // 4, 1),
            nn.GELU(),
            nn.Conv2d(channels // 4, 1, 1),
            nn.Sigmoid(),
        )

    def _aggregate_neighbors(self, f_t, f_omega):
        """窗口 dot-product 相关聚合相邻帧（不含中心帧）。"""
        b, t, c, h, w = f_omega.shape
        feat = f_omega.reshape(b * t, c, h, w)
        feat, pad_hw = pad_to_window(feat, self.window_size)
        hp, wp = feat.shape[-2:]
        feat = feat.reshape(b, t, c, hp, wp)

        f_t_padded, _ = pad_to_window(f_t, self.window_size)

        center_windows = window_partition_2d(f_t_padded, self.window_size)
        feat_windows = window_partition_video(feat, self.window_size)

        corr = torch.matmul(center_windows, feat_windows.transpose(-1, -2)) / math.sqrt(c)
        corr = torch.softmax(corr, dim=-1)

        aligned_windows = torch.matmul(corr, feat_windows)
        aligned = window_reverse_2d(aligned_windows, self.window_size, hp, wp)
        aligned = unpad_from_window(aligned, pad_hw)
        return aligned

    def forward(self, F_aligned_list, center_idx, sigma_t_clean=None,
                C_omega_list=None, F_t_aligned=None):
        f_t_aligned = F_aligned_list[center_idx]
        f_neighbors = torch.stack(
            [F_aligned_list[i] for i in range(len(F_aligned_list)) if i != center_idx],
            dim=1,
        )

        f_omega_aligned = self._aggregate_neighbors(f_t_aligned, f_neighbors)

        # Delta: motion magnitude from C_omega_list (full motion_estimator)
        motion_mag = None
        if C_omega_list is not None and len(C_omega_list) > 0:
            diag_vals = []
            for C_t in C_omega_list:
                diag = C_t.diagonal(dim1=-2, dim2=-1)  # (B, N)
                diag_vals.append(diag)
            diag_stack = torch.stack(diag_vals, dim=1)  # (B, T-1, N)
            B_val = diag_stack.shape[0]
            ds = int(diag_stack.shape[-1] ** 0.5)
            H_ref, W_ref = f_t_aligned.shape[-2:]
            diag_2d = diag_stack.reshape(B_val, len(C_omega_list), ds, ds)
            motion_mag = self.motion_estimator(diag_2d)  # (B, 1, ds, ds)
            motion_mag = F.interpolate(motion_mag, size=(H_ref, W_ref),
                                       mode='bilinear', align_corners=False)

        # Delta: use F_t_aligned as alignment reference
        if F_t_aligned is not None:
            f_t_aligned = F_t_aligned

        # blur_mask gate
        blur_mask = None
        if sigma_t_clean is not None:
            sigma_1ch = sigma_t_clean.mean(dim=1, keepdim=True)
            frame_diff = (f_omega_aligned - f_t_aligned).abs()
            blur_input = torch.cat([sigma_1ch, frame_diff], dim=1)
            blur_mask = self.blur_estimator(blur_input)

        z_t = torch.cat([f_t_aligned, f_omega_aligned], dim=1)
        g_t = torch.sigmoid(self.gate(z_t))

        # Delta: motion magnitude modulates gate
        if blur_mask is not None:
            g_t = g_t * (1.0 - blur_mask) + blur_mask * 0.3
        if motion_mag is not None:
            g_t = g_t * (1.0 - motion_mag) + motion_mag * 0.3

        f_t_fuse = g_t * f_t_aligned + (1.0 - g_t) * f_omega_aligned
        hat_f_t = self.refine(f_t_fuse) + f_t_aligned

        return {
            "f_omega_aligned": f_omega_aligned,
            "z_t": z_t,
            "G_t": g_t,
            "f_t_fuse": f_t_fuse,
            "f_motion_out": hat_f_t,
        }


# ============================================================
# CrossFusionGate — NDPN/MRPN 输出端交叉门控
# ============================================================


class CrossFusionGate(nn.Module):
    """Charlie3 P1 + Delta: NDPN/MRPN 交叉门控 (结构性重参数化)

    训练时: 动态交叉门控协调梯度冲突
    推理时: 融合为静态逐通道缩放 (deploy=True)
    参考 DRNet (CVPR 2026) DRMLP 重参数化范式
    """

    def __init__(self, channels: int, deploy: bool = False):
        super().__init__()
        self.deploy = deploy
        self.channels = channels

        if not deploy:
            self.gate_noise = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )
            self.gate_motion = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Conv2d(channels, channels // 4, 1),
                nn.ReLU(inplace=True),
                nn.Conv2d(channels // 4, channels, 1),
                nn.Sigmoid(),
            )
        else:
            self.scale_noise = nn.Parameter(torch.ones(1, channels, 1, 1))
            self.scale_motion = nn.Parameter(torch.ones(1, channels, 1, 1))

    def forward(self, f_noise: torch.Tensor, f_motion: torch.Tensor):
        if self.deploy:
            return f_noise * self.scale_noise, f_motion * self.scale_motion
        g_n = self.gate_noise(f_motion)
        g_m = self.gate_motion(f_noise)
        return f_noise * g_n, f_motion * g_m

    def get_deploy(self, avg_gate_n: torch.Tensor, avg_gate_m: torch.Tensor):
        """训练结束后调用: 用统计平均门控权重创建推理融合模块"""
        deploy_mod = CrossFusionGate(self.channels, deploy=True)
        deploy_mod.scale_noise.data = avg_gate_n.view(1, -1, 1, 1)
        deploy_mod.scale_motion.data = avg_gate_m.view(1, -1, 1, 1)
        return deploy_mod



class TFSNet(nn.Module):
    """
    Args:
        in_channels    : 输入图像通道 (默认 3)
        level_channels : 编码器三尺度通道 (默认 (32, 64, 96))
        fused_channels : 编码器融合输出通道 (默认 48)
        eps            : TFSI 数值稳定项
        n_groups       : SACE 可变形分组数
        kernel_size    : SACE 可变形核大小
        share_lff      : SACE 是否与 TFSI 共享 LFF (默认 True)
    """

    def __init__(
        self,
        in_channels: int = 3,
        level_channels: Tuple[int, ...] = (32, 64, 96),
        fused_channels: int = 64,
        eps: float = 1e-6,
        n_groups: int = 4,
        kernel_size: int = 3,
        share_lff: bool = True,
        sace_phase_preserving: bool = True,
        use_soft_clamp: bool = True,
        sace_offset_use_norm: bool = True,
        sace_offset_kaiming_init: bool = True,
        use_soft_median: bool = True,
        use_cross_rwkv: bool = False,
        use_dwt_lff: bool = False,
        use_pure_rwkv: bool = False,
        use_nafblock: bool = False,
        num_bottleneck_blocks: int = 0,
        num_igrf_res_blocks: int = 2,
        use_amp_enhance: bool = False,
        charlie_mode: bool = False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.fused_channels = fused_channels
        self.share_lff = share_lff
        # v5.5: 默认 3 级编码器, coarse_channels=96
        coarse_channels = level_channels[-1]  # 最粗层通道数

        # v6 Charlie: 数据流路径重改 (Charlie-plan P0)
        self.charlie_mode = charlie_mode

        # v5.9: AmpEnhance — 图像级频域幅度增强 (Encoder 前处理)
        self.use_amp_enhance = use_amp_enhance
        if use_amp_enhance:
            self.amp_enhance = AmpEnhance(in_channels=in_channels, hidden=16, min_amps=0.1)
        else:
            self.amp_enhance = None

        # Stage 0: PyramidEncoder
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
            num_bottleneck_blocks=num_bottleneck_blocks,
        )

        # Charlie2 D2: IFPN 图像输入 → encoder 特征投影 (64→3)
        self.feat_to_img = nn.Conv2d(fused_channels, in_channels, 3, 1, 1)

        # v6 Bravo: DWT-LFF — TFSI 用 center (α=0.6 锚定), SACE 自建双实例
        self.use_dwt_lff = use_dwt_lff
        if use_dwt_lff:
            dwt_lff_tfsi = SpatialDWTLFFAdapter(in_channels=fused_channels, alpha_init=0.6)
        else:
            dwt_lff_tfsi = None

        # Stage 1: TFSI
        self.tfsi = TFSI(
            channels=fused_channels,
            fused_channels=fused_channels,
            eps=eps,
            use_soft_median=use_soft_median,
            dwt_lff=dwt_lff_tfsi,
        )

        # Stage 2: SACE / PureRWKVSACE
        if use_dwt_lff and not use_pure_rwkv:
            shared_lff = dwt_lff_tfsi
        else:
            shared_lff = self.tfsi.freq_branch.lff if share_lff else None

        if use_pure_rwkv:
            # v6.5: 纯 RWKV (内部自建双 DWT-LFF 实例 + Bravo V raw)
            self.sace = PureRWKVSACE(channels=fused_channels, n_layer=1)
        else:
            self.sace = SACE(
                channels=fused_channels,
                n_groups=n_groups,
                kernel_size=kernel_size,
                use_optimized=True,
                lff_module=shared_lff,
                phase_preserving=sace_phase_preserving,
                offset_use_norm=sace_offset_use_norm,
                offset_kaiming_init=sace_offset_kaiming_init,
                use_cross_rwkv=use_cross_rwkv,
            )

        # Stage 3: 三源恢复分支
        self.ifpn = IFPN(
            fused_channels=fused_channels,
            coarse_channels=coarse_channels,
            img_channels=in_channels,
        )
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # Charlie3 P1: NDPN/MRPN 输出端交叉门控 (VSRELL cross-modulation 风格)
        self.cross_fuse = CrossFusionGate(channels=fused_channels)

        # Stage 4: IGRF
        self.igrf = IGRF(channels=fused_channels, out_channels=in_channels,
                         use_soft_clamp=use_soft_clamp, use_nafblock=use_nafblock,
                         num_res_blocks=num_igrf_res_blocks)

        # 逐帧特征缓存 (推理时滑动窗口复用)
        self.frame_cache: Dict[int, Dict[str, torch.Tensor]] = {}

    def clear_frame_cache(self):
        """清空逐帧特征缓存（切换序列或释放显存时调用）。"""
        self.frame_cache.clear()

    def forward(
        self,
        x: torch.Tensor,
        frame_indices: Optional[List[int]] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, T, 3, H, W) 多帧低光输入

        Returns:
            dict with keys: res_t, delta, f_fused_igrf, s_illum, s_noise, ...
        """
        B, T, C_in, H, W = x.shape
        center_idx = T // 2

        # v5.9: AmpEnhance — 图像级频域幅度增强 (Encoder 前处理)
        # 全帧共用 center frame 的 curve_amps，保证 SACE 对齐一致性
        if self.amp_enhance is not None:
            # 对 center frame 估计 curve_amps
            img_center = x[:, center_idx]  # (B, 3, H, W)
            with torch.no_grad():
                curve_amps = self.amp_enhance.amp_net(img_center).clamp(min=0.1, max=1.0)
            # 对每帧用同一 curve_amps 做幅度增强
            x_enhanced = torch.empty_like(x)
            for i in range(T):
                y_i = x[:, i]  # (B, 3, H, W)
                F_i = torch.fft.fft2(y_i, dim=(-2, -1), norm='ortho')
                mag_i = torch.abs(F_i) / curve_amps
                pha_i = torch.angle(F_i)
                F_new = torch.polar(mag_i, pha_i)
                x_enhanced[:, i] = torch.fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real
            x = x_enhanced

        # Stage 0: 编码（支持逐帧缓存）
        feats_list: List[torch.Tensor] = []
        for i in range(T):
            gidx = frame_indices[i] if frame_indices else None
            if gidx is not None and gidx in self.frame_cache:
                feats_list.append(self.frame_cache[gidx]["feat"])
            else:
                f = self.encoder.forward_single(x[:, i], return_coarse=False)
                feats_list.append(f)
                if gidx is not None:
                    self.frame_cache[gidx] = {"feat": f}
        feats = torch.stack(feats_list, dim=1)

        # Stage 1: TFSI
        tfsi_out = self.tfsi(feats)
        F_fused  = tfsi_out["F_fused"]
        s_illum  = tfsi_out["s_illum"]
        s_noise  = tfsi_out["s_noise"]

        # Stage 2: SACE（LFF 支持逐帧缓存）
        cached_lff: Dict[int, torch.Tensor] = {}
        if frame_indices:
            for i, gidx in enumerate(frame_indices):
                if gidx in self.frame_cache and "lff" in self.frame_cache[gidx]:
                    cached_lff[i] = self.frame_cache[gidx]["lff"]

        sace_out_dict = self.sace(
            feats, tfsi_out,
            cached_lff=cached_lff if cached_lff else None,
        )

        # Delta: 新 SACE 输出接口
        sace_feats      = sace_out_dict["sace_out"]   # (B, T, C, H, W)
        C_omega_list    = sace_out_dict.get("C_omega_list", [])
        F_t_aligned     = sace_out_dict["F_t_aligned"] # (B, C, H, W)
        mu_t_clean      = sace_out_dict["mu_t_clean"]
        sigma_t_clean   = sace_out_dict["sigma_t_clean"]

        # 兼容旧接口: F_aligned_list = list of (B, C, H, W) per frame
        F_aligned_list = [sace_feats[:, t] for t in range(T)]

        # Stage 3: 三源恢复
        image_center = x[:, center_idx]

        # v5.3: IFPN 改用 SACE 对齐特征（不再使用 Encoder 粗特征）
        aligned_feats = torch.stack(F_aligned_list, dim=1)  # (B, T, C_f, H, W)

        # Charlie2 D2: IFPN 输入用 Encoder 浅层特征 (避免原始噪声泄漏)
        h_c, w_c = H // 4, W // 4
        f_center_for_ifpn = feats[:, center_idx]  # (B, 64, H, W) encoder 中心帧
        image_down = F.interpolate(
            self.feat_to_img(f_center_for_ifpn),
            size=(h_c, w_c), mode='bilinear', align_corners=False
        )
        # 邻帧图像: 对 encoder 邻帧做同样投影
        imgs_flat = feats.reshape(B * T, feats.shape[2], H, W)
        imgs_proj = self.feat_to_img(imgs_flat)
        imgs_down = F.interpolate(
            imgs_proj, size=(h_c, w_c), mode='bilinear', align_corners=False
        ).view(B, T, imgs_proj.shape[1], h_c, w_c)

        ifpn_out = self.ifpn(
            I_t_down=image_down,
            aligned_feats=aligned_feats,
            center_idx=center_idx,
            imgs_down=imgs_down,
            s_illum=s_illum,
            F_t_aligned=F_t_aligned,
        )

        ndpn_out = self.ndpn(
            feats=feats,
            F_aligned_list=F_aligned_list,
            mu_t_clean=mu_t_clean,
            sigma_t_clean=sigma_t_clean,
            s_noise=s_noise,
            center_idx=center_idx,
            C_omega_list=C_omega_list,         # Delta: correspondence confidence
            F_t_aligned=F_t_aligned,            # Delta: aligned reference
        )

        # Charlie P2: sigma_sace 调制 s_noise
        if self.charlie_mode and s_noise is not None:
            sigma_sace = sigma_t_clean.mean(dim=1, keepdim=True)
            scale_sigma = torch.sigmoid(-sigma_sace * 2.0)
            s_noise = s_noise * scale_sigma

        mrpn_out = self.mrpn(
            F_aligned_list=F_aligned_list,
            center_idx=center_idx,
            sigma_t_clean=sigma_t_clean if self.charlie_mode else None,
            C_omega_list=C_omega_list,         # Delta: motion magnitude
            F_t_aligned=F_t_aligned,            # Delta: alignment baseline
        )

        # Extract outputs for IGRF
        lit_up_map_raw = ifpn_out["lit_up_map_raw"]
        f_illum_feat   = ifpn_out["f_illum_feat"]
        A_illu         = ifpn_out.get("A_illu")        # Delta: from IFPN
        f_noise_out_raw = ndpn_out["f_noise_out"]
        f_motion_out_raw = mrpn_out["f_motion_out"]

        # Charlie3 P1: 交叉门控 — 互补置信度交换
        f_noise_out, f_motion_out = self.cross_fuse(f_noise_out_raw, f_motion_out_raw)

        # Stage 4: IGRF (Delta: A_illu 由 IFPN 传入)
        igrf_out = self.igrf(
            f_illum_feat=f_illum_feat,
            f_noise_out=f_noise_out,
            f_motion_out=f_motion_out,
            lit_up_map_raw=lit_up_map_raw,
            image_center=image_center,
            A_illu=A_illu,
            s_noise=s_noise,
        )

        return {
            "res_t":          igrf_out["res_t"],
            "img_s1":         igrf_out["img_s1"],
            "img_s2":         igrf_out["img_s2"],
            "lit_up_map":     igrf_out["lit_up_map"],
            "image_center":   image_center,
            "s_illum":        s_illum,
            "s_noise":        s_noise,
            "f_illum_feat":   ifpn_out["f_illum_feat"],
            "f_noise_out":    ndpn_out["f_noise_out"],
            "f_motion_out":   mrpn_out["f_motion_out"],
            "L_t":            ifpn_out["L_t"],
            "L_ref":          ifpn_out["L_ref"],
            "L_ratio":        ifpn_out["L_ratio"],
            "A_illu":         A_illu,                       # Delta: from IFPN
            "ifpn_side":      ifpn_out.get("ifpn_side"),
            "mu_t_clean":     mu_t_clean,
            "s_snr":          ndpn_out["s_snr"],
            "motion_weights": mrpn_out["G_t"],
            "tfsi_out":       tfsi_out,
        }



# ============================================================
# IGRF: Intensity-Guided Residual Fusion
# ============================================================

def soft_clamp(x: torch.Tensor, sharpness: float = 20.0) -> torch.Tensor:
    """v5.6 P0-4: soft clamp with non-zero gradient everywhere.

    sigmoid(sharpness * (x - 0.5)) maps R -> (0, 1), gradient = sharpness * s * (1-s) > 0.
    At x=0 or x=1, gradient is tiny but non-zero (vs hard clamp's 0 gradient).
    """
    return torch.sigmoid(sharpness * (x - 0.5))


def _make_res_blocks(channels: int, n: int, use_nafblock: bool = False):
    """创建 n 个残差块，可选 NAFBlock 或 ResBlock。"""
    Block = NAFBlock if use_nafblock else ResBlock
    return nn.Sequential(*[Block(channels) for _ in range(n)])



class StageBlock(nn.Module):
    """Single-stage restoration block: branch feature + current image + optional intensity -> delta -> restored image

    v5.5: When use_intensity=True, s_intensity (s_noise) is injected as additive correction
    to delta via intensity_corr (Conv2d 1->img_channels, zero-initialized).
    v5.6 P0-4: use_soft_clamp controls whether intermediate stages use soft clamp (default True).
    v5.6 P1-7: use_nafblock controls whether to use NAFBlock instead of ResBlock in fuse.
    """

    def __init__(self, channels: int, img_channels: int = 3, use_intensity: bool = False,
                 use_soft_clamp: bool = False, use_nafblock: bool = False,
                 num_res_blocks: int = 2):
        super().__init__()
        self.use_intensity = use_intensity
        self.use_soft_clamp = use_soft_clamp
        self.img_proj = nn.Conv2d(img_channels, channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, 1, 0),
            nn.GELU(),
            *[Block(channels) for _ in range(num_res_blocks)],
            nn.Conv2d(channels, img_channels, 3, 1, 1),
        )
        if use_intensity:
            # v5.5: s_noise direct injection — zero-init so initial delta unchanged
            self.intensity_corr = nn.Conv2d(1, img_channels, kernel_size=3, padding=1, bias=True)
            nn.init.zeros_(self.intensity_corr.weight)
            nn.init.zeros_(self.intensity_corr.bias)

    def forward(self, f_branch: torch.Tensor, img_current: torch.Tensor,
                s_intensity: torch.Tensor = None):
        img_feat = self.img_proj(img_current)
        combined = torch.cat([f_branch, img_feat], dim=1)
        delta = self.fuse(combined)
        # v5.5: additive intensity correction (zero-init -> initial behavior unchanged)
        if self.use_intensity and s_intensity is not None:
            delta = delta + self.intensity_corr(s_intensity)
        # v5.6 P0-4: soft clamp for intermediate stages (gradient non-zero in dark/bright regions)
        if self.use_soft_clamp:
            img_next = soft_clamp(img_current + delta)
        else:
            img_next = torch.clamp(img_current + delta, 0.0, 1.0)
        return img_next, delta



class BrightenStage(nn.Module):
    """
    Hybrid brightening stage (v5.5, restored from v5.7 gating experiment).

    v5.7 gating (lit_up_map = 1 + s_illum * (lit_up_map_full - 1)) solved s_illum collapse
    but reduced PSNR by 0.6 dB (constrains per-channel freedom). Reverted to v5.5 additive.

    Multiplicative base (Retinex):
        lit_up_map = lit_up_map_raw * (1 + tanh(delta) * max_delta)
        brighten_base = img_dark * lit_up_map

    Additive s_illum correction:
        corr_mag = illum_corr(f_illum_feat)           # 64->3, zero-init
        illum_residual = s_illum * corr_mag
        res_t = clamp(brighten_base + illum_residual, 0, 1)
    """

    def __init__(self, channels: int, img_channels: int = 3, max_delta: float = 0.5,
                 use_nafblock: bool = False):
        super().__init__()
        self.max_delta = max_delta
        self.feat_proj = nn.Conv2d(channels, img_channels, 3, 1, 1)
        self.img_proj = nn.Conv2d(img_channels, img_channels, 3, 1, 1)
        Block = NAFBlock if use_nafblock else ResBlock
        self.delta_refine = nn.Sequential(
            nn.Conv2d(img_channels * 2, img_channels, 1, 1, 0),
            nn.GELU(),
            Block(img_channels),
            nn.Conv2d(img_channels, img_channels, 3, 1, 1),
        )
        # Delta: unified_illu 已移除 — A_illu 改由 IFPN 生成并传入

    def forward(self, lit_up_map_raw: torch.Tensor, f_illum_feat: torch.Tensor,
                img_dark: torch.Tensor, A_illu: torch.Tensor = None):
        """Delta: A_illu 由 IFPN 传入，不再内部生成"""
        feat_cond = self.feat_proj(f_illum_feat)
        img_cond = self.img_proj(img_dark)
        delta = self.delta_refine(torch.cat([feat_cond, img_cond], dim=1))
        lit_up_map = lit_up_map_raw * (1.0 + torch.tanh(delta) * self.max_delta)
        lit_up_map = lit_up_map.clamp(min=0.5)

        if A_illu is not None:
            A_resized = F.interpolate(A_illu, size=lit_up_map.shape[-2:],
                                      mode='bilinear', align_corners=False)
            lit_up_map = lit_up_map * (1.0 + A_resized)

        res_t = torch.clamp(img_dark * lit_up_map, 0.0, 1.0)
        return res_t, lit_up_map



class IGRF(nn.Module):
    """
    IGRF v5.7 - Denoise -> Motion -> Brighten (sequential cascade)

    Stage 1 (denoise):   img_s1 = clamp(img_center + delta_noise(f_noise, img, s_noise))
                          s_noise 作为 additive correction 直接参与 delta
    Stage 2 (motion):    img_s2 = clamp(img_s1 + delta_motion(f_motion, img_s1))
    Stage 3 (brighten):  lit_up_map = 1 + s_illum * (lit_up_map_full - 1)   (v5.7 乘法门控)
                          res_t = clamp(img_s2 * lit_up_map, 0, 1)
                          s_illum 门控 lit_up_map: s_illum=0→无提亮, s_illum=1→完全提亮
                          (NO .detach(): L_recon gradient flows through to NDPN/MRPN/IFPN)
    v5.7: 移除加法修正路径, 改用乘法门控, 消除 s_illum 功能冗余
    """

    def __init__(self, channels: int = 64, out_channels: int = 3, use_soft_clamp: bool = False,
                 use_nafblock: bool = False, num_res_blocks: int = 2):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels

        self.stage_noise = StageBlock(channels, out_channels, use_intensity=True,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock,
                                       num_res_blocks=num_res_blocks)
        self.stage_motion = StageBlock(channels, out_channels, use_intensity=False,
                                       use_soft_clamp=use_soft_clamp,
                                       use_nafblock=use_nafblock,
                                       num_res_blocks=num_res_blocks)
        self.brighten = BrightenStage(channels, out_channels,
                                      use_nafblock=use_nafblock)    # hybrid brighten + s_illum (final hard clamp)

    def forward(
        self,
        f_illum_feat: torch.Tensor,
        f_noise_out: torch.Tensor,
        f_motion_out: torch.Tensor,
        lit_up_map_raw: torch.Tensor,
        image_center: torch.Tensor,
        A_illu: torch.Tensor = None,
        s_noise: torch.Tensor = None,
    ) -> dict:
        """Delta: A_illu 由 IFPN 生成传入"""
        img_s1, delta_s1 = self.stage_noise(f_noise_out, image_center, s_intensity=s_noise)
        img_s2, delta_s2 = self.stage_motion(f_motion_out, img_s1)
        res_t, lit_up_map = self.brighten(lit_up_map_raw, f_illum_feat, img_s2, A_illu=A_illu)

        return {
            "res_t":       res_t,
            "img_s1":      img_s1,
            "img_s2":      img_s2,
            "lit_up_map":  lit_up_map,
            "delta_s1":    delta_s1,
            "delta_s2":    delta_s2,
        }
