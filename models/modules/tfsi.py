"""
TFSI — 时频源指示器 (Temporal-Frequency Source Indicator)
============================================================

v5 设计：输出双源独立强度图 [s_illum, s_noise]，s_motion 已废弃。

实现状态：
  - 空间分支（时域统计量 → Conv）：✅ 已实现
  - 拼接融合：✅ 已实现（v5.2 从 sigmoid 门控改为 concat + Conv）
  - 强度输出头（2 路 Sigmoid）：✅ 已实现
  - 频域分支（LFF 可学习频率滤波）：✅ 已实现 (B.2)

设计参考：TFSv3-result.md § 4.1
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock, LayerNorm2d
from .lff import LFFFeatureAdapter
from .dwt_lff import SpatialDWTLFFAdapter
from .dwt_lff import SpatialDWTLFFAdapter


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


class TFDE(nn.Module):
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
