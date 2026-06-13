"""
TFSI — 时频源指示器 (Temporal-Frequency Source Indicator)
============================================================

v3 设计：替代 MINSBlock，输出三源独立强度图 [s_illum, s_noise, s_motion]。

实现状态：
  - 空间分支（时域统计量 → Conv）：✅ 已实现
  - 门控融合：✅ 已实现
  - 强度输出头（3 路 Sigmoid）：✅ 已实现
  - 频域分支（LFF 可学习频率滤波）：✅ 已实现 (B.2)

设计参考：TFSv3-result.md § 4.1
"""

import torch
import torch.nn as nn

from .blocks import ConvBlock, LayerNorm2d
from .lff import LFFFeatureAdapter


class SpatialBranch(nn.Module):
    """
    TFSI 空间分支：时域统计量 → Conv → F_s

    对多帧特征 {F_i} 沿时间维计算：
        μ_t(x,y)  = Median_i{F_i(x,y)}          # 时域中位值（结构先验）
        σ_t²(x,y) = Var_i{F_i(x,y)}             # 时域方差（噪声/运动度量）
        SNR(x,y)  = μ_t / (σ_t + ε)             # 信噪比估计

    拼接后经 3×3 Conv 得到空间特征 F_s ∈ R^{C_f × H × W}。
    """

    def __init__(self, channels: int, fused_channels: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        # 输入：[μ_t, σ_t, SNR]，共 3 通道（每通道为 C 维特征的压缩统计）
        # 注：Median/Var 对每个空间位置 (x,y) 在通道维度上分别统计，
        #     结果 shape 为 (B, C, H, W)，拼接后为 (B, 3C, H, W)
        self.conv = nn.Sequential(
            ConvBlock(channels * 3, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
            ConvBlock(fused_channels, fused_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

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
        # 时域中位值：对 T 维取 median
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
    TFSI 频域分支 — 基于 LFFFeatureAdapter 的可学习频率滤波。

    实现状态: ✅ 已接入 LFF (B.2)

    数据流:
        feats[:, center_idx]                  # (B, C, H, W)
        → optional in_proj (若 C != fused_channels)
        → LFFFeatureAdapter (频域 RBF 整形)
        → optional out_proj (若 C != fused_channels)
        → F_f                                 # (B, fused_channels, H, W)
    """

    def __init__(
        self,
        channels: int,
        fused_channels: int,
        K: int = 10,
        n_ang_freq: int = 1,
        per_channel_rbf: bool = False,
    ):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels

        if channels == fused_channels:
            self.in_proj = nn.Identity()
            self.out_proj = nn.Identity()
            lff_channels = channels
        else:
            self.in_proj = nn.Conv2d(channels, fused_channels, kernel_size=1, bias=True)
            self.out_proj = nn.Identity()
            lff_channels = fused_channels

        # 核心: LFF 频率域适配器
        self.lff = LFFFeatureAdapter(
            channels=lff_channels,
            K=K,
            n_ang_freq=n_ang_freq,
            per_channel_rbf=per_channel_rbf,
        )

    def forward(self, feats: torch.Tensor, center_idx: int) -> torch.Tensor:
        """
        Args:
            feats      : (B, T, C, H, W) 多帧归一化后的编码器特征
            center_idx : 中心帧索引

        Returns:
            F_f: (B, fused_channels, H, W) 频域整形后的特征
        """
        f_center = feats[:, center_idx]           # (B, C, H, W)
        f_center = self.in_proj(f_center)         # (B, lff_C, H, W)
        f_f = self.lff(f_center)                  # (B, lff_C, H, W)
        f_f = self.out_proj(f_f)                  # (B, fused_channels, H, W)
        return f_f


class GatedFusion(nn.Module):
    """
    TFSI 门控融合（替代 v2 的跨域注意力）

    公式：
        g = σ(Conv1x1(Concat[F_s, F_f]))
        F_fused = g ⊙ F_s + (1-g) ⊙ F_f

    参考：Restormer (CVPR 2022)、NAFNet (ECCV 2022) 的标准门控融合设计。
    """

    def __init__(self, fused_channels: int):
        super().__init__()
        # 输入：Concat[F_s, F_f]，共 2*C_f 通道
        self.gate_conv = nn.Conv2d(fused_channels * 2, fused_channels, 1, 1, 0)

    def forward(self, f_s: torch.Tensor, f_f: torch.Tensor) -> torch.Tensor:
        """
        Args:
            f_s: (B, C_f, H, W) 空间分支特征
            f_f: (B, C_f, H, W) 频域分支特征

        Returns:
            F_fused: (B, C_f, H, W) 门控融合特征
        """
        concat = torch.cat([f_s, f_f], dim=1)   # (B, 2*C_f, H, W)
        g = torch.sigmoid(self.gate_conv(concat))  # (B, C_f, H, W)
        f_fused = g * f_s + (1.0 - g) * f_f    # (B, C_f, H, W)
        return f_fused


class IntensityHead(nn.Module):
    """
    TFSI 三源独立强度输出头

    公式：
        [s_illum, s_noise, s_motion] = σ(Conv1x1(F_fused))
        每个 s_* ∈ [0,1]，独立 Sigmoid，允许多源叠加（物理正确）

    参考：TFSv3-result.md § 4.1 Step 4
    """

    def __init__(self, fused_channels: int):
        super().__init__()
        # 输出 3 通道，分别对应 illum / noise / motion 强度
        self.conv = nn.Conv2d(fused_channels, 3, 1, 1, 0)

    def forward(self, f_fused: torch.Tensor) -> dict:
        """
        Args:
            f_fused: (B, C_f, H, W) 门控融合特征

        Returns:
            dict:
                s_illum : (B, 1, H, W) 光照退化强度，∈ [0,1]
                s_noise : (B, 1, H, W) 噪声退化强度，∈ [0,1]
                s_motion: (B, 1, H, W) 运动退化强度，∈ [0,1]
        """
        raw = self.conv(f_fused)             # (B, 3, H, W)
        intensities = torch.sigmoid(raw)     # (B, 3, H, W)
        s_illum = intensities[:, 0:1]        # (B, 1, H, W)
        s_noise = intensities[:, 1:2]        # (B, 1, H, W)
        s_motion = intensities[:, 2:3]       # (B, 1, H, W)
        return {
            "s_illum": s_illum,
            "s_noise": s_noise,
            "s_motion": s_motion,
        }


class TFSI(nn.Module):
    """
    TFSI 时频源指示器：整合空间分支、频域分支（占位）、门控融合、强度输出

    数据流：
        feats (B,T,C,H,W)
            ├── SpatialBranch  → F_s (B,C_f,H,W), μ_t, σ_t, snr
            └── FrequencyBranch → F_f (B,C_f,H,W)  [当前为零张量]
                    ↓
              GatedFusion(F_s, F_f) → F_fused (B,C_f,H,W)
                    ↓
              IntensityHead(F_fused) → s_illum, s_noise, s_motion (B,1,H,W)

    freq_branch 属性可为 None（当前状态）或注入外部 LFF 模块（待实现），
    供后续 SACE 共享 LFF 时扩展。
    """

    def __init__(self, channels: int = 48, fused_channels: int = 48, eps: float = 1e-6):
        super().__init__()
        self.channels = channels
        self.fused_channels = fused_channels
        self.eps = eps

        self.norm = LayerNorm2d(channels)
        self.spatial_branch = SpatialBranch(channels, fused_channels, eps=eps)
        # 频域分支：LFF 已实现
        self.freq_branch = FrequencyBranch(
            channels=channels,
            fused_channels=fused_channels,
            K=10,
            n_ang_freq=1,
            per_channel_rbf=False,
        )
        self.gated_fusion = GatedFusion(fused_channels)
        self.intensity_head = IntensityHead(fused_channels)

    def forward(self, feats: torch.Tensor) -> dict:
        """
        Args:
            feats: (B, T, C, H, W) 多帧编码器特征（所有帧）

        Returns:
            dict:
                F_fused   : (B, C_f, H, W) 门控融合特征
                F_s       : (B, C_f, H, W) 空间分支特征（供调试/可视化）
                F_f       : (B, C_f, H, W) 频域分支特征（供调试/可视化）
                mu_t      : (B, C, H, W) 时域中位值
                sigma_t   : (B, C, H, W) 时域标准差
                snr       : (B, C, H, W) 原始 SNR 估计
                s_illum   : (B, 1, H, W) 光照强度
                s_noise   : (B, 1, H, W) 噪声强度
                s_motion  : (B, 1, H, W) 运动强度
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

        # 门控融合
        f_fused = self.gated_fusion(f_s, f_f)

        # 强度输出
        intensities = self.intensity_head(f_fused)

        return {
            "F_fused": f_fused,
            "F_s": f_s,
            "F_f": f_f,
            "mu_t": spatial_out["mu_t"],
            "sigma_t": spatial_out["sigma_t"],
            "snr": spatial_out["snr"],
            "s_illum": intensities["s_illum"],
            "s_noise": intensities["s_noise"],
            "s_motion": intensities["s_motion"],
        }
