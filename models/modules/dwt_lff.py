"""
DWT-LFF Adapter — v6.3 小波-频域特征适配器 (2026-06-27)
==========================================================
基于 DWT (Haar 小波) 将特征分为低频子带和三个高频子带。

设计 (v6.3 严格版):
  低频子带 LL → Conv (可学习光照幅度提取器)
  feat_sace = DWT_HF + DWT_LF_phase + Conv(LL)_amplitude
    → SACE 对齐: 高频结构 + 低频相位(对齐信息) + 卷积提取的"光照参考幅度"
  feat_tfsi = DWT_HF + DWT_LF_phase + (LL_original - Conv(LL))_amplitude
    → TFSI 诊断: 高频 + 低频相位 + 被移除了多少光照(= 退化诊断)

物理意义:
  - Conv 学习提取"正常光照应有的幅度"→ SACE 的归一化参考
  - LL - Conv = "光照退化幅度"(γ_t 的影响)→ TFSI 诊断 s_illum/s_noise
  - 两者共享低频相位(= 结构/噪声指纹)和高频(= 纹理/运动)
  - 互补设计: SACE 拿"光照参考", TFSI 拿"光照残差"
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft


class HaarDWT2D(nn.Module):
    """2D Haar 小波变换。"""

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        lo = (x[:, :, 0::2, :] + x[:, :, 1::2, :]) / 2.0
        hi = (x[:, :, 0::2, :] - x[:, :, 1::2, :]) / 2.0
        LL = (lo[:, :, :, 0::2] + lo[:, :, :, 1::2]) / 2.0
        LH = (lo[:, :, :, 0::2] - lo[:, :, :, 1::2]) / 2.0
        HL = (hi[:, :, :, 0::2] + hi[:, :, :, 1::2]) / 2.0
        HH = (hi[:, :, :, 0::2] - hi[:, :, :, 1::2]) / 2.0
        return LL, LH, HL, HH


class HaarIDWT2D(nn.Module):
    """2D Haar 逆小波变换。"""

    def forward(self, LL, LH, HL, HH):
        B, C, h2, w2 = LL.shape
        lo_up = torch.zeros(B, C, h2, w2 * 2, device=LL.device, dtype=LL.dtype)
        hi_up = torch.zeros_like(lo_up)
        lo_up[:, :, :, 0::2] = LL + LH
        lo_up[:, :, :, 1::2] = LL - LH
        hi_up[:, :, :, 0::2] = HL + HH
        hi_up[:, :, :, 1::2] = HL - HH
        x_up = torch.zeros(B, C, h2 * 2, w2 * 2, device=LL.device, dtype=LL.dtype)
        x_up[:, :, 0::2, :] = lo_up + hi_up
        x_up[:, :, 1::2, :] = lo_up - hi_up
        return x_up


class DWTLFFAdapter(nn.Module):
    """DWT-FFT 特征适配器 — v6.3 严格版: SACE/TFSI 互补低频幅度设计。

    Args:
        channels: 特征通道数
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # 低频卷积: 提取"正常光照"的幅度参考 (3×3 conv, 保持尺寸)
        self.illum_conv = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=True),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0, bias=True),
        )

        # 高频融合 (LH/HL/HH → C)
        self.high_fusion = nn.Sequential(
            nn.Conv2d(channels * 3, channels, 1, 1, 0, bias=False),
        )
        nn.init.zeros_(self.high_fusion[0].weight)

    def forward(self, x: torch.Tensor):
        """
        Returns:
            x_out: (B, C, H, W) IDWT 重建
            feat_tfsi: (B, C, H, W) TFSI 诊断特征
            feat_sace: (B, C, H, W) SACE 对齐特征
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # --- Step 1: DWT 分解 ---
        LL, LH, HL, HH = self.dwt(x)  # 各 (B, C, H/2, W/2)

        # --- Step 2: LL 子带 FFT 提取相位 ---
        F = fft.fft2(LL, dim=(-2, -1), norm='ortho')
        mag_original = torch.abs(F)    # (B, C, H/2, W/2) — 原始幅度
        pha = torch.angle(F)           # (B, C, H/2, W/2) — 低频相位

        # --- Step 3: Conv 提取"正常光照幅度" ---
        # Conv 学习从 LL 中提取光照参考幅度
        conv_out = self.illum_conv(LL)  # (B, C, H/2, W/2) — 正常光照参考
        mag_conv = torch.abs(conv_out)  # Conv 输出幅度

        # --- Step 4: 低频相位基座 ---
        # 相位携带结构和噪声指纹, TFSI/SACE 各需
        # 用单位幅度 + 低频相位 → IFFT → 上采样
        F_phase_base = torch.polar(torch.ones_like(mag_original), pha)
        phase_base = fft.ifft2(F_phase_base, dim=(-2, -1), norm='ortho').real
        phase_base_up = torch.nn.functional.interpolate(
            phase_base, size=(H, W), mode='nearest')

        # --- Step 5: 高频子带上采样 + 融合 ---
        LH_up = torch.nn.functional.interpolate(LH, size=(H, W), mode='nearest')
        HL_up = torch.nn.functional.interpolate(HL, size=(H, W), mode='nearest')
        HH_up = torch.nn.functional.interpolate(HH, size=(H, W), mode='nearest')
        high_feat = self.high_fusion(torch.cat([LH_up, HL_up, HH_up], dim=1))  # (B, C, H, W)

        # --- Step 6: SACE 特征 = HF + LF_phase + Conv(LF)_amplitude ---
        # Conv(LF) 幅度 → IFFT(用原始相位) → 上采样
        F_sace_amp = torch.polar(mag_conv, pha)
        sace_amp = fft.ifft2(F_sace_amp, dim=(-2, -1), norm='ortho').real
        sace_amp_up = torch.nn.functional.interpolate(
            sace_amp, size=(H, W), mode='nearest')

        feat_sace = phase_base_up + high_feat + sace_amp_up  # LF_phase + HF + Conv_amp

        # --- Step 7: TFSI 特征 = HF + LF_phase + (LL_original - Conv(LL))_amplitude ---
        # 残差幅度 = 原始幅度 - Conv幅度 = 被移除的光照退化信息
        mag_residual = mag_original - mag_conv  # 光照退化残差
        F_tfsi_amp = torch.polar(mag_residual, pha)
        tfsi_amp = fft.ifft2(F_tfsi_amp, dim=(-2, -1), norm='ortho').real
        tfsi_amp_up = torch.nn.functional.interpolate(
            tfsi_amp, size=(H, W), mode='nearest')

        feat_tfsi = phase_base_up + high_feat + tfsi_amp_up  # LF_phase + HF + residual_amp

        # --- Step 8: 逆 DWT 重建 (Conv(LL) 作为低频重建) ---
        x_out = self.idwt(conv_out, LH, HL, HH)

        return {
            "x_out": x_out,
            "feat_tfsi": feat_tfsi,
            "feat_sace": feat_sace,
        }
