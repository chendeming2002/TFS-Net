"""
DWT-LFF Adapter — v6.2 小波-频域特征适配器 (2026-06-27)
==========================================================
基于 DWT (Haar 小波) 将特征分为低频子带和三个高频子带。

设计动机 (解决 TFSI ↔ SACE 共享 LFF 的矛盾):
  - TFSI 需要低频信息 (光照 \u03b3_t 在幅度谱低频段, 噪声 n_t 全频段含低频)
  - SACE 需要抑制低频 (光照差异是"对齐噪声"), 保留相位高频结构
  - 共享同一 LFF 实例导致矛盾折中 — s_illum 长期塌缩的深层物理原因

新设计:
  1. DWT 分解: LL (低频), LH/HL/HH (高频)
  2. LL 子带做 phase-preserving FFT 卷积 (RBF 幅度整形, 相位保留)
  3. TFSI 输出: LL 幅度 + 原始相位 (= 光照+低频噪声诊断)
  4. SACE 输出: 相位差 (∠原始−∠处理后) + 全频段 (高频原始+低频处理后)
  5. 逆 DWT: 处理后的 LL + 原始 LH/HL/HH → 空间域特征
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.fft as fft

from .lff import RadialBasisFilter


class HaarDWT2D(nn.Module):
    """2D Haar 小波变换 — 将 (B,C,H,W) 分解为 4 个子带 LL/LH/HL/HH."""

    def forward(self, x: torch.Tensor):
        B, C, H, W = x.shape
        # 行方向 L/H
        lo = (x[:, :, 0::2, :] + x[:, :, 1::2, :]) / 2.0   # L
        hi = (x[:, :, 0::2, :] - x[:, :, 1::2, :]) / 2.0   # H
        # 列方向 LL/LH/HL/HH
        LL = (lo[:, :, :, 0::2] + lo[:, :, :, 1::2]) / 2.0
        LH = (lo[:, :, :, 0::2] - lo[:, :, :, 1::2]) / 2.0
        HL = (hi[:, :, :, 0::2] + hi[:, :, :, 1::2]) / 2.0
        HH = (hi[:, :, :, 0::2] - hi[:, :, :, 1::2]) / 2.0
        return LL, LH, HL, HH  # 各 (B, C, H/2, W/2)


class HaarIDWT2D(nn.Module):
    """2D Haar 逆小波变换 — 4 个子带 LL/LH/HL/HH → (B,C,H,W)."""

    def forward(self, LL, LH, HL, HH):
        B, C, h2, w2 = LL.shape
        # 逆列方向
        lo_up = torch.zeros(B, C, h2, w2 * 2, device=LL.device, dtype=LL.dtype)
        hi_up = torch.zeros_like(lo_up)
        lo_up[:, :, :, 0::2] = LL + LH
        lo_up[:, :, :, 1::2] = LL - LH
        hi_up[:, :, :, 0::2] = HL + HH
        hi_up[:, :, :, 1::2] = HL - HH
        # 逆行方向
        x_up = torch.zeros(B, C, h2 * 2, w2 * 2, device=LL.device, dtype=LL.dtype)
        x_up[:, :, 0::2, :] = lo_up + hi_up
        x_up[:, :, 1::2, :] = lo_up - hi_up
        return x_up


class DWTLFFAdapter(nn.Module):
    """DWT-FFT 特征适配器 — 小波多频段解耦, 解决共享 LFF 矛盾。

    Args:
        channels: 特征通道数
        K: RBF 基函数个数
        n_ang_freq: 角度调制频率
    """

    def __init__(self, channels: int, K: int = 10, n_ang_freq: int = 1):
        super().__init__()
        self.channels = channels
        self.dwt = HaarDWT2D()
        self.idwt = HaarIDWT2D()

        # RBF 频域滤波器 — 作用于 LL 子带
        self.rbf = RadialBasisFilter(K=K, n_ang_freq=n_ang_freq)

        # 后处理 Conv (对 IFFT 结果)
        self.post_conv_ll = nn.Conv2d(channels, channels, 1, 1, 0)
        nn.init.eye_(self.post_conv_ll.weight.squeeze(-1).squeeze(-1))
        nn.init.zeros_(self.post_conv_ll.bias)

        # 高频融合 Conv
        self.high_fusion = nn.Conv2d(channels * 3, channels, 1, 1, 0)
        nn.init.zeros_(self.high_fusion.weight)
        nn.init.zeros_(self.high_fusion.bias)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            x_out: (B, C, H, W) IDWT 重建后的特征
            feat_tfsi: (B, C, H, W) TFSI 诊断特征 (低频幅度 + 噪声信息)
            feat_sace: (B, C, H, W) SACE 对齐特征 (相位差 + 高频结构)
        """
        B, C, H, W = x.shape
        device, dtype = x.device, x.dtype

        # --- Step 1: DWT 分解 ---
        LL, LH, HL, HH = self.dwt(x)  # 各 (B, C, H/2, W/2)
        h2, w2 = LL.shape[-2:]

        # --- Step 2: LL 子带 phase-preserving FFT ---
        F = fft.fft2(LL, dim=(-2, -1), norm='ortho')
        mag, pha = torch.abs(F), torch.angle(F)

        # RBF 幅度整形 (使用共享 RBF, 幅度做乘性残差)
        diff_mag, diff_phase = self.rbf(h2, w2, device, dtype)
        diff_mag = diff_mag.unsqueeze(0)  # (1, 1, h2, w2)

        # 幅度: 仅低频幅度被 RBF 调整 (= 光照幅度归一化)
        mag_processed = mag * (1.0 + diff_mag)
        # 相位: 保留 (phase-preserving, ExpoMamba 物理: 光照=幅度, 结构=相位)
        pha_processed = pha

        # IFFT 重建处理后的 LL
        F_new = torch.polar(mag_processed, pha_processed)
        LL_processed = fft.ifft2(F_new, dim=(-2, -1), norm='ortho').real
        LL_processed = self.post_conv_ll(LL_processed)

        # --- Step 3: TFSI 诊断特征 (低频幅度 + 原始相位 = 光照+低频噪声) ---
        # 用处理后的低频幅度 + 原始相位重建 → 包含光照回归 + 原始噪声相位指纹
        mag_tfsi = mag_processed.detach()  # 不向 TFSI 传梯度 (SACE 领域独立)
        F_tfsi = torch.polar(mag_tfsi, pha)
        feat_tfsi_low = fft.ifft2(F_tfsi, dim=(-2, -1), norm='ortho').real
        # 上采样 LL→H×W (最近邻)
        feat_tfsi_low = torch.nn.functional.interpolate(
            feat_tfsi_low, size=(H, W), mode='nearest')

        # --- Step 4: SACE 对齐特征 (相位差 + 高频结构) ---
        # 相位差: ∠原始 − ∠处理后 (保留相位中的对齐信息)
        pha_diff = pha - pha_processed  # (B, C, h2, w2) 复数角度差值
        F_sace_phase = torch.polar(torch.ones_like(mag), pha_diff)
        feat_sace_low_phase = fft.ifft2(F_sace_phase, dim=(-2, -1), norm='ortho').real
        feat_sace_low_phase = torch.nn.functional.interpolate(
            feat_sace_low_phase, size=(H, W), mode='nearest')

        # 高频子带上采样
        LH_up = torch.nn.functional.interpolate(LH, size=(H, W), mode='nearest')
        HL_up = torch.nn.functional.interpolate(HL, size=(H, W), mode='nearest')
        HH_up = torch.nn.functional.interpolate(HH, size=(H, W), mode='nearest')

        # 高频融合 (三个高频子带 concat + 1×1 Conv)
        high_combined = self.high_fusion(torch.cat([LH_up, HL_up, HH_up], dim=1))

        # SACE 特征 = 相位差信息 + 高频结构
        feat_sace = feat_sace_low_phase + high_combined

        # --- Step 5: 逆 DWT 重建 ---
        x_out = self.idwt(LL_processed, LH, HL, HH)

        return {
            "x_out": x_out,
            "feat_tfsi": feat_tfsi_low,
            "feat_sace": feat_sace,
        }
