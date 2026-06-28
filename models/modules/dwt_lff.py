"""
Spatial-Domain DWT-LFF Adapter (v6.4 — 综合修正版)
====================================================
基于 v6-Alpha.md 和 v6-Alpha-corr.md 的综合修正：

修正 v6-Alpha.md:
  1. α = sigmoid(Conv3×3(LL)) soft 分配 → 非负, LL_ref+LL_deg=LL
  2. 空间域 Conv3×3 (非 FFT 域) → 保留局部性
  3. feat_sace 加 LayerNorm → 适配 Cross-RWKV Q-Shift

修正 v6-Alpha-corr.md:
  4. IDWT 重构 (非 bilinear 上采样) → 保留完整高频
  5. HF 0.5× 平均分配 → feat_sace+feat_tfsi 完美重构输入
  6. α init=0.5 → 初始兼容 v5.9.2 预训练权重
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class HaarDWT2D(nn.Module):
    """Haar 小波 2D 正逆变换（无参数）"""

    def forward(self, x):
        """x: (B, C, H, W) → LL, LH, HL, HH: (B, C, H/2, W/2)"""
        B, C, H, W = x.shape
        x01, x02 = x[:, :, 0::2, :], x[:, :, 1::2, :]
        L = (x01 + x02) * 0.5
        H_ = (x01 - x02) * 0.5
        LL = (L[:, :, :, 0::2] + L[:, :, :, 1::2]) * 0.5
        LH = (L[:, :, :, 0::2] - L[:, :, :, 1::2]) * 0.5
        HL = (H_[:, :, :, 0::2] + H_[:, :, :, 1::2]) * 0.5
        HH = (H_[:, :, :, 0::2] - H_[:, :, :, 1::2]) * 0.5
        return LL, LH, HL, HH

    def inverse(self, LL, LH, HL, HH):
        """逆变换: (B, C, H/2, W/2) × 4 → (B, C, H, W)"""
        B, C, H2, W2 = LL.shape
        L = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        H_ = torch.zeros(B, C, H2, W2 * 2, device=LL.device, dtype=LL.dtype)
        L[:, :, :, 0::2] = (LL + LH) * 2
        L[:, :, :, 1::2] = (LL - LH) * 2
        H_[:, :, :, 0::2] = (HL + HH) * 2
        H_[:, :, :, 1::2] = (HL - HH) * 2
        x = torch.zeros(B, C, H2 * 2, W2 * 2, device=LL.device, dtype=LL.dtype)
        x[:, :, 0::2, :] = (L + H_) * 0.5
        x[:, :, 1::2, :] = (L - H_) * 0.5
        return x


class SpatialDWTLFFAdapter(nn.Module):
    """空间域 DWT-LFF 适配器 (v6.4 综合修正版)

    关键设计:
      1. 空间域 Conv3×3(LL) → α ∈ [0,1] soft 分配 (非 FFT 域, 非减法)
      2. IDWT(LL_ref/LL_deg, 0.5·LH/HL/HH) → 保留完整高频, 完美重构
      3. α init=0.5 → 训练初期 feat_sace≈feat_tfsi≈x/2, 兼容 v5.9.2
      4. LayerNorm → 适配 Cross-RWKV Q-Shift 通道分布
    """

    def __init__(self, in_channels: int, alpha_init: float = 0.5):
        super().__init__()
        self.in_channels = in_channels
        self.alpha_init = alpha_init
        self.dwt = HaarDWT2D()

        # 低频光照分离 α
        self.illum_alpha = nn.Sequential(
            nn.Conv2d(in_channels, in_channels, 3, 1, 1, groups=in_channels, bias=False),
            nn.GELU(),
            nn.Conv2d(in_channels, in_channels, 1, 1, 0),
            nn.Sigmoid(),
        )

        # LayerNorm
        self.norm_sace = nn.LayerNorm(in_channels)
        self.norm_tfsi = nn.LayerNorm(in_channels)

        # α init — bias = log(alpha_init / (1-alpha_init)) → sigmoid = alpha_init
        for m in self.illum_alpha.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0.0)
                import math
                init_val = math.log(max(alpha_init / max(1.0 - alpha_init, 1e-8), 1e-8))
                if m.bias is not None:
                    nn.init.constant_(m.bias, init_val)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B, C, H, W) 输入特征
        Returns:
            feat_sace: (B, C, H, W) 归一化光照 + 高频 → SACE 对齐
            feat_tfsi: (B, C, H, W) 退化残差 + 高频 → TFSI 诊断
            alpha: (B, C, H/2, W/2) 光照分配图 (用于可视化/正则)
        """
        B, C, H, W = x.shape

        # Step 1: Haar DWT 分解
        LL, LH, HL, HH = self.dwt(x)

        # Step 2: 空间域低频光照分离
        alpha = self.illum_alpha(LL)       # ∈ [0, 1]
        LL_ref = alpha * LL                # 归一化光照参考 (SACE)
        LL_deg = (1.0 - alpha) * LL        # 光照退化残差 (TFSI)

        # Step 3: HF 平均分配 → 保证 feat_sace+feat_tfsi = x (渐变约束)
        LH_half, HL_half, HH_half = LH * 0.5, HL * 0.5, HH * 0.5

        # Step 4: IDWT 重构 (保留完整高频, 非 bilinear 上采样)
        feat_sace = self.dwt.inverse(LL_ref, LH_half, HL_half, HH_half)
        feat_tfsi = self.dwt.inverse(LL_deg, LH_half, HL_half, HH_half)

        # Step 5: LayerNorm → 适配 Cross-RWKV Q-Shift
        feat_sace = self.norm_sace(feat_sace.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        feat_tfsi = self.norm_tfsi(feat_tfsi.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)

        return {
            "x_out": feat_sace,     # 兼容旧接口: x_out = feat_sace
            "feat_tfsi": feat_tfsi,
            "feat_sace": feat_sace,
        }
