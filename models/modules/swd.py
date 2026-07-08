"""
SWD (Spatial Wavelet Diverter) — v6 Delta Mark1 (2026-07-01)
=============================================================
Haar DWT 子带级分流: 不做 inverse DWT，直接在子带空间完成信号分离。

  LL ──┬── alpha × LL ──→ feat_tfde (光照估计)
       └── (1-alpha) × LL + InstanceNorm ──→ feat_tca (对齐匹配)

  HF ──┬── noise_gate × HF ──→ feat_tfde (噪声能量)
       └── struct_gate × HF + LayerNorm ──→ feat_tca (结构边缘)

核心改进: 旧 DWT-LFF 通过 inverse DWT 重建全分辨率，导致两分支拿到的信息几乎相同。
新 SWD 在子带级用 alpha/noise_gate 显式分离"光照+噪声"和"光照无关结构"。
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .blocks import LayerNorm2d


class HaarDWT2D(nn.Module):
    """Haar 小波 2D 正逆变换（无参数）"""

    def forward(self, x):
        """x: (B, C, H, W) → LL, LH, HL, HH: (B, C, H/2, W/2)"""
        B, C, H, W = x.shape
        x01, x02 = x[:, :, 0::2, :], x[:, :, 1::2, :]
        L = (x01 + x02) * 0.5
        H_ = (x01 - x02) * 0.5
        L0, L1 = L[:, :, :, 0::2], L[:, :, :, 1::2]
        H0, H1 = H_[:, :, :, 0::2], H_[:, :, :, 1::2]
        LL = (L0 + L1) * 0.5
        LH = (L0 - L1) * 0.5
        HL = (H0 + H1) * 0.5
        HH = (H0 - H1) * 0.5
        return LL, LH, HL, HH

    @staticmethod
    def inverse(LL, LH, HL, HH):
        """四子带 → (B, C, H, W)"""
        B, C, H, W = LL.shape
        L0 = LL + LH
        L1 = LL - LH
        H0 = HL + HH
        H1 = HL - HH
        L = torch.empty(B, C, H, W * 2, device=LL.device)
        H_ = torch.empty(B, C, H, W * 2, device=LL.device)
        L[:, :, :, 0::2] = L0
        L[:, :, :, 1::2] = L1
        H_[:, :, :, 0::2] = H0
        H_[:, :, :, 1::2] = H1
        out = torch.empty(B, C, H * 2, W * 2, device=LL.device)
        out[:, :, 0::2, :] = L
        out[:, :, 1::2, :] = H_
        return out


class WFR(nn.Module):
    """WFR (Wavelet Feature Router): 子带级分流，不做 inverse DWT

    Args:
        channels: encoder 特征通道数
        alpha_init: 初始 LL 分配比例 (默认 0.6，偏向 TFDE)
    """

    def __init__(self, channels: int, alpha_init: float = 0.6):
        super().__init__()
        self.dwt = HaarDWT2D()
        self.channels = channels

        # === 低频分流 ===
        self.alpha_net = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, groups=channels, bias=False),
            nn.GELU(),
            nn.Conv2d(channels, channels, 1, 1, 0),
            nn.Sigmoid(),
        )
        self.ll_tca_norm = nn.InstanceNorm2d(channels, affine=True)

        # === 高频处理 (Flight3: 取消噪声门控, proj 层隐式分化) ===
        # noise_gate 和 hf_tca_norm 已移除 — 两路共享完整 HF

        # === 输出投影 (子带→统一通道) ===
        self.proj_tfde = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0),
            nn.GELU(),
            LayerNorm2d(channels),
        )
        self.proj_tca = nn.Sequential(
            nn.Conv2d(channels * 4, channels, 1, 1, 0),
            nn.GELU(),
            LayerNorm2d(channels),
        )

        self._init_alpha(alpha_init)

    def _init_alpha(self, alpha_init):
        for m in self.alpha_net.modules():
            if isinstance(m, nn.Conv2d) and m.kernel_size == (1, 1):
                nn.init.constant_(m.weight, 0.0)
                if m.bias is not None:
                    logit = math.log(alpha_init / max(1 - alpha_init, 1e-8))
                    nn.init.constant_(m.bias, logit)

    def forward(self, x: torch.Tensor):
        """
        Args:
            x: (B*T, C, H, W) encoder 特征
        Returns:
            feat_tfde: (B*T, C, H/2, W/2) — 含光照+噪声信号
            feat_tca:  (B*T, C, H/2, W/2) — 光照无关，结构清晰
            alpha: 用于可视化
            noise_gate: 用于可视化
        """
        LL, LH, HL, HH = self.dwt(x)

        # --- 低频分流 ---
        alpha = self.alpha_net(LL)
        ll_tfde = alpha * LL
        ll_tca = self.ll_tca_norm((1 - alpha) * LL)

        # --- 高频处理 (Flight3: 取消噪声门控, 两路共享完整 HF) ---
        hf_cat = torch.cat([LH, HL, HH], dim=1)

        # 各自 proj 层 (独立权重) 隐式学到不同的 HF 关注模式
        feat_tfde = self.proj_tfde(torch.cat([ll_tfde, hf_cat], dim=1))
        feat_tca = self.proj_tca(torch.cat([ll_tca, hf_cat], dim=1))

        return {
            "feat_tfde": feat_tfde,
            "feat_tca": feat_tca,
            "alpha": alpha,
        }
