"""
Branch-L: Illumination Correction Branch
==========================================
处理光照扰动 (Type III: 时间相关 + 空间全局)

设计原则:
  1. Retinex 分解: I = R ⊗ L
  2. 光照图 L_t 时序锚定 (EMA) — 避免帧间闪烁
  3. 空间平滑先验抑制光照噪声
  4. Gamma 校正输出

输入: F_L (TCA 解耦的光照分量) + X_t (中心帧原图)
输出: Y_L (光照校正后的图像), L_t (光照图), R_t (反射图)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class RetinexHead(nn.Module):
    """Retinex 分解头: 从特征估计光照图 L_t
    
    L_t = Sigmoid(Conv(F_L)) ∈ [0, 1]
    R_t = X_t / (L_t + ε)
    """
    
    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True)
        # 初始化为接近恒等 (L≈0.5 附近)
        nn.init.zeros_(self.conv2.weight)
        nn.init.constant_(self.conv2.bias, 0.0)
    
    def forward(self, f_l: torch.Tensor) -> torch.Tensor:
        """
        f_l: (B, C, H, W)
        Returns: L: (B, 1, H, W) ∈ (0, 1)
        """
        return torch.sigmoid(self.conv2(self.act(self.conv1(f_l))))


class BranchL(nn.Module):
    """Branch-L: Illumination Correction
    
    Args:
        channels: 输入特征通道数 (默认128)
        num_blocks: 特征细化 NAFBlock 数 (默认2)
        ema_alpha: EMA 锚定系数 (默认0.7，越大越信任当前帧)
        gamma: 校正指数 (默认2.0)
        out_channels: 输出通道数 (默认3)
    
    Input:
        F_L: (B, C, H/2, W/2) — TCA 解耦的光照分量
        F_L_seq: (B, T, C, H/2, W/2) — 全时序光照特征 (用于 EMA 锚定), 可选
        X_t: (B, 3, H, W) — 中心帧原图 (RGB, [0,1])
    
    Output:
        Y_L: (B, 3, H, W) — 校正后的图像
        L_t: (B, 1, H, W) — 光照图 (供融合/损失)
        R_t: (B, 3, H, W) — 反射图
    """
    
    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 ema_alpha: float = 0.7, gamma: float = 2.0, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        self.ema_alpha = ema_alpha
        self.gamma = gamma
        
        # 特征细化
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.refine_norm = LayerNorm2d(channels)
        
        # Retinex 头
        self.retinex_head = RetinexHead(channels)
        
        # 上采样到原分辨率
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=True),
            nn.PixelShuffle(2),
            LayerNorm2d(channels),
        )
        
        # 光照图精化 (在原分辨率)
        self.illum_refine = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
        )
        
        # 反射图残差补偿 (可选)
        self.residual_conv = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)
    
    def _ema_anchor(self, L_center: torch.Tensor, 
                    F_L_seq: torch.Tensor) -> torch.Tensor:
        """时序锚定: L_t ← α·L_t + (1-α)·EMA(L_neighbors)
        
        L_center: (B, 1, H, W) — 中心帧光照
        F_L_seq: (B, T, C, H/2, W/2) — 全时序光照特征
        
        Returns: (B, 1, H, W) — 锚定后的光照图
        """
        B, T, C, H_ds, W_ds = F_L_seq.shape
        center_idx = T // 2
        
        # 计算各帧光照图 (低分辨率)
        L_seq = []
        for t in range(T):
            L_t = torch.sigmoid(
                self.retinex_head.conv2(self.retinex_head.act(
                    self.retinex_head.conv1(F_L_seq[:, t])))
            )  # (B, 1, H_ds, W_ds)
            L_seq.append(L_t)
        L_seq = torch.stack(L_seq, dim=1)  # (B, T, 1, H_ds, W_ds)
        
        # EMA 跨时序
        ema = L_seq[:, 0]
        ema_list = [ema]
        for t in range(1, T):
            ema = self.ema_alpha * L_seq[:, t] + (1 - self.ema_alpha) * ema
            ema_list.append(ema)
        L_anchor_ds = ema_list[center_idx]  # (B, 1, H_ds, W_ds)
        
        # 上采样锚定光照到原分辨率
        L_anchor = F.interpolate(L_anchor_ds, size=L_center.shape[-2:],
                                 mode='bilinear', align_corners=False)
        # 锚定融合
        alpha = self.ema_alpha
        L_anchored = alpha * L_center + (1 - alpha) * L_anchor
        return L_anchored
    
    def forward(self, F_L: torch.Tensor, X_t: torch.Tensor,
                F_L_seq: torch.Tensor = None) -> dict:
        """
        F_L: (B, C, H/2, W/2)
        X_t: (B, 3, H, W) — 中心帧原图 [0,1]
        F_L_seq: (B, T, C, H/2, W/2) 可选，用于时序 EMA 锚定
        
        Returns dict: Y_L, L_t, R_t
        """
        B, C, H_ds, W_ds = F_L.shape
        H, W = X_t.shape[-2:]
        
        # 特征细化
        x = self.refine_blocks(F_L)
        x = self.refine_norm(x)
        
        # 光照图估计 (低分辨率)
        L_ds = self.retinex_head(x)  # (B, 1, H_ds, W_ds)
        
        # 上采样特征到原分辨率并精化光照图
        x_up = self.upsample(x)  # (B, C, H, W)
        L_res = self.illum_refine(x_up)  # (B, 1, H, W)
        L_raw = torch.sigmoid(F.interpolate(
            L_ds, size=(H, W), mode='bilinear', align_corners=False) + L_res)
        
        # 时序锚定 (若提供时序特征)
        if F_L_seq is not None:
            L_t = self._ema_anchor(L_raw, F_L_seq)
        else:
            L_t = L_raw
        
        # Retinex 分解
        eps = 1e-4
        L_t_clamped = L_t.clamp(min=eps, max=1.0)
        R_t = X_t / L_t_clamped  # 反射图
        R_t = R_t.clamp(max=10.0)  # 防止爆炸
        
        # Gamma 校正
        L_gamma = L_t_clamped.pow(self.gamma)
        Y_L = R_t * L_gamma
        
        # 残差补偿 (网络可学习的修正)
        residual = self.residual_conv(x_up)
        Y_L = Y_L + residual
        Y_L = Y_L.clamp(0.0, 1.0)
        
        return {
            "Y_L": Y_L,
            "L_t": L_t,
            "R_t": R_t,
        }
