"""
Branch-L: Illumination Correction Branch
==========================================
处理光照扰动 (Type III: 时间相关 + 空间全局)

设计原则:
  1. Retinex 分解: I = R ⊗ L，从 TCA 的光照分量特征估计
  2. 空间平滑先验抑制光照噪声
  3. 可学习 Gamma 校正输出
  4. 用可学习残差补偿 Retinex 近似误差

注：Branch-L 接收 TCA 输出的单帧特征 F_L（TCA 已在特征空间融合了时序光照信息）
    不需要 per-batch 时序 EMA（EMA 状态在 batch 之间不连续无意义）
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class BranchL(nn.Module):
    """Branch-L: Illumination Correction via Retinex
    
    Args:
        channels: 输入特征通道数 (默认128, 与 tca_channels 一致)
        num_blocks: 特征细化 NAFBlock 数 (默认2)
        gamma_init: gamma 初始值 (默认2.0, 低光增强倾向大 gamma)
        out_channels: 输出通道数 (默认3)
    
    Input:
        F_L: (B, C, H/2, W/2) — TCA 解耦的光照分量特征 (已融合时序光照信息)
        X_t: (B, 3, H, W)     — 中心帧原图 (RGB, [0,1])
    
    Output dict:
        Y_L:  (B, 3, H, W) — Retinex 校正后的图像
        L_t:  (B, 1, H, W) — 光照图 (供损失监督/可视化)
        R_t:  (B, 3, H, W) — 反射图
    """
    
    def __init__(self, channels: int = 128, num_blocks: int = 2,
                 gamma_init: float = 2.0, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        
        # 特征细化
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        self.refine_norm = LayerNorm2d(channels)
        
        # 光照图预测 (低分辨率)
        self.illum_head = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
        )
        # 零初始化 → 初始 L≈0.5
        nn.init.zeros_(self.illum_head[-1].weight)
        nn.init.zeros_(self.illum_head[-1].bias)
        
        # 上采样到原分辨率
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=True),
            nn.PixelShuffle(2),
            LayerNorm2d(channels),
        )
        
        # 光照图精化 (原分辨率附加修正)
        self.illum_refine = nn.Sequential(
            nn.Conv2d(channels, 1, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.illum_refine[-1].weight)
        nn.init.zeros_(self.illum_refine[-1].bias)
        
        # 可学习 gamma 参数 (log 空间, 保证正数)
        self.log_gamma = nn.Parameter(torch.tensor(float(gamma_init)).log())
        
        # 残差补偿 (可学习修正 Retinex 近似误差)
        self.residual_conv = nn.Conv2d(channels, out_channels, 3, 1, 1, bias=True)
        nn.init.zeros_(self.residual_conv.weight)
        nn.init.zeros_(self.residual_conv.bias)
    
    def forward(self, F_L: torch.Tensor, X_t: torch.Tensor) -> dict:
        """
        F_L: (B, C, H/2, W/2)
        X_t: (B, 3, H, W) — 中心帧原图 [0,1]
        
        Returns dict: Y_L, L_t, R_t
        """
        B, C, H_ds, W_ds = F_L.shape
        H, W = X_t.shape[-2:]
        
        # 特征细化
        x = self.refine_norm(self.refine_blocks(F_L))
        
        # 光照图 (低分辨率)
        L_ds = torch.sigmoid(self.illum_head(x))  # (B, 1, H_ds, W_ds)
        
        # 上采样 + 精化光照图
        x_up = self.upsample(x)  # (B, C, H, W)
        L_res = self.illum_refine(x_up)  # (B, 1, H, W)
        L_raw = F.interpolate(L_ds, size=(H, W), mode='bilinear', align_corners=False)
        L_t = torch.sigmoid(L_raw + L_res)  # 融合低/高分辨率估计, ∈ (0,1)
        
        # Retinex 分解: R = X / L
        eps = 1e-4
        R_t = (X_t / L_t.clamp(min=eps)).clamp(max=10.0)
        
        # Gamma 校正后重建: Y = R × L^γ
        gamma = self.log_gamma.exp()
        L_gamma = L_t.pow(gamma)
        Y_L = R_t * L_gamma
        
        # 残差补偿
        Y_L = Y_L + self.residual_conv(x_up)
        Y_L = Y_L.clamp(0.0, 1.0)
        
        return {"Y_L": Y_L, "L_t": L_t, "R_t": R_t}
