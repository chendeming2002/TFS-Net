"""
Branch-N: Imaging Noise Denoising Branch
=========================================
处理成像噪声 (Type I+II: Read Noise + Shot Noise 合并)

设计原则:
  1. TCA 已完成时序平均，此处做残差空间去噪
  2. 使用方差图 (variance map) 引导自适应去噪强度
  3. NAFBlock 堆叠 + Channel Attention
  4. 不引入时序建模 (避免与 Branch-M 冲突)

输入: F_N (TCA 解耦的噪声分量) + var_map (帧间方差)
输出: Y_N (去噪后的图像)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class ChannelAttention(nn.Module):
    """通道注意力 (SE-like)，自适应调节各通道去噪强度"""
    
    def __init__(self, channels: int, reduction: int = 8):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False),
            nn.Sigmoid(),
        )
    
    def forward(self, x):
        w = self.fc(self.avg_pool(x))
        return x * w


class BranchN(nn.Module):
    """Branch-N: Imaging Noise Denoising
    
    Args:
        channels: 输入特征通道数 (与 TCA 输出一致, 默认128)
        num_blocks: NAFBlock 堆叠层数 (默认3)
        out_channels: 输出通道数 (默认3 for RGB)
    
    Input:
        F_N: (B, C, H, W) — TCA 解耦的噪声分量特征
        var_map: (B, 1, H, W) — 帧间方差图 (暗区噪声大)
    
    Output:
        Y_N: (B, 3, H_orig, W_orig) — 去噪后的 RGB 图像
    """
    
    def __init__(self, channels: int = 128, num_blocks: int = 3, out_channels: int = 3):
        super().__init__()
        self.channels = channels
        
        # 方差图投影 (1 → C，与 F_N 拼接)
        self.var_proj = nn.Sequential(
            nn.Conv2d(1, channels // 4, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 4, channels, 3, 1, 1, bias=True),
        )
        
        # 融合 F_N + var_map
        self.fuse = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        
        # NAFBlock 堆叠去噪
        self.denoise_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])
        
        # Channel Attention
        self.channel_attn = ChannelAttention(channels)
        
        # 上采样到原始分辨率 (H/2 → H)
        self.upsample = nn.Sequential(
            nn.Conv2d(channels, channels * 4, 1, bias=True),
            nn.PixelShuffle(2),  # C*4 → C, H → 2H
            LayerNorm2d(channels),
        )
        
        # 输出投影 (C → 3)
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, out_channels, 3, 1, 1, bias=True),
        )
    
    def forward(self, F_N: torch.Tensor, var_map: torch.Tensor = None) -> dict:
        """
        F_N: (B, C, H/2, W/2) — TCA 输出
        var_map: (B, 1, H/2, W/2) — 方差图 (与 F_N 同分辨率), 若 None 则用零图
        
        Returns dict:
            Y_N: (B, 3, H, W) — 去噪后的 RGB 图像
            sigma_map: (B, 1, H, W) — 上采样后的方差图
        """
        B, C, H_ds, W_ds = F_N.shape
        
        # 若无方差图则用零图
        if var_map is None:
            var_map = torch.zeros(B, 1, H_ds, W_ds, device=F_N.device, dtype=F_N.dtype)
        
        # 方差图投影 + 拼接
        var_feat = self.var_proj(var_map)  # (B, C, H/2, W/2)
        fused = torch.cat([F_N, var_feat], dim=1)  # (B, 2C, H/2, W/2)
        x = self.fuse(fused)  # (B, C, H/2, W/2)
        
        # NAFBlock 去噪
        x = self.denoise_blocks(x)
        
        # Channel Attention
        x = self.channel_attn(x)
        
        # 上采样到原始分辨率
        x = self.upsample(x)  # (B, C, H, W)
        
        # 输出 RGB
        Y_N = self.to_rgb(x)  # (B, 3, H, W)
        
        # 上采样方差图 (用于可视化/监督)
        sigma_map = F.interpolate(var_map, scale_factor=2, mode='bilinear', align_corners=False)
        
        return {
            "Y_N": Y_N,
            "sigma_map": sigma_map,
        }
