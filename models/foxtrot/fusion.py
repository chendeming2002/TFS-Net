"""
Adaptive Fusion + Center-Frame Residual
========================================
三分支输出 + 中心帧的自适应融合

设计 (来自 TSD-Foxtort.md §3.6):
  1. 门控加权融合: [ω_N, ω_L, ω_M] = Softmax(Conv([Y_N, Y_L, Y_M, X_t]))
  2. Y_fused = ω_N·Y_N + ω_L·Y_L + ω_M·Y_M
  3. 中心帧残差保护: Ô_t = Y_fused + γ·(X_t - Detach(Y_fused_lowfreq))

空间自适应权重的物理意义:
  - 暗区: ω_N 大 (噪声主导)
  - 过曝/平滑区: ω_L 大 (光照主导)
  - 边界/运动区: ω_M 大 (运动主导)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class AdaptiveFusion(nn.Module):
    """自适应融合 + 中心帧残差
    
    Args:
        in_channels: 每个分支输入的通道数 (默认3 for RGB)
        hidden_channels: 中间层通道数 (默认64)
        num_blocks: 融合精化 block 数 (默认2)
        from_features: 若 True，输入是特征图 (C通道) 而非 RGB
    
    Input:
        Y_N, Y_L, Y_M: (B, 3, H, W) — 三分支输出
        X_t: (B, 3, H, W) — 中心帧原图
    
    Output:
        O_t: (B, 3, H, W) — 最终输出
        weights: (B, 3, H, W) — 三分支权重 [ω_N, ω_L, ω_M]
    """
    
    def __init__(self, in_channels: int = 3, hidden_channels: int = 64,
                 num_blocks: int = 2):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        
        # 权重生成网络
        # 输入: 3分支 * in_channels + 中心帧 in_channels
        gate_in = in_channels * 4
        self.weight_net = nn.Sequential(
            nn.Conv2d(gate_in, hidden_channels, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, 3, 1, 1, 0, bias=True),  # 3 路权重
        )
        # 初始化为均匀权重 (各分支等权)
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)
        
        # 融合后的精化网络
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(in_channels) for _ in range(num_blocks)
        ])
        
        # 中心帧残差门控 (可学习 γ，零初始化 → 初始无残差)
        self.residual_gamma = nn.Parameter(torch.zeros(1, 1, 1, 1))
        
        # 低频提取 (用于中心帧残差的低频保护)
        self.lowfreq_pool = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)
    
    def forward(self, Y_N: torch.Tensor, Y_L: torch.Tensor,
                Y_M: torch.Tensor, X_t: torch.Tensor) -> dict:
        """
        Y_N, Y_L, Y_M: (B, 3, H, W)
        X_t: (B, 3, H, W)
        
        Returns dict: O_t, weights, Y_fused
        """
        # Step 1: 权重生成
        gate_input = torch.cat([Y_N, Y_L, Y_M, X_t], dim=1)
        weight_logits = self.weight_net(gate_input)  # (B, 3, H, W)
        weights = F.softmax(weight_logits, dim=1)    # (B, 3, H, W)
        
        w_N = weights[:, 0:1]
        w_L = weights[:, 1:2]
        w_M = weights[:, 2:3]
        
        # Step 2: 加权融合
        Y_fused = w_N * Y_N + w_L * Y_L + w_M * Y_M  # (B, 3, H, W)
        
        # Step 3: 精化
        Y_refined = self.refine_blocks(Y_fused)
        
        # Step 4: 中心帧残差保护
        # Ô_t = Y_refined + γ·(X_t - Detach(lowfreq(Y_refined)))
        lowfreq = self.lowfreq_pool(Y_refined)
        residual = X_t - lowfreq.detach()  # 中心帧提供高频细节锚点
        gamma = torch.sigmoid(self.residual_gamma)  # 限制到 (0,1)
        O_t = Y_refined + gamma * residual
        O_t = O_t.clamp(0.0, 1.0)
        
        return {
            "O_t": O_t,
            "weights": weights,
            "Y_fused": Y_fused,
        }
