"""
Golf-R5 Adaptive Fusion + Dynamic Temporal Gating
===================================================
R5 核心改动 (解决 tile 边界伪影):
  - 动态时序权重门控: 基于运动残差自适应调整中心帧/时序融合权重
  - 静态区 (residual 小) → 高时序权重 (降噪)
  - 动态区/tile 边界 (residual 大) → 低时序权重 (保留中心帧)

与 R4 的区别:
  1. 新增 ω_temporal 门控 (借鉴 DWTA-Net 动态权重思想)
  2. 保留 R4 的权重网络深度 + 中心帧残差
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class AdaptiveFusion(nn.Module):
    """自适应融合 + 中心帧残差保护 + R5 动态时序门控

    Input:
        Y_N, Y_L, Y_M: (B, 3, H, W)
        X_t:           (B, 3, H, W)
    Output:
        O_t, weights, Y_fused
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64,
                 num_blocks: int = 2, temporal_alpha: float = 2.0):
        super().__init__()
        gate_in = in_channels * 4
        self.temporal_alpha = temporal_alpha

        self.weight_net = nn.Sequential(
            nn.Conv2d(gate_in, hidden_channels, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels, hidden_channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden_channels // 2, 3, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

        self.refine_blocks = nn.Sequential(*[
            NAFBlock(in_channels) for _ in range(num_blocks)
        ])

        # 中心帧残差门控 — Golf: gamma 上界从 0.5 放宽到 0.9
        self.residual_gamma = nn.Parameter(torch.zeros(1, 1, 1, 1))
        self.lowfreq_pool = nn.AvgPool2d(kernel_size=9, stride=1, padding=4)

        # R5: 动态时序门控估计器 (基于运动残差)
        # 输入: |Y_M - X_t| (运动残差, 1 通道) + 融合权重 (3 通道)
        # 输出: ω_temporal ∈ (0,1), 静态区→1 (信任时序), 动态区→0 (回退中心帧)
        self.temporal_gate = nn.Sequential(
            nn.Conv2d(in_channels + 3, 16, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(16, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        # 偏置初始化为 +1.0 → sigmoid(1+residual)/... 初始偏时序 (但非饱和)
        nn.init.zeros_(self.temporal_gate[-2].weight)
        nn.init.constant_(self.temporal_gate[-2].bias, 1.0)

    def forward(self, Y_N: torch.Tensor, Y_L: torch.Tensor,
                Y_M: torch.Tensor, X_t: torch.Tensor) -> dict:
        gate_input = torch.cat([Y_N, Y_L, Y_M, X_t], dim=1)
        weights = F.softmax(self.weight_net(gate_input), dim=1)

        w_N, w_L, w_M = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        Y_fused = w_N * Y_N + w_L * Y_L + w_M * Y_M

        Y_refined = self.refine_blocks(Y_fused)

        # R5-1: 动态时序门控
        # 运动残差: M 分支偏离中心帧的程度 (高 → 动态/边界)
        motion_residual = (Y_M - X_t).abs()
        gate_in = torch.cat([motion_residual, weights], dim=1)
        omega = self.temporal_gate(gate_in)   # (B,1,H,W) ∈ (0,1)
        # 静态区信任时序增强, 动态区回退中心帧
        O_gated = omega * Y_refined + (1.0 - omega) * X_t

        lowfreq = self.lowfreq_pool(Y_refined)
        residual = X_t - lowfreq.detach()
        gamma = 0.9 * torch.sigmoid(self.residual_gamma)  # 上界 0.9
        O_t = (O_gated + gamma * residual).clamp(0.0, 1.0)

        return {"O_t": O_t, "weights": weights, "Y_fused": Y_fused,
                "temporal_gate": omega}
