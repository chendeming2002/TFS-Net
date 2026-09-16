"""
Golf Adaptive Fusion + Center-Frame Residual
=============================================
与 Foxtrot 的区别:
  1. 权重网络加深 (3层), 支持更精细的空间自适应
  2. 输出后接 resize-conv 精化 (无棋盘格)
  3. 中心帧残差 gamma 上界放宽 (Foxtrot 的 sigmoid 限制在 0.5)

三分支融合 + 中心帧锚定
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d


class AdaptiveFusion(nn.Module):
    """自适应融合 + 中心帧残差保护

    Input:
        Y_N, Y_L, Y_M: (B, 3, H, W)
        X_t:           (B, 3, H, W)
    Output:
        O_t, weights, Y_fused
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 64,
                 num_blocks: int = 2):
        super().__init__()
        gate_in = in_channels * 4

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

    def forward(self, Y_N: torch.Tensor, Y_L: torch.Tensor,
                Y_M: torch.Tensor, X_t: torch.Tensor) -> dict:
        gate_input = torch.cat([Y_N, Y_L, Y_M, X_t], dim=1)
        weights = F.softmax(self.weight_net(gate_input), dim=1)

        w_N, w_L, w_M = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        Y_fused = w_N * Y_N + w_L * Y_L + w_M * Y_M

        Y_refined = self.refine_blocks(Y_fused)

        lowfreq = self.lowfreq_pool(Y_refined)
        residual = X_t - lowfreq.detach()
        gamma = 0.9 * torch.sigmoid(self.residual_gamma)  # 上界 0.9
        O_t = (Y_refined + gamma * residual).clamp(0.0, 1.0)

        return {"O_t": O_t, "weights": weights, "Y_fused": Y_fused}
