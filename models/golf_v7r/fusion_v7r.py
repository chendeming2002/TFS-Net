#!/usr/bin/env python3
"""
v7r 专用三分支融合 (替代 AdaptiveFusion 在 RGB 上跑 NAFBlock 的做法)

问题背景 (docs/v7/02 §10.2b):
  原 AdaptiveFusion 在 3 通道 RGB 上先做 NAFBlock 精化, 再叠加中心帧残差。
  消融显示:
    - Y_fused (分支加权)              18.94 dB
    - 经 NAFBlock 精化后             14.20 dB   <-- 毁掉图像 (-4.75 dB)
    - 再叠加中心帧残差               18.58 dB
  NAFBlock 是特征级模块, 直接作用于 3 通道 RGB 会破坏图像。

本模块的修正:
  1. 不在 RGB 上跑 NAFBlock; 只用轻量 3x3 卷积做局部精化 (恒等初始化)
  2. 融合权重末层零初始化 -> 初始输出为三分支均值 (稳定起点)
  3. 中心帧残差门控从 0 起 (tanh(0)=0), 由训练自行决定是否需要

接口与 AdaptiveFusion 完全一致, 便于直接替换:
  forward(Y_N, Y_L, Y_M, X_t) -> {"O_t", "weights", "Y_fused"}
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class V7RFusion(nn.Module):
    """v7r 三分支融合 + 可选中心帧残差 (不破坏图像)"""

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32,
                 num_blocks: int = 2):
        super().__init__()
        gate_in = in_channels * 4  # Y_N, Y_L, Y_M, X_t

        # 逐像素自适应权重网络
        layers = []
        c_in = gate_in
        for i in range(max(num_blocks, 1)):
            layers += [
                nn.Conv2d(c_in, hidden_channels, 3, 1, 1, bias=True),
                nn.GELU(),
            ]
            c_in = hidden_channels
        layers += [nn.Conv2d(hidden_channels, in_channels, 1, 1, 0, bias=True)]
        self.weight_net = nn.Sequential(*layers)
        # 零初始化末层 -> 初始权重均匀 (softmax 后 1/3)
        nn.init.zeros_(self.weight_net[-1].weight)
        nn.init.zeros_(self.weight_net[-1].bias)

        # 轻量局部精化 (恒等初始化, 起点为纯融合结果)
        self.refine = nn.Conv2d(in_channels, in_channels, 3, 1, 1, bias=False)
        self._init_identity_(self.refine, in_channels)

        # 中心帧残差门控: 从 0 起 (tanh)
        self.residual_gamma = nn.Parameter(torch.zeros(1))

    @staticmethod
    def _init_identity_(conv: nn.Conv2d, channels: int):
        with torch.no_grad():
            conv.weight.zero_()
            for c in range(channels):
                conv.weight[c, c, conv.kernel_size[0] // 2, conv.kernel_size[1] // 2] = 1.0

    def forward(self, Y_N: torch.Tensor, Y_L: torch.Tensor,
                Y_M: torch.Tensor, X_t: torch.Tensor) -> Dict[str, torch.Tensor]:
        gate_input = torch.cat([Y_N, Y_L, Y_M, X_t], dim=1)
        weights = F.softmax(self.weight_net(gate_input), dim=1)

        w_N, w_L, w_M = weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]
        Y_fused = w_N * Y_N + w_L * Y_L + w_M * Y_M

        # 轻量精化 (起点恒等, 不会破坏)
        Y_refined = self.refine(Y_fused)

        # 可选中心帧残差 (gamma 从 0 起)
        gamma = torch.tanh(self.residual_gamma)
        O_t = (Y_refined + gamma * X_t).clamp(0.0, 1.0)

        return {"O_t": O_t, "weights": weights, "Y_fused": Y_fused}


if __name__ == '__main__':
    B, H, W = 1, 64, 64
    YN = torch.rand(B, 3, H, W)
    YL = torch.rand(B, 3, H, W)
    YM = torch.rand(B, 3, H, W)
    Xc = torch.rand(B, 3, H, W)

    fusion = V7RFusion()
    out = fusion(YN, YL, YM, Xc)
    print("O_t:", out['O_t'].shape, "weights:", out['weights'].shape)

    # 恒等初始化验证: 初始应等于三分支均值
    mean = (YN + YL + YM) / 3.0
    print(f"初始输出 == 三分支均值: {torch.allclose(out['O_t'], mean, atol=1e-5)}")
    print(f"初始权重 (应≈1/3): {out['weights'].mean(dim=(0,2,3)).tolist()}")
    print(f"初始 gamma: {torch.tanh(fusion.residual_gamma).item():.4f}")
    print(f"参数量: {sum(p.numel() for p in fusion.parameters())/1e3:.1f}K")
