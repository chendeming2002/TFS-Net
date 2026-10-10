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

────────────────────────────────────────────────────────────────────────
⚠️ 2026-10-10 关键修正: 融合算子的梯度病理 (docs/v7/03 §6.16)
────────────────────────────────────────────────────────────────────────
实测发现 `mode="softmax"` (历史实现) 存在**正反馈塌缩**:

    ∂Y_fused/∂Y_k = w_k      (softmax 权重直接充当分支梯度衰减系数)

`phaseA_l` 实测 w_L = 0.0044 ⇒ Branch-L 梯度仅为最大路的 **1.36%**
⇒ 分支学不动 ⇒ 输出更差 ⇒ 融合更不信任它 ⇒ w_L 继续降 …… 自锁死。
(同一现象在 v5 时代已记录为「归一化加权导致分支梯度衰减至 1-8%」.)

故新增 `mode="floored"` (推荐), 用**权重下限**取代纯竞争 softmax:

    g_k = ε + (1 − 3ε) · softmax(logits)_k        (ε = weight_floor, 默认 0.1)

梯度为  ∂Y/∂Y_k = g_k ≥ ε        ← **硬下界, 结构性保证**

设计要点与两处被实测否决的替代方案:
  - bias 零初始化 ⇒ softmax = 1/3 ⇒ g_k = 1/3 (与旧模式**完全同起点**, Σg=1 仍为凸组合)
  - ❌ `Y_base + Σσ_k(Y_k−Y_base)` (曾实现后废弃): 无效。其梯度 `σ_k + (1−Σσ)/3` 中 base 项与 σ
    **耦合** —— 塌缩时 σ=(0.5, 0.001, 0.5) ⇒ Σσ=1.001 ⇒ (1−Σσ)/3 = −0.0003,
    ∂/∂Y_L = 0.0007, 仍近零。**base 项不是独立兜底路径。**
  - ⚠️ 纯加性 `Σ Y_k` (IGRF Stage1/2 形式) 未被采用的原因:**它与本模块的既有分工不符** ——
    `V7RFusion` 的存在意义就是"让网络自己决定信任哪个分支"(见上文 §10.2b:
    `best_branch` 18.96 dB 高于任一固定分支); 固定等权相加会退化为不可学的平均。
    故保留**可学习门控**, 只对其加下界, 从而兼顾"可学"与"不塌缩"。

实测 (σ_L 塌缩至 0.001 时, ∂/∂Y_L 占最大路的比例):
    softmax  ε=0     : **0.0010**  ← 自锁死
    floored  ε=0.05  : 0.0570     (×57)
    floored  ε=0.10  : **0.1270** (×127, 默认)
    floored  ε=0.20  : 0.3362     (×336)

⚠️ `ε` 的取舍: 越大越防塌缩, 但越限制门控的表达力 (ε=1/3 时退化为固定均值, 完全不可学)。
默认 0.1 是"允许 10× 权重差异但仍保底"的折中, 属**待验证的超参**, 建议与 0.05/0.2 一起消融。
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict


class V7RFusion(nn.Module):
    """v7r 三分支融合 + 可选中心帧残差 (不破坏图像)

    mode:
      - "softmax" : 历史实现, 纯竞争归一化 `Σ w_k·Y_k` (∂/∂Y_k = w_k, 无下界 → 会塌缩)
      - "floored" : 新增 (推荐), 加权重下限 `g_k = ε + (1−3ε)·softmax_k` (∂/∂Y_k ≥ ε)
    """

    def __init__(self, in_channels: int = 3, hidden_channels: int = 32,
                 num_blocks: int = 2, mode: str = "softmax",
                 weight_floor: float = 0.1):
        super().__init__()
        if mode not in ("softmax", "floored"):
            raise ValueError(f"mode must be 'softmax' or 'floored', got {mode!r}")
        if not (0.0 <= weight_floor < 1.0 / 3.0):
            raise ValueError(f"weight_floor must be in [0, 1/3), got {weight_floor}")
        self.mode = mode
        self.weight_floor = float(weight_floor)
        self.mode = mode
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
        # 末层零初始化 → softmax 均匀 1/3 → g_k = ε + (1−3ε)/3 = 1/3
        # 故 floored 模式的初始输出与 softmax 完全一致 (受控对照前提)
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
        logits = self.weight_net(gate_input)

        if self.mode == "softmax":
            weights = F.softmax(logits, dim=1)
        else:
            # 权重下限: g = ε + (1−3ε)·softmax  ⇒  Σg = 1 (仍是凸组合),
            # 且 ∂Y/∂Y_k = g_k ≥ ε, 结构性地杜绝梯度自锁死
            eps = self.weight_floor
            weights = eps + (1.0 - 3.0 * eps) * F.softmax(logits, dim=1)

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
