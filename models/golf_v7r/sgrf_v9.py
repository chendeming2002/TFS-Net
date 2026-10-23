#!/usr/bin/env python3
"""SGRFv9: TSDR-aware Gated Progressive Restoration (三源门控渐进式恢复)。

## 与 SGRFv8 的差别

SGRFv8 是三段加性串接, 每段用一个全局标量决定贡献。30 轮实验显示三个问题:

1. 阶段依赖过强: L 段置零代价 −17.2 dB, N 段 −0.44 dB, M 段 −0.13 dB,
   最终图像几乎完全由 L 段决定;
2. 阶段系数失衡: 收敛到 N 1.628 / L 1.002 / M 0.405, M 段的有效贡献被压制;
3. 串接误差累积: 后一段直接接收前一段预测的图像, 前段的误差会改变后段的输入分布,
   而 SGRFv8 没有任何机制去抑制这种误差传播。

SGRFv9 保留"渐进式分段修正"这一范式, 但把每段的贡献机制从"全局标量"改为
"空间自适应门控", 并让每段预测的是**针对当前状态的修正量**。

## 更新式

    X_0 = 0                                  (三路频带目标之和等于 GT, 从零累加)
    Δ_N = Y_N(F_N)                           N 段: 去噪增量
    G_N = σ(gate_N([X_0, F_N]))              逐像素门控
    X_N = X_0 + s_N · G_N ⊙ Δ_N

    F_L' = F_L + proj_L(X_N)                 把当前状态映回特征空间
    Δ_L = Y_L(F_L', X_0)                     L 段: 复用分支自身的 Retinex 重建
    G_L = σ(gate_L([X_N, F_L']))
    X_L = X_N + s_L · G_L ⊙ Δ_L

    F_M' = F_M + proj_M(X_L)
    Δ_M = Y_M(F_M', F2_single)
    G_M = σ(gate_M([X_L, F_M']))
    X_M = X_L + s_M · G_M ⊙ Δ_M

    O_t = clamp(X_M, 0, 1)

## 三条设计要点

**门控必须有初值且不能饱和。** 门是一个 3×3 卷积的轻量网络, 输出 3 通道,
最后一层权重零初始化、偏置零初始化, 因此初始 `σ(0) = 0.5` —— 每段初始只贡献一半,
且各处一致。若用大偏置初始化 (如 2.0) 会接近全开, 失去门控意义; 用负偏置会让
早期阶段被压制, 正是 SGRFv8 的病灶。

**阶段系数有界。** `s_k = 1 + 0.5·tanh(a_k)`, 取值范围 [0.5, 1.5], 初值 1。
它只调整整体幅度, 不承担逐位置的选择职责。这样既不会出现 SGRFv8 那种 0.405 的
压制, 也不会强制三段等权。

**阶段专属的修正形式。**
    N: 增量经一次深度卷积 (逐通道 3×3) 处理, 强调局部纹理保留, 避免过度平滑;
    L: 直接复用分支的 Retinex 重建 (曲线 + 增益), 不额外堆叠通用卷积头;
    M: 增量经一次深度卷积, 利用分支已有时序特征, 不引入光流。

三个分支网络本身完全不变, 各自的归纳偏置 (N 的通道注意力、L 的 Retinex、M 的运动精化)
由此保留。

## 梯度

模块内部没有任何 `detach`。最终重建损失可以一路传回 N 段, 因此不存在"前段拿不到
最终损失梯度"的问题。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F


class StageGate(nn.Module):
    """逐像素门控: 由当前重建状态与该段特征共同决定修正强度。

    Args:
        feat_channels: 该段特征通道数
        img_channels: 图像通道数 (3)
        hidden: 隐层通道数

    初始输出恒为 0 (末层权重与偏置均零初始化), 因此 `σ(0) = 0.5`:
    每段初始贡献一半修正量, 所有位置一致, 不偏向某一段。
    """

    def __init__(self, feat_channels: int, img_channels: int = 3,
                 hidden: int = 16):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(img_channels + feat_channels, hidden, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(hidden, img_channels, 1, 1, 0, bias=True),
        )
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, state: torch.Tensor, feat: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.net(torch.cat([state, feat], dim=1)))


class SGRFv9(nn.Module):
    """三段门控渐进式恢复: N (去噪) → L (光照) → M (运动)。

    Args:
        feat_channels: 分支特征通道数 (与 TCA 输出一致)
        stage_order: 三段顺序, 默认 "NLM"
        clamp_output: 是否对最终输出做 clamp(0, 1)
    """

    def __init__(self, feat_channels: int = 32, stage_order: str = "NLM",
                 clamp_output: bool = True, delta_bound: float = 0.5):
        super().__init__()
        if sorted(stage_order) != ["L", "M", "N"]:
            raise ValueError(
                f"stage_order 必须是 N/L/M 的一个排列, 收到 {stage_order!r}")
        self.feat_channels = feat_channels
        self.stage_order = stage_order
        self.clamp_output = bool(clamp_output)

        # 把当前重建状态映回特征空间: 1×1 卷积, 零初始化为恒等起点 (不改变初始特征)
        self.proj_L = nn.Conv2d(3, feat_channels, 1, 1, 0, bias=False)
        self.proj_M = nn.Conv2d(3, feat_channels, 1, 1, 0, bias=False)
        for proj in (self.proj_L, self.proj_M):
            nn.init.zeros_(proj.weight)

        # 逐段门控
        self.gate = nn.ModuleDict({
            name: StageGate(feat_channels) for name in "NLM"
        })

        # 阶段专属的增量处理: N 与 M 用逐通道 3×3, 强调局部结构; L 不额外处理。
        self.lift_N = nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False)
        self.lift_M = nn.Conv2d(3, 3, 3, 1, 1, groups=3, bias=False)
        for conv in (self.lift_N, self.lift_M):
            # 恒等初始化: 初始时该处理不改变分支输出
            nn.init.zeros_(conv.weight)
            with torch.no_grad():
                for c in range(3):
                    conv.weight[c, 0, 1, 1] = 1.0

        # 阶段幅度校准: s_k = 1 + 0.5·tanh(a_k) ∈ [0.5, 1.5], 初值 1
        self.stage_amp = nn.Parameter(torch.zeros(3))

        # 增量限幅。三个分支的初始化并不一致: N 与 L 内部有零初始化或 clamp, M 的
        # to_rgb 是普通卷积, 输出幅度不受控。实测冒烟 6 步内 M 段增量从 0.024 涨到
        # 0.431 而 L 段收缩, 会重现阶段失衡。这里对每段修正量做软限幅, 使其在
        # |Δ| << bound 时近似恒等, 超出后饱和, 从而避免某一段独占更新。
        self.delta_bound = float(delta_bound)

    def _amp(self, idx: int) -> torch.Tensor:
        return 1.0 + 0.5 * torch.tanh(self.stage_amp[idx])

    def _amp_all(self) -> torch.Tensor:
        return 1.0 + 0.5 * torch.tanh(self.stage_amp)

    def forward(self, feats: Dict[str, torch.Tensor], X_t: torch.Tensor,
                branch_N: nn.Module, branch_L: nn.Module, branch_M: nn.Module,
                F2_single: torch.Tensor,
                zero_stage: str | None = None) -> Dict[str, torch.Tensor]:
        """
        feats:      {"N": F_N, "L": F_L, "M": F_M}, 各 [B, C, H, W]
        X_t:        [B, 3, H, W]  中心帧图像 (只作为 L 段的辅助输入, 不进累加)
        F2_single:  [B, C2, h, w] M 段的辅助输入

        Returns: O_t, 各段输出与统计 (增量幅度、门控均值、门控饱和率)、Y_N/L/M
        """
        branches = {"N": branch_N, "L": branch_L, "M": branch_M}
        projs = {"N": None, "L": self.proj_L, "M": self.proj_M}

        x = torch.zeros_like(X_t)          # 从零累加: 三路目标之和等于 GT
        y_branch: Dict[str, torch.Tensor] = {}
        stats: Dict[str, torch.Tensor] = {}
        gate_reg = x.new_zeros(())

        for k, name in enumerate(self.stage_order):
            feat = feats[name]
            if projs[name] is not None:
                feat = feat + projs[name](x)     # 当前状态参与该段特征

            if name == "N":
                delta = branches["N"](feat)["Y_N"]
                delta = self.lift_N(delta)       # 纹理保留的局部残差
            elif name == "L":
                delta = branches["L"](feat, X_t)["Y_L"]   # 复用 Retinex 重建
            else:
                delta = branches["M"](feat, F2_single.unsqueeze(1))["Y_M"]
                delta = self.lift_M(delta)

            # 软限幅: |Δ| << bound 时近似恒等, 超出后饱和到 ±bound
            delta_b = self.delta_bound * torch.tanh(delta / self.delta_bound)

            g = self.gate[name](x, feat)
            amp = self._amp(k)
            if zero_stage is not None and name == zero_stage:
                amp = amp * 0.0

            step = amp * g * delta_b
            x = x + step

            y_branch[name] = delta
            stats[f"inc_{name}"] = step.detach().abs().mean()
            stats[f"gate_{name}"] = g.detach().mean()
            # 饱和率: 门值落在 (0.9, 1] 或 [0, 0.1) 的比例, 反映门是否退化为开关
            stats[f"gate_{name}_sat"] = (
                ((g.detach() > 0.9) | (g.detach() < 0.1)).float().mean())
            # 饱和惩罚: 门接近 0 或 1 时 sigmoid 导数趋零, 该段就失去调节能力。
            # 用 (g−0.5)² 把门保持在可调区间, 而不是让某段整体退出。
            gate_reg = gate_reg + ((g - 0.5) ** 2).mean()
            # 限幅前后的幅度: 用于判断限幅是否压制了该段真正需要的修正。
            # 压缩比例 = 进入明显非线性区 (|Δ| > 0.5·δ) 的元素占比
            d = delta.detach()
            db = delta_b.detach()
            stats[f"rms_pre_{name}"] = d.pow(2).mean().sqrt()
            stats[f"rms_post_{name}"] = db.pow(2).mean().sqrt()
            stats[f"absmax_pre_{name}"] = d.abs().max()
            stats[f"absmax_post_{name}"] = db.abs().max()
            stats[f"frac_compress_{name}"] = (d.abs() > 0.5 * self.delta_bound).float().mean()

        out = x.clamp(0.0, 1.0) if self.clamp_output else x
        return {"O_t": out, "stages": {},
                "Y_N": y_branch["N"], "Y_L": y_branch["L"], "Y_M": y_branch["M"],
                "stage_stats": stats, "stage_amp": self._amp_all(),
                "gate_reg": gate_reg / len(self.stage_order)}


if __name__ == "__main__":
    print(__doc__.split("## 更新式")[0].strip()[:300])
