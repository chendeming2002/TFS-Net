#!/usr/bin/env python3
"""SGRFv8: 三分支分阶段串接融合 (Stage-wise Guided Restoration & Fusion)。

## 与 V7RFusion 的根本差别

V7RFusion 是并行融合: 三路分支各自从同一个 F1 出发, 各自输出一张图, 最后用逐像素权重
加权求和。三路之间没有任何信息依赖 —— N 不知道 L 在做什么, L 不知道 M 在做什么。

SGRFv8 是串接融合: 从中心帧出发, 先由 N 路做去噪, 得到一级结果; L 路以这一级结果为输入
做光照校正, 得到二级结果; M 路再以二级结果为输入做运动补偿, 输出最终图。三路共享同一组
分支网络, 但每路的输入不再是共享的 F1, 而是上一级的输出。

这个顺序对应 TSDR 的物理依赖: 曝光决定了噪声的绝对强度, 噪声又叠加在同一个曝光结果上,
因此先去噪、再定曝光、最后修运动, 比三路各自独立猜测更接近成像过程。

## 三条设计原则

1. **加性串接, 不用乘性**。每一级写成 `y_k = y_{k-1} + Δ_k`, 因此
   `∂y_3/∂Δ_k = 1` (k = 1, 2, 3), 三路梯度完全相等, 不存在衰减。
   早期 IGRF 的 Stage 3 是乘性提亮 `base ⊙ lit`, 偏导为 `lit/base`,
   实测对某一路衰减 100 到 346 倍 (docs/v7/03-v7r-v3-design.md §6.16), 本实现不用乘性。

2. **级间留在特征域, 不做 clamp**。每一级的输出是与输入同形状的残差项, 累加结果不进
   [0, 1]; 只有最后一级写回图像域时才做一次 clamp。中间 clamp 会在极暗区截断梯度
   (v5 的记录), 这里避免。

3. **每级的输入是上一级的输出**。实现上把上一级的累加结果用一个轻量投影映回特征空间,
   与共享的 F1 相加后送入下一路分支。这样既保留了"上一级结果驱动下一级"的语义,
   又不需要三路分支改变接口 (它们仍然接收一个 C 通道的特征图)。

## 与分支的接口

三路分支本身完全不变: `NBranch(F_N)`, `LBranch(F_L, X_center)`, `MBranch(F_M, F2)`。
本模块只负责组织它们的输入与串接它们的输出:

    F_N = F_N,   F_L = F_L + proj(y_1),   F_M = F_M + proj(y_2)
    y_1 = Y_N(F_N)
    y_2 = y_1 + Y_L(F_L, X_0)
    y_3 = y_2 + Y_M(F_M)
    O_t = clamp(y_3, 0, 1)

串接的起点是**零**而不是中心帧图像。原因是三路频带目标之和恒等于 GT
(`datasets/branch_targets.py` 的定义), 加性融合在权重全为 1 时精确还原 GT。
若把 `X_0` 也加进来, 等于把光照计入两次 —— L 路的目标本身已经包含 GT 的低频部分。

其中 `proj` 是 3 通道到 C 通道的 1×1 卷积, 零初始化, 因此初始化时 `F_L = F_M = F1`,
三路行为与并行版完全一致, 训练可以平稳起步。
"""
from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn as nn


class SGRFv8(nn.Module):
    """分阶段串接融合: N → L → M 三级加性累加。

    Args:
        channels: 分支特征的通道数 (与 F1 一致)
        stage_order: 三段的执行顺序, 取 "NLM" / "LNM" 等排列。
            默认 "NLM", 依据是"先去噪、再定曝光、最后修运动"这一物理顺序。
        clamp_output: 是否在最终输出上做 clamp(0, 1)。只作用于最后一级。
    """

    def __init__(self, channels: int = 32, stage_order: str = "NLM",
                 clamp_output: bool = True):
        super().__init__()
        if sorted(stage_order) != ["L", "M", "N"]:
            raise ValueError(
                f"stage_order 必须是 N/L/M 的一个排列, 收到 {stage_order!r}")
        self.channels = channels
        self.stage_order = stage_order
        self.clamp_output = bool(clamp_output)

        # 把上一级的图像域累加结果映回特征空间。
        # 零初始化 → 初始化时 F_L = F_M = F1, 与并行版同起点。
        self.proj_to_feat_1 = nn.Conv2d(3, channels, 1, 1, 0, bias=False)
        self.proj_to_feat_2 = nn.Conv2d(3, channels, 1, 1, 0, bias=False)
        for proj in (self.proj_to_feat_1, self.proj_to_feat_2):
            nn.init.zeros_(proj.weight)

        # 每段的逐像素缩放, 初值 1 → 初始输出 = ΣY_k, 与 additive 融合同起点。
        # 用 softplus 的正参数化避免符号翻转带来的抵消。
        self.stage_scale = nn.Parameter(torch.zeros(3))

    # softplus(0) = ln 2; 除以它使初值精确为 1
    _LN2 = math.log(2.0)

    def _scale(self, idx: int) -> torch.Tensor:
        return torch.nn.functional.softplus(self.stage_scale[idx]) / self._LN2

    def _scale_all(self) -> torch.Tensor:
        return torch.nn.functional.softplus(self.stage_scale) / self._LN2

    def forward(self, feats: Dict[str, torch.Tensor], X_t: torch.Tensor,
                branch_N: nn.Module, branch_L: nn.Module, branch_M: nn.Module,
                F2_single: torch.Tensor,
                zero_stage: str | None = None) -> Dict[str, torch.Tensor]:
        """
        feats:      {"N": F_N, "L": F_L, "M": F_M}, 各 [B, C, H, W]
                    TaskQueryAttention 输出的三路特征, 保持各自的专门化
        X_t:        [B, 3, H, W]  中心帧图像, 作为串接起点
        F2_single:  [B, C2, h, w] M 支路的辅助输入, 接口保持不变

        zero_stage: 诊断用。把指定段的缩放置零, 即跳过该段的贡献,
            用于测量每一段对最终图像的实际影响 (对应并行融合里的"置零代价")。

        Returns: {"O_t", "Y_N", "Y_L", "Y_M", "stages"}
            Y_* 仍是各段自身的输出, 供分支频带目标监督使用。
        """
        projs = {"N": None, "L": self.proj_to_feat_1, "M": self.proj_to_feat_2}
        branches = {"N": branch_N, "L": branch_L, "M": branch_M}

        # 起点为零: 三路目标之和等于 GT, 加到中心帧上会重复计入光照
        y = torch.zeros_like(X_t)
        prev = None
        stage_img = {}
        y_branch = {}
        for k, name in enumerate(self.stage_order):
            proj = projs[name]
            f_in = feats[name]
            if proj is not None and prev is not None:
                # 上一段的累加结果映回特征空间, 与本段自己的 TCA 特征相加
                f_in = f_in + proj(prev)

            if name == "N":
                y_new = branches["N"](f_in)["Y_N"]
            elif name == "L":
                y_new = branches["L"](f_in, X_t)["Y_L"]
            else:
                y_new = branches["M"](f_in, F2_single.unsqueeze(1))["Y_M"]

            scale = self._scale(k)
            if zero_stage is not None and name == zero_stage:
                scale = scale * 0.0
            y = y + scale * y_new
            prev = y
            stage_img[name] = y
            y_branch[name] = y_new

        out = y.clamp(0.0, 1.0) if self.clamp_output else y
        return {"O_t": out, "stages": stage_img,
                "Y_N": y_branch["N"], "Y_L": y_branch["L"], "Y_M": y_branch["M"],
                "stage_scales": self._scale_all()}


if __name__ == "__main__":
    print(__doc__.split("\n\n")[1])
