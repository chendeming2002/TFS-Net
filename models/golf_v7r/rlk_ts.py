#!/usr/bin/env python3
"""RLK-TS: 残差大核 TokenShift (v8.8N)。

> 设计见 docs/v8/20-v8.8N-RLKTS设计.md。

形式:
    RLKTS(x) = x + γ · (DW_5x5(x) − x)

- DW_5x5 初始化为中心 1、其余 0 → 初始时 DW(x) = x;
- γ 初始化 0.1 → 初始时整体输出 = x (双重保证的恒等起点);
- 训练中 5×5 核学习邻域增量, γ 控制混合强度;
- 推理时可用 merge() 融合为单个 5×5 核 ((1−γ)·δ + γ·W)。

设计依据: v8.8-B 的 45 组置零消融显示 1×1/3×3 分支无功能贡献、
5×5 是唯一有实质贡献的卷积分支。RLK-TS 是该证据支持的简化形式。
"""
from __future__ import annotations

import torch
import torch.nn as nn


class RLKTS(nn.Module):
    """残差大核 TokenShift: x + γ·(DW5x5(x) − x), 初始恒等。

    Args:
        channels: 通道数 (depthwise, 参数量 = 25·channels + 1)
    """

    def __init__(self, channels: int):
        super().__init__()
        self.channels = channels
        self.dw = nn.Conv2d(channels, channels, 5, 1, 2,
                            groups=channels, bias=False)
        self.gamma = nn.Parameter(torch.tensor(0.1))
        with torch.no_grad():                     # 中心 1 其余 0 → DW(x) = x
            self.dw.weight.zero_()
            for c in range(channels):
                self.dw.weight[c, 0, 2, 2] = 1.0
        # 推理用的融合核缓存; 放在列表里避免被注册为子模块污染 state_dict
        self._merged_cache = [None]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            if self._merged_cache[0] is None:
                self.reparameterize()
            return self._merged_cache[0](x)
        return x + self.gamma * (self.dw(x) - x)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._merged_cache[0] = None
        return self

    @torch.no_grad()
    def reparameterize(self):
        """把恒等项与卷积项融合为单个 5×5 depthwise 核。

        RLKTS(x) = (1−γ)·x + γ·DW(x) 的核形式: (1−γ)·δ + γ·W。
        """
        w = self.dw.weight
        merged = (1.0 - self.gamma) * torch.zeros_like(w)
        for c in range(self.channels):
            merged[c, 0, 2, 2] = 1.0 - float(self.gamma)
        merged = merged + self.gamma * w
        conv = nn.Conv2d(self.channels, self.channels, 5, 1, 2,
                         groups=self.channels, bias=False)
        conv.weight.copy_(merged)
        conv = conv.to(device=w.device, dtype=w.dtype)
        self._merged_cache[0] = conv
        return conv


if __name__ == "__main__":
    torch.manual_seed(0)
    m = RLKTS(32)
    x = torch.randn(2, 32, 16, 16)
    y = m(x)
    print(f"RLKTS: {tuple(y.shape)}, 初始恒等误差 {(y - x).abs().max().item():.2e}")
    # 非恒等验证
    with torch.no_grad():
        m.dw.weight += 0.1
        m.gamma.fill_(0.5)
    y2 = m.train()(x)
    print(f"调整后与输入差 {(y2 - x).abs().mean().item():.4f} (应非零)")
    # 融合等价
    m.eval()
    with torch.no_grad():
        y_train_form = m.train()(x)
        y_eval_form = m(x)
    print(f"训练/推理形式差 {(y_train_form - y_eval_form).abs().max().item():.2e}")
