#!/usr/bin/env python3
"""v8 B1: 用空间 RWKV block 替换编码器里的 NAFBlock (docs/v8/01-设计.md §四)。

## 动机

当前编码器 (`models/golf/encoder.py::SharedEncoder`) 是 NAFBlock 构建的三尺度金字塔,
它是整个模型里唯一不使用 RWKV 的主干。把编码器换成 RWKV 可以让结构统一。

## 做法

编码器是**逐帧独立**跑的, 没有时序轴, 因此只能用空间扫描。这里直接复用已实现的
`SpatialRWKVHeadV8` (OmniShift + 双向扫描 + Channel Mix), 把它的两条输入都指向同一张特征图
(单帧自注意力: Q 与 K/V 同源, 但仍走不同的投影矩阵)。

包装成一个 `forward(x) -> x` 的同形状模块, 以便在 `SharedEncoder` 里与 `NAFBlock` 互换。

## 与 NAFBlock 的差异 (评估时要留意)

NAFBlock 是卷积 + SimpleGate + 通道注意力, 感受野由卷积层数决定;
本 block 是全局双向扫描, 感受野覆盖整图。因此替换后编码器的**归纳偏置从局部变成全局**,
这是本次实验真正的变量, 不只是"换了个主干"。

## 参数量

NAFBlock(C) 约 3C² 量级 (两段式, expand=2);
SpatialRWKVHeadV8(C) 约 14C² (四个投影 + Channel Mix 4× 隐层)。
因此替换后参数会上涨, 属预期, 评估时只看指标是否明显变差与显存是否可控。
"""
from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn

from models.golf_v7r.spatial_rwkv_head_v8 import SpatialRWKVHeadV8


class RWKVEncoderBlock(nn.Module):
    """编码器用的空间 RWKV block: 与 NAFBlock 同接口 (同形状进出)。

    Args:
        channels: 特征通道数
        num_directions: 扫描方向数
        hidden_rate: Channel Mix 隐层倍数
        num_loops: 循环次数 M (默认 1)。

            Restore-RWKV 的 Re-WKV 把同一个模块循环 M 次 (参数不变, 计算 ×M),
            与本项目历史上"堆 N 个 block (参数与计算都 ×N)"不同。
            这里给出 M 的开关以便后续消融, 默认 1 即单次。
    """

    def __init__(self, channels: int, num_directions: int = 4,
                 hidden_rate: int = 4, num_loops: int = 1):
        super().__init__()
        assert num_loops >= 1, f"num_loops 必须 ≥ 1, 收到 {num_loops}"
        self.num_loops = int(num_loops)
        self.head = SpatialRWKVHeadV8(channels, num_directions=num_directions,
                                      hidden_rate=hidden_rate)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for _ in range(self.num_loops):
            x = self.head(x, x)
        return x


if __name__ == '__main__':
    for c in (32, 64, 128):
        blk = RWKVEncoderBlock(c)
        x = torch.randn(2, c, 16, 16)
        y = blk(x)
        n = sum(p.numel() for p in blk.parameters())
        print(f"RWKVEncoderBlock({c}): in {tuple(x.shape)} -> out {tuple(y.shape)}, "
              f"params={n/1e3:.1f}K")
        y.sum().backward()
        dead = [name for name, p in blk.named_parameters()
                if p.grad is None or p.grad.abs().sum() == 0]
        print(f"  首步无梯度参数: {dead if dead else '无'}")
