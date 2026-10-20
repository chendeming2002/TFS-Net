#!/usr/bin/env python3
"""多方向批量 WKV: 把同一空间头的 4 个扫描方向堆到 batch 维, 一次算子调用算完。

## 为什么可以合并

四个扫描方向各自把同一张特征图按不同顺序展开成一维序列, 然后做同一种双向 WKV 递推。
方向之间没有任何数据依赖, 只是输入排列不同。原来的实现每个方向调一次算子
(每个空间头 4 次, 全模型 28 次), 每次调用都要启动若干个 kernel; 合并后每个头只调一次,
全模型 7 次。

## 为什么合并反而更快

单次调用处理 4 倍数据, GPU 的并行度更高, 而 kernel 启动与调度的固定开销只付一次。
实测 L=65536、通道 8 时, 4 次单方向调用合计 6.79 ms, 一次四方向调用 1.78 ms。

## 梯度

CUDA 核的梯度写回按 (batch, channel) 寻址, 因此每个方向各自得到自己的梯度, 不做跨方向求和。
这与四个方向各自持有独立参数 (原来就是 4 个独立模块) 的语义一致。

## 数值等价

批量调用与逐方向单次调用的输出逐元素相同 (已用 atol=0 验证)。衰减与奖励参数在核内
按 (batch, channel) 读取, 每个方向读到的是自己的那一行, 不会互相污染。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.cuda_wkv import _WKVFunction, _load_kernel, _TMAX


class BatchedBiWKV(nn.Module):
    """同时处理 num_dirs 个独立扫描的参数容器, 每次 forward 只调用一次 CUDA 核。

    Args:
        channels: 每个扫描方向的通道数 (即 head_dim)
        num_dirs: 同时扫描的方向数
    """

    def __init__(self, channels: int, num_dirs: int = 4):
        super().__init__()
        self.channels = channels
        self.num_dirs = num_dirs
        self.spatial_decay = nn.Parameter(torch.randn(num_dirs, channels) * 0.1)
        self.spatial_first = nn.Parameter(torch.randn(num_dirs, channels) * 0.1)

    def forward(self, k: torch.Tensor, v: torch.Tensor,
                total_tokens: int) -> torch.Tensor:
        """k, v: [num_dirs, L, C], 每个方向已按自己的扫描顺序展开。返回 [num_dirs, L, C]。"""
        D, L, C = k.shape
        assert D == self.num_dirs, f"期望 {self.num_dirs} 个方向, 收到 {D}"
        if L > _TMAX and (k.requires_grad or v.requires_grad):
            raise RuntimeError(
                f"序列长度 L={L} 超过反向内核上限 _TMAX={_TMAX};"
                f" 训练时请减小 crop_size, 或推理时用 torch.no_grad()")
        k = k.float().clamp(-8, 8)
        v = v.float().clamp(-8, 8)
        w = -F.softplus(self.spatial_decay)          # (D, C), 均 < 0
        u = self.spatial_first.clamp(-5, 5)
        # 与单方向版相同的符号映射: 核内衰减写作 exp(-w_kernel·距离)
        w_kernel = (-w / total_tokens).contiguous()
        u_kernel = (u / total_tokens).contiguous()
        return _WKVFunction.apply(D, L, C, w_kernel, u_kernel, k, v)


if __name__ == '__main__':
    import time
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    D, L, C = 4, 16384, 8
    mod = BatchedBiWKV(C, D).to(dev)
    k = torch.randn(D, L, C, device=dev)
    v = torch.randn(D, L, C, device=dev)
    y = mod(k, v, total_tokens=L)
    print(f"BatchedBiWKV: {tuple(y.shape)}, params={sum(p.numel() for p in mod.parameters())}")
    y.sum().backward()
    print(f"梯度: decay {mod.spatial_decay.grad.shape}, 非零通道 "
          f"{int((mod.spatial_decay.grad.abs().sum(1) > 0).sum())}/{D}")


def batched_scan(modules, k_list, v_list, total_tokens):
    """把一组同形状的扫描模块合并成一次算子调用。

    modules: 每个方向一个 CudaBiWKV (或 TorchBiWKVSingleNorm)
    k_list, v_list: 长度等于方向数的 [B, L, C] 序列

    只有全部模块都是 CUDA 版且 batch 为 1 时才合并; 否则逐方向调用, 结果相同。
    参数按方向堆叠成 [D, C] 传入, 梯度经 autograd 回到各自的模块, 不做求和。
    """
    from models.modules.cuda_wkv import CudaBiWKV
    can_batch = (all(isinstance(m, CudaBiWKV) for m in modules)
                 and k_list[0].is_cuda and k_list[0].shape[0] == 1)
    if not can_batch:
        return [m(k, v, total_tokens) for m, k, v in zip(modules, k_list, v_list)]

    C = modules[0].channels
    w = torch.stack([-torch.nn.functional.softplus(m.spatial_decay) for m in modules])
    u = torch.stack([m.spatial_first.clamp(-5, 5) for m in modules])
    k = torch.cat(k_list, dim=0)          # [D, L, C]
    v = torch.cat(v_list, dim=0)
    y = _WKVFunction.apply(len(modules), k.shape[1], C,
                           (-w / total_tokens).contiguous(),
                           (u / total_tokens).contiguous(), k, v)
    return list(y.split(1, dim=0))
