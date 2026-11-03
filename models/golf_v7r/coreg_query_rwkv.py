#!/usr/bin/env python3
"""CoReg-QRB: 同网格关系条件化 QueryRWKVBlock。

TSD-Net 的 Query 与 Context 都是 ``[B,C,H,W]`` 全分辨率特征, 且同一
空间位置具有明确的对应关系。原 CrossRWKVHeadV8 只用 Q 产生 R/G,
而 WKV 的 K/V 状态在看到 Q 之前已完成构造。本块在保留原路径的基础上
增加三个严格可消融的关系条件化作用:

1. **Write**: 成对关系逐位置修正 K, 让 Q 控制什么写入 WKV 状态;
2. **Read / Gate**: 成对关系以残差方式增强 R/G, 分别控制读出与输出;
3. **Aligned / State**: 在全局 Bi-WKV 读出与同位置 V 读出之间做关系感知修正。

四个作用尺度均为零初始化, 因此新块初始时与 CrossRWKVHeadV8
逐元素一致; 它们只在训练证明有用后逐步打开。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8


class CoRegisteredQueryRWKVHead(CrossRWKVHeadV8):
    """同网格关系条件化 Cross-RWKV 头。

    ``center_feat`` 是查询流, ``context_feat`` 是内容流。两者必须完全
    同形, 因为成对描述子显式使用了同位置乘积与差异。

    Args:
        pair_rank: 成对关系 MLP 的瓶颈宽度; 默认 ``max(C/4, 8)``。
    """

    def __init__(self, channels: int, pair_rank: int | None = None, **kwargs):
        super().__init__(channels, **kwargs)
        rank = int(pair_rank or max(channels // 4, 8))
        self.pair_rank = rank

        # [Q, X, Q·X, |Q-X|] → 低秩关系特征。
        self.pair_in = nn.Linear(channels * 4, rank, bias=False)
        self.pair_out = nn.Linear(rank, channels, bias=False)
        # 零初始化的局部残差为小幅位移/边缘提供 3×3 容忍度。
        self.pair_local = nn.Conv2d(
            channels, channels, 3, 1, 1, groups=channels, bias=False)
        self.pair_norm = nn.LayerNorm(channels)

        self.pair_to_r = nn.Linear(channels, channels, bias=False)
        self.pair_to_g = nn.Linear(channels, channels, bias=False)
        self.pair_to_k = nn.Linear(channels, channels, bias=False)
        # 每个扫描方向一个同位置置信度, 在 head_dim 上广播。
        self.pair_to_aligned = nn.Linear(channels, self.num_dirs, bias=True)

        # tanh(scale) 有界且在 0 处梯度为 1; 初始时严格回退基线。
        self.read_scale = nn.Parameter(torch.zeros(()))
        self.gate_scale = nn.Parameter(torch.zeros(()))
        self.write_scale = nn.Parameter(torch.zeros(()))
        self.aligned_scale = nn.Parameter(torch.zeros(()))

        self._init_coreg_weights()

    def _init_coreg_weights(self):
        nn.init.xavier_uniform_(self.pair_in.weight)
        nn.init.xavier_uniform_(self.pair_out.weight)
        nn.init.zeros_(self.pair_local.weight)
        bound = 0.25 / math.sqrt(self.channels)
        for proj in (self.pair_to_r, self.pair_to_g, self.pair_to_k):
            nn.init.uniform_(proj.weight, -bound, bound)
        nn.init.zeros_(self.pair_to_aligned.weight)
        nn.init.zeros_(self.pair_to_aligned.bias)

    def _prepare_cross_context(self, center_feat: torch.Tensor,
                               context_feat: torch.Tensor):
        if center_feat.shape != context_feat.shape:
            raise ValueError(
                "CoReg-QRB 需要 Query/Context 完全同形, "
                f"收到 {tuple(center_feat.shape)} 与 {tuple(context_feat.shape)}")

        B, C, H, W = center_feat.shape
        q = self.pre_norm(center_feat.flatten(2).transpose(1, 2))
        x = self.pre_norm(context_feat.flatten(2).transpose(1, 2))
        relation = torch.cat((q, x, q * x, (q - x).abs()), dim=-1)
        pair = self.pair_out(F.silu(self.pair_in(relation)))
        pair_2d = pair.transpose(1, 2).reshape(B, C, H, W)
        pair = pair + self.pair_local(pair_2d).flatten(2).transpose(1, 2)
        pair = self.pair_norm(pair)
        aligned_rho = torch.sigmoid(self.pair_to_aligned(pair))

        self._last_coreg = {
            "read_scale": torch.tanh(self.read_scale.detach()),
            "gate_scale": torch.tanh(self.gate_scale.detach()),
            "write_scale": torch.tanh(self.write_scale.detach()),
            "aligned_scale": torch.tanh(self.aligned_scale.detach()),
            "aligned_rho_mean": aligned_rho.detach().mean(),
        }
        return {"pair": pair, "aligned_rho": aligned_rho}

    def _make_r_with_context(self, center_feat: torch.Tensor, cross_context):
        r = self._make_r(center_feat)
        delta = self.pair_to_r(cross_context["pair"])
        return r + torch.tanh(self.read_scale) * delta

    def _make_g_with_context(self, center_feat: torch.Tensor, cross_context):
        g = self._make_g(center_feat)
        delta = self.pair_to_g(cross_context["pair"])
        return g + torch.tanh(self.gate_scale) * delta

    def _modulate_k_with_context(self, center_feat: torch.Tensor,
                                 context_feat: torch.Tensor,
                                 k: torch.Tensor, cross_context):
        del center_feat, context_feat
        # 加性 log-weight 修正: 进入 WKV 前有界, 不直接改动 V。
        delta = torch.tanh(self.pair_to_k(cross_context["pair"]))
        return k + torch.tanh(self.write_scale) * delta

    def _mix_cross_head(self, r_i: torch.Tensor, g_i: torch.Tensor,
                        wkv_i: torch.Tensor, value_i: torch.Tensor,
                        direction: int, cross_context) -> torch.Tensor:
        state_read = self.head_norm(r_i * wkv_i)
        aligned_read = self.head_norm(r_i * value_i)
        rho = cross_context["aligned_rho"][:, :, direction:direction + 1]
        # 残差式对齐修正避免重复累加当前 token; scale=0 时严格为基线。
        mixed = state_read + torch.tanh(self.aligned_scale) * rho * (
            aligned_read - state_read)
        return g_i * mixed


if __name__ == "__main__":
    torch.manual_seed(0)
    head = CoRegisteredQueryRWKVHead(32, wkv_mode="legacy")
    q = torch.randn(1, 32, 16, 16, requires_grad=True)
    x = torch.randn(1, 32, 16, 16, requires_grad=True)
    y = head(q, x)
    y.mean().backward()
    print(f"CoReg-QRB: {tuple(y.shape)}, params={sum(p.numel() for p in head.parameters()):,}")
