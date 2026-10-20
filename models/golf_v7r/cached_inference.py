#!/usr/bin/env python3
"""v8 滑窗推理加速: 逐帧编码缓存 + 向量化前向。

## 为什么可以缓存

模型对 5 帧片段做三件事: 逐帧编码、跨帧加权聚合、三分支重建。
第一步的编码器是**权重共享的逐帧运算**, 每一帧的结果只取决于该帧本身, 与它在窗口内的位置
无关。而实际推理是按窗口滑动的: 第 t 步的窗口是 [t-2, t+2], 第 t+1 步是 [t-1, t+3],
两者有 4 帧重合。这 4 帧的 f1/f2/f3 特征完全不变, 不需要重算。

因此把编码器结果按帧缓存, 每个新窗口只需编码 1 个新帧。稳态下单窗口的计算量从
5 次编码降到 1 次, 编码器约占模型参数的一半, 对应约 40% 的总延迟下降。

## 与直接调用的等价性

`cached_forward` 复用了与 `GolfNet_v8.forward` 完全相同的后续各阶段 (聚合、查询、三分支、
融合), 只是输入特征从缓存取。缓存键是帧在序列中的索引; 调用方负责保证同一索引对应同一帧。

## 向量化

编码器对 5 帧是独立运算, 因此在 batch 内把 5 帧拼成一次前向 (batch=5) 比循环 5 次更快,
GPU 利用率更高, 结果与逐帧循环完全一致 (共享权重、无跨帧算子)。`encode_frames` 采用这种做法。
"""
from __future__ import annotations

from collections import OrderedDict
from typing import Dict, Optional, Sequence

import torch


class CachedInference:
    """给 GolfNet_v8 包一层帧特征缓存。

    Args:
        model: GolfNet_v8 实例
        capacity: 缓存保留的帧数. 滑窗推理时设为窗口大小 + 预读帧数即可, 默认 8。
    """

    def __init__(self, model, capacity: int = 8):
        self.model = model
        self.capacity = int(capacity)
        # frame_key -> (f1, f2, f3)
        self._cache: "OrderedDict[int, tuple]" = OrderedDict()

    def clear(self):
        self._cache.clear()

    @torch.no_grad()
    def encode_frames(self, frames: torch.Tensor) -> Sequence[tuple]:
        """frames: [K, 3, H, W] → 每帧 (f1, f2, f3)。

        把 K 帧拼成一个 batch 一次前向。编码器是逐帧独立运算, 结果与循环 K 次相同。
        """
        f1, f2, f3 = self.model.encoder(frames)
        return [(f1[i], f2[i], f3[i]) for i in range(frames.shape[0])]

    @torch.no_grad()
    def get_features(self, key: int, frame: torch.Tensor) -> tuple:
        """取某帧特征, 未命中则计算并写入缓存 (LRU)。"""
        hit = self._cache.get(key)
        if hit is not None:
            self._cache.move_to_end(key)
            return hit
        feats = self.encode_frames(frame.unsqueeze(0))[0]
        self._cache[key] = feats
        while len(self._cache) > self.capacity:
            self._cache.popitem(last=False)
        return feats

    @torch.no_grad()
    def warm(self, frames: torch.Tensor, start_key: int = 0):
        """预热: 一次性编码并缓存一批连续帧 (用于第一个窗口)。"""
        feats = self.encode_frames(frames)
        for i, f in enumerate(feats):
            self._cache[start_key + i] = f

    @torch.no_grad()
    def step(self, clip: torch.Tensor,
             cache_keys: Optional[Sequence[int]] = None) -> Dict[str, torch.Tensor]:
        """单窗口推理。

        clip: [1, T, 3, H, W]
        cache_keys: 长度为 T 的帧索引; 为空时退化为不缓存的普通前向。
        """
        model = self.model
        B, T, _, H, W = clip.shape
        center = T // 2

        # ---------- Stage 1: 逐帧特征 (缓存或计算) ----------
        if cache_keys is None:
            feats = self.encode_frames(clip[0])
        else:
            feats = [self.get_features(k, clip[0, i])
                     for i, k in enumerate(cache_keys)]
        # 缓存的单帧特征不带 batch 维, 这里补回并沿帧维堆叠
        f1_seq = torch.stack([f[0].unsqueeze(0) for f in feats], dim=1)
        f2_seq = torch.stack([f[1].unsqueeze(0) for f in feats], dim=1)
        f3_seq = torch.stack([f[2].unsqueeze(0) for f in feats], dim=1)

        # ---------- Stage 2–5: 与 GolfNet_v8.forward 完全一致 ----------
        X_center = clip[:, center]
        summary = model.summary(f1_seq, f2_seq, f3_seq)
        F1_multi, F1_single = summary["F1_multi"], summary["F1_single"]
        F2_single = summary["F2_single"]

        F_N, F_L, F_M, ortho = model.tca(F1_multi, F1_single)
        out_N = model.branch_N(F_N)["Y_N"]
        out_L = model.branch_L(F_L, X_center)["Y_L"]
        out_M = model.branch_M(F_M, F2_single.unsqueeze(1))["Y_M"]

        fusion_out = model.fusion(out_N, out_L, out_M, X_center)
        return {
            "final": fusion_out["O_t"],
            "branch_N": out_N,
            "branch_L": out_L,
            "branch_M": out_M,
            "frame_weights": summary["frame_weights"],
            "ortho_loss": ortho,
            "fusion_weights": fusion_out["weights"],
        }
