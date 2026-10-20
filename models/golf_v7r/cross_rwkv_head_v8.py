#!/usr/bin/env python3
"""Cross RWKV Gate 空间头: 按 Video Action Recognition Based RWKV (arXiv 2411.05636)
第 3.2 节的 Cross RWKV gate 组织查询, 替换 SpatialRWKVHeadV8 里的 R 式单门。

## 论文结构与本实现的对应

论文式 (2)–(4)::

    R = Q-Shift_R(Cauconv(x_t, h_{t-1})) W_R
    G = Q-Shift_G(Cauconv(x_t, h_{t-1})) W_G
    K = Q-Shift_K(x_t^e) W_K
    V = Q-Shift_V(x_t^e) W_V
    a = concat( SiLU(G) ⊙ LN(R · Bi-wKV) ) W_a

关键点有三处, 与旧实现 (sigmoid(q) * wkv) 不同:

1. R 和 G 是**两个独立投影**, 来自同一路输入; K 和 V 来自**另一路**输入。
2. R 是带符号的线性投影, 直接与 WKV 输出逐元素相乘, 不经过 sigmoid。
3. 乘法之后接**逐头 LayerNorm**, 最后由 SiLU(G) 门控。

因此在这个头里, R 与 G 就是三路任务各自的查询机制本身: R 决定以什么符号和比例
调制 WKV 的输出, G 决定在归一化之后保留多少。论文没有给出 Q-Shift 的实现细节,
图像上采用 OmniShift (空间邻域的等价物), 四个投影各一个。

## 不迁移的部分

LSTM 的隐状态与细胞状态、式 (10)(11) 的递推更新、边缘信息作为遗忘门、Canny 边缘图
与 tube masking 均不迁移。KV 仍由调用方按支路频带提供。理由见
`docs/v8/06-CrossRWKVGate-Q重构设计.md` 第三节与第七节。

## 输入约定

`center_feat` 是 R/G 的输入 (N、L 传多帧聚合特征, M 传中心帧特征),
`context_feat` 是 K/V 的输入 (各支路自己的频带特征)。两者形状均为 (B, C, H, W)。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.golf_v7r.spatial_rwkv_head_v8 import OmniShift, SpatialRWKVHeadV8
from models.modules.batched_wkv import batched_scan


class CrossRWKVHeadV8(SpatialRWKVHeadV8):
    """Cross RWKV Gate 形态的空间语义头 (R/G 双查询, 逐头 LN, 论文式 6–8 的 Channel Mix)。

    Args:
        channels: 特征通道数
        num_directions: 扫描方向数, 同时是"头"的个数 (逐头 LN 以它为分组)
        hidden_rate: Channel Mix 的隐层倍数
        use_omni_shift: 是否对四个投影的输入做 Q-Shift (默认 True)
        use_channel_mix: 是否启用 Channel Mix (默认 True)
        wkv_mode: "single_norm" (官方双向定义, 优先 CUDA) 或 "legacy"
        cm_act: Channel Mix 的激活。"leaky_sq" 为 LeakyReLU(0.01) 的平方,
            保留负半轴梯度; "relu_sq" 为论文原样的 ReLU 平方。
    """

    def __init__(self, channels: int, num_directions: int = 4, hidden_rate: int = 4,
                 use_omni_shift: bool = True, use_channel_mix: bool = True,
                 wkv_mode: str = "legacy", cm_act: str = "leaky_sq"):
        super().__init__(channels, num_directions=num_directions,
                         hidden_rate=hidden_rate,
                         use_omni_shift=use_omni_shift,
                         use_channel_mix=use_channel_mix,
                         wkv_mode=wkv_mode)
        if cm_act not in ("leaky_sq", "relu_sq"):
            raise ValueError(f"cm_act 必须是 'leaky_sq' 或 'relu_sq', 收到 {cm_act!r}")
        self.cm_act = cm_act

        # 旧头的单查询投影与普通 MLP 式 Channel Mix 在这里不适用, 替换掉。
        del self.proj_q
        del self.channel_mix
        self.channel_mix = None

        # ---------- 四个 Q-Shift (论文式 2、3 各有两处) ----------
        if self.use_omni_shift:
            self.shift_r = OmniShift(channels)
            self.shift_g = OmniShift(channels)
            self.shift_k = OmniShift(channels)
            self.shift_v = OmniShift(channels)
        else:
            self.shift_r = self.shift_g = self.shift_k = self.shift_v = None

        # ---------- R/G 双查询 ----------
        self.proj_r = nn.Linear(channels, channels, bias=False)
        self.proj_g = nn.Linear(channels, channels, bias=True)

        # ---------- 逐头 LayerNorm ----------
        self.head_norm = nn.LayerNorm(self.head_dim)

        # ---------- Channel Mix: 式 (6)–(8) ----------
        hidden = channels * hidden_rate
        if self.use_channel_mix:
            # 论文式 (6)–(8) 的形状: R'、K' 为隐层维度, V' 投影回通道维度,
            # 门 σ(R') 与 V' 同为隐层维度, 最后 W_a 回通道维度。
            self.cm_r = nn.Linear(channels, hidden, bias=False)
            self.cm_k = nn.Linear(channels, hidden, bias=False)
            self.cm_v = nn.Linear(hidden, hidden, bias=False)
            self.cm_out = nn.Linear(hidden, channels, bias=False)
        else:
            self.cm_r = self.cm_k = self.cm_v = self.cm_out = None

        self._init_cross_weights()

    def _init_cross_weights(self):
        c = self.channels
        # R 的幅度会被逐头 LayerNorm 抹掉, 初始不放大即可
        nn.init.uniform_(self.proj_r.weight, -0.5 / math.sqrt(c), 0.5 / math.sqrt(c))
        nn.init.uniform_(self.proj_g.weight, -0.5 / math.sqrt(c), 0.5 / math.sqrt(c))
        # SiLU(0) = 0, G 若从 0 开始则整条 WKV 通路在第一步没有梯度。
        # 偏置设 1 → SiLU(1) ≈ 0.731, 门初始为开启状态。
        nn.init.constant_(self.proj_g.bias, 1.0)
        if self.use_channel_mix:
            nn.init.kaiming_uniform_(self.cm_r.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.cm_k.weight, a=math.sqrt(5))
            nn.init.kaiming_uniform_(self.cm_v.weight, a=math.sqrt(5))
            # 末层零初始化 → Channel Mix 初始为恒等, 但梯度第一步即可达末层
            nn.init.zeros_(self.cm_out.weight)

    def _project(self, x_2d: torch.Tensor, shift, proj):
        """Q-Shift → 归一化 → 线性投影。"""
        if shift is not None:
            x_2d = shift(x_2d)
        tokens = self.pre_norm(x_2d.flatten(2).transpose(1, 2))
        return proj(tokens)

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        """center_feat 生成 R/G, context_feat 生成 K/V。两者均为 (B, C, H, W)。"""
        B, C, H, W = center_feat.shape
        D = self.num_dirs
        hd = self.head_dim

        r = self._project(center_feat, self.shift_r, self.proj_r)
        g = self._project(center_feat, self.shift_g, self.proj_g)
        k = self._project(context_feat, self.shift_k, self.proj_k)
        v = self._project(context_feat, self.shift_v, self.proj_v)

        r = r.view(B, H * W, D, hd)
        g = torch.nn.functional.silu(g.view(B, H * W, D, hd))

        # 四个方向合并成一次扫描算子调用 (见 models/modules/batched_wkv.py)
        scan_fns = self._scan_fns()
        k_list, v_list = [], []
        for i, scan_fn in enumerate(scan_fns):
            k_2d = k[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            v_2d = v[:, :, i * hd:(i + 1) * hd].transpose(1, 2).reshape(B, hd, H, W)
            k_list.append(scan_fn(k_2d))
            v_list.append(scan_fn(v_2d))
        wkv_list = batched_scan(self.bi_wkv_list, k_list, v_list, total_tokens=H * W)

        heads = []
        for i, scan_fn in enumerate(scan_fns):
            wkv_i = wkv_list[i][:, self._inv_index(scan_fn, B, hd, H, W,
                                                   center_feat.device)]
            # 论文式 (4): R 先乘, 再逐头归一化, 最后由 SiLU(G) 门控
            heads.append(g[:, :, i] * self.head_norm(r[:, :, i] * wkv_i))
        out = torch.cat(heads, dim=-1)          # (B, N, C)

        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        if self.use_channel_mix:
            a = self.post_norm(out.flatten(2).transpose(1, 2))   # (B, N, C)
            r2 = self.cm_r(a)
            k2 = self.cm_k(a)
            if self.cm_act == "relu_sq":
                v2 = F.relu(k2).pow(2)
            else:
                v2 = F.leaky_relu(k2, 0.01).pow(2)
            a2 = torch.sigmoid(r2) * self.cm_v(v2)
            out = out + self.cm_out(a2).transpose(1, 2).reshape(B, C, H, W)

        return out


if __name__ == "__main__":
    B, C, H, W = 2, 128, 32, 32
    head = CrossRWKVHeadV8(C)
    y = head(torch.randn(B, C, H, W), torch.randn(B, C, H, W))
    n = sum(p.numel() for p in head.parameters())
    print(f"CrossRWKVHeadV8: {tuple(y.shape)}, params={n/1e3:.1f}K")
    y.sum().backward()
    dead = [name for name, p in head.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    print(f"无梯度参数: {dead if dead else '无'}")
