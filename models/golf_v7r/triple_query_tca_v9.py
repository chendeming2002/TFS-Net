#!/usr/bin/env python3
"""v8 TCA (F1 尺度): 三源查询在**全分辨率**上工作, 且三路输入来源不同。

## 相对旧版 TripleQueryTCA 的两处结构变化

1. **尺度从 F2 移到 F1**。旧版在 H/2 尺度上做注意力, 所有细节都要先被下采样丢掉再学回来;
   而 M 支路要建模的"局部结构偏移"恰恰存在于最高频带上。移到 F1 后三路注意力直接在全分辨率
   上工作, 配合 WKV 的线性复杂度, 计算量可控 (L=H·W=65536 时每次扫描约 3.4 ms)。

2. **输入按支路来源分流, 不再共用同一张特征图**:
   - N、L: 用 U-Net 聚合出的**多帧**特征 (F1_multi), 时序平均已抑制 i.i.d. 噪声;
   - M: 用**中心帧**特征 (F1_single), 完全不经跨帧运算。
   KV 也按各自的物理含义构造:
   - N 的 KV 来自多帧特征的帧间统计 (低通后的均值);
   - L 的 KV 来自多帧特征的低频趋势;
   - M 的 KV 来自单帧特征的高频残差 (单帧减去其低通)。

这样分工发生在数据流上, 不依赖损失函数把三路"逼"成不同 —— 这是前几轮反复失败的症结
(见 docs/v8/01-设计.md §六·补·续 与 03-M支路重构设计.md)。

## 与旧版的兼容性

旧类 `TripleQueryTCA` 原样保留 (它的 24 个历史 run 依赖其行为)。本类是新结构专用,
接口不同: 输入是 (F1_multi, F1_single), 而不是 (feat_spatial, feats_seq, rwkv_ctx)。
"""
from __future__ import annotations

from typing import Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import LayerNorm2d
from models.golf_r4.tca_rwkv import RWKVSpatialHead
from models.golf_v7r.spatial_rwkv_head_v8 import SpatialRWKVHeadV8
from models.golf_v7r.cross_rwkv_head_v8 import CrossRWKVHeadV8
from models.golf_v7r.qrb_variants import (
    CrossRWKVHeadKvMod, CrossRWKVHeadQPath, CrossRWKVHeadGateCtx)


class TaskQueryAttention(nn.Module):
    """F1 尺度三源查询 TCA: 多帧 (N、L) + 单帧 (M)。

    Args:
        feat_dim: F1 特征通道数 (与 encoder_channels[0] 一致)
        spatial_head: "v8" (OmniShift + Channel Mix) 或 "legacy"
        wkv_mode: "single_norm" (官方双向定义 + CUDA 算子) 或 "legacy"
        lowpass_kernel: 低频分离用的平均池化核 (M 的高频残差与 L 的低频趋势共用)
        q_mode: "receptance" 为旧的 sigmoid(q) 门控; "cross_rwkv" 使用
            CrossRWKVHeadV8, R/G 直接接收特征图 (N/L 传多帧聚合, M 传中心帧),
            不再经过 query_* 卷积。见 docs/v8/06-CrossRWKVGate-Q重构设计.md。
            另支持三个单变量变体, 见 docs/v8/12-QRB改进方向评估.md:
            "kvmod"  Q 条件化 K 调制; "qpath"  受控 Q 内容支路; "gatectx" R/G 局部上下文
    """

    def __init__(self, feat_dim: int = 32, spatial_head: str = "v8",
                 wkv_mode: str = "single_norm", lowpass_kernel: int = 9,
                 num_directions: int = 4, channel_mix_rate: int = 4,
                 use_omni_shift: bool = True, use_channel_mix: bool = True,
                 q_mode: str = "receptance"):
        super().__init__()
        self.feat_dim = feat_dim
        self.lowpass_kernel = int(lowpass_kernel)
        if spatial_head not in ("legacy", "v8"):
            raise ValueError(f"spatial_head 必须是 'legacy' 或 'v8', 收到 {spatial_head!r}")
        if q_mode not in ("receptance", "cross_rwkv", "kvmod", "qpath", "gatectx"):
            raise ValueError(
                f"q_mode 必须是 receptance/cross_rwkv/kvmod/qpath/gatectx, 收到 {q_mode!r}")
        if q_mode != "receptance" and spatial_head != "v8":
            raise ValueError("cross_rwkv 查询只实现于 spatial_head='v8'")
        self.spatial_head_kind = spatial_head
        self.q_mode = q_mode

        # ---------- 三路 Query: N/L 来自多帧, M 来自单帧 ----------
        # cross_rwkv 模式下 R/G 直接接收特征图, 由 CrossRWKVHeadV8 内部的 Q-Shift 处理,
        # 这里不再需要额外的 1x1 卷积。
        if q_mode == "receptance":
            self.query_N = nn.Sequential(
                LayerNorm2d(feat_dim), nn.Conv2d(feat_dim, feat_dim, 1, bias=False))
            self.query_L = nn.Sequential(
                LayerNorm2d(feat_dim), nn.Conv2d(feat_dim, feat_dim, 1, bias=False))
            self.query_M = nn.Sequential(
                LayerNorm2d(feat_dim), nn.Conv2d(feat_dim, feat_dim, 1, bias=False))
        else:
            self.query_N = self.query_L = self.query_M = None

        # ---------- 三路 KV: 来源各不相同 ----------
        # N: 多帧特征的低通 (帧间平均后的主成分, 对应 i.i.d. 噪声被抑制后的结果)
        self.kv_proj_N = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1, bias=True), LayerNorm2d(feat_dim))
        # L: 多帧特征的低频趋势 (更大的核)
        self.kv_proj_L = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1, bias=True), LayerNorm2d(feat_dim))
        # M: 单帧特征的高频残差 (单帧 - 其低通)
        self.kv_proj_M = nn.Sequential(
            nn.Conv2d(feat_dim, feat_dim, 1, bias=True), LayerNorm2d(feat_dim))

        # ---------- 三路 RWKV 空间注意力头 ----------
        if spatial_head == "v8":
            head_kwargs = {"hidden_rate": channel_mix_rate,
                           "use_omni_shift": use_omni_shift,
                           "use_channel_mix": use_channel_mix,
                           "num_directions": num_directions,
                           "wkv_mode": wkv_mode}
            head_cls = {"cross_rwkv": CrossRWKVHeadV8,
                        "kvmod": CrossRWKVHeadKvMod,
                        "qpath": CrossRWKVHeadQPath,
                        "gatectx": CrossRWKVHeadGateCtx}.get(q_mode, SpatialRWKVHeadV8)
        else:
            head_kwargs = {"num_directions": num_directions, "wkv_mode": wkv_mode}
            head_cls = RWKVSpatialHead
        self.attn_N = head_cls(feat_dim, **head_kwargs)
        self.attn_L = head_cls(feat_dim, **head_kwargs)
        self.attn_M = head_cls(feat_dim, **head_kwargs)

        # ---------- LayerScale 残差 (零初始化 → 初始恒等) ----------
        self.scale_N = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.scale_L = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))
        self.scale_M = nn.Parameter(torch.zeros(1, feat_dim, 1, 1))

        # ---------- 输出归一化 ----------
        self.out_norm_N = LayerNorm2d(feat_dim)
        self.out_norm_L = LayerNorm2d(feat_dim)
        self.out_norm_M = LayerNorm2d(feat_dim)

    def _lowpass(self, x: torch.Tensor, kernel: int | None = None) -> torch.Tensor:
        k = int(kernel or self.lowpass_kernel)
        pad = k // 2
        return F.avg_pool2d(x, kernel_size=k, stride=1, padding=pad)

    def forward(self, F1_multi: torch.Tensor, F1_single: torch.Tensor
                ) -> Tuple[torch.Tensor, ...]:
        """
        F1_multi:  [B, C, H, W]  多帧完整聚合 (U-Net 输出)
        F1_single: [B, C, H, W]  中心帧特征 (U-Net 内旁路)

        Returns: F_N, F_L, F_M ([B, C, H, W]), ortho_loss (scalar)
        """
        # ---------- 三路 Query (R/G) ----------
        # cross_rwkv: R/G 直接由特征图产生, Q-Shift 在头内部完成。
        #   L: 多帧聚合特征 (光照是低频、跨帧稳定, 平均后更干净)
        #   M: 中心帧特征 (运动是单帧内的局部结构偏移, 不做跨帧运算)
        #   N: 单帧减多帧聚合。这个差值就是帧间不一致量, 物理上对应成像噪声:
        #      场景结构在帧间稳定, 相减后保留下来的是各帧独立的噪声及其相位,
        #      而多帧聚合本身已经把噪声平均掉, 用它做查询等于问一个看不到噪声的信号。
        if self.q_mode != "receptance":
            noise_query = F1_single - F1_multi
            Q_N, Q_L, Q_M = noise_query, F1_multi, F1_single
        else:
            Q_N = self.query_N(F1_multi)
            Q_L = self.query_L(F1_multi)
            Q_M = self.query_M(F1_single)

        # ---------- 三路 KV ----------
        # cross_rwkv: 全带 KV, 不在检索入口做固定滤波。频带分工交由目标与输出侧,
        # 不再把内容池在注意力之前切掉一块 (见 docs/v8/07)。
        #   N: 多帧聚合。噪声是帧间独立量, 它的内容池是多帧平均后的结果。
        #   L: 中心帧。光照是当前帧自身的属性, 多帧平均会把曝光变化抹平,
        #      与要预测的中心帧光照图错位。R/G 仍取多帧聚合, 提供跨帧结构参照。
        #   M: 中心帧。与它不引入跨帧运算的约定一致。
        if self.q_mode != "receptance":
            kv_N = self.kv_proj_N(F1_multi)
            kv_L = self.kv_proj_L(F1_single)
            kv_M = self.kv_proj_M(F1_single)
        else:
            multi_low = self._lowpass(F1_multi)
            single_low = self._lowpass(F1_single)
            kv_N = self.kv_proj_N(multi_low)
            kv_L = self.kv_proj_L(self._lowpass(
                F1_multi, kernel=max(self.lowpass_kernel * 3, 9)))
            kv_M = self.kv_proj_M(F1_single - single_low)

        # ---------- 三路注意力 ----------
        attn_N = self.attn_N(Q_N, kv_N)
        attn_L = self.attn_L(Q_L, kv_L)
        attn_M = self.attn_M(Q_M, kv_M)

        # ---------- 残差 + 归一化 ----------
        # 残差基按支路来源选取:
        #   N: 以多帧聚合为基。查询是单帧与聚合的差值, 即噪声估计, 把它加回
        #      已经去噪的聚合结果上, 正好是"干净结构 + 噪声"的组合。
        #   L: 全带 KV 时以中心帧为基。注意力在中心帧的光照内容上检索, 结果应加回
        #      同一来源, 否则等于把中心帧的光照修正量叠加到多帧平均上, 两处位移不一致。
        #   M: 以单帧为基, 避免"注意力算的是单帧、残差却加多帧"的混用。
        L_base = F1_single if self.q_mode != "receptance" else F1_multi
        F_N = self.out_norm_N(F1_multi + attn_N * self.scale_N)
        F_L = self.out_norm_L(L_base + attn_L * self.scale_L)
        F_M = self.out_norm_M(F1_single + attn_M * self.scale_M)

        ortho_loss = self._ortho_loss(F_N, F_L, F_M)
        return F_N, F_L, F_M, ortho_loss

    @staticmethod
    def _ortho_loss(F_N: torch.Tensor, F_L: torch.Tensor,
                    F_M: torch.Tensor) -> torch.Tensor:
        """特征层正交约束 (辅助项, 权重很小)。"""
        f_N = F.normalize(F_N.flatten(2), dim=1)
        f_L = F.normalize(F_L.flatten(2), dim=1)
        f_M = F.normalize(F_M.flatten(2), dim=1)
        cos_NL = (f_N * f_L).sum(dim=1).abs().mean()
        cos_NM = (f_N * f_M).sum(dim=1).abs().mean()
        cos_LM = (f_L * f_M).sum(dim=1).abs().mean()
        return (cos_NL + cos_NM + cos_LM) / 3.0


if __name__ == '__main__':
    import torch as _t
    dev = 'cuda' if _t.cuda.is_available() else 'cpu'
    B, C, H, W = 2, 32, 64, 64
    tca = TaskQueryAttention(C).to(dev)
    multi = _t.randn(B, C, H, W, device=dev)
    single = _t.randn(B, C, H, W, device=dev)
    F_N, F_L, F_M, ortho = tca(multi, single)
    print(f"F_N/L/M: {tuple(F_N.shape)}  ortho={ortho.item():.4f}")
    print(f"params = {sum(p.numel() for p in tca.parameters())/1e6:.3f}M")
    (F_N.sum() + F_L.sum() + F_M.sum() + ortho).backward()
    print("反传正常")


# 旧名别名: 本类在历史 run 与文档中记为 TripleQueryTCAv9。
TripleQueryTCAv9 = TaskQueryAttention
