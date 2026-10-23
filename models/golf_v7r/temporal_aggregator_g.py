#!/usr/bin/env python3
"""TemporalAggregator_G: 双流共享 top-down 的多尺度聚合 (G = 改进, 用户命名 2026-10-23)。

> 规格原文与六项裁决见 docs/v8/11-TemporalAggregator_G设计.md。
> 未接入 golfnet_v8, 未训练 (裁决 6: 只实现不启动)。

## 重构范围 (用户界定)

frame_encoder 原样保留 (F3 逐帧 RWKV 编码, 不在重构范围); 本类重构的是它的输出与
F1_single / F1_multi 产生之间的接线:

1. 每尺度特征拆成两股: 中心帧 fi_t 与全帧均值 fi_agg (均值**含中心帧**, 裁决 4);
2. 上采样 Conv3x3 + PixelShuffle 打包为 `Up` 组件 (规格第 2 条);
3. 两股**共享同一组** Up / fuse (裁决 3), fuse = Conv3x3 + LayerNorm2d + SiLU
   (搬运 lite 的非线性, 裁决 1), 各走一遍 F3 → F2 → F1 的 top-down:
     agg 股 → F1_multi;   t 股 → F1_single (F2 级中间量 = F2_single)

## 与 lite 的差异 (模块集合与 lite 完全相同 → 参数量相同, 只是接线不同;
## state_dict 键名与 lite 对齐, 可直接热启动已训练 lite checkpoint 的聚合器权重)

| 项 | lite | G |
|----|------|---|
| frame_encoder (F3 逐帧 RWKV) | 有 | 原样保留 |
| 跨帧聚合 | 仅 F3 一处, 非中心帧均值 | 每尺度 (F1/F2/F3) 全 T 帧均值 (含中心) |
| multi 输出是否拼接中心帧 | 每级 fuse 拼 f_center | 否 (中心帧只经均值进入) |
| top-down 走几次 | 1 次 (只产 F1_multi) | 同一组参数走 2 次 (产 F1_multi 与 F1_single) |
| F1_single | 原始 f1_t (不经 top-down) | t 股 top-down 输出 (论文 §3.3 "same top-down path" 成立) |
| 邻帧的 F1/F2 编码特征 | 死计算 | 被每尺度均值消费 |

## 结构性质 (冒烟断言, 见 __main__)

1. F1_single / F2_single 对邻帧扰动逐位不变 (t 股只消费中心帧);
2. F1_multi 对中心帧与邻帧的扰动都响应 (均值含中心, 裁决 4);
3. T 帧输入相同时 F1_multi ≈ F1_single (两股同路同输入, 裁决 3 的直接推论)。
"""
from __future__ import annotations

import torch
import torch.nn as nn

from models.modules.blocks import LayerNorm2d


class Up(nn.Module):
    """2 倍上采样组件: Conv3x3 把通道扩到 4 倍, PixelShuffle 重排到空间维。

    与 lite 的 UpsampleConvShuffle 同构同名 (proj / shuffle), 保证 checkpoint 键名兼容。
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class TemporalAggregator_G(nn.Module):
    """双流共享 top-down 聚合: 全帧均值股产 F1_multi, 中心帧股产 F1_single。

    Args:
        encoder_channels: [C1, C2, C3] 三层金字塔通道数
        num_frames: 帧数 T (仅记录; 中心帧位置由实际输入 T 推出)
        wkv_mode: 传给 frame_encoder, 与 lite 一致
    """

    def __init__(self, encoder_channels, num_frames: int = 5,
                 wkv_mode: str = "single_norm"):
        super().__init__()
        C1, C2, C3 = encoder_channels
        self.C1, self.C2, self.C3 = C1, C2, C3
        self.num_frames = num_frames

        # F3 逐帧编码: lite 原样保留 (重构范围之外, 裁决 2)
        from models.golf_v7r.rwkv_encoder_block import RWKVEncoderBlock
        self.frame_encoder = RWKVEncoderBlock(C3, wkv_mode=wkv_mode)

        # 共享 top-down (两股经过同一组参数): 与 lite 同构, 键名对齐
        self.up3_to_2 = Up(C3, C2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(C2 * 2, C2, 3, 1, 1, bias=True),
            LayerNorm2d(C2),
            nn.SiLU(inplace=True),
        )
        self.up2_to_1 = Up(C2, C1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(C1 * 2, C1, 3, 1, 1, bias=True),
            LayerNorm2d(C1),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def split_streams(seq: torch.Tensor, center: int):
        """拆出 (中心帧特征 fi_t, 全帧均值 fi_agg)。均值含中心帧 (裁决 4)。"""
        return seq[:, center], seq.mean(dim=1)

    def _top_down(self, f3: torch.Tensor, f2: torch.Tensor, f1: torch.Tensor):
        """共享路径走一遍 F3 → F2 → F1。返回 (F1 级输出, F2 级中间量)。"""
        s = self.up3_to_2(f3)
        s = self.fuse2(torch.cat([s, f2], dim=1))
        mid = s
        s = self.up2_to_1(s)
        s = self.fuse1(torch.cat([s, f1], dim=1))
        return s, mid

    def forward(self, f1_seq: torch.Tensor, f2_seq: torch.Tensor,
                f3_seq: torch.Tensor) -> dict:
        """
        f1_seq: [B, T, C1, H, W]
        f2_seq: [B, T, C2, H/2, W/2]
        f3_seq: [B, T, C3, H/4, W/4]
        """
        B, T = f1_seq.shape[:2]
        center = T // 2

        # frame_encoder: F3 逐帧编码 (全部 T 帧), lite 原样保留
        f3_flat = f3_seq.reshape(B * T, *f3_seq.shape[2:])
        f3_enc = self.frame_encoder(f3_flat).reshape(B, T, *f3_seq.shape[2:])

        # 每尺度拆两股: 中心帧 _t 与 全帧均值 _agg
        f3_t, f3_agg = self.split_streams(f3_enc, center)
        f2_t, f2_agg = self.split_streams(f2_seq, center)
        f1_t, f1_agg = self.split_streams(f1_seq, center)

        # 两股各走一遍同一组 top-down (共享参数, 裁决 3)
        F1_multi, _ = self._top_down(f3_agg, f2_agg, f1_agg)
        F1_single, F2_single = self._top_down(f3_t, f2_t, f1_t)

        return {
            "F1_multi": F1_multi,
            "F1_single": F1_single,
            "F2_single": F2_single,
        }


if __name__ == "__main__":
    torch.manual_seed(0)
    B, T = 2, 5
    ch = [32, 64, 128]
    H = W = 32
    net = TemporalAggregator_G(ch, num_frames=T)
    f1 = torch.randn(B, T, ch[0], H, W)
    f2 = torch.randn(B, T, ch[1], H // 2, W // 2)
    f3 = torch.randn(B, T, ch[2], H // 4, W // 4)

    out = net(f1, f2, f3)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    n = sum(p.numel() for p in net.parameters())
    print(f"  params = {n:,}  (lite 为 662,948, 模块集合相同应一致)")
    assert n == 662_948, f"参数量与 lite 不一致: {n}"

    c = T // 2

    # 性质 1: 扰动邻帧 → F1_single / F2_single 逐位不变; F1_multi 必须响应
    f1n, f2n, f3n = f1.clone(), f2.clone(), f3.clone()
    for i in range(T):
        if i != c:
            f1n[:, i] += 10.0
            f2n[:, i] += 10.0
            f3n[:, i] += 10.0
    out_n = net(f1n, f2n, f3n)
    assert torch.equal(out["F1_single"], out_n["F1_single"]), "F1_single 受邻帧影响!"
    assert torch.equal(out["F2_single"], out_n["F2_single"]), "F2_single 受邻帧影响!"
    assert not torch.equal(out["F1_multi"], out_n["F1_multi"]), "F1_multi 对邻帧无响应!"

    # 性质 2: 扰动中心帧 → F1_multi 也必须响应 (均值含中心, 裁决 4)
    f1b, f2b, f3b = f1.clone(), f2.clone(), f3.clone()
    f1b[:, c] += 10.0
    f2b[:, c] += 10.0
    f3b[:, c] += 10.0
    out_b = net(f1b, f2b, f3b)
    assert not torch.equal(out["F1_multi"], out_b["F1_multi"]), "F1_multi 对中心帧无响应!"
    assert not torch.equal(out["F1_single"], out_b["F1_single"]), "F1_single 对中心帧无响应!"

    # 性质 3: T 帧相同输入 → 两股同路同输入, F1_multi ≈ F1_single
    f1s = f1[:, c:c + 1].repeat(1, T, 1, 1, 1)
    f2s = f2[:, c:c + 1].repeat(1, T, 1, 1, 1)
    f3s = f3[:, c:c + 1].repeat(1, T, 1, 1, 1)
    out_s = net(f1s, f2s, f3s)
    d = (out_s["F1_multi"] - out_s["F1_single"]).abs().max().item()
    print(f"  同帧输入 |F1_multi - F1_single|max = {d:.2e}")
    assert d < 1e-4, "同帧输入下两股输出差异超出容差!"

    (out["F1_multi"].sum() + out["F1_single"].sum()
     + out["F2_single"].sum()).backward()
    dead = [nm for nm, p in net.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    print(f"  反传正常; 无梯度参数: {dead if dead else '无'}")
    print("  结构性质断言全部通过")
