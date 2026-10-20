#!/usr/bin/env python3
"""v8.5 A 方案: 轻量版 TemporalAggregator。

## 相对原 TemporalAggregator 的改动

| 项 | 原实现 | 本实现 |
|----|--------|--------|
| 帧描述子 | 由 `frame_encoder` 输出全局平均池化得到 | 删除 |
| 帧权重 | 帧级 RWKV 块 + 打分网络, softmax 归一化 | 删除, 改为固定均值 |
| F3 聚合 | 按权重对 5 帧加权求和 | 非中心帧的算术平均 |
| `frame_encoder` | 保留 | 保留, 语义不变 |
| 尺度变换 | `Conv3x3` + `F.interpolate` 双线性 | `Conv3x3` + `PixelShuffle` |
| 跨尺度融合 | `up3_to_2 → fuse2 → up2_to_1 → fuse1` | 完全保留, 连接与通道不变 |
| 单帧旁路 | `F1_single = f1_seq[:, center]` | 完全保留 |

## 为什么可以删除帧权重

实测 24 个验证窗口, 训练完成的模型给出的帧权重全部是 0.2000, 标准差为 0, 窗口内最大值与
最小值之差为 0。原因是 F3 尺度上帧间特征差异只有 4.8e-6, 全局平均池化后的帧描述子几乎
完全相同, softmax 自然给出等权。因此"学习式加权"在当前结构下等价于算术平均, 保留它只
增加参数与计算, 不提供信息。

需要说明的是, 这不代表"加权聚合"与"均值聚合"在数学上等价。加权路径上还叠了一次空间
RWKV 编码 (`frame_encoder`) 的非线性, 本实现保留了这个编码, 只去掉打分与加权。

## 为什么把差值上采样换成 Conv + PixelShuffle

`F.interpolate(mode="bilinear")` 是固定的双线性插值, 不参与学习。换成 PixelShuffle 后,
上采样的权重由前面的卷积学习得到, 表达能力更强。代价是参数增加: 原来的
`Conv2d(C3, C2, 3)` 变为 `Conv2d(C3, 4*C2, 3)` + `PixelShuffle(2)`。

## 中心帧的处理

`F1_multi` 的构造是 `fuse1(cat[up(F2_multi), f1_seq[:, center]])`, 即中心帧在每一级融合时
都会参与拼接。因此"中心帧是语义锚点"这一点保持不变; 本实现只在 F3 的初始聚合上排除中心帧,
避免中心帧先被计入一次平均、又在融合时被拼接一次。
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import LayerNorm2d


class UpsampleConvShuffle(nn.Module):
    """2 倍上采样: 3×3 卷积把通道扩到 4 倍, 再由 PixelShuffle 重排到空间维。

    Args:
        in_ch: 输入通道数
        out_ch: 输出通道数
    """

    def __init__(self, in_ch: int, out_ch: int):
        super().__init__()
        self.proj = nn.Conv2d(in_ch, out_ch * 4, 3, 1, 1, bias=False)
        self.shuffle = nn.PixelShuffle(2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.shuffle(self.proj(x))


class TemporalAggregatorLite(nn.Module):
    """轻量版多尺度聚合: 非中心帧均值 + 两级上采样融合 + 中心帧拼接。

    对外接口与原 `TemporalAggregator` 完全一致, 返回 `F1_multi` / `F1_single` /
    `F2_single` 三个键, 供 `TaskQueryAttention` 按原有接法使用。

    Args:
        encoder_channels: [C1, C2, C3] 三层金字塔通道数
        num_frames: 帧数 T
        wkv_mode: 仅传递给 `frame_encoder`, 与原实现一致
    """

    def __init__(self, encoder_channels, num_frames: int = 5,
                 wkv_mode: str = "single_norm"):
        super().__init__()
        C1, C2, C3 = encoder_channels
        self.C1, self.C2, self.C3 = C1, C2, C3
        self.num_frames = num_frames

        # 逐帧空间编码: 与原实现相同, 保留
        from models.golf_v7r.rwkv_encoder_block import RWKVEncoderBlock
        self.frame_encoder = RWKVEncoderBlock(C3, wkv_mode=wkv_mode)

        # 自顶向下融合: 尺度变换改为 Conv + PixelShuffle
        self.up3_to_2 = UpsampleConvShuffle(C3, C2)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(C2 * 2, C2, 3, 1, 1, bias=True),
            LayerNorm2d(C2),
            nn.SiLU(inplace=True),
        )
        self.up2_to_1 = UpsampleConvShuffle(C2, C1)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(C1 * 2, C1, 3, 1, 1, bias=True),
            LayerNorm2d(C1),
            nn.SiLU(inplace=True),
        )

    @staticmethod
    def aggregate_neighbors(f3_enc: torch.Tensor, center: int) -> torch.Tensor:
        """对非中心帧的 F3 编码特征取算术平均。

        f3_enc: [B, T, C3, H/4, W/4] -> [B, C3, H/4, W/4]
        """
        T = f3_enc.shape[1]
        idx = [i for i in range(T) if i != center]
        if not idx:
            raise ValueError("至少需要一帧非中心帧才能做聚合")
        return f3_enc[:, idx].mean(dim=1)

    def forward(self, f1_seq: torch.Tensor, f2_seq: torch.Tensor,
                f3_seq: torch.Tensor) -> dict:
        """
        f1_seq: [B, T, C1, H, W]
        f2_seq: [B, T, C2, H/2, W/2]
        f3_seq: [B, T, C3, H/4, W/4]
        """
        B, T = f1_seq.shape[:2]
        center = T // 2

        # 逐帧 F3 编码 (保留原 frame_encoder)
        f3_flat = f3_seq.reshape(B * T, *f3_seq.shape[2:])
        f3_enc = self.frame_encoder(f3_flat).reshape(B, T, *f3_seq.shape[2:])

        # 非中心帧均值聚合 (替代原来的学习式加权)
        f3_agg = self.aggregate_neighbors(f3_enc, center)

        # 自顶向下融合: 与原来相同的拼接顺序与通道设计
        up2 = self.up3_to_2(f3_agg)
        F2_multi = self.fuse2(torch.cat([up2, f2_seq[:, center]], dim=1))
        up1 = self.up2_to_1(F2_multi)
        F1_multi = self.fuse1(torch.cat([up1, f1_seq[:, center]], dim=1))

        # 单帧旁路: 与原来相同, 不经任何跨帧运算
        F1_single = f1_seq[:, center]
        F2_single = f2_seq[:, center]

        return {
            "F1_multi": F1_multi,
            "F1_single": F1_single,
            "F2_single": F2_single,
        }


if __name__ == '__main__':
    dev = 'cuda' if torch.cuda.is_available() else 'cpu'
    B, T, ch = 2, 5, [32, 64, 128]
    net = TemporalAggregatorLite(ch, num_frames=T).to(dev)
    f1 = torch.randn(B, T, ch[0], 32, 32, device=dev)
    f2 = torch.randn(B, T, ch[1], 16, 16, device=dev)
    f3 = torch.randn(B, T, ch[2], 8, 8, device=dev)
    out = net(f1, f2, f3)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"  params = {sum(p.numel() for p in net.parameters())/1e6:.3f}M")
    out["F1_multi"].sum().backward()
    print("  反传正常")
