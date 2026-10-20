#!/usr/bin/env python3
"""v8 U-Net 式空间聚合: 把 SpatialSummary + TripleQueryTCA 重构成"编码器-解码器"式结构。

## 为什么这样改

原结构里有三段互相重叠的处理, 职责不清:

1. `SpatialSummary` 把 F2 压成 [B, T, D] 的**帧级 token** (空间信息被池化抹平);
2. `MatrixRWKVBlock` 堆 2 层, 只在这串 token 上做时序建模, 结果 `ctx` 仅用于给 KV 做门控;
3. `TripleQueryTCA` 又在 F2 尺度上重新建 KV (均值 / 低频 / 差分), 与 1、2 的产物几乎无关。

三段都在算"多帧该怎么汇总", 但只有第 3 段真正进入后续分支。第 1、2 段算出的东西
基本是死计算 —— 合并实验里 gate 诊断也印证了这一点 (gate 的 N/L 分量在训练中几乎不动)。

新结构把它们合并成一条自上而下的聚合链 (F3 → F2 → F1), 只有一个职责:
**为三路 Query 准备空间特征, 并把"单帧"与"多帧"两种输入分开**。

## 结构

```
逐帧编码 (SharedEncoder, 权重共享)
   每个 t: X_t → f1 [C1, H, W], f2 [C2, H/2, W/2], f3 [C3, H/4, W/4]

多帧支路 (N、L 共用, 在 U-Net 里完成"完整聚合"):
   f3_seq ──逐帧 RWKV Block (循环 M 次)──> 帧描述子 ──时序 RWKV──> 逐帧权重
        └─ 按权重对各帧 f3 加权求和 → F3_agg [C3, H/4, W/4]
   F3_agg ─上采样+融合 f2_center─> F2_agg ─上采样+融合 f1_center─> F1_agg [C1, H, W]

单帧支路 (M 用):
   F1_single = f1_center (中心帧的 F1 特征, 不经任何时序聚合)
   F2_single = f2_center (F2 尺度上的单帧特征, 供 TCA 的 M 路计算统计量)
```

关键点: **单帧支路在 U-Net 内部直接旁路得出, 不经过任何跨帧运算**。这对应 TSDR 里
"运动表现为单帧内的局部结构偏移"这一描述, 也是之前 M 支路重构 (docs/v8/03) 的延续。

## 与旧结构的对应

| 旧组件 | 新位置 |
|--------|--------|
| `SpatialSummary` (池化成帧 token) | 删除; 改为逐帧 RWKV 块直接处理空间特征 |
| `MatrixRWKVBlock` ×2 (帧 token 时序建模) | 保留为 U-Net 内的"逐帧描述子 → 时序权重"部分, 改为单模块循环 M 次 |
| `TripleQueryTCA` 的统计 KV (均值/低频/差分) | 保留, 改为在 F1 尺度计算, 输入换成 F1_single / F1_agg |
| `PixelTemporalAttentionSimple` (F2 像素对齐) | 删除; 其作用被 U-Net 的加权聚合替代 |
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import LayerNorm2d


class TemporalWeighting(nn.Module):
    """把逐帧描述子变成帧间权重, 并对空间特征做加权聚合。

    用帧级 RWKV 在 T 维上建模 (对应原 MatrixRWKVBlock 的作用), 输出每帧一个标量权重。
    权重经 softmax 归一化, 因此聚合结果是各帧特征的凸组合 —— 不会放大噪声,
    这与"时序平均抑制 i.i.d. 噪声"的物理直觉一致。
    """

    def __init__(self, dim: int, num_heads: int = 4, num_loops: int = 2,
                 wkv_mode: str = "single_norm"):
        super().__init__()
        from models.golf_v7r.matrix_rwkv import MatrixRWKVBlock
        # 单模块循环 M 次 (Re-WKV 的做法: 参数不变, 计算 ×M),
        # 区别于"堆 N 个独立 block"(参数与计算都 ×N)。
        self.block = MatrixRWKVBlock(dim=dim, num_heads=num_heads,
                                     head_size=dim // num_heads)
        self.num_loops = int(num_loops)
        self.scorer = nn.Sequential(
            nn.Linear(dim, dim // 2),
            nn.GELU(),
            nn.Linear(dim // 2, 1),
        )

    def forward(self, frame_tokens: torch.Tensor,
                feats_seq: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        frame_tokens: [B, T, D]  每帧的描述子
        feats_seq:    [B, T, C, H, W]  每帧的待聚合特征

        Returns:
            aggregated: [B, C, H, W]  按权重聚合后的特征
            weights:    [B, T]       每帧权重 (和为 1)
        """
        ctx = frame_tokens
        for _ in range(self.num_loops):
            ctx = self.block(ctx)
        logits = self.scorer(ctx).squeeze(-1)          # [B, T]
        weights = F.softmax(logits, dim=1)             # [B, T], 和为 1
        # 加权求和: [B, 1, T, 1, 1] * [B, T, C, H, W] -> sum over T
        aggregated = (feats_seq * weights[:, :, None, None, None]).sum(dim=1)
        return aggregated, weights


class TemporalAggregator(nn.Module):
    """自上而下的空间聚合 (F3 → F2 → F1), 同时给出单帧旁路特征。

    Args:
        encoder_channels: [C1, C2, C3] 三层金字塔通道数 (与 SharedEncoder 一致)
        num_frames: 帧数 T
        num_loops: 帧时序建模模块的循环次数 M
        wkv_mode: WKV 实现 ("legacy" / "single_norm")
    """

    def __init__(self, encoder_channels, num_frames: int = 5,
                 num_loops: int = 2, wkv_mode: str = "single_norm"):
        super().__init__()
        C1, C2, C3 = encoder_channels
        self.C1, self.C2, self.C3 = C1, C2, C3
        self.num_frames = num_frames

        # ---- 逐帧空间编码 (每帧独立, 对应旧 SpatialSummary 的替代品) ----
        # 用空间 RWKV 块在 F3 上提取每帧的紧凑描述; 不池化, 保留空间结构。
        from models.golf_v7r.rwkv_encoder_block import RWKVEncoderBlock
        self.frame_encoder = RWKVEncoderBlock(C3, wkv_mode=wkv_mode)

        # 帧描述子: 把 [B, C3, H/4, W/4] 池化成 [B, D]
        self.descriptor = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
        )
        self.to_token = nn.Linear(C3, C3)

        # ---- 时序加权聚合 (N、L 两支路共用的"完整聚合") ----
        self.temporal = TemporalWeighting(C3, num_heads=4 if C3 % 4 == 0 else 1,
                                          num_loops=num_loops, wkv_mode=wkv_mode)

        # ---- 自顶向下融合: F3 → F2 → F1 ----
        # 每级: 上采样聚合结果 + 与中心帧的对应尺度特征拼接 → 卷积融合
        self.up3_to_2 = nn.Conv2d(C3, C2, 3, 1, 1, bias=False)
        self.fuse2 = nn.Sequential(
            nn.Conv2d(C2 * 2, C2, 3, 1, 1, bias=True),
            LayerNorm2d(C2),
            nn.SiLU(inplace=True),
        )
        self.up2_to_1 = nn.Conv2d(C2, C1, 3, 1, 1, bias=False)
        self.fuse1 = nn.Sequential(
            nn.Conv2d(C1 * 2, C1, 3, 1, 1, bias=True),
            LayerNorm2d(C1),
            nn.SiLU(inplace=True),
        )

    def forward(self, f1_seq: torch.Tensor, f2_seq: torch.Tensor,
                f3_seq: torch.Tensor) -> dict:
        """
        f1_seq: [B, T, C1, H, W]
        f2_seq: [B, T, C2, H/2, W/2]
        f3_seq: [B, T, C3, H/4, W/4]

        Returns dict:
            F1_multi:  [B, C1, H, W]      多帧完整聚合 (N、L 使用)
            F1_single: [B, C1, H, W]      中心帧单帧特征 (M 使用)
            F2_single: [B, C2, H/2, W/2]  中心帧 F2 特征 (TCA 统计量用)
            frame_weights: [B, T]          每帧被采纳的权重 (诊断用)
        """
        B, T = f1_seq.shape[:2]
        center = T // 2

        # ========== 1. 逐帧空间编码 (仅 F3, 计算量最小) ==========
        f3_flat = f3_seq.reshape(B * T, *f3_seq.shape[2:])
        f3_enc = self.frame_encoder(f3_flat).reshape(B, T, *f3_seq.shape[2:])

        # ========== 2. 帧描述子 + 时序权重 ==========
        tokens = self.to_token(self.descriptor(f3_enc.reshape(B * T, *f3_enc.shape[2:])))
        tokens = tokens.reshape(B, T, -1)                    # [B, T, C3]
        f3_agg, frame_w = self.temporal(tokens, f3_enc)      # [B, C3, h3, w3]

        # ========== 3. 自顶向下融合 (多帧支路) ==========
        up2 = F.interpolate(self.up3_to_2(f3_agg), size=f2_seq.shape[-2:],
                            mode="bilinear", align_corners=False)
        F2_multi = self.fuse2(torch.cat([up2, f2_seq[:, center]], dim=1))

        up1 = F.interpolate(self.up2_to_1(F2_multi), size=f1_seq.shape[-2:],
                            mode="bilinear", align_corners=False)
        F1_multi = self.fuse1(torch.cat([up1, f1_seq[:, center]], dim=1))

        # ========== 4. 单帧旁路 (M 支路), 不经过任何跨帧运算 ==========
        F1_single = f1_seq[:, center]
        F2_single = f2_seq[:, center]

        return {
            "F1_multi": F1_multi,
            "F1_single": F1_single,
            "F2_single": F2_single,
            "frame_weights": frame_w,
        }


if __name__ == '__main__':
    import torch as _t
    B, T = 2, 5
    ch = [32, 64, 128]
    dev = 'cuda' if _t.cuda.is_available() else 'cpu'
    net = TemporalAggregator(ch, num_frames=T).to(dev)
    f1 = _t.randn(B, T, ch[0], 32, 32, device=dev)
    f2 = _t.randn(B, T, ch[1], 16, 16, device=dev)
    f3 = _t.randn(B, T, ch[2], 8, 8, device=dev)
    out = net(f1, f2, f3)
    for k, v in out.items():
        print(f"  {k}: {tuple(v.shape)}")
    print(f"  params = {sum(p.numel() for p in net.parameters())/1e6:.3f}M")
    out["F1_multi"].sum().backward()
    print("  反传正常")


# 旧名别名: 历史文档与实验记录使用 SpatialSummaryUNet。
SpatialSummaryUNet = TemporalAggregator
