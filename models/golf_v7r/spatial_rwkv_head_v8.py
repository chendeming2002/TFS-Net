#!/usr/bin/env python3
"""v8 空间 RWKV 头: 规范的 Vision-RWKV / Restore-RWKV 形态 (docs/v8/01-设计.md §三)。

与旧实现 (`models/golf_r4/tca_rwkv.py::RWKVSpatialHead`) 的差别:

| 项 | 旧实现 | 本实现 |
|----|--------|--------|
| 局部上下文 | 无 (BiWKV 只接收 k/v) | **OmniShift** (多分支 depthwise, 可重参数化) |
| Channel Mix | 无 | **有** (hidden_rate × dim 的 MLP + 残差) |
| 衰减 w | `-softplus(spatial_decay)` 逐通道可学习 | 相同 (与 Vision-RWKV 一致, 见 §2.2 核对) |
| 扫描 | 4 方向, 每方向独立 BiWKV | 相同 (扫描策略属后续消融项, 本版不动) |
| Q 的作用 | `sigmoid(q) * wkv` 门控 | 相同 (三路任务查询是我们的创新点, §3.2) |

## 为什么用 LeakyReLU 而非 GELU

channel mix 与 to_rgb 都用 LeakyReLU。GELU 在负半轴梯度严格为零, 一旦某路输入变负
就会永久冻结 (BranchM 实测权重梯度恒为 0)。LeakyReLU 保留了负半轴的微小梯度通道,
使模块不会因为一次方向性的数值偏移而彻底死掉。

## 三路 Q 的角色

OmniShift 只作用于 K/V 的生成 (与 Restore-RWKV 一致: token shift 注入的是 k/r 的输入,
不是 q)。三路 Query 仍独立由 `feat_spatial` 投影得到, 与共享 KV 做匹配 —— 这是我们相对
所有 RWKV 变体的区别所在, 不在本次整理中改动。
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.golf_r4.tca_rwkv import FastBiWKV


class OmniShift(nn.Module):
    """多分支 depthwise 局部上下文 (Restore-RWKV / EvRWKV 的 Omni-Shift)。

    形式: alpha[0]·x + alpha[1]·DWConv1x1(x) + alpha[2]·DWConv3x3(x) + alpha[3]·DWConv5x5(x)

    提供 RWKV 在图像上缺失的局部归纳偏置: 时序 RWKV 靠 token shift 把前一时刻的信息
    混入当前输入, 而图像没有时间轴, 必须用空间邻域替代。三个不同感受野的分支让网络
    自己选择需要的尺度。

    参考实现: reference_repos/EvRWKV/egllie/models/Cross_Restore_RWKV.py:103
    alpha 初始化为 randn (与参考一致); conv 权重用 kaiming, 避免输出尺度随分支数放大。
    """

    def __init__(self, dim: int):
        super().__init__()
        self.conv1x1 = nn.Conv2d(dim, dim, 1, 1, 0, groups=dim, bias=False)
        self.conv3x3 = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim, bias=False)
        self.conv5x5 = nn.Conv2d(dim, dim, 5, 1, 2, groups=dim, bias=False)
        self.alpha = nn.Parameter(torch.randn(4))
        # 初始化: 各分支 kaiming, alpha 用 1/sqrt(分支数) 缩放
        # → 初始输出尺度与输入同量级, 不会因 4 路叠加而放大 4 倍
        for conv in (self.conv1x1, self.conv3x3, self.conv5x5):
            nn.init.kaiming_normal_(conv.weight, mode="fan_out", nonlinearity="linear")
        with torch.no_grad():
            self.alpha.fill_(1.0 / math.sqrt(4.0))
        self.reparam_ready = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return (self.alpha[0] * x
                + self.alpha[1] * self.conv1x1(x)
                + self.alpha[2] * self.conv3x3(x)
                + self.alpha[3] * self.conv5x5(x))

    @torch.no_grad()
    def reparameterize(self):
        """把四个分支合并为单个 5×5 depthwise 卷积 (推理加速, 输出等价)。

        不做这一步也不影响正确性; 训练阶段保持多分支即可。
        """
        w1 = F.pad(self.conv1x1.weight, (2, 2, 2, 2))
        w3 = F.pad(self.conv3x3.weight, (1, 1, 1, 1))
        ident = F.pad(torch.ones_like(self.conv1x1.weight), (2, 2, 2, 2))
        combined = (self.alpha[0] * ident + self.alpha[1] * w1
                    + self.alpha[2] * w3 + self.alpha[3] * self.conv5x5.weight)
        conv = nn.Conv2d(self.conv5x5.in_channels, self.conv5x5.out_channels,
                         5, 1, 2, groups=self.conv5x5.groups, bias=False)
        conv.weight.copy_(combined)
        self.reparam_ready = True
        return conv


class SpatialRWKVHeadV8(nn.Module):
    """单路空间 RWKV 头: OmniShift + 双向 WKV 扫描 + Channel Mix。

    Args:
        channels: 特征通道数
        num_directions: 扫描方向数 (默认 4)
        hidden_rate: Channel Mix 的隐层倍数 (与 Vision-RWKV 默认 4 一致)
        use_omni_shift: 是否在 K/V 输入前注入局部邻域 (默认 True)。
        use_channel_mix: 是否启用两段式里的 Channel Mix (默认 True)。

    两个开关默认都为 True, 即完整形态; 单独关掉一个可用于做单变量对照
    (只加 OmniShift, 或只加 Channel Mix), 见 `docs/v8/02-实验计划.md` 组 A。
    """

    def __init__(self, channels: int, num_directions: int = 4, hidden_rate: int = 4,
                 use_omni_shift: bool = True, use_channel_mix: bool = True):
        super().__init__()
        assert channels % num_directions == 0, \
            f"channels({channels}) 必须能被 num_directions({num_directions}) 整除"
        self.channels = channels
        self.num_dirs = num_directions
        self.head_dim = channels // num_directions
        self.use_omni_shift = bool(use_omni_shift)
        self.use_channel_mix = bool(use_channel_mix)

        # ---------- 局部上下文 (可选) ----------
        self.omni_shift = OmniShift(channels) if self.use_omni_shift else None

        # ---------- Q/K/V 投影 ----------
        self.proj_q = nn.Linear(channels, channels, bias=False)
        self.proj_k = nn.Linear(channels, channels, bias=False)
        self.proj_v = nn.Linear(channels, channels, bias=False)
        self.proj_out = nn.Linear(channels, channels, bias=False)

        self.pre_norm = nn.LayerNorm(channels)
        self.post_norm = nn.LayerNorm(channels)

        self.bi_wkv_list = nn.ModuleList([
            FastBiWKV(self.head_dim) for _ in range(num_directions)
        ])

        # ---------- Channel Mix (可选) ----------
        # 经典 RWKV block 的两段式: Time Mix 做跨位置传递, Channel Mix 做逐位置跨通道变换。
        # 用 LeakyReLU 而非 GELU/ReLU², 避免负半轴梯度为零导致永久冻结。
        if self.use_channel_mix:
            hidden = channels * hidden_rate
            self.channel_mix = nn.Sequential(
                nn.Linear(channels, hidden, bias=False),
                nn.LeakyReLU(0.01, inplace=True),
                nn.Linear(hidden, channels, bias=False),
            )
        else:
            self.channel_mix = None

        self._init_weights()
        self._idx_cache: dict = {}

    def _init_weights(self):
        """RWKV-7 风格的小初始化 (key 更小, 抑制早期 key 数值溢出)。"""
        c = self.channels
        nn.init.uniform_(self.proj_k.weight, -0.05 / math.sqrt(c), 0.05 / math.sqrt(c))
        for proj in (self.proj_q, self.proj_v, self.proj_out):
            nn.init.uniform_(proj.weight, -0.5 / math.sqrt(c), 0.5 / math.sqrt(c))
        if self.use_channel_mix:
            nn.init.kaiming_uniform_(self.channel_mix[0].weight, a=math.sqrt(5))
            nn.init.zeros_(self.channel_mix[2].weight)

    # ---------- 扫描 (与旧实现一致, 扫描策略属后续消融项) ----------
    @staticmethod
    def _scan_h(x):
        return x.flatten(2).transpose(1, 2)

    @staticmethod
    def _scan_v(x):
        return x.permute(0, 1, 3, 2).flatten(2).transpose(1, 2)

    def _scan_d1(self, x):
        return self._diag_scan(x, anti=False)

    def _scan_d2(self, x):
        return self._diag_scan(x, anti=True)

    def _diag_scan(self, x, anti: bool):
        B, C, H, W = x.shape
        idx = self._diag_index(H, W, anti, x.device)
        return x.flatten(2)[:, :, idx].transpose(1, 2)

    def _diag_index(self, H, W, anti: bool, device) -> torch.Tensor:
        key = (H, W, anti, str(device))
        if key in self._idx_cache:
            return self._idx_cache[key]
        coords, used = [], set()
        for s in range(H + W - 1):
            for i in range(max(0, s - W + 1), min(s + 1, H)):
                j = (s - i) if not anti else (W - 1 - (s - i))
                if 0 <= j < W:
                    ij = i * W + j
                    coords.append(ij)
                    used.add(ij)
        for ij in range(H * W):          # 反对角扫描会漏掉部分位置, 补齐
            if ij not in used:
                coords.append(ij)
        idx = torch.tensor(coords[:H * W], device=device)
        self._idx_cache[key] = idx
        return idx

    def _inv_index(self, scan_fn, B, C, H, W, device) -> torch.Tensor:
        key = ("inv", scan_fn.__name__, H, W, str(device))
        if key in self._idx_cache:
            return self._idx_cache[key]
        identity = torch.arange(H * W, device=device).float().view(1, 1, H, W)
        scanned = scan_fn(identity).squeeze(0).squeeze(-1).long()
        inv = torch.empty_like(scanned)
        inv[scanned] = torch.arange(H * W, device=device)
        self._idx_cache[key] = inv
        return inv

    def _scan_fns(self):
        return (self._scan_h, self._scan_v, self._scan_d1, self._scan_d2)

    def forward(self, center_feat: torch.Tensor,
                context_feat: torch.Tensor) -> torch.Tensor:
        """center_feat 生成 Q, context_feat 生成 K/V。两者均为 (B, C, H, W)。

        与 Restore-RWKV 一致: OmniShift 只注入 K/V 的输入 (token shift 的对应物),
        不作用于 Q —— Q 保持来自中心帧的纯净表示, 以便三路语义分化。
        """
        B, C, H, W = center_feat.shape

        # Q 来自中心帧; K/V 可选地先经 OmniShift 注入局部上下文
        q_tokens = self.pre_norm(center_feat.flatten(2).transpose(1, 2))
        ctx = self.omni_shift(context_feat) if self.use_omni_shift else context_feat
        kv_tokens = self.pre_norm(ctx.flatten(2).transpose(1, 2))

        q = self.proj_q(q_tokens)
        k = self.proj_k(kv_tokens)
        v = self.proj_v(kv_tokens)

        heads = []
        for i, scan_fn in enumerate(self._scan_fns()):
            c0, c1 = i * self.head_dim, (i + 1) * self.head_dim
            k_2d = k[:, :, c0:c1].transpose(1, 2).reshape(B, self.head_dim, H, W)
            v_2d = v[:, :, c0:c1].transpose(1, 2).reshape(B, self.head_dim, H, W)
            wkv_seq = self.bi_wkv_list[i](scan_fn(k_2d), scan_fn(v_2d),
                                          total_tokens=H * W)
            heads.append(wkv_seq[:, self._inv_index(scan_fn, B, self.head_dim, H, W,
                                                    center_feat.device)])
        wkv = torch.cat(heads, dim=-1)          # (B, N, C)

        out = torch.sigmoid(q) * wkv            # 三路 Q 门控 (我们的创新点)
        out = self.proj_out(out)
        out = out.transpose(1, 2).reshape(B, C, H, W)

        # Channel Mix: 逐位置跨通道变换 (末层零初始化 → 初始恒等, 梯度可达)
        if self.use_channel_mix:
            cm = self.channel_mix(self.post_norm(out.flatten(2).transpose(1, 2)))
            out = out + cm.transpose(1, 2).reshape(B, C, H, W)

        return out


if __name__ == "__main__":
    B, C, H, W = 2, 128, 32, 32
    head = SpatialRWKVHeadV8(C)
    y = head(torch.randn(B, C, H, W), torch.randn(B, C, H, W))
    n = sum(p.numel() for p in head.parameters())
    print(f"SpatialRWKVHeadV8: {tuple(y.shape)}, params={n/1e3:.1f}K")
    y.sum().backward()
    dead = [n for n, p in head.named_parameters()
            if p.grad is None or p.grad.abs().sum() == 0]
    print(f"无梯度参数: {dead if dead else '无'}")
