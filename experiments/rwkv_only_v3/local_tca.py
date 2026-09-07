"""
LocalTCA — T-BC 实验专用 (替代共享 TCA 的时序聚合路径)
======================================================
保留: HaarDWT anchor + MVC-Shift + SpatialWKV2D + ChannelMix (与共享 TCA 完全一致)
替换: TemporalCorrespondence/Aggregation (32² 全局 softmax warp, 结构性模糊)
  → LocalWindowAlignment (H/2 全分辨率 R×R 窗口 softmax + per-frame sigmoid 门
    + 置信度回退门控)
附注: 对齐在增强后特征上计算 (修复 P4 信号/载体错位); 返回 conf_map 作为
  motion_map 的真实信号源 (替代退化的 C_omega 对角线)。
"""
from __future__ import annotations

from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.modules.blocks import LayerNorm2d
from models.modules.pure_rwkv_sace import TCA


def _pad_crop(x: torch.Tensor, dx: int, dy: int, r: int) -> torch.Tensor:
    """shifted_v(p) = v(p + [dx, dy])，边界 replicate。"""
    B, C, H, W = x.shape
    xp = F.pad(x, (r, r, r, r), mode="replicate")
    return xp[:, :, r + dy:r + dy + H, r + dx:r + dx + W]


class LocalWindowAlignment(nn.Module):
    """H/2 全分辨率 R×R 窗口相关对齐。

    对中心帧每个位置 p:
      sim(p, d) = cos(embed(center_p), embed(neighbor_{p+d})) / temp,  d ∈ R×R 窗口
      prob = softmax_d(sim)
      warp(p) = Σ_d prob·neighbor_{p+d}      ← 局部凸组合 (非全帧)
      conf(p) = mean_t max_d prob_t(p, d)    ← 匹配唯一性 (运动→平坦→低)
      F_out = conf·gate_agg + (1-conf)·center

    bootstrap 参数 (T-BC1b, 修复 T-BC1 的乘性自门控冷启动死锁):
      "none" — 原版: conf=max-prob, 初始≈1/81 → 支路梯度被 conf 缩放致死 (已证实死锁)
      "bias" — 恒等偏移: (0,0) 偏移的 logit 加可学习偏置 (init 4.5, 初始 conf≈0.86)
               "先信任对齐, 学会在运动边界不信任" —— 符合 SDSD 静态主导统计
      "gate" — 独立可学习门: conf = sigmoid(conv([center, F_agg])), 与对齐 softmax 解耦,
               零初始化 → conf=0.5 起步, 门自身的输入路径保证 F_agg 支路梯度不灭
    """

    def __init__(self, channels: int = 64, radius: int = 4, embed_dim: int = 16,
                 bootstrap: str = "none"):
        super().__init__()
        self.radius = radius
        self.bootstrap = bootstrap
        self.embed = nn.Conv2d(channels, embed_dim, 1, bias=False)
        self.temp_raw = nn.Parameter(torch.zeros(1))
        self.frame_gate = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 3, 1, 1),
            nn.GELU(),
            nn.Conv2d(channels, 1, 1),
        )
        if bootstrap == "bias":
            self.identity_bias = nn.Parameter(torch.tensor(5.8))
        elif bootstrap == "gate":
            self.conf_gate = nn.Sequential(
                nn.Conv2d(channels * 2, channels, 3, 1, 1),
                nn.GELU(),
                nn.Conv2d(channels, 1, 1),
            )
        self.out_norm = LayerNorm2d(channels)
        # (0,0) 恒等偏移在 offsets 展开序列中的索引 (dx 外层, dy 内层)
        self.id_idx = radius * (2 * radius + 1) + radius

    @property
    def temp(self) -> torch.Tensor:
        return F.softplus(self.temp_raw) + 0.02

    def forward(self, center: torch.Tensor, neighbors: torch.Tensor):
        """center: (B,C,H,W) 增强后中心帧; neighbors: (B,Tn,C,H,W) 增强后邻帧。
        返回 (F_aligned (B,C,H,W), conf_map (B,1,H,W))"""
        B, _, C, H, W = neighbors.shape
        Tn = neighbors.shape[1]
        r = self.radius
        R2 = (2 * r + 1) ** 2

        ck = F.normalize(self.embed(center), dim=1)
        nk = F.normalize(self.embed(neighbors.reshape(-1, C, H, W)), dim=1) \
            .view(B, Tn, -1, H, W)

        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]

        # pass 1: 全部窗口 logits → 窗口内 softmax
        logits = []
        for t in range(Tn):
            for dx, dy in offsets:
                sk = _pad_crop(nk[:, t], dx, dy, r)
                logits.append((ck * sk).sum(1, keepdim=True))
        logits = torch.cat(logits, dim=1) / self.temp                 # (B, Tn·R², H, W)

        if self.bootstrap == "bias":
            bias_map = torch.zeros_like(logits)
            for t in range(Tn):
                bias_map[:, t * R2 + self.id_idx] = self.identity_bias
            logits = logits + bias_map

        # T-BC1b: 逐帧窗口 softmax (每帧 R² 内归一) —— 联合 softmax 会让 Tn 个恒等
        # 偏移互相竞争, 把 conf 上限钳在 1/Tn; 逐帧归一后帧内对齐置信与帧间门控解耦
        probs = F.softmax(logits.view(B, Tn, R2, H, W), dim=2) \
                   .reshape(B, Tn * R2, H, W)

        # pass 2: 逐帧加权聚合 (sigmoid 帧门允许弃权, 非 softmax 强制归一)
        conf_maps = []
        gate_sum = None
        agg = None
        for t in range(Tn):
            p_t = probs[:, t * R2:(t + 1) * R2]                       # (B, R², H, W)
            conf_maps.append(p_t.amax(dim=1, keepdim=True))
            acc = None
            i = 0
            for dx, dy in offsets:
                sv = _pad_crop(neighbors[:, t], dx, dy, r)
                acc = p_t[:, i:i + 1] * sv if acc is None else acc + p_t[:, i:i + 1] * sv
                i += 1
            warp_t = acc                                              # (B, C, H, W)
            g = torch.sigmoid(self.frame_gate(torch.cat([center, warp_t], dim=1)))
            agg = g * warp_t if agg is None else agg + g * warp_t
            gate_sum = g if gate_sum is None else gate_sum + g

        F_agg = agg / (gate_sum + 1e-6)                               # 全弃权 → →0

        if self.bootstrap == "gate":
            conf = torch.sigmoid(self.conf_gate(torch.cat([center, F_agg], dim=1)))
        else:
            conf = torch.stack(conf_maps, dim=1).mean(dim=1)          # (B,1,H,W)
        conf = F.avg_pool2d(conf, 3, 1, 1)                            # 轻度平滑

        out = conf * F_agg + (1.0 - conf) * center
        return self.out_norm(out), conf


class LocalTCA(TCA):
    """共享 TCA 的实验子类: 空间路径不变, 时序聚合替换为 LocalWindowAlignment。"""

    def __init__(self, channels: int = 64, num_frames: int = 5, radius: int = 4,
                 bootstrap: str = "none"):
        super().__init__(channels=channels, num_frames=num_frames)
        self.local_align = LocalWindowAlignment(channels, radius=radius,
                                                bootstrap=bootstrap)

    def forward(self, feats: torch.Tensor) -> Dict:
        B, T, C, H_ds, W_ds = feats.shape

        # ── 与共享 TCA 完全一致的空间路径 ──
        feats_flat = feats.reshape(B * T, C, H_ds, W_ds)
        LL, LH, HL, HH = self._haar_dwt(feats_flat)
        anchor = self.dwt_anchor(LL)
        anchor_up = F.interpolate(anchor, size=(H_ds, W_ds), mode='bilinear',
                                  align_corners=False)
        hf_cat = torch.cat([LH, HL, HH], dim=1)
        hf_up = F.interpolate(hf_cat, size=(H_ds, W_ds), mode='bilinear',
                              align_corners=False)
        hf_feat = self.dwt_hf_proj(hf_up)
        x_enhanced = self.anchor_fuse(torch.cat([feats_flat, anchor_up, hf_feat], dim=1))

        x_shifted = self.mvc_shift(x_enhanced)
        x_wkv = self.spatial_wkv(x_shifted)
        x_cm = self.channel_mix(x_wkv)
        sace_out_ds = x_enhanced + x_cm * self.spatial_gamma
        sace_4d = sace_out_ds.reshape(B, T, C, H_ds, W_ds)

        mu_t_clean = sace_4d[:, self.center_idx]
        sigma_t_clean = sace_4d.std(dim=1, unbiased=False)

        # ── T-BC: 局部窗口对齐 (增强后特征, 修复 P4 错位) ──
        center_idx = self.center_idx
        neighbor_idx = [t for t in range(T) if t != center_idx]
        F_t_aligned, conf_map = self.local_align(
            sace_4d[:, center_idx], sace_4d[:, neighbor_idx])

        return {
            "tca_out":        sace_4d,
            "mu_t_clean":     mu_t_clean,
            "sigma_t_clean":  sigma_t_clean,
            "C_omega_list":   [],
            "F_t_aligned":    F_t_aligned,
            "conf_map":       conf_map,
            "lff_feats":      [],
            "attn_maps":      [],
        }
