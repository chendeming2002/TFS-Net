"""
LocalTCA — Flight11 主干模块 (T-BC1b 转正)
==========================================
保留共享 TCA 的空间路径 (HaarDWT anchor + MVC-Shift + SpatialWKV2D + ChannelMix)，
时序聚合替换为 LocalWindowAlignment:
  - H/2 全分辨率 R×R 窗口 softmax (逐帧归一，修复联合 softmax 的 conf 上限钳制)
  - bootstrap=bias: 恒等偏移 logit bias 破自举死锁 (T-BC1 实测 conf 恒 0.013)
  - per-frame sigmoid 帧门 (可弃权) + 置信度回退门控
新增 aux 接口 (供三分支对接, RESULTS.md §6 P1/P2):
  - warped_list: 全分辨率逐帧对齐特征 (NDPN 时间平均材料)
  - disp_field:  soft-argmax 位移场 (MCPN 运动幅度)
  - conf_map:    匹配唯一性先验 (NDPN conf_proj 替换)
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
    def __init__(self, channels: int = 64, radius: int = 4, embed_dim: int = 16,
                 bootstrap: str = "bias"):
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
        self.id_idx = radius * (2 * radius + 1) + radius

    @property
    def temp(self) -> torch.Tensor:
        return F.softplus(self.temp_raw) + 0.02

    def forward(self, center: torch.Tensor, neighbors: torch.Tensor):
        """center (B,C,H,W); neighbors (B,Tn,C,H,W)。
        返回 (F_aligned, conf_map, aux)；
        aux: {"warped_list": [Tn×(B,C,H,W)], "disp_field": (B,2,H,W)}"""
        B, _, C, H, W = neighbors.shape
        Tn = neighbors.shape[1]
        r = self.radius
        R2 = (2 * r + 1) ** 2

        ck = F.normalize(self.embed(center), dim=1)
        nk = F.normalize(self.embed(neighbors.reshape(-1, C, H, W)), dim=1) \
            .view(B, Tn, -1, H, W)

        offsets = [(dx, dy) for dx in range(-r, r + 1) for dy in range(-r, r + 1)]
        dev = center.device
        # offsets 序列 = [(dx,dy) for dx in -r..r for dy in -r..r]
        ox = torch.arange(-r, r + 1, device=dev).repeat_interleave(2 * r + 1)  # dx 慢变
        oy = torch.arange(-r, r + 1, device=dev).repeat(2 * r + 1)             # dy 快变

        logits = []
        for t in range(Tn):
            for dx, dy in offsets:
                sk = _pad_crop(nk[:, t], dx, dy, r)
                logits.append((ck * sk).sum(1, keepdim=True))
        logits = torch.cat(logits, dim=1) / self.temp

        if self.bootstrap == "bias":
            bias_map = torch.zeros_like(logits)
            for t in range(Tn):
                bias_map[:, t * R2 + self.id_idx] = self.identity_bias
            logits = logits + bias_map

        # 逐帧 softmax（联合 softmax 会把 conf 上限钳在 1/Tn）
        probs = F.softmax(logits.view(B, Tn, R2, H, W), dim=2) \
                   .reshape(B, Tn * R2, H, W)

        conf_maps, warped_list, disp_list, gate_sum, agg = [], [], [], None, None
        for t in range(Tn):
            p_t = probs[:, t * R2:(t + 1) * R2]
            conf_maps.append(p_t.amax(dim=1, keepdim=True))
            dxmap = (p_t * ox.view(1, -1, 1, 1)).sum(1, keepdim=True)
            dymap = (p_t * oy.view(1, -1, 1, 1)).sum(1, keepdim=True)
            disp_list.append(torch.cat([dxmap, dymap], dim=1))
            acc = None
            i = 0
            for dx, dy in offsets:
                sv = _pad_crop(neighbors[:, t], dx, dy, r)
                acc = p_t[:, i:i + 1] * sv if acc is None else acc + p_t[:, i:i + 1] * sv
                i += 1
            warp_t = acc
            warped_list.append(warp_t)
            g = torch.sigmoid(self.frame_gate(torch.cat([center, warp_t], dim=1)))
            agg = g * warp_t if agg is None else agg + g * warp_t
            gate_sum = g if gate_sum is None else gate_sum + g

        F_agg = agg / (gate_sum + 1e-6)

        if self.bootstrap == "gate":
            conf = torch.sigmoid(self.conf_gate(torch.cat([center, F_agg], dim=1)))
        else:
            conf = torch.stack(conf_maps, dim=1).mean(dim=1)
        conf = F.avg_pool2d(conf, 3, 1, 1)

        w = torch.stack(conf_maps, dim=1)                          # (B,Tn,1,H,W)
        disp_field = (torch.stack(disp_list, dim=1) * w).sum(dim=1) \
            / (w.sum(dim=1) + 1e-6)                                # (B,2,H,W)

        out = conf * F_agg + (1.0 - conf) * center
        aux = {"warped_list": warped_list, "disp_field": disp_field}
        return self.out_norm(out), conf, aux


class LocalTCA(TCA):
    """共享 TCA 子类: 空间路径不变，时序聚合 = LocalWindowAlignment。
    额外返回 conf_map / warped_list / disp_field。"""

    def __init__(self, channels: int = 64, num_frames: int = 5, radius: int = 4,
                 bootstrap: str = "bias"):
        super().__init__(channels=channels, num_frames=num_frames)
        self.local_align = LocalWindowAlignment(channels, radius=radius,
                                                bootstrap=bootstrap)

    def forward(self, feats: torch.Tensor) -> Dict:
        B, T, C, H_ds, W_ds = feats.shape

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

        neighbor_idx = [t for t in range(T) if t != self.center_idx]
        F_t_aligned, conf_map, aux = self.local_align(
            sace_4d[:, self.center_idx], sace_4d[:, neighbor_idx])

        return {
            "tca_out":        sace_4d,
            "mu_t_clean":     mu_t_clean,
            "sigma_t_clean":  sigma_t_clean,
            "C_omega_list":   [],
            "F_t_aligned":    F_t_aligned,
            "conf_map":       conf_map,
            "warped_list":    aux["warped_list"],
            "disp_field":     aux["disp_field"],
            "lff_feats":      [],
            "attn_maps":      [],
        }
