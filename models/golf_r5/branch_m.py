"""
Golf Branch-M: 运动补偿 (修复版)
==================================
与 Foxtrot 的区别:
  1. 上采样改为 resize-conv (消除棋盘格)
  2. F1 skip 提供原分辨率运动边界细节
  3. 置信度门控保留不变 (已验证有效)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from models.modules.blocks import NAFBlock, LayerNorm2d
from .upsample import UpsampleBlock
from typing import Tuple


class FlowEstimator(nn.Module):
    """轻量级光流估计 (特征域, coarse-to-fine)"""

    def __init__(self, in_channels: int = 128):
        super().__init__()
        self.corr_channels = 32
        self.corr_proj = nn.Sequential(
            nn.Conv2d(in_channels * 2, self.corr_channels, 1, bias=True),
            nn.GELU(),
        )
        self.flow_decoder = nn.Sequential(
            nn.Conv2d(self.corr_channels, 64, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(64, 32, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(32, 2, 3, 1, 1, bias=True),
        )
        nn.init.zeros_(self.flow_decoder[-1].weight)
        nn.init.zeros_(self.flow_decoder[-1].bias)

    def forward(self, f_center: torch.Tensor, f_neigh: torch.Tensor) -> torch.Tensor:
        corr = self.corr_proj(torch.cat([f_center, f_neigh], dim=1))
        return self.flow_decoder(corr)


class DeformableAlign(nn.Module):
    """基于 flow field 的可变形采样"""

    def forward(self, feat: torch.Tensor, flow: torch.Tensor) -> torch.Tensor:
        B, C, H, W = feat.shape
        grid_y, grid_x = torch.meshgrid(
            torch.linspace(-1, 1, H, device=feat.device),
            torch.linspace(-1, 1, W, device=feat.device),
            indexing='ij',
        )
        grid = torch.stack([grid_x, grid_y], dim=-1).unsqueeze(0).expand(B, -1, -1, -1)
        flow_perm = flow.permute(0, 2, 3, 1)
        scale = torch.tensor([W / 2.0, H / 2.0], device=flow.device, dtype=flow.dtype)
        sample_grid = grid + flow_perm / scale
        return F.grid_sample(feat, sample_grid, mode='bilinear',
                             padding_mode='border', align_corners=True)


class ConfidenceEstimator(nn.Module):
    """遮挡/对齐失败置信度检测

    R3-A 修复: bias 从 2.0 → 0.0
      - 原来 bias=2.0 → sigmoid(2.0)≈0.88，训练中 conf 均值0.978，
        几乎从不抑制任何帧，门控失效（Golf-R2 实测 >0.99 占 45%）
      - 修正为 bias=0.0 → sigmoid(0)=0.5，中性起点，让模型真正学习遮挡判断
    """

    def __init__(self, channels: int = 128):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(channels * 2, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, 1, 3, 1, 1, bias=True),
            nn.Sigmoid(),
        )
        nn.init.zeros_(self.conv[-2].weight)
        nn.init.constant_(self.conv[-2].bias, 0.0)   # R3-A: 0.0（中性）而非 2.0（恒高）

    def forward(self, warped: torch.Tensor, center: torch.Tensor) -> torch.Tensor:
        return self.conv(torch.cat([warped, center], dim=1))


class BranchM(nn.Module):
    """Branch-M: Motion Compensation (Golf-R3)

    R3-A: ConfidenceEstimator bias 0.0 (中性，真正学习遮挡)
    R3-B: 高分辨率 warp — 光流在 H/2 估计，warp 在 H 执行再降采样
          (DGAF-VSR: 高分辨率域 warp 保留更多高频，pair45 运动场景收益)

    Input:
        F_M:    (B, channels, H/2, W/2)
        F2_seq: (B, T, enc_channels, H/2, W/2)
        skip:   (B, C1, H, W) F1 skip

    Output dict: Y_M, flow_vis, conf_map
    """

    def __init__(self, channels: int = 128, enc_channels: int = 64,
                 num_frames: int = 5, num_blocks: int = 3,
                 out_channels: int = 3, skip_channels: int = 32,
                 use_hires_warp: bool = True):          # R3-B 开关
        super().__init__()
        self.channels = channels
        self.num_frames = num_frames
        self.center_idx = num_frames // 2
        self.use_hires_warp = use_hires_warp

        self.neigh_proj = nn.Sequential(
            nn.Conv2d(enc_channels, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        self.flow_estimator = FlowEstimator(channels)
        self.deform_align = DeformableAlign()
        self.conf_estimator = ConfidenceEstimator(channels)   # R3-A: bias=0.0
        self.fusion_conv = nn.Sequential(
            nn.Conv2d(channels * num_frames, channels, 1, bias=True),
            LayerNorm2d(channels),
        )
        self.refine_blocks = nn.Sequential(*[
            NAFBlock(channels) for _ in range(num_blocks)
        ])

        # resize-conv + F1 skip
        self.upsample = UpsampleBlock(
            in_channels=channels,
            out_channels=channels,
            skip_channels=skip_channels,
            norm=True,
        )
        self.to_rgb = nn.Sequential(
            nn.Conv2d(channels, channels // 2, 3, 1, 1, bias=True),
            nn.GELU(),
            nn.Conv2d(channels // 2, out_channels, 3, 1, 1, bias=True),
        )

    def _align_and_aggregate(self, F_M, F2_seq):
        B, T, C_enc, H, W = F2_seq.shape
        center = F_M
        aligned_list = [center]
        flow_list, conf_list = [], []

        for t in range(T):
            if t == self.center_idx:
                continue
            neigh_h2 = self.neigh_proj(F2_seq[:, t])   # (B,C,H/2,W/2)
            # 光流在 H/2 估计（省算力）
            flow_h2 = self.flow_estimator(center, neigh_h2)   # (B,2,H/2,W/2)
            flow_list.append(flow_h2)

            if self.use_hires_warp:
                # R3-B: 高分辨率 warp
                # 上采样到 H 域做 warp，保留更多高频
                neigh_h = F.interpolate(neigh_h2, scale_factor=2,
                                        mode='bilinear', align_corners=False)  # (B,C,H,W)
                flow_h  = F.interpolate(flow_h2, scale_factor=2,
                                        mode='bilinear', align_corners=False) * 2  # scale flow
                warped_h = self.deform_align(neigh_h, flow_h)    # warp at H
                warped   = F.avg_pool2d(warped_h, 2)             # 降回 H/2
            else:
                warped = self.deform_align(neigh_h2, flow_h2)

            conf = self.conf_estimator(warped, center)
            conf_list.append(conf)
            aligned_list.append(warped * conf)

        aggregated = self.fusion_conv(torch.cat(aligned_list, dim=1))
        flow_center = (torch.stack(flow_list, dim=1).mean(dim=1)
                       if flow_list else torch.zeros(B, 2, H, W, device=F_M.device))
        conf_avg = (torch.stack(conf_list, dim=1).mean(dim=1)
                    if conf_list else torch.ones(B, 1, H, W, device=F_M.device))
        return aggregated, flow_center, conf_avg

    def forward(self, F_M: torch.Tensor, F2_seq: torch.Tensor,
                skip: torch.Tensor = None) -> dict:
        aggregated, flow_vis, conf_map = self._align_and_aggregate(F_M, F2_seq)
        x = self.refine_blocks(aggregated)
        x = self.upsample(x, skip)
        Y_M = self.to_rgb(x)
        return {"Y_M": Y_M, "flow_vis": flow_vis, "conf_map": conf_map}
