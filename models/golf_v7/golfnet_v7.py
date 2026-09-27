#!/usr/bin/env python3
"""
GolfNet v7: RWKV-Hybrid 简化版

架构简化但保留核心创新:
1. Encoder (复用 Golf R2)
2. PixelTemporalAttention (Window=3)
3. FrameLevelRWKV (8头, 帧级语义) ⭐
4. 简化解码器 (单分支 + 3×3 Upsample)
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

from models.golf.encoder import SharedEncoder
from models.golf_v7.frame_rwkv import FrameLevelRWKV
from models.golf_v7.pixel_temporal import PixelTemporalAttentionSimple
from models.golf_v7.upsample import Upsample3x3
from models.modules.blocks import NAFBlock


class GolfNet_v7(nn.Module):
    """GolfNet v7: RWKV-Hybrid 简化版"""
    
    def __init__(self, 
                 num_frames: int = 5,
                 encoder_channels: list = [32, 64, 128],
                 feature_dim: int = 128,
                 rwkv_heads: int = 8,
                 rwkv_blocks: int = 2,
                 adaptive_decay: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.C1, self.C2, self.C3 = encoder_channels
        self.feature_dim = feature_dim
        
        # Stage 1: Encoder
        self.encoder = SharedEncoder(
            in_channels=3,
            out_channels_list=encoder_channels,
            num_blocks_per_scale=[2, 2, 2]
        )
        
        # Stage 2: Feature projection + Pixel Temporal
        self.feature_proj = nn.Conv2d(self.C2, feature_dim, 1)
        self.pixel_temporal = PixelTemporalAttentionSimple(dim=feature_dim)
        
        # Stage 3: Frame-level RWKV ⭐
        self.frame_rwkv = FrameLevelRWKV(
            dim=feature_dim,
            num_heads=rwkv_heads,
            num_blocks=rwkv_blocks,
            adaptive_decay=adaptive_decay
        )
        
        # Stage 4: Decoder (简化单分支)
        self.decoder_blocks = nn.Sequential(
            NAFBlock(feature_dim),
            NAFBlock(feature_dim)
        )
        
        # Upsample H/2 → H (3×3 Conv 修复棋盘伪影)
        self.upsample = Upsample3x3(in_ch=feature_dim, out_ch=64, scale=2)
        
        # Output projection
        self.to_rgb = nn.Conv2d(64, 3, kernel_size=3, padding=1)
        
        # Residual connection
        self.residual_gamma = nn.Parameter(torch.tensor(0.1))
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, 3, H, W]
        Returns: dict with 'final', 'frame_ctx'
        """
        B, T, C_in, H, W = x.shape
        assert T == self.num_frames
        
        center_idx = T // 2
        X_center = x[:, center_idx]  # [B, 3, H, W]
        
        # Stage 1: Encoder (逐帧)
        feats_f2 = []
        for t in range(T):
            _, f2, _ = self.encoder(x[:, t])  # f2: [B, 64, H/2, W/2]
            feats_f2.append(f2)
        feats_f2 = torch.stack(feats_f2, dim=1)  # [B, T, 64, H/2, W/2]
        
        # 投影到 feature_dim
        feats_proj = []
        for t in range(T):
            feat_t = self.feature_proj(feats_f2[:, t])
            feats_proj.append(feat_t)
        feats_proj = torch.stack(feats_proj, dim=1)  # [B, T, 128, H/2, W/2]
        
        # Stage 2: Pixel temporal alignment
        feat_aligned = self.pixel_temporal(feats_proj)  # [B, 128, H/2, W/2]
        
        # Stage 3: Frame-level RWKV
        feats_rwkv, frame_ctx = self.frame_rwkv(feats_proj)  # [B, T, 128, H/2, W/2]
        feat_center_rwkv = feats_rwkv[:, center_idx]
        
        # Fuse: pixel-aligned + RWKV semantic
        feat_fused = feat_aligned + feat_center_rwkv
        
        # Stage 4: Decode
        feat_decoded = self.decoder_blocks(feat_fused)  # [B, 128, H/2, W/2]
        
        # Upsample
        feat_up = self.upsample(feat_decoded)  # [B, 64, H, W]
        
        # To RGB
        Y = self.to_rgb(feat_up)  # [B, 3, H, W]
        
        # Residual
        out = Y + self.residual_gamma * X_center
        
        return {
            'final': out,
            'frame_ctx': frame_ctx,
            'Y_base': Y,
        }


if __name__ == '__main__':
    model = GolfNet_v7(
        num_frames=5,
        encoder_channels=[32, 64, 128],
        feature_dim=128,
        rwkv_heads=8,
        rwkv_blocks=2
    )
    
    x = torch.randn(2, 5, 3, 256, 256)
    out = model(x)
    
    print(f"Input: {x.shape}")
    print(f"Output: {out['final'].shape}")
    print(f"Frame ctx: {out['frame_ctx'].shape}")
    print(f"\nParameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
