#!/usr/bin/env python3
"""
GolfNet v7: RWKV-Hybrid 分层混合架构

核心创新:
1. Stage 1: Encoder (复用 Golf R2, 逐帧独立)
2. Stage 2: PixelTemporalAttention (Window=3, 像素级对齐)
3. Stage 3: FrameLevelRWKV (8头, 帧级语义聚合) ⭐
4. Stage 4: 三分支 + 3×3 Upsample + Fusion

设计哲学: 让 RWKV 做擅长的事情 (高层帧级), 像素对齐交给 Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Tuple

from models.golf.encoder import SharedEncoder
from models.golf.branch_n import BranchN
from models.golf.branch_l import BranchL
from models.golf.branch_m import BranchM
from models.golf.fusion import AdaptiveFusion
from models.golf_v7.frame_rwkv import FrameLevelRWKV
from models.golf_v7.pixel_temporal import PixelTemporalAttentionSimple
from models.golf_v7.upsample import Upsample3x3


class GolfNet_v7(nn.Module):
    """
    GolfNet v7: RWKV-Hybrid
    
    输入: [B, T, 3, H, W] (T=5 帧)
    输出: [B, 3, H, W] (中心帧增强)
    """
    def __init__(self, 
                 num_frames: int = 5,
                 encoder_channels: list = [32, 64, 128],
                 rwkv_heads: int = 8,
                 rwkv_blocks: int = 2,
                 adaptive_decay: bool = True):
        super().__init__()
        self.num_frames = num_frames
        self.C1, self.C2, self.C3 = encoder_channels
        
        # ========== Stage 1: Encoder (逐帧独立) ==========
        self.encoder = SharedEncoder(
            in_channels=3,
            out_channels_list=encoder_channels,
            num_blocks_per_scale=[2, 2, 2]
        )
        
        # ========== Stage 2: 像素级时序对齐 ==========
        # 从 C2=64 提升到 128 (为后续模块准备)
        self.feature_proj = nn.Conv2d(self.C2, 128, 1)
        
        self.pixel_temporal = PixelTemporalAttentionSimple(dim=128)
        
        # ========== Stage 3: 帧级 RWKV (核心创新) ⭐ ==========
        # 注意: FrameLevelRWKV 需要时序输入 [B, T, C, H, W]
        # 但 PixelTemporalAttention 输出 [B, C, H, W] (单帧)
        # 解决: 扩展维度或直接在 encoder 特征上应用
        
        # 方案: 在 encoder 输出的 F2 上应用 FrameLevelRWKV
        self.frame_rwkv = FrameLevelRWKV(
            dim=128, 
            num_heads=rwkv_heads, 
            num_blocks=rwkv_blocks,
            adaptive_decay=adaptive_decay
        )
        
        # ========== Stage 4: 三分支 + Upsample + Fusion ==========
        # 输入: [B, 128, H/2, W/2] (中心帧特征)
        
        # 三分支 (复用 Golf R2, 但需要调整输入通道)
        self.branch_N = BranchN(channels=128)
        self.branch_L = BranchL(channels=128)
        self.branch_M = BranchM(channels=128)
        
        # Upsample (3×3 Conv 修复棋盘伪影)
        self.upsample_N = Upsample3x3(in_ch=128, out_ch=64, scale=2)
        self.upsample_L = Upsample3x3(in_ch=128, out_ch=64, scale=2)
        self.upsample_M = Upsample3x3(in_ch=128, out_ch=64, scale=2)
        
        # 投影到 RGB 空间 (Fusion 需要 3 通道输入)
        self.to_rgb_N = nn.Conv2d(64, 3, 1)
        self.to_rgb_L = nn.Conv2d(64, 3, 1)
        self.to_rgb_M = nn.Conv2d(64, 3, 1)
        
        # Fusion (需要中心帧输入)
        self.fusion = AdaptiveFusion(in_channels=3, hidden_channels=64, num_blocks=2)
    
    def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        x: [B, T, 3, H, W]
        
        Returns: dict with keys:
            - 'final': [B, 3, H, W] 最终输出
            - 'branch_N/L/M': 三分支输出 (用于 loss)
            - 'frame_ctx': [B, T, D] 帧级上下文 (用于诊断)
        """
        B, T, C_in, H, W = x.shape
        assert T == self.num_frames, f"期望 T={self.num_frames}, 得到 T={T}"
        
        # 中心帧 (用于 Fusion 残差)
        center_idx = T // 2
        X_center = x[:, center_idx]  # [B, 3, H, W]
        
        # ========== Stage 1: 逐帧编码 ==========
        feats_f1 = []
        feats_f2 = []
        feats_f3 = []
        for t in range(T):
            f1, f2, f3 = self.encoder(x[:, t])  # f2: [B, 64, H/2, W/2]
            feats_f1.append(f1)
            feats_f2.append(f2)
            feats_f3.append(f3)
        
        feats_f2 = torch.stack(feats_f2, dim=1)  # [B, T, 64, H/2, W/2]
        
        # 投影到 128 维
        feats_f2_proj = []
        for t in range(T):
            feat_t = self.feature_proj(feats_f2[:, t])  # [B, 128, H/2, W/2]
            feats_f2_proj.append(feat_t)
        feats_f2_proj = torch.stack(feats_f2_proj, dim=1)  # [B, T, 128, H/2, W/2]
        
        # ========== Stage 2: 像素级时序对齐 ==========
        feat_aligned = self.pixel_temporal(feats_f2_proj)  # [B, 128, H/2, W/2]
        
        # ========== Stage 3: 帧级 RWKV ==========
        # 在原始时序特征上应用 (保留时序信息)
        feats_rwkv, frame_ctx = self.frame_rwkv(feats_f2_proj)  # [B, T, 128, H/2, W/2], [B, T, 128]
        
        # 取中心帧 (融合像素对齐 + RWKV 语义)
        feat_center_rwkv = feats_rwkv[:, center_idx]  # [B, 128, H/2, W/2]
        
        # 融合策略: 像素对齐 + RWKV 调制
        feat_fused = feat_aligned + feat_center_rwkv  # 简单相加
        
        # ========== Stage 4: 三分支处理 ==========
        out_N = self.branch_N(feat_fused)  # [B, 128, H/2, W/2]
        out_L = self.branch_L(feat_fused)
        out_M = self.branch_M(feat_fused)
        
        # Upsample (3×3 Conv)
        Y_N = self.upsample_N(out_N)  # [B, 64, H, W]
        Y_L = self.upsample_L(out_L)
        Y_M = self.upsample_M(out_M)
        
        # 投影到 RGB 空间
        Y_N_rgb = self.to_rgb_N(Y_N)
        Y_L_rgb = self.to_rgb_L(Y_L)
        Y_M_rgb = self.to_rgb_M(Y_M)
        
        # Fusion (需要 Y_N, Y_L, Y_M, X_center)
        fusion_out = self.fusion(Y_N_rgb, Y_L_rgb, Y_M_rgb, X_center)
        
        return {
            'final': fusion_out['output'],
            'branch_N': Y_N_rgb,
            'branch_L': Y_L_rgb,
            'branch_M': Y_M_rgb,
            'frame_ctx': frame_ctx,  # 用于诊断
            'fusion_weights': fusion_out['weights'],
        }


if __name__ == '__main__':
    # Test
    model = GolfNet_v7(
        num_frames=5,
        encoder_channels=[32, 64, 128],
        rwkv_heads=8,
        rwkv_blocks=2,
        adaptive_decay=True
    )
    
    x = torch.randn(2, 5, 3, 256, 256)
    
    print("Testing GolfNet_v7...")
    out = model(x)
    
    print(f"\nInput: {x.shape}")
    print(f"Output final: {out['final'].shape}")
    print(f"Branch N: {out['branch_N'].shape}")
    print(f"Frame ctx: {out['frame_ctx'].shape}")
    
    total_params = sum(p.numel() for p in model.parameters())
    print(f"\nTotal parameters: {total_params / 1e6:.2f}M")
    
    # 模块参数分布
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    pixel_params = sum(p.numel() for p in model.pixel_temporal.parameters())
    rwkv_params = sum(p.numel() for p in model.frame_rwkv.parameters())
    branch_params = (sum(p.numel() for p in model.branch_N.parameters()) +
                    sum(p.numel() for p in model.branch_L.parameters()) +
                    sum(p.numel() for p in model.branch_M.parameters()))
    upsample_params = (sum(p.numel() for p in model.upsample_N.parameters()) +
                      sum(p.numel() for p in model.upsample_L.parameters()) +
                      sum(p.numel() for p in model.upsample_M.parameters()))
    
    print("\n=== 参数分布 ===")
    print(f"Encoder:         {encoder_params / 1e6:.2f}M")
    print(f"PixelTemporal:   {pixel_params / 1e6:.3f}M")
    print(f"FrameRWKV:       {rwkv_params / 1e6:.3f}M")
    print(f"三分支:          {branch_params / 1e6:.2f}M")
    print(f"Upsample (3×3):  {upsample_params / 1e6:.2f}M")
    print(f"Fusion:          {(total_params - encoder_params - pixel_params - rwkv_params - branch_params - upsample_params) / 1e6:.3f}M")
