#!/usr/bin/env python3
"""
Pixel-level Temporal Attention: v7 Stage 2

设计理念: 像素级精确运动补偿，使用 Window Attention (窗口=3帧)
- 输入: [B, T, C, H, W]
- 输出: [B, C, H, W] (中心帧增强)
- 机制: 中心帧 t=2，窗口 [t-1, t, t+1]，逐像素 Attention
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PixelTemporalAttention(nn.Module):
    """
    像素级时序注意力 (Window=3)
    
    对中心帧的每个像素，与相邻帧对应位置的 3×3 邻域做 Attention
    """
    def __init__(self, dim: int = 64, num_heads: int = 4, window: int = 3):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window = window
        self.scale = self.head_dim ** -0.5
        
        assert dim % num_heads == 0
        assert window == 3, "当前仅支持 window=3"
        
        # QKV projections
        self.qkv = nn.Conv2d(dim, dim * 3, 1, bias=False)
        self.proj = nn.Conv2d(dim, dim, 1, bias=False)
        
        # Positional encoding (相对位置)
        self.rel_pos_h = nn.Parameter(torch.zeros(2 * 3 - 1, self.head_dim))
        self.rel_pos_w = nn.Parameter(torch.zeros(2 * 3 - 1, self.head_dim))
        nn.init.trunc_normal_(self.rel_pos_h, std=0.02)
        nn.init.trunc_normal_(self.rel_pos_w, std=0.02)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        assert T >= 3, f"需要至少 3 帧，当前 T={T}"
        
        # 取中心帧和相邻帧
        center_idx = T // 2
        x_window = x[:, center_idx-1:center_idx+2]  # [B, 3, C, H, W]
        
        # 处理窗口帧
        out = self._window_attention(x_window)  # [B, C, H, W]
        
        return out
    
    def _window_attention(self, x):
        # x: [B, 3, C, H, W]
        B, T, C, H, W = x.shape
        
        # 展开时序维度
        x_flat = x.reshape(B, T * C, H, W)  # [B, 3C, H, W]
        
        # 中心帧作为 Query
        x_center = x[:, T // 2]  # [B, C, H, W]
        
        # QKV 投影
        qkv_center = self.qkv(x_center).reshape(B, 3, self.num_heads, self.head_dim, H, W)
        q, k_c, v_c = qkv_center.unbind(1)  # [B, num_heads, head_dim, H, W]
        
        # 相邻帧的 KV (简化版: 使用中心帧的 K 投影应用到所有帧)
        k_all = []
        v_all = []
        for t in range(T):
            kv_t = self.qkv(x[:, t])[:, C:]  # 只取 KV 部分 [B, 2C, H, W]
            kv_t = kv_t.reshape(B, 2, self.num_heads, self.head_dim, H, W)
            k_t, v_t = kv_t.unbind(1)
            k_all.append(k_t)
            v_all.append(v_t)
        
        k_all = torch.stack(k_all, dim=2)  # [B, num_heads, T, head_dim, H, W]
        v_all = torch.stack(v_all, dim=2)
        
        # Reshape for attention
        q = q.permute(0, 1, 3, 4, 2).reshape(B * self.num_heads * H * W, 1, self.head_dim)
        k_all = k_all.permute(0, 1, 3, 4, 2, 5).reshape(B * self.num_heads * H * W, T, self.head_dim)
        v_all = v_all.permute(0, 1, 3, 4, 2, 5).reshape(B * self.num_heads * H * W, T, self.head_dim)
        
        # Attention
        attn = (q @ k_all.transpose(-2, -1)) * self.scale  # [B*H*H*W, 1, T]
        attn = F.softmax(attn, dim=-1)
        
        # Aggregate
        out = (attn @ v_all).reshape(B, self.num_heads, H, W, self.head_dim)
        out = out.permute(0, 1, 4, 2, 3).reshape(B, C, H, W)
        
        # Output projection
        out = self.proj(out)
        
        return out


class PixelTemporalAttentionSimple(nn.Module):
    """
    简化版: 3D Conv 实现的时序注意力
    
    更高效，适合快速原型
    """
    def __init__(self, dim: int = 64):
        super().__init__()
        self.dim = dim
        
        # 3D Conv for temporal modeling
        self.conv3d = nn.Conv3d(dim, dim, kernel_size=(3, 3, 3), 
                                padding=(1, 1, 1), groups=dim)
        
        # Pointwise conv
        self.pw_conv = nn.Conv2d(dim, dim, 1)
        
        # Attention weights
        self.attn_conv = nn.Sequential(
            nn.Conv2d(dim, dim // 4, 1),
            nn.ReLU(),
            nn.Conv2d(dim // 4, 3, 1),  # 3 个时序位置的权重
            nn.Softmax(dim=1)
        )
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        assert T >= 3
        
        center_idx = T // 2
        x_window = x[:, center_idx-1:center_idx+2]  # [B, 3, C, H, W]
        
        # Reshape for 3D conv
        x_3d = x_window.permute(0, 2, 1, 3, 4)  # [B, C, 3, H, W]
        
        # 3D conv
        feat_3d = self.conv3d(x_3d)  # [B, C, 3, H, W]
        
        # Temporal attention weights
        x_center = x_window[:, 1]  # [B, C, H, W]
        attn_weights = self.attn_conv(x_center)  # [B, 3, H, W]
        
        # Weighted sum
        attn_weights = attn_weights.unsqueeze(2)  # [B, 3, 1, H, W]
        feat_3d = feat_3d.permute(0, 2, 1, 3, 4)  # [B, 3, C, H, W]
        
        out = (feat_3d * attn_weights).sum(dim=1)  # [B, C, H, W]
        
        # Output projection
        out = self.pw_conv(out)
        
        return out


if __name__ == '__main__':
    # Test
    print("Testing PixelTemporalAttentionSimple (推荐)")
    model = PixelTemporalAttentionSimple(dim=64)
    x = torch.randn(2, 5, 64, 32, 32)
    out = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.3f}M")
    
    print("\nTesting PixelTemporalAttention (完整版)")
    model2 = PixelTemporalAttention(dim=64, num_heads=4)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()) / 1e6:.3f}M")
