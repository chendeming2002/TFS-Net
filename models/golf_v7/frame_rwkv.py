#!/usr/bin/env python3
"""
Frame-level RWKV: v7 核心创新模块

设计理念: RWKV 用于高层帧级语义聚合，而非像素级对齐
- 输入: [B, T, C, H, W] 空间特征
- 提取: Global Pool → [B, T, D] 帧级 token
- 建模: Multi-head RWKV → 捕捉帧间依赖
- 注入: FiLM 调制 → 全局上下文注入回空间
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class RWKVChannelMix(nn.Module):
    """RWKV Channel Mixing (单头)"""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        
        # RWKV-4 style channel mix
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        
        # Time-mixing weights
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))  # Shift along time
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))
    
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Time shift
        xx = self.time_shift(x.transpose(1, 2)).transpose(1, 2)  # [B, T, D]
        
        # Time mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        # RWKV operations
        k = self.key(xk)
        v = self.value(x)
        r = self.receptance(xr)
        
        # Channel mix: element-wise
        out = torch.sigmoid(r) * torch.square(torch.relu(k)) * v
        
        return out


class RWKVTimeMix(nn.Module):
    """RWKV Time Mixing with Adaptive Decay (单头)"""
    def __init__(self, dim: int, adaptive_decay: bool = True):
        super().__init__()
        self.dim = dim
        self.adaptive_decay = adaptive_decay
        
        # RWKV projections
        self.key = nn.Linear(dim, dim, bias=False)
        self.value = nn.Linear(dim, dim, bias=False)
        self.receptance = nn.Linear(dim, dim, bias=False)
        self.output = nn.Linear(dim, dim, bias=False)
        
        # Time shift
        self.time_shift = nn.ZeroPad2d((0, 0, 1, -1))
        
        # Time mixing weights
        self.time_mix_k = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_v = nn.Parameter(torch.ones(1, 1, dim))
        self.time_mix_r = nn.Parameter(torch.ones(1, 1, dim))
        
        # Decay (base)
        self.time_decay = nn.Parameter(torch.ones(dim))
        
        # Adaptive decay (运动自适应)
        if self.adaptive_decay:
            self.motion_net = nn.Sequential(
                nn.Linear(dim, dim // 4),
                nn.ReLU(),
                nn.Linear(dim // 4, 1),
                nn.Sigmoid()
            )
    
    def forward(self, x):
        # x: [B, T, D]
        B, T, D = x.shape
        
        # Time shift
        xx = self.time_shift(x.transpose(1, 2)).transpose(1, 2)
        
        # Time mixing
        xk = x * self.time_mix_k + xx * (1 - self.time_mix_k)
        xv = x * self.time_mix_v + xx * (1 - self.time_mix_v)
        xr = x * self.time_mix_r + xx * (1 - self.time_mix_r)
        
        # RWKV projections
        k = self.key(xk)
        v = self.value(xv)
        r = self.receptance(xr)
        
        # Adaptive decay weights
        if self.adaptive_decay and T > 1:
            motion = x[:, 1:] - x[:, :-1]  # [B, T-1, D]
            motion_scores = self.motion_net(motion)  # [B, T-1, 1]
            # 运动大 → 衰减快 (权重小)
            decay_adaptive = torch.exp(-motion_scores.squeeze(-1))  # [B, T-1]
        else:
            decay_adaptive = None
        
        # RWKV recurrent processing
        outputs = []
        state = torch.zeros(B, D, device=x.device, dtype=x.dtype)
        
        for t in range(T):
            k_t = k[:, t]  # [B, D]
            v_t = v[:, t]
            r_t = r[:, t]
            
            # Update state with decay
            if t > 0:
                if decay_adaptive is not None:
                    decay_t = self.time_decay * decay_adaptive[:, t-1:t]  # [B, D]
                    state = state * torch.exp(-decay_t) + k_t * v_t
                else:
                    state = state * torch.exp(-self.time_decay) + k_t * v_t
            else:
                state = k_t * v_t
            
            # Output
            o_t = torch.sigmoid(r_t) * state
            outputs.append(o_t)
        
        outputs = torch.stack(outputs, dim=1)  # [B, T, D]
        outputs = self.output(outputs)
        
        return outputs


class RWKVBlock(nn.Module):
    """Single RWKV Block (Time Mix + Channel Mix)"""
    def __init__(self, dim: int, adaptive_decay: bool = True):
        super().__init__()
        self.ln1 = nn.LayerNorm(dim)
        self.ln2 = nn.LayerNorm(dim)
        self.time_mix = RWKVTimeMix(dim, adaptive_decay=adaptive_decay)
        self.channel_mix = RWKVChannelMix(dim)
    
    def forward(self, x):
        # x: [B, T, D]
        x = x + self.time_mix(self.ln1(x))
        x = x + self.channel_mix(self.ln2(x))
        return x


class FrameLevelRWKV(nn.Module):
    """
    Frame-level RWKV: v7 核心模块
    
    输入: [B, T, C, H, W] 空间特征
    输出: [B, T, C, H, W] FiLM 调制后的特征
    
    Pipeline:
    1. Global Pool: 提取帧级 token [B, T, D]
    2. Multi-head RWKV: 捕捉帧间语义依赖
    3. FiLM: 注入全局上下文回空间特征
    """
    def __init__(self, dim: int = 128, num_heads: int = 8, 
                 num_blocks: int = 2, adaptive_decay: bool = True):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        
        assert dim % num_heads == 0, f"dim={dim} must be divisible by num_heads={num_heads}"
        
        # Frame-level feature extraction
        self.frame_pool = nn.AdaptiveAvgPool2d(1)  # H×W → 1×1
        
        # Multi-head RWKV blocks
        self.rwkv_heads = nn.ModuleList([
            nn.Sequential(*[
                RWKVBlock(self.head_dim, adaptive_decay=adaptive_decay)
                for _ in range(num_blocks)
            ])
            for _ in range(num_heads)
        ])
        
        # Context injection (FiLM modulation)
        self.film_scale = nn.Conv2d(dim, dim, 1)
        self.film_shift = nn.Conv2d(dim, dim, 1)
        
        # Initialize FiLM to identity
        nn.init.zeros_(self.film_scale.weight)
        nn.init.zeros_(self.film_scale.bias)
        nn.init.zeros_(self.film_shift.weight)
        nn.init.zeros_(self.film_shift.bias)
    
    def forward(self, x):
        # x: [B, T, C, H, W]
        B, T, C, H, W = x.shape
        
        # 1. Extract frame-level tokens
        frame_tokens = []
        for t in range(T):
            token = self.frame_pool(x[:, t])  # [B, C, 1, 1]
            frame_tokens.append(token.squeeze(-1).squeeze(-1))  # [B, C]
        frame_tokens = torch.stack(frame_tokens, dim=1)  # [B, T, C]
        
        # 2. Multi-head RWKV processing
        head_outputs = []
        for h in range(self.num_heads):
            start = h * self.head_dim
            end = start + self.head_dim
            head_input = frame_tokens[:, :, start:end]  # [B, T, head_dim]
            head_out = self.rwkv_heads[h](head_input)
            head_outputs.append(head_out)
        
        frame_ctx = torch.cat(head_outputs, dim=-1)  # [B, T, C]
        
        # 3. Inject context back to spatial features (FiLM)
        out = []
        for t in range(T):
            ctx_t = frame_ctx[:, t, :, None, None]  # [B, C, 1, 1]
            scale = self.film_scale(ctx_t)  # [B, C, 1, 1]
            shift = self.film_shift(ctx_t)
            
            # FiLM modulation
            out_t = x[:, t] * (1 + scale) + shift
            out.append(out_t)
        
        out = torch.stack(out, dim=1)  # [B, T, C, H, W]
        
        return out, frame_ctx  # 返回 ctx 用于诊断


if __name__ == '__main__':
    # Test
    model = FrameLevelRWKV(dim=128, num_heads=8, num_blocks=2, adaptive_decay=True)
    x = torch.randn(2, 5, 128, 32, 32)
    
    out, ctx = model(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out.shape}")
    print(f"Context: {ctx.shape}")
    print(f"Parameters: {sum(p.numel() for p in model.parameters()) / 1e6:.2f}M")
