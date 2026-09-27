#!/usr/bin/env python3
"""
Upsample with 3×3 Conv: 修复棋盘伪影

Golf R2-R5 使用 1×1 Conv + PixelShuffle → 棋盘格伪影 (2px 周期)
v7 改用 3×3 Conv + PixelShuffle → 引入空间连续性约束
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class Upsample3x3(nn.Module):
    """
    3×3 Conv + PixelShuffle 上采样
    
    修复 Golf R2-R5 的棋盘格伪影
    """
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        
        # 3×3 卷积 (关键: 引入空间邻域信息)
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 
                             kernel_size=3, padding=1, bias=True)
        
        self.pixel_shuffle = nn.PixelShuffle(scale)
        
        # Initialize to avoid checkerboard
        nn.init.kaiming_normal_(self.conv.weight, mode='fan_out', nonlinearity='relu')
        if self.conv.bias is not None:
            nn.init.zeros_(self.conv.bias)
    
    def forward(self, x):
        # x: [B, in_ch, H, W]
        x = self.conv(x)  # [B, out_ch * scale^2, H, W]
        x = self.pixel_shuffle(x)  # [B, out_ch, H*scale, W*scale]
        return x


class Upsample1x1(nn.Module):
    """
    1×1 Conv + PixelShuffle 上采样 (Golf R2-R5 原版)
    
    用于对比实验
    """
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        
        # 1×1 卷积 (无空间邻域信息)
        self.conv = nn.Conv2d(in_ch, out_ch * scale * scale, 
                             kernel_size=1, bias=True)
        
        self.pixel_shuffle = nn.PixelShuffle(scale)
    
    def forward(self, x):
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        return x


class UpsampleTranspose(nn.Module):
    """
    转置卷积上采样 (备选方案)
    
    可能引入其他伪影，但避免 PixelShuffle 的周期性问题
    """
    def __init__(self, in_ch: int, out_ch: int, scale: int = 2):
        super().__init__()
        self.scale = scale
        
        # 转置卷积
        self.deconv = nn.ConvTranspose2d(
            in_ch, out_ch, 
            kernel_size=scale * 2, 
            stride=scale, 
            padding=scale // 2,
            output_padding=0
        )
        
        # 细化卷积 (消除转置卷积的伪影)
        self.refine = nn.Conv2d(out_ch, out_ch, kernel_size=3, padding=1)
    
    def forward(self, x):
        x = self.deconv(x)
        x = self.refine(x)
        return x


if __name__ == '__main__':
    # Test and compare
    x = torch.randn(2, 128, 32, 32)
    
    print("Testing Upsample3x3 (v7 推荐)")
    model1 = Upsample3x3(in_ch=128, out_ch=64, scale=2)
    out1 = model1(x)
    print(f"Input: {x.shape}")
    print(f"Output: {out1.shape}")
    print(f"Parameters: {sum(p.numel() for p in model1.parameters()) / 1e3:.1f}K")
    
    print("\nTesting Upsample1x1 (Golf R2-R5 原版)")
    model2 = Upsample1x1(in_ch=128, out_ch=64, scale=2)
    out2 = model2(x)
    print(f"Output: {out2.shape}")
    print(f"Parameters: {sum(p.numel() for p in model2.parameters()) / 1e3:.1f}K")
    
    print("\nTesting UpsampleTranspose (备选)")
    model3 = UpsampleTranspose(in_ch=128, out_ch=64, scale=2)
    out3 = model3(x)
    print(f"Output: {out3.shape}")
    print(f"Parameters: {sum(p.numel() for p in model3.parameters()) / 1e3:.1f}K")
    
    # Check checkerboard pattern (简单测试: 自相关)
    def check_checkerboard(tensor):
        # tensor: [B, C, H, W]
        # 计算相邻像素的平均差异
        diff_h = (tensor[:, :, :-1, :] - tensor[:, :, 1:, :]).abs().mean()
        diff_w = (tensor[:, :, :, :-1] - tensor[:, :, :, 1:]).abs().mean()
        return diff_h.item(), diff_w.item()
    
    print("\n棋盘格测试 (差异越大 → 伪影越明显):")
    diff1 = check_checkerboard(out1)
    diff2 = check_checkerboard(out2)
    diff3 = check_checkerboard(out3)
    print(f"3×3 Conv: H_diff={diff1[0]:.4f}, W_diff={diff1[1]:.4f}")
    print(f"1×1 Conv: H_diff={diff2[0]:.4f}, W_diff={diff2[1]:.4f}")
    print(f"Transpose: H_diff={diff3[0]:.4f}, W_diff={diff3[1]:.4f}")
