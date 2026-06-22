import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvBlock(nn.Module):
    """v5.6: Pre-LN Conv block — LayerNorm2d → Conv2d → GELU.

    Args:
        use_norm: True 启用 Pre-LN（默认）；False 退回 v4.3 行为（向后兼容）。
    """

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True, use_norm=True):
        super().__init__()
        layers = []
        if use_norm:
            layers.append(LayerNorm2d(in_channels))
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=True))
        if act:
            layers.append(nn.GELU())
        self.block = nn.Sequential(*layers)

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    """v5.6: Pre-LN ResBlock with LayerScale (beta zero-init → 初始恒等).

    LayerNorm2d → Conv1(3x3) → GELU → Conv2(3x3) → ×beta + skip
    beta 初始化为 0 → 训练初始时残差块为恒等映射，深层堆叠稳定。
    """

    def __init__(self, channels):
        super().__init__()
        self.norm = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(channels, channels, 3, 1, 1)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        residual = self.conv2(self.act(self.conv1(self.norm(x))))
        return x + residual * self.beta


class SimpleGate(nn.Module):
    """NAFNet SimpleGate: channel split × multiply, 替代非线性激活。"""
    def forward(self, x):
        x1, x2 = x.chunk(2, dim=1)
        return x1 * x2


class NAFBlock(nn.Module):
    """v5.6 P1-7: NAFNet-style block — 无非线性激活，SimpleGate + SCA + LayerScale.

    结构 (参照 reference_repos/NAFNet/NAFNet_arch.py):
        Branch 1 (attention):  LN → Conv1(1x1,c→2c) → Conv2(3x3 DW,2c) → SG(2c→c) → ×SCA → Conv3(1x1,c→c) → ×beta + skip
        Branch 2 (FFN):        LN → Conv4(1x1,c→2c) → SG(2c→c) → Conv5(1x1,c→c) → ×gamma + skip

    beta/gamma 零初始化 → 初始恒等映射，深层堆叠稳定。
    """

    def __init__(self, channels, dw_expand=2, ffn_expand=2):
        super().__init__()
        dw_channel = channels * dw_expand

        # Branch 1: "attention" (depthwise conv + SCA)
        self.norm1 = LayerNorm2d(channels)
        self.conv1 = nn.Conv2d(channels, dw_channel, 1, 1, 0, bias=True)
        self.conv2 = nn.Conv2d(dw_channel, dw_channel, 3, 1, 1, groups=dw_channel, bias=True)
        self.conv3 = nn.Conv2d(dw_channel // 2, channels, 1, 1, 0, bias=True)
        self.sca = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(dw_channel // 2, dw_channel // 2, 1, 1, 0, bias=True),
        )
        self.sg1 = SimpleGate()

        # Branch 2: FFN
        ffn_channel = ffn_expand * channels
        self.norm2 = LayerNorm2d(channels)
        self.conv4 = nn.Conv2d(channels, ffn_channel, 1, 1, 0, bias=True)
        self.conv5 = nn.Conv2d(ffn_channel // 2, channels, 1, 1, 0, bias=True)
        self.sg2 = SimpleGate()

        # LayerScale (零初始化 → 初始恒等)
        self.beta = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.gamma = nn.Parameter(torch.zeros(1, channels, 1, 1))

    def forward(self, x):
        # Branch 1
        y = self.norm1(x)
        y = self.conv1(y)
        y = self.conv2(y)
        y = self.sg1(y)
        y = y * self.sca(y)
        y = self.conv3(y)
        y = x + y * self.beta

        # Branch 2
        z = self.norm2(y)
        z = self.conv4(z)
        z = self.sg2(z)
        z = self.conv5(z)
        return y + z * self.gamma


class LayerNorm2d(nn.Module):
    def __init__(self, channels, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, channels, 1, 1))
        self.bias = nn.Parameter(torch.zeros(1, channels, 1, 1))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(dim=1, keepdim=True)
        var = (x - mean).pow(2).mean(dim=1, keepdim=True)
        x = (x - mean) / torch.sqrt(var + self.eps)
        return x * self.weight + self.bias


def pad_to_window(x, window_size):
    _, _, h, w = x.shape
    pad_h = (window_size - h % window_size) % window_size
    pad_w = (window_size - w % window_size) % window_size
    if pad_h == 0 and pad_w == 0:
        return x, (0, 0)
    x = F.pad(x, (0, pad_w, 0, pad_h), mode="reflect")
    return x, (pad_h, pad_w)


def unpad_from_window(x, pad_hw):
    pad_h, pad_w = pad_hw
    if pad_h > 0:
        x = x[:, :, :-pad_h, :]
    if pad_w > 0:
        x = x[:, :, :, :-pad_w]
    return x


def window_partition_2d(x, window_size):
    b, c, h, w = x.shape
    x = x.view(b, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 2, 4, 3, 5, 1).contiguous()
    return x.view(b, -1, window_size * window_size, c)


def window_reverse_2d(windows, window_size, h, w):
    b, nw, tokens, c = windows.shape
    num_h = h // window_size
    num_w = w // window_size
    x = windows.view(b, num_h, num_w, window_size, window_size, c)
    x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
    return x.view(b, c, h, w)


def window_partition_video(x, window_size):
    b, t, c, h, w = x.shape
    x = x.view(b, t, c, h // window_size, window_size, w // window_size, window_size)
    x = x.permute(0, 3, 5, 1, 4, 6, 2).contiguous()
    return x.view(b, -1, t * window_size * window_size, c)


def window_reverse_video(windows, window_size, t, h, w):
    b, nw, tokens, c = windows.shape
    num_h = h // window_size
    num_w = w // window_size
    x = windows.view(b, num_h, num_w, t, window_size, window_size, c)
    x = x.permute(0, 3, 6, 1, 4, 2, 5).contiguous()
    return x.view(b, t, c, h, w)


def safe_divide(x, y, eps=1e-6):
    return x / (y + eps)


def pairwise_cosine_logits(center, neighbors):
    b, t, c, h, w = neighbors.shape
    center_flat = center.flatten(2)
    center_flat = F.normalize(center_flat, dim=-1)
    neigh_flat = neighbors.flatten(3)
    neigh_flat = F.normalize(neigh_flat, dim=-1)
    logits = (center_flat.unsqueeze(1) * neigh_flat).mean(dim=-1).mean(dim=-1)
    return logits

