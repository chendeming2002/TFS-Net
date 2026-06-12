import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import ConvBlock


class EncoderStage(nn.Module):
    def __init__(self, in_channels, out_channels, stride):
        super().__init__()
        self.block = nn.Sequential(
            ConvBlock(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, act=True),
            ConvBlock(out_channels, out_channels, kernel_size=3, stride=1, padding=1, act=True),
        )

    def forward(self, x):
        return self.block(x)


class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=48):
        super().__init__()
        c1, c2, c3 = level_channels
        self.stage1 = EncoderStage(in_channels, c1, stride=1)
        self.stage2 = EncoderStage(c1, c2, stride=2)
        self.stage3 = EncoderStage(c2, c3, stride=2)

        self.lateral3 = nn.Conv2d(c3, fused_channels, 1, 1, 0)
        self.lateral2 = nn.Conv2d(c2, fused_channels, 1, 1, 0)
        self.lateral1 = nn.Conv2d(c1, fused_channels, 1, 1, 0)
        self.fuse = nn.Sequential(
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
            ConvBlock(fused_channels, fused_channels, 3, 1, 1, act=True),
        )

    def forward_single(self, x):
        l1 = self.stage1(x)
        l2 = self.stage2(l1)
        l3 = self.stage3(l2)

        p3 = self.lateral3(l3)
        p2 = self.lateral2(l2) + F.interpolate(p3, size=l2.shape[-2:], mode="bilinear", align_corners=False)
        p1 = self.lateral1(l1) + F.interpolate(p2, size=l1.shape[-2:], mode="bilinear", align_corners=False)
        return self.fuse(p1)

    def forward(self, x):
        b, t, c, h, w = x.shape
        x = x.view(b * t, c, h, w)
        feat = self.forward_single(x)
        _, cf, hf, wf = feat.shape
        return feat.view(b, t, cf, hf, wf)

