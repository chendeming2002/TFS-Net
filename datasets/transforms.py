import random

import torch


def random_crop_pair(clip, target, crop_size):
    _, _, h, w = clip.shape
    if h < crop_size or w < crop_size:
        raise ValueError("Crop size is larger than input image size.")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    clip = clip[:, :, top : top + crop_size, left : left + crop_size]
    target = target[:, top : top + crop_size, left : left + crop_size]
    return clip, target


def random_flip_pair(clip, target):
    if random.random() < 0.5:
        clip = torch.flip(clip, dims=[-1])
        target = torch.flip(target, dims=[-1])
    if random.random() < 0.5:
        clip = torch.flip(clip, dims=[-2])
        target = torch.flip(target, dims=[-2])
    return clip, target


def random_time_reverse(clip):
    if random.random() < 0.5:
        clip = torch.flip(clip, dims=[0])
    return clip


def random_dark_gamma(clip, prob=0.5, gamma_range=(1.5, 3.0)):
    """§6.6-A 极暗增广: 对 LQ 施加更极端的 gamma 压暗 (GT 不动)。

    动机 (§6.6 量化证据): DID 上失败的 video19/video20 平均亮度 0.019/0.023, 比 SDSD 训练集
    最暗序列 (0.043) 还暗约 2×, 中位亮度 (0.090) 是它们的 4–5×。模型从未见过该亮度区间。
    对 LQ 做 out = in ** gamma (gamma>1 变暗) 可把训练亮度分布下探到 ~0.02。

    与 train_golf_v7r_v3.py 中"目标退化强度"的关系: 该增广【只改 LQ】, GT 保持原亮度,
    因此它是"让模型见过更暗输入"而不是"改变任务定义"——与推理时极暗场景一致。

    clip: [T, C, H, W], 值域 [0, 1] (改动前)。
    返回: 同形状 tensor。
    """
    if prob <= 0.0 or random.random() >= prob:
        return clip
    gamma = random.uniform(gamma_range[0], gamma_range[1])
    return clip.clamp(0.0, 1.0).pow(gamma)

