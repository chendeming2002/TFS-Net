import random
import torch


def random_crop_pair(clip, target, crop_size):
    _, _, h, w = clip.shape
    if h < crop_size or w < crop_size:
        raise ValueError("Crop size exceeds image size.")
    top = random.randint(0, h - crop_size)
    left = random.randint(0, w - crop_size)
    clip = clip[:, :, top: top + crop_size, left: left + crop_size]
    target = target[:, top: top + crop_size, left: left + crop_size]
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
