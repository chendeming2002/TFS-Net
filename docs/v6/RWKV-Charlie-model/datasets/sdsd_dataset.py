"""
SDSD 数据集加载器 — RWKV-Charlie
================================
直接从 .npy 加载视频帧
"""

import os
import glob
import random

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, ConcatDataset


def load_npy_video(video_path: str) -> np.ndarray:
    data = np.load(video_path)
    if data.ndim == 3:
        data = data[..., None]
    if data.shape[-1] == 1:
        data = np.repeat(data, 3, axis=-1)
    data = data.transpose(0, 3, 1, 2)
    return data.astype(np.float32)


class SDSDDataset(Dataset):
    def __init__(
        self,
        root_dir: str,
        split: str = "train",
        num_frames: int = 5,
        max_tries: int = 100,
        transform=None,
        sigma_range: tuple[float, float] = (0.0, 0.0),
    ):
        self.root_dir = root_dir
        self.split = split
        self.num_frames = num_frames
        self.max_tries = max_tries
        self.transform = transform
        self.sigma_range = sigma_range
        self.pairs = []

        pair_dir = os.path.join(root_dir, split)
        video_dirs = sorted(os.listdir(pair_dir))
        for vd in video_dirs:
            clean_dir = os.path.join(pair_dir, vd, "clean")
            noisy_dir = os.path.join(pair_dir, vd, "noisy")
            if not os.path.isdir(clean_dir) or not os.path.isdir(noisy_dir):
                continue
            clean_files = sorted(glob.glob(os.path.join(clean_dir, "*.npy")))
            noisy_files = sorted(glob.glob(os.path.join(noisy_dir, "*.npy")))
            if not clean_files:
                for f in sorted(glob.glob(os.path.join(clean_dir, "*"))):
                    if f.endswith('.npy'):
                        clean_files.append(f)
            if not noisy_files:
                for f in sorted(glob.glob(os.path.join(noisy_dir, "*"))):
                    if f.endswith('.npy'):
                        noisy_files.append(f)
            if not clean_files or not noisy_files:
                continue
            n_frames = min(len(clean_files), len(noisy_files))
            for start in range(0, n_frames - num_frames + 1):
                self.pairs.append((
                    clean_files[start:start + num_frames],
                    noisy_files[start:start + num_frames],
                ))

    def __len__(self) -> int:
        return len(self.pairs)

    def __getitem__(self, idx: int) -> dict:
        clean_paths, noisy_paths = self.pairs[idx]
        clean_frames = [load_npy_video(f) for f in clean_paths]
        noisy_frames = [load_npy_video(f) for f in noisy_paths]
        clean_frames = np.concatenate(clean_frames, axis=0)
        noisy_frames = np.concatenate(noisy_frames, axis=0)
        assert clean_frames.shape == noisy_frames.shape

        # sigma_t: 模拟噪声水平 (Bravo2/Charlie — 用于 MRPN)
        if self.sigma_range[1] > 0:
            sigma_t = np.random.uniform(self.sigma_range[0], self.sigma_range[1],
                                        size=(self.num_frames,)).astype(np.float32)
        else:
            sigma_t = np.zeros(self.num_frames, dtype=np.float32)

        sample = {
            "clean": clean_frames,
            "noisy": noisy_frames,
            "sigma_t": sigma_t,
        }

        if self.transform:
            sample = self.transform(sample)

        return sample


def create_sdsd_dataloader(
    root_dir: str,
    split: str = "train",
    num_frames: int = 5,
    batch_size: int = 4,
    shuffle: bool = True,
    num_workers: int = 4,
    transform=None,
    sigma_range=(0.0, 0.0),
):
    dataset = SDSDDataset(
        root_dir=root_dir,
        split=split,
        num_frames=num_frames,
        transform=transform,
        sigma_range=sigma_range,
    )
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=(split == "train"),
    )
