import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import random_crop_pair, random_flip_pair, random_time_reverse


def read_image(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


class SDSDDataset(Dataset):
    def __init__(self, input_root, target_root, window_size=5, mode="train", crop_size=256, max_seqs=None):
        super().__init__()
        self.input_root = input_root
        self.target_root = target_root
        self.window_size = window_size
        self.half_window = window_size // 2
        self.mode = mode
        self.crop_size = crop_size
        self.max_seqs = max_seqs
        self.samples = self._build_samples()

    def _build_samples(self):
        samples = []
        seq_names = sorted([name for name in os.listdir(self.input_root) if os.path.isdir(os.path.join(self.input_root, name))])
        if self.max_seqs is not None:
            seq_names = seq_names[: self.max_seqs]
        for seq_name in seq_names:
            lq_paths = sorted(glob(os.path.join(self.input_root, seq_name, "*")))
            gt_paths = sorted(glob(os.path.join(self.target_root, seq_name, "*")))
            lq_map = {os.path.basename(path): path for path in lq_paths}
            gt_map = {os.path.basename(path): path for path in gt_paths}
            common_names = sorted(set(lq_map.keys()) & set(gt_map.keys()))
            if not common_names:
                raise RuntimeError("Sequence {} has no overlapping input/gt frames.".format(seq_name))
            lq_paths = [lq_map[name] for name in common_names]
            gt_paths = [gt_map[name] for name in common_names]
            for idx in range(len(lq_paths)):
                samples.append(
                    {
                        "sequence": seq_name,
                        "index": idx,
                        "lq_paths": lq_paths,
                        "gt_paths": gt_paths,
                    }
                )
        return samples

    def __len__(self):
        return len(self.samples)

    def _gather_window(self, paths, center_idx):
        frames = []
        max_idx = len(paths) - 1
        for offset in range(-self.half_window, self.half_window + 1):
            idx = min(max(center_idx + offset, 0), max_idx)
            frames.append(read_image(paths[idx]))
        return torch.stack(frames, dim=0)

    def __getitem__(self, index):
        sample = self.samples[index]
        center_idx = sample["index"]
        clip = self._gather_window(sample["lq_paths"], center_idx)
        target = read_image(sample["gt_paths"][center_idx])

        if self.mode == "train":
            clip, target = random_crop_pair(clip, target, self.crop_size)
            clip, target = random_flip_pair(clip, target)
            clip = random_time_reverse(clip)

        meta = {
            "sequence": sample["sequence"],
            "index": center_idx,
            "frame_name": os.path.basename(sample["lq_paths"][center_idx]),
        }
        return clip, target, meta
