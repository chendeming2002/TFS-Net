import os
from glob import glob

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from .transforms import random_flip_pair, random_time_reverse

try:
    import cv2
    _HAS_CV2 = True
except ImportError:
    _HAS_CV2 = False


class _FrameLRU:
    """Per-worker LRU cache of decoded uint8 frames (process-local, no locks)."""

    def __init__(self, capacity=24):
        self.capacity = capacity
        self._cache = {}
        self._order = []

    def get(self, path):
        if path in self._cache:
            self._order.remove(path)
            self._order.append(path)
            return self._cache[path]
        arr = _decode_u8(path)
        self._cache[path] = arr
        self._order.append(path)
        if len(self._order) > self.capacity:
            evict = self._order.pop(0)
            self._cache.pop(evict, None)
        return arr


def _decode_u8(path):
    """Decode PNG to HWC uint8 RGB (cv2 fast path, PIL fallback)."""
    if _HAS_CV2:
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            raise IOError("cv2 failed to read: {}".format(path))
        return img[:, :, ::-1]  # BGR -> RGB
    return np.array(Image.open(path).convert("RGB"), dtype=np.uint8)


def _u8_to_tensor(arr):
    """(H,W,C) uint8 -> (C,H,W) float32 /255."""
    t = torch.from_numpy(np.ascontiguousarray(arr))
    return t.permute(2, 0, 1).float().div_(255.0)


def read_image(path):
    """Full-res float tensor (kept for val / external callers)."""
    return _u8_to_tensor(_decode_u8(path))


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
        self._frame_cache = _FrameLRU(capacity=24)

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

    def _gather_window_u8(self, paths, center_idx):
        frames = []
        max_idx = len(paths) - 1
        cache = self._frame_cache
        for offset in range(-self.half_window, self.half_window + 1):
            idx = min(max(center_idx + offset, 0), max_idx)
            frames.append(cache.get(paths[idx]))
        return frames

    def _gather_window(self, paths, center_idx):
        frames = []
        max_idx = len(paths) - 1
        cache = self._frame_cache
        for offset in range(-self.half_window, self.half_window + 1):
            idx = min(max(center_idx + offset, 0), max_idx)
            frames.append(_u8_to_tensor(cache.get(paths[idx])))
        return torch.stack(frames, dim=0)

    def __getitem__(self, index):
        import random as _random

        sample = self.samples[index]
        center_idx = sample["index"]
        cache = self._frame_cache

        if self.mode == "train":
            # Fast path: uint8 decode -> shared random crop -> small float tensors
            # (numerically identical to decode->float->crop, ~100x less memory traffic)
            lq_u8 = self._gather_window_u8(sample["lq_paths"], center_idx)
            gt_u8 = cache.get(sample["gt_paths"][center_idx])

            h, w = lq_u8[0].shape[:2]
            cs = self.crop_size
            top = _random.randint(0, h - cs)
            left = _random.randint(0, w - cs)

            clip = np.stack([f[top:top + cs, left:left + cs] for f in lq_u8], axis=0)
            target = gt_u8[top:top + cs, left:left + cs]

            clip = torch.from_numpy(np.ascontiguousarray(clip)).permute(0, 3, 1, 2).float().div_(255.0)
            target = _u8_to_tensor(target)

            clip, target = random_flip_pair(clip, target)
            clip = random_time_reverse(clip)
        else:
            clip = self._gather_window(sample["lq_paths"], center_idx)
            target = _u8_to_tensor(cache.get(sample["gt_paths"][center_idx]))

        meta = {
            "sequence": sample["sequence"],
            "index": center_idx,
            "frame_name": os.path.basename(sample["lq_paths"][center_idx]),
        }
        return clip, target, meta
