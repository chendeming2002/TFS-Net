"""Charlie: 视频数据增强"""

import random
import numpy as np
import torch
import torch.nn.functional as F


class VideoToTensor:
    def __call__(self, sample: dict) -> dict:
        for key in ["clean", "noisy"]:
            arr = sample[key]
            arr = torch.from_numpy(arr).float() / 255.0
            sample[key] = arr
        sample["sigma_t"] = torch.from_numpy(sample["sigma_t"]).float()
        return sample


class VideoNormalize:
    def __call__(self, sample: dict) -> dict:
        return sample


class VideoRandomCrop:
    def __init__(self, patch_size: int = 128):
        self.patch_size = patch_size

    def __call__(self, sample: dict) -> dict:
        clean = sample["clean"]
        T, C, H, W = clean.shape
        ps = self.patch_size
        if H < ps or W < ps:
            return sample
        top = random.randint(0, H - ps)
        left = random.randint(0, W - ps)
        sample["clean"] = clean[:, :, top:top + ps, left:left + ps]
        sample["noisy"] = sample["noisy"][:, :, top:top + ps, left:left + ps]
        return sample


class VideoRandomFlip:
    def __call__(self, sample: dict) -> dict:
        for key in ["clean", "noisy"]:
            arr = sample[key]
            if random.random() > 0.5:
                arr = arr[..., ::-1].copy()
            if random.random() > 0.5:
                arr = arr[..., ::-1, :].copy()
            sample[key] = arr
        return sample


class TestTimeAug:
    def __init__(self):
        self.augmentations = [
            ("identity", lambda x: x),
            ("hflip", lambda x: x.flip(-1)),
            ("vflip", lambda x: x.flip(-2)),
            ("hvflip", lambda x: x.flip(-1).flip(-2)),
        ]

    def __call__(self, model, noisy: torch.Tensor) -> torch.Tensor:
        B, T, C, H, W = noisy.shape
        outputs = []
        for _, aug_fn in self.augmentations:
            x = aug_fn(noisy)
            with torch.no_grad():
                out = model(x)
            if isinstance(out, list):
                out = out[0]
            outputs.append(aug_fn(out))
        return torch.stack(outputs).mean(dim=0)
