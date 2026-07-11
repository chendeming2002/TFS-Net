import os

import numpy as np
import torch
from PIL import Image


def ensure_dir(path):
    os.makedirs(path, exist_ok=True)


def save_checkpoint(state, path):
    ensure_dir(os.path.dirname(path))
    torch.save(state, path)


def load_checkpoint(path, map_location="cpu"):
    return torch.load(path, map_location=map_location)


def save_image_tensor(tensor, path):
    ensure_dir(os.path.dirname(path))
    image = tensor.detach().clamp(0.0, 1.0).cpu().permute(1, 2, 0).numpy()
    image = (image * 255.0).round().astype(np.uint8)
    Image.fromarray(image).save(path)

