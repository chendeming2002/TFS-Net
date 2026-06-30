import argparse
import os
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image

try:
    from tqdm import tqdm
except Exception:
    class _TqdmFallback(object):
        def __init__(self, iterable=None, *args, **kwargs):
            self.iterable = iterable

        def __iter__(self):
            return iter(self.iterable)

        def set_postfix(self, **kwargs):
            return None

    def tqdm(iterable=None, *args, **kwargs):
        return _TqdmFallback(iterable, *args, **kwargs)

from models import TFSNet
from utils.inference import tiled_forward
from utils.io import ensure_dir, load_checkpoint, save_image_tensor


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--checkpoint", type=str, required=True)
    parser.add_argument("--input_root", type=str, required=True)
    parser.add_argument("--output_root", type=str, required=True)
    return parser.parse_args()


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def read_image(path):
    image = Image.open(path).convert("RGB")
    array = np.asarray(image, dtype=np.float32) / 255.0
    return torch.from_numpy(array).permute(2, 0, 1).contiguous()


def list_sequences(root):
    dirs = [p for p in sorted(glob(os.path.join(root, "*"))) if os.path.isdir(p)]
    if dirs:
        return dirs
    return [root]


def gather_clip(paths, center_idx, window_size):
    half = window_size // 2
    max_idx = len(paths) - 1
    frames = []
    for offset in range(-half, half + 1):
        idx = min(max(center_idx + offset, 0), max_idx)
        frames.append(read_image(paths[idx]))
    return torch.stack(frames, dim=0)


def main():
    args = parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = TFSNet(
        in_channels=cfg["model"]["in_channels"],
        level_channels=tuple(cfg["model"]["level_channels"]),
        fused_channels=cfg["model"]["fused_channels"],
        use_dwt_lff=cfg["model"].get("use_dwt_lff", False),
        use_pure_rwkv=cfg["model"].get("use_pure_rwkv", False),
        use_soft_clamp=cfg["model"].get("use_soft_clamp", False),
        sace_offset_use_norm=cfg["model"].get("sace_offset_use_norm", False),
        sace_offset_kaiming_init=cfg["model"].get("sace_offset_kaiming_init", False),
        use_soft_median=cfg["model"].get("use_soft_median", True),
        use_nafblock=cfg["model"].get("use_nafblock", False),
        num_bottleneck_blocks=cfg["model"].get("num_bottleneck_blocks", 0),
        num_igrf_res_blocks=cfg["model"].get("num_igrf_res_blocks", 2),
        use_amp_enhance=cfg["model"].get("use_amp_enhance", False),
    ).to(device)
    checkpoint = load_checkpoint(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model"], strict=True)
    model.eval()

    ensure_dir(args.output_root)
    sequences = list_sequences(args.input_root)
    window_size = cfg["dataset"]["window_size"]

    with torch.no_grad():
        for seq_path in sequences:
            model.clear_frame_cache()
            frame_paths = sorted(glob(os.path.join(seq_path, "*")))
            seq_name = os.path.basename(seq_path.rstrip("\\/"))
            save_dir = os.path.join(args.output_root, seq_name)
            ensure_dir(save_dir)
            half = window_size // 2
            max_idx = len(frame_paths) - 1
            iterator = list(enumerate(frame_paths))
            for idx, frame_path in tqdm(iterator, total=len(frame_paths), desc=seq_name, leave=False):
                # 构建全局帧索引（用于逐帧缓存）
                indices = [
                    min(max(idx + offset, 0), max_idx)
                    for offset in range(-half, half + 1)
                ]
                clip = gather_clip(frame_paths, idx, window_size).unsqueeze(0).to(device)
                output = tiled_forward(
                    model=model,
                    clip=clip,
                    tile_size=cfg["eval"]["tile_size"],
                    tile_overlap=cfg["eval"]["tile_overlap"],
                    use_amp=cfg["eval"]["amp"] and device.type == "cuda",
                    frame_indices=indices,
                )[0]
                save_image_tensor(output, os.path.join(save_dir, os.path.basename(frame_path)))
                del clip, output


if __name__ == "__main__":
    main()
