"""Mark4 推理：best/latest 权重 × SDSD/DID 各 1 序列，逐帧保存"""
import os, sys, glob
import torch
import yaml
import numpy as np
from PIL import Image
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(__file__))
from models import TFSNet
from utils.inference import tiled_forward

DEVICE = torch.device("cuda")
SDSD_ROOT = "/home/a1005/yzy/dataset/SDSD/test/low-light"
DID_ROOT = "/home/a1005/yzy/dataset/DID/test/low-light"
CONFIG = "configs/v6_bravo.yaml"
OUT_BASE = "outputs/sdsd_delta/infer"

# sequences to test
SEQS = {"SDSD": "pair19", "DID": "video10"}
WEIGHTS = {"best": "outputs/sdsd_delta/best.pth", "latest": "outputs/sdsd_delta/latest.pth"}


def read_image(path):
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def gather_clip(paths, center_idx, window_size=5):
    half = window_size // 2
    max_idx = len(paths) - 1
    frames = []
    for offset in range(-half, half + 1):
        idx = min(max(center_idx + offset, 0), max_idx)
        frames.append(read_image(paths[idx]))
    return torch.stack(frames, dim=0).unsqueeze(0)


if __name__ == "__main__":
    with open(CONFIG) as f:
        cfg = yaml.safe_load(f)

    cfg_model = cfg["model"]
    model = TFSNet(
        in_channels=cfg_model["in_channels"],
        level_channels=tuple(cfg_model["level_channels"]),
        fused_channels=cfg_model["fused_channels"],
        use_dwt_lff=cfg_model.get("use_dwt_lff", False),
        use_pure_rwkv=cfg_model.get("use_pure_rwkv", False),
        use_soft_clamp=cfg_model.get("use_soft_clamp", False),
        use_soft_median=cfg_model.get("use_soft_median", True),
        use_nafblock=cfg_model.get("use_nafblock", False),
        num_igrf_res_blocks=cfg_model.get("num_igrf_res_blocks", 2),
        use_amp_enhance=cfg_model.get("use_amp_enhance", False),
    ).to(DEVICE)

    for wname, wpath in WEIGHTS.items():
        ckpt = torch.load(wpath, map_location=DEVICE, weights_only=False)
        model.load_state_dict(ckpt["model"], strict=False)
        model.eval()
        ep = ckpt.get("epoch", "?")
        print(f"\n{'='*60}")
        print(f"权重: {wname} (epoch {ep})")

        for dname, seq_name in SEQS.items():
            root = SDSD_ROOT if dname == "SDSD" else DID_ROOT
            seq_path = os.path.join(root, seq_name)
            frames = sorted(glob.glob(os.path.join(seq_path, "*")))
            out_dir = os.path.join(OUT_BASE, f"{dname}_{seq_name}_{wname}_ep{ep}")
            os.makedirs(out_dir, exist_ok=True)

            print(f"  {dname}/{seq_name}: {len(frames)} frames → {out_dir}")

            with torch.no_grad():
                for idx in tqdm(range(len(frames)), desc=f"  {wname} {dname}/{seq_name}", leave=False):
                    clip = gather_clip(frames, idx, 5).to(DEVICE)
                    out = tiled_forward(model, clip, tile_size=cfg["eval"]["tile_size"],
                                        tile_overlap=cfg["eval"]["tile_overlap"],
                                        use_amp=False, phase='phase2')
                    out_img = out[0].cpu().clamp(0, 1)
                    out_arr = (out_img.permute(1, 2, 0).numpy() * 255).astype(np.uint8)
                    fname = os.path.basename(frames[idx])
                    Image.fromarray(out_arr).save(os.path.join(out_dir, fname))
                    del clip, out, out_img

        torch.cuda.empty_cache()

    print(f"\n完成！输出目录: {OUT_BASE}/")
