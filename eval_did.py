"""DID 跨数据集评测：在 SDSD 训练的模型上跑 DID test，评估泛化能力。

用法:
    python eval_did.py --config configs/v58_deep.yaml --checkpoint outputs/sdsd_v58_deep/best.pth
"""
import argparse, os, glob, time
import numpy as np, torch, yaml
from PIL import Image
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models import TFSNet
from utils.inference import tiled_forward
from utils.metrics import tensor_psnr, tensor_ssim


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--config", type=str, required=True)
    p.add_argument("--checkpoint", type=str, required=True)
    p.add_argument("--did_root", type=str, default="/home/a1005/yzy/dataset/DID/test")
    p.add_argument("--max_videos", type=int, default=0, help="0=all, N=只跑前N个视频")
    p.add_argument("--max_frames", type=int, default=0, help="0=all, N=每视频只跑前N帧")
    return p.parse_args()


def read_image(path):
    im = Image.open(path).convert("RGB")
    arr = np.asarray(im, dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).contiguous()


def main():
    args = parse_args()
    cfg = yaml.safe_load(open(args.config))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {dev}", flush=True)

    model = TFSNet(
        in_channels=cfg["model"]["in_channels"],
        level_channels=tuple(cfg["model"]["level_channels"]),
        fused_channels=cfg["model"]["fused_channels"],
        use_soft_clamp=cfg["model"].get("use_soft_clamp", False),
        sace_offset_use_norm=cfg["model"].get("sace_offset_use_norm", False),
        sace_offset_kaiming_init=cfg["model"].get("sace_offset_kaiming_init", False),
        use_soft_median=cfg["model"].get("use_soft_median", True),
        use_nafblock=cfg["model"].get("use_nafblock", False),
        num_bottleneck_blocks=cfg["model"].get("num_bottleneck_blocks", 0),
        num_igrf_res_blocks=cfg["model"].get("num_igrf_res_blocks", 2),
    ).to(dev)

    ckpt = torch.load(args.checkpoint, map_location=dev, weights_only=False)
    sd = ckpt["model"]
    # Remap: v5.8 旧 checkpoint 用 Sequential 包裹 ResBlock (fuse.N.0.conv1 → fuse.N.conv1)
    remapped = {}
    for k, v in sd.items():
        parts = k.split(".")
        if len(parts) >= 4 and parts[-3] == "fuse" and parts[-2].isdigit() and parts[-1] == "0":
            # fuse.N.0.rest → fuse.N.rest
            remapped[".".join(parts[:-2] + parts[-1:])] = v
        elif len(parts) >= 5 and "fuse" in parts:
            # 更通用的 remap: fuse.X.0.Y → fuse.X.Y
            for i in range(len(parts) - 1):
                if parts[i] == "fuse" and i + 2 < len(parts) and parts[i + 2] == "0":
                    new_parts = parts[:i + 2] + parts[i + 3:]
                    remapped[".".join(new_parts)] = v
                    break
            else:
                remapped[k] = v
        else:
            remapped[k] = v
    model.load_state_dict(remapped, strict=False)
    model.eval()
    ep = ckpt.get("epoch", "?")
    best_psnr = ckpt.get("best_psnr", "?")
    print(f"Loaded {args.checkpoint} (epoch={ep}, best_psnr={best_psnr})", flush=True)

    ws = cfg["dataset"]["window_size"]
    half = ws // 2
    tile_size = cfg["eval"]["tile_size"]
    tile_overlap = cfg["eval"]["tile_overlap"]

    ll_root = os.path.join(args.did_root, "low-light")
    gt_root = os.path.join(args.did_root, "GT")
    videos = sorted([d for d in glob.glob(os.path.join(ll_root, "*")) if os.path.isdir(d)])
    if args.max_videos > 0:
        videos = videos[:args.max_videos]

    all_psnr, all_ssim = [], []
    t0 = time.time()

    with torch.no_grad():
        for vpath in videos:
            vname = os.path.basename(vpath)
            gtdir = os.path.join(gt_root, vname)
            if not os.path.isdir(gtdir):
                print(f"  skip {vname}: no GT", flush=True)
                continue

            model.clear_frame_cache()
            frames = sorted(glob.glob(os.path.join(vpath, "*")))
            gt_frames = sorted(glob.glob(os.path.join(gtdir, "*")))
            if args.max_frames > 0:
                frames = frames[:args.max_frames]
                gt_frames = gt_frames[:args.max_frames]

            psnrs, ssims = [], []
            for idx, fp in enumerate(frames):
                gtfp = gt_frames[idx] if idx < len(gt_frames) else gt_frames[-1]
                mx = len(frames) - 1
                indices = [min(max(idx + o, 0), mx) for o in range(-half, half + 1)]
                clip = torch.stack([read_image(frames[j]) for j in indices], 0).unsqueeze(0).to(dev)

                out = tiled_forward(
                    model=model, clip=clip, tile_size=tile_size,
                    tile_overlap=tile_overlap, use_amp=False, frame_indices=indices,
                )[0]
                tgt = read_image(gtfp).to(dev)
                if out.dim() == 3:
                    out = out.unsqueeze(0)
                if tgt.dim() == 3:
                    tgt = tgt.unsqueeze(0)
                psnrs.append(tensor_psnr(out, tgt))
                ssims.append(tensor_ssim(out, tgt))

                del clip, out, tgt
                if (idx + 1) % 20 == 0:
                    print(f"  {vname}: {idx+1}/{len(frames)} frames, PSNR={np.mean(psnrs):.2f}", flush=True)

            ap = np.mean(psnrs)
            as_ = np.mean(ssims)
            all_psnr.extend(psnrs)
            all_ssim.extend(ssims)
            print(f"{vname}: frames={len(psnrs)} PSNR={ap:.3f} SSIM={as_:.4f}", flush=True)

    dt = time.time() - t0
    print(f"\n=== DID OVERALL ({len(all_psnr)} frames, {dt:.1f}s) ===", flush=True)
    print(f"PSNR={np.mean(all_psnr):.4f} dB  SSIM={np.mean(all_ssim):.4f}", flush=True)


if __name__ == "__main__":
    main()
