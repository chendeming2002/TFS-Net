"""X3 分支消融链式实验 — Flight11 终判回退分析 (FSD-plan §回退路径).

用 sdsd_f11_simple2/best.pth (ep30) 在 pair45 (131帧全量) 上做三组单开关消融,
顺序自动执行: base → X3-a (luckiness→恒等) → X3-b (disp_field→零) → X3-c (s_illum→常数0).

消融实现 (模型实例 monkey-patch, 不改源码):
  X3-a: ndpn.luck_delta ← log(1e4) → luck_weights = exp(-r²/2δ²) ≡ 1.0 (fp32 下精确恒等)
  X3-b: tca.forward 包裹 → 输出 dict 的 disp_field 置零 → motion_mag = 1-exp(0) ≈ 0
  X3-c: ispn.forward 包裹 → s_illum 置零 (通道 concat 的条件输入无信息化)

判据: 哪组相对 base 回升 → 定位拖累源 (A4: luckiness/disp_field, A1: s_illum 条件).
评估: 位置配对 (输入 0047 起 / GT 0107 起, 帧号偏移 60), RGB PSNR + 全局统计 SSIM
      (与 2026-09-13 终判 13.51/0.686 同方法, 组间可比).
"""
import json
import math
import os
import sys
import time
from glob import glob

import numpy as np
import torch
import yaml
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from models import TFSNet
from utils.inference import tiled_forward
from utils.io import ensure_dir, save_image_tensor

CFG_PATH = "configs/delta_flight11_simple.yaml"
CKPT_PATH = "outputs/sdsd_f11_simple2/best.pth"
IN_ROOT = "/home/a1005/yzy/dataset/SDSD/test/low-light/pair45"
GT_ROOT = "/home/a1005/yzy/dataset/SDSD/test/GT/pair45"
OUT_ROOT = "outputs/x3_ablation"
GROUPS = ["base", "x3a_luck_off", "x3b_disp_off", "x3c_sillum_const"]


def build_model(device):
    cfg = yaml.safe_load(open(CFG_PATH))
    m = cfg["model"]
    model = TFSNet(
        in_channels=m["in_channels"],
        level_channels=tuple(m["level_channels"]),
        fused_channels=m["fused_channels"],
        use_pure_rwkv=m.get("use_pure_rwkv", False),
        use_soft_clamp=m.get("use_soft_clamp", False),
        use_soft_median=m.get("use_soft_median", True),
        use_nafblock=m.get("use_nafblock", False),
        num_bottleneck_blocks=m.get("num_bottleneck_blocks", 0),
        num_igrf_res_blocks=m.get("num_igrf_res_blocks", 2),
        use_local_tca=m.get("use_local_tca", False),
        tca_bootstrap=m.get("tca_bootstrap", "bias"),
    ).to(device)
    ckpt = torch.load(CKPT_PATH, map_location=device)
    model.load_state_dict(ckpt["model"], strict=True)
    model.eval()
    return model, cfg


def apply_ablation(model, name):
    if name == "base":
        return
    if name == "x3a_luck_off":
        with torch.no_grad():
            model.ndpn.luck_delta.data.fill_(math.log(1e4))
    elif name == "x3b_disp_off":
        orig = model.tca.forward

        def patched(*a, **k):
            out = orig(*a, **k)
            out["disp_field"] = torch.zeros_like(out["disp_field"])
            return out

        model.tca.forward = patched
    elif name == "x3c_sillum_const":
        orig = model.ispn.forward

        def patched(f_enc, s_illum, *a, **k):
            return orig(f_enc, torch.zeros_like(s_illum), *a, **k)

        model.ispn.forward = patched
    else:
        raise ValueError(name)


def read_image(path):
    return np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0


def gather_clip(paths, center_idx, window_size):
    half = window_size // 2
    max_idx = len(paths) - 1
    frames = []
    for offset in range(-half, half + 1):
        idx = min(max(center_idx + offset, 0), max_idx)
        frames.append(torch.from_numpy(np.asarray(Image.open(paths[idx]).convert("RGB")).transpose(2, 0, 1).astype(np.float32) / 255.0))
    return torch.stack(frames, dim=0)


def run_group(name, device):
    t0 = time.time()
    model, cfg = build_model(device)
    apply_ablation(model, name)
    save_dir = os.path.join(OUT_ROOT, name, "pair45")
    ensure_dir(save_dir)

    frame_paths = sorted(glob(os.path.join(IN_ROOT, "*")))
    window_size = cfg["dataset"]["window_size"]
    half = window_size // 2
    max_idx = len(frame_paths) - 1

    model.clear_frame_cache()
    with torch.no_grad():
        for idx, frame_path in enumerate(frame_paths):
            indices = [min(max(idx + o, 0), max_idx) for o in range(-half, half + 1)]
            clip = gather_clip(frame_paths, idx, window_size).unsqueeze(0).to(device)
            output = tiled_forward(
                model=model, clip=clip,
                tile_size=cfg["eval"]["tile_size"],
                tile_overlap=cfg["eval"]["tile_overlap"],
                use_amp=False, frame_indices=indices,
            )[0]
            save_image_tensor(output, os.path.join(save_dir, os.path.basename(frame_path)))
            del clip, output
    del model
    torch.cuda.empty_cache()

    outs = sorted(glob(os.path.join(save_dir, "*")))
    gts = sorted(glob(os.path.join(GT_ROOT, "*")))
    assert len(outs) == len(gts) == 131, f"{name}: {len(outs)} vs {len(gts)}"
    C1, C2 = 0.01 ** 2, 0.03 ** 2
    psnrs, ssims = [], []
    for op, gp in zip(outs, gts):
        a, b = read_image(op), read_image(gp)
        psnrs.append(10 * np.log10(1.0 / max(((a - b) ** 2).mean(), 1e-10)))
        ss = []
        for c in range(3):
            x, y = a[:, :, c], b[:, :, c]
            mx, my = x.mean(), y.mean()
            vx, vy = ((x - mx) ** 2).mean(), ((y - my) ** 2).mean()
            cxy = ((x - mx) * (y - my)).mean()
            ss.append(((2 * mx * my + C1) * (2 * cxy + C2)) / ((vx + vy + C1) * (mx * mx + my * my + C2)))
        ssims.append(np.mean(ss))
    return {
        "group": name, "psnr": round(float(np.mean(psnrs)), 3),
        "ssim": round(float(np.mean(ssims)), 4),
        "frames": len(psnrs), "seconds": round(time.time() - t0, 1),
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ensure_dir(OUT_ROOT)
    results_path = os.path.join(OUT_ROOT, "results.json")
    results = []
    print(f"[X3] device={device} ckpt={CKPT_PATH}", flush=True)
    for name in GROUPS:
        print(f"[X3] === {name} 开始 ===", flush=True)
        try:
            r = run_group(name, device)
        except Exception as e:
            r = {"group": name, "error": str(e)}
        results.append(r)
        print(f"[X3] {name}: {json.dumps(r, ensure_ascii=False)}", flush=True)
        with open(results_path, "w") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
    ok = [r for r in results if "psnr" in r]
    if len(ok) >= 2:
        base = next((r["psnr"] for r in ok if r["group"] == "base"), None)
        print("\n[X3] ===== 汇总 (pair45, 131帧, 位置配对) =====", flush=True)
        print(f"[X3] {'组':<16} {'PSNR':>7} {'ΔvsBase':>8} {'SSIM':>7}", flush=True)
        for r in ok:
            d = (r["psnr"] - base) if base is not None else float("nan")
            print(f"[X3] {r['group']:<16} {r['psnr']:>7.2f} {d:>+8.2f} {r['ssim']:>7.3f}", flush=True)
        print("[X3] 参照: 终判 base=13.51/0.686 | TBC1B@40=15.69/0.697 | v2@49=15.04", flush=True)


if __name__ == "__main__":
    main()
