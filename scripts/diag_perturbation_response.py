#!/usr/bin/env python3
"""三分支分工的扰动响应测试。

问题: 三路分支的设计分工是 N 管噪声、L 管光照、M 管运动。这个分工在模型里是否成立?
做法: 对同一批测试片段施加三类退化, 观察每一路的输出如何变化:
    dark  — 整体乘以系数 (压暗), 考察 L 路是否对亮度变化最敏感;
    noise — 每帧独立叠加高斯噪声, 考察 N 路是否对噪声最敏感;
    blur  — 每帧施加同一方向的水平运动模糊, 考察 M 路是否对运动最敏感。
GT 保持不变, 因此最终 PSNR 反映的是模型对退化的稳健程度。

每路的指标:
    change  — 扰动后该路输出相对干净输入时输出的变化量 (平均绝对值, 除以该路目标的平均绝对值;
              目标尺度与输入无关, 三路因此可以直接比较);
    rel_err — 该路输出与【自身频带目标】的平均绝对误差, 除以目标自身的平均绝对值。
              1.0 等于输出全零, 小于 1 才说明这一路真的学到了内容。
              目标由干净 GT 构造, 扰动前后保持不变;
    drop_dB — 加性融合下把该路置零 (不重新归一化) 后, 最终 PSNR 的变化量。

注意: 三路的训练目标是 GT 的频带 (低频 / 中频 / 高频), 不是输入中的噪声或运动。
因此"N 对噪声敏感"这一判读是对设计意图的检验, 结果为负并不自动说明代码有错, 而可能说明
频带划分与退化类型并不一一对应。

用法:
    PYTHONPATH=. python scripts/diag_perturbation_response.py \
        --ckpt outputs/v8_unet_refactor_15ep/best.pth \
        --config configs/v8/unet_refactor_15ep.yaml --n_seqs 10 --crop 256
"""
import argparse
import os

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.branch_targets import build_branch_targets
from utils.io import load_state_dict_compat

CONDITIONS = [
    ("clean", 0.0),
    ("dark", 0.5),
    ("dark", 0.25),
    ("noise", 0.02),
    ("noise", 0.05),
    ("blur", 5),
    ("blur", 9),
]

BRANCH_KEYS = ("N", "L", "M")
TARGET_KEYS = {"N": "noise_target", "L": "illum_target", "M": "motion_target"}


def read_image(path, crop):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    img = torch.from_numpy(arr).permute(2, 0, 1)
    if crop:
        h, w = img.shape[-2:]
        top, left = (h - crop) // 2, (w - crop) // 2
        img = img[:, top:top + crop, left:left + crop]
    return img


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def mean_abs(x):
    return x.abs().mean().item()


def degrade(clip, kind, level):
    """clip: [T, 3, H, W], 值域 [0, 1]。"""
    if kind == "clean":
        return clip
    if kind == "dark":
        return clip * level
    if kind == "noise":
        gen = torch.Generator().manual_seed(0)
        noise = torch.randn(clip.shape, generator=gen).to(clip.device) * level
        return (clip + noise).clamp(0.0, 1.0)
    if kind == "blur":
        k = int(level)
        kernel = torch.full((3, 1, 1, k), 1.0 / k, device=clip.device)
        padded = F.pad(clip, (k // 2, k // 2, 0, 0), mode="reflect")
        return F.conv2d(padded, kernel, groups=3)
    raise ValueError(f"未知退化类型: {kind}")


def compose(model, outs, weights, x_center):
    """与模型融合路径一致的加性融合, 不做重新归一化。"""
    fused = (weights[:, 0:1] * outs["N"] + weights[:, 1:2] * outs["L"]
             + weights[:, 2:3] * outs["M"])
    refined = model.fusion.refine(fused)
    if model.fusion.residual_freeze:
        gamma = torch.zeros((), device=refined.device, dtype=refined.dtype)
    else:
        gamma = torch.tanh(model.fusion.residual_gamma)
    return (refined + gamma * x_center).clamp(0.0, 1.0)


def run_model(model, clip):
    out = model(clip.unsqueeze(0))
    branch = {"N": out["branch_N"], "L": out["branch_L"], "M": out["branch_M"]}
    return branch, out["fusion_weights"], out["final"]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/test/low-light")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/test/GT")
    p.add_argument("--n_seqs", type=int, default=10)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    import yaml
    from models.golf_v7r import GolfNet_v8

    with open(args.config) as f:
        cfg = yaml.safe_load(f)
    model_kwargs = dict(cfg.get("model", {}) or {})
    arch = model_kwargs.pop("arch", "v7r_v3")
    if arch != "v8":
        raise SystemExit(f"本脚本只支持 arch=v8, 收到 {arch}")

    dev = args.device if torch.cuda.is_available() else "cpu"
    model = GolfNet_v8(**model_kwargs).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    load_state_dict_compat(model, ck['model_state_dict'])
    if getattr(model.fusion, "mode", "") != "additive":
        raise SystemExit("本脚本的置零消融只适用于 additive 融合")
    print(f"RESULT loaded epoch {ck.get('epoch')}, fusion=additive, "
          f"residual_freeze={model.fusion.residual_freeze}", flush=True)

    tgt_cfg = {
        "kernel_low": int(cfg["dataset"].get("branch_target_kernel_low", 31)),
        "kernel_high": int(cfg["dataset"].get("branch_target_kernel_high", 5)),
        "motion_share_mid": float(cfg["dataset"].get("branch_target_motion_share_mid", 0.5)),
        "motion_share_high": float(cfg["dataset"].get("branch_target_motion_share_high", 0.2)),
    }

    names = sorted(os.listdir(args.seq_root),
                   key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0)
    seqs = names[:args.n_seqs]

    keys = ["psnr"] + [f"change_{b}" for b in BRANCH_KEYS] \
        + [f"rel_{b}" for b in BRANCH_KEYS] + [f"drop_{b}" for b in BRANCH_KEYS] \
        + [f"w_{b}" for b in BRANCH_KEYS]
    table = {cond: {k: [] for k in keys} for cond in CONDITIONS}

    with torch.no_grad():
        for seq in seqs:
            low_dir = os.path.join(args.seq_root, seq)
            gt_dir = os.path.join(args.gt_root, seq)
            low_files = sorted(os.listdir(low_dir))
            gt_files = sorted(os.listdir(gt_dir))
            if not low_files or not gt_files:
                continue
            center = len(low_files) // 2
            idxs = [min(max(center + off, 0), len(low_files) - 1) for off in (-2, -1, 0, 1, 2)]
            clean_clip = torch.stack(
                [read_image(os.path.join(low_dir, low_files[i]), args.crop) for i in idxs], 0
            ).to(dev)
            gt = read_image(os.path.join(gt_dir, gt_files[min(center, len(gt_files) - 1)]),
                            args.crop).unsqueeze(0).to(dev)

            targets = build_branch_targets(gt, **tgt_cfg)
            targets = {b: targets[TARGET_KEYS[b]] for b in BRANCH_KEYS}

            clean_branch, _, _ = run_model(model, clean_clip)

            for kind, level in CONDITIONS:
                clip = degrade(clean_clip, kind, level)
                branch, weights, final = run_model(model, clip)
                x_center = clip[2:3]

                row = table[(kind, level)]
                row["psnr"].append(psnr(final, gt))
                for b in BRANCH_KEYS:
                    row[f"change_{b}"].append(
                        mean_abs(branch[b] - clean_branch[b]) / max(mean_abs(targets[b]), 1e-8))
                    row[f"rel_{b}"].append(
                        mean_abs(branch[b] - targets[b]) / max(mean_abs(targets[b]), 1e-8))
                    row[f"w_{b}"].append(weights[:, BRANCH_KEYS.index(b)].mean().item())

                base = psnr(compose(model, branch, weights, x_center), gt)
                for drop_idx, b in enumerate(BRANCH_KEYS):
                    w = weights.clone()
                    w[:, drop_idx:drop_idx + 1] = 0.0
                    dropped = compose(model, branch, w, x_center)
                    row[f"drop_{b}"].append(psnr(dropped, gt) - base)

    print(f"RESULT {len(seqs)} sequences, crop={args.crop}", flush=True)
    header = (f"{'条件':<12}{'PSNR':>8}  {'变化 N/L/M':>22}  {'相对误差 N/L/M':>24}  "
              f"{'置零代价 dB N/L/M':>26}  {'权重 N/L/M':>22}")
    print(header, flush=True)
    for cond in CONDITIONS:
        row = table[cond]
        name = "clean" if cond[0] == "clean" else f"{cond[0]}{cond[1]}"

        def triple(prefix, fmt):
            return "/".join(fmt.format(np.mean(row[f"{prefix}_{b}"])) for b in BRANCH_KEYS)

        print(f"{name:<12}{np.mean(row['psnr']):>8.2f}  "
              f"{triple('change', '{:.3f}'):>22}  {triple('rel', '{:.3f}'):>24}  "
              f"{triple('drop', '{:+.3f}'):>26}  {triple('w', '{:.3f}'):>22}", flush=True)


if __name__ == "__main__":
    main()
