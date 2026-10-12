#!/usr/bin/env python3
"""多 seed 双臂配对评估 (对应 docs/v7/03-v7r-v3-design.md §6.17, §6.14)。

在训练序列上逐序列算 PSNR, 按臂聚合, 输出:
  · 每个 run 的 micro / 中位 PSNR;
  · 臂间配对差 (处理臂 seed 均值 − 对照臂 seed 均值): 中位差、均值差、逐序列胜负计数;
  · 臂内 seed 间差 —— 用于判断臂间差异是否被训练随机性覆盖。

判读原则 (§6.14): 本机 5ep 训练的 seed 间方差实测可达 3 dB,
故【臂间差必须显著大于臂内 seed 差】才能算作效应。逐序列 JSON 落到
outputs/multiseed_summary/per_seq_psnr.json, 供后续复查。

用法:
    PYTHONPATH=. python scripts/eval_multiseed_arms.py \
        --control outputs/multiseed_ref_softmax_runs \
        --control_seeds 0 1 2 \
        --treated outputs/multiseed_decorr_runs \
        --treated_seeds 0 1
"""
import argparse
import json
import os

import numpy as np
import torch
from PIL import Image

from models.golf_v7r import GolfNet_v7r_v3


def read_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def find_config(run_dir, seed):
    """临时配置优先 (保存了实际训练用的完整参数), 否则退回臂的基准配置。"""
    for name in (f"_cfg_seed{seed}.yaml", f"seed{seed}/_cfg.yaml"):
        path = os.path.join(run_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"找不到 seed{seed} 的配置: {run_dir}")


def eval_run(ckpt, cfg, seq_root, gt_root, dev):
    import yaml
    with open(cfg) as f:
        model_kwargs = (yaml.safe_load(f) or {}).get("model", {}) or {}
    model = GolfNet_v7r_v3(**model_kwargs).to(dev).eval()
    model.load_state_dict(torch.load(ckpt, map_location=dev, weights_only=False)["model_state_dict"])

    per_seq = {}
    for seq in sorted(os.listdir(seq_root), key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0):
        seq_in = os.path.join(seq_root, seq)
        seq_gt = os.path.join(gt_root, seq)
        if not os.path.isdir(seq_in) or not os.path.isdir(seq_gt):
            continue
        lq = sorted(os.path.join(seq_in, f) for f in os.listdir(seq_in))
        gq = sorted(os.path.join(seq_gt, f) for f in os.listdir(seq_gt))
        if not lq or not gq:
            continue
        center = len(lq) // 2
        idxs = [min(max(center + off, 0), len(lq) - 1) for off in (-2, -1, 0, 1, 2)]
        clip = torch.stack([read_image(lq[i]) for i in idxs], 0).unsqueeze(0).to(dev)
        gt = read_image(gq[min(center, len(gq) - 1)]).unsqueeze(0).to(dev)
        with torch.no_grad():
            out = model(clip)
        per_seq[seq] = psnr(out["final"].clamp(0, 1), gt)
    return per_seq


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--control", default="outputs/multiseed_ref_softmax_runs")
    p.add_argument("--control_seeds", type=int, nargs="+", default=[0, 1, 2])
    p.add_argument("--treated", default="outputs/multiseed_decorr_runs")
    p.add_argument("--treated_seeds", type=int, nargs="+", default=[0, 1])
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/indoor/input")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/indoor/GT")
    p.add_argument("--out_json", default="outputs/multiseed_summary/per_seq_psnr.json")
    args = p.parse_args()

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    runs = {}
    for arm, root, seeds in (("control", args.control, args.control_seeds),
                             ("treated", args.treated, args.treated_seeds)):
        for sd in seeds:
            ckpt = os.path.join(root, f"seed{sd}", "best.pth")
            if not os.path.isfile(ckpt):
                print(f"RESULT skip {arm}-seed{sd}: 无 best.pth", flush=True)
                continue
            tag = f"{arm}-seed{sd}"
            runs[tag] = eval_run(ckpt, find_config(root, sd), args.seq_root, args.gt_root, dev)
            arr = np.array(list(runs[tag].values()))
            print(f"RESULT {tag}: micro={arr.mean():.3f} dB  median={np.median(arr):.3f}  n={len(arr)}", flush=True)

    os.makedirs(os.path.dirname(args.out_json), exist_ok=True)
    with open(args.out_json, "w") as f:
        json.dump({"per_seq": runs}, f, indent=1)
    print(f"RESULT saved {args.out_json}", flush=True)

    ctrl = [v for k, v in runs.items() if k.startswith("control-")]
    trt = [v for k, v in runs.items() if k.startswith("treated-")]
    if len(ctrl) < 2 or len(trt) < 2:
        print("RESULT 每臂需 >=2 个 seed 才能比较", flush=True)
        return

    seqs = sorted(set(ctrl[0]) & set(trt[0]))
    # 臂内 seed 差 (对照臂): 用于判断臂间差是否落在随机性范围内
    within = max(np.mean([abs(c1[s] - c2[s]) for s in seqs])
                 for i, c1 in enumerate(ctrl) for c2 in ctrl[i + 1:])
    c_mean = np.array([np.mean([c[s] for c in ctrl]) for s in seqs])
    t_mean = np.array([np.mean([t[s] for t in trt]) for s in seqs])
    delta = t_mean - c_mean
    print(f"RESULT 臂内 seed 差 (对照臂最大) = {within:.3f} dB", flush=True)
    print(f"RESULT 臂间配对差: median={np.median(delta):+.3f}, mean={delta.mean():+.3f} dB", flush=True)
    print(f"RESULT 处理臂胜 {int((delta > 0).sum())}/{len(delta)}, 负 {int((delta < 0).sum())}/{len(delta)}", flush=True)
    verdict = "臂间差小于臂内 seed 差 → 无效应" if delta.mean() < within else "臂间差超过臂内 seed 差 → 需更多 seed 确认"
    print(f"RESULT 判读: {verdict}", flush=True)


if __name__ == "__main__":
    main()
