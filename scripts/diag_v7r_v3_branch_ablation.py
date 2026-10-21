#!/usr/bin/env python3
"""v7r-v3 三分支功能贡献消融 (对应 docs/v7/03-v7r-v3-design.md §6.17 附带发现)。

问题: 三路分支 (N/L/M) 各自都能重建完整图像, 那么融合层保留了几个真正有用的?
做法: 不重新训练, 只改已训练模型的融合环节 —— 把某一路的融合权重置 0,
      其余两路按比例重新归一化 (保持凸组合), 再走完精化卷积与中心帧残差,
      与 GT 比 PSNR, 与不做消融的结果相减。

判读: 若去掉某一路后指标基本不变 (Δ≈0), 说明该路对最终输出几乎无贡献;
      若去掉后显著变差, 说明该路确实承担了重建任务。
      注意这是【事后】改动融合权重, 不是重新训练; 结论应读作
      "在已达成的解附近, 各路的可替代程度", 而非"删掉一路能提升指标"。

用法:
    PYTHONPATH=. python scripts/diag_v7r_v3_branch_ablation.py \
        --ckpt outputs/multiseed_ref_softmax_runs/seed0/best.pth \
        --config outputs/multiseed_ref_softmax_runs/_cfg_seed0.yaml \
        --n_seqs 24
"""
import argparse
import os

import numpy as np
import torch
from PIL import Image

from models.golf_v7r import GolfNet_v7r_v3, GolfNet_v8
from utils.io import load_state_dict_compat


def read_image(path):
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1)


def psnr(pred, target):
    mse = ((pred - target) ** 2).mean().item()
    return 10.0 * np.log10(1.0 / max(mse, 1e-10))


def list_sequences(root, limit):
    names = os.listdir(root)
    names.sort(key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0)
    return names[:limit]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True,
                   help="训练配置文件; 必须传, 否则 motion_aware_diff 等无参数开关会静默走错路径")
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/indoor/input")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/indoor/GT")
    p.add_argument("--n_seqs", type=int, default=24)
    args = p.parse_args()

    import yaml
    with open(args.config) as f:
        model_kwargs = dict((yaml.safe_load(f) or {}).get("model", {}) or {})

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    arch = model_kwargs.pop("arch", "v7r_v3")
    model_cls = GolfNet_v8 if arch == "v8" else GolfNet_v7r_v3
    model = model_cls(**model_kwargs).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    load_state_dict_compat(model, ck['model_state_dict'])
    print(f"RESULT loaded epoch {ck.get('epoch')} from {args.ckpt}", flush=True)
    print(f"RESULT fusion mode={getattr(model.fusion, 'mode', 'softmax')}, "
          f"residual_freeze={bool(getattr(model.fusion, 'residual_freeze', False))} "
          f"(消融口径: {'加性-置零' if getattr(model.fusion, 'mode', '') == 'additive' else '凸组合-归一化'})",
          flush=True)

    seqs = list_sequences(args.seq_root, args.n_seqs)
    deltas = {"minusN": [], "minusL": [], "minusM": []}

    for seq in seqs:
        lq = sorted(os.path.join(args.seq_root, seq, f)
                    for f in os.listdir(os.path.join(args.seq_root, seq)))
        gq = sorted(os.path.join(args.gt_root, seq, f)
                    for f in os.listdir(os.path.join(args.gt_root, seq)))
        if not lq or not gq:
            continue
        center = len(lq) // 2
        idxs = [min(max(center + off, 0), len(lq) - 1) for off in (-2, -1, 0, 1, 2)]
        clip = torch.stack([read_image(lq[i]) for i in idxs], 0).unsqueeze(0).to(dev)
        gt = read_image(gq[min(center, len(gq) - 1)]).unsqueeze(0).to(dev)

        with torch.no_grad():
            out = model(clip)
            weights = out["fusion_weights"]
            x_center = clip[:, 2]
            fusion = model.fusion
            y_n, y_l, y_m = out["branch_N"], out["branch_L"], out["branch_M"]

            # ⚠️ 融合语义必须与模型实际一致, 否则消融数字失真 (2026-10-16 修正):
            #   softmax / floored → 凸组合 Σw=1, 去掉一路后其余按比例归一化才是"等预算"消融;
            #   additive          → 权重【不归一化】(softplus, 初始=1, ΣS=GT 的物理前提),
            #                       去掉一路应直接把该项置 0, 归一化会把剩余路放大 ~1.8×,
            #                       制造出 +8 dB 量级的假衰减 (实测: 脚本口径 -8.11 dB
            #                       vs 正确加性口径 +0.08 dB, 见 §6.20)。
            additive = getattr(fusion, "mode", "softmax") == "additive"
            freeze = bool(getattr(fusion, "residual_freeze", False))

            def compose(w_n, w_l, w_m):
                fused = w_n * y_n + w_l * y_l + w_m * y_m
                refined = fusion.refine(fused)
                # 冻结时模型硬置 gamma=0 (零张量不参与), 此处必须一致
                gamma = (torch.zeros((), device=refined.device, dtype=refined.dtype)
                         if freeze else torch.tanh(fusion.residual_gamma))
                return (refined + gamma * x_center).clamp(0.0, 1.0)

            base = psnr(compose(weights[:, 0:1], weights[:, 1:2], weights[:, 2:3]), gt)

            # 去掉某一路
            for name, drop_idx in (("minusN", 0), ("minusL", 1), ("minusM", 2)):
                w = weights.clone()
                w[:, drop_idx:drop_idx + 1] = 0.0
                if not additive:
                    # 凸组合: 其余两路按比例归一化 (保持 Σw=1)
                    w = w / w.sum(dim=1, keepdim=True).clamp_min(1e-8)
                deltas[name].append(psnr(compose(w[:, 0:1], w[:, 1:2], w[:, 2:3]), gt) - base)

    print(f"RESULT {len(seqs)} sequences", flush=True)
    for name, vals in deltas.items():
        v = np.array(vals)
        better = int((v > 0).sum())
        worse = int((v < 0).sum())
        print(f"RESULT {name}: mean_delta={v.mean():+.3f} dB, median={np.median(v):+.3f}, "
              f"better={better}/{len(v)}, worse={worse}/{len(v)}", flush=True)


if __name__ == "__main__":
    main()
