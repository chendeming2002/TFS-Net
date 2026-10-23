#!/usr/bin/env python3
"""SGRFv9 结构诊断: 门控健康度、限幅统计、阶段反事实测试。

## 三类诊断

1. **门控健康度** (只对 SGRFv9 有意义): 门控的均值、标准差、分位数与饱和比例。
   只看均值不够 —— 门若在全图都是 0.5, 均值同样"健康", 但它没有做任何空间选择。
   因此同时报告标准差与分位数, 并统计门在空间上的变异系数。

2. **限幅统计**: 记录软限幅前后的 RMS 与最大绝对值, 以及进入明显压缩区间的元素比例。
   只报限幅后的数值无法判断限幅是否是瓶颈。

3. **阶段反事实**: 分别关闭 N / L / M 其中一段, 记录 PSNR / SSIM / LPIPS。
   关闭前序阶段会改变后续阶段的输入分布, 因此这一项只作为依赖程度的诊断,
   不能当作"该阶段学到了正确物理退化"的证据。

## 用法

    python scripts/diag_sgrf_meta.py --ckpt <ckpt> --config <cfg> [--n_seqs 10 --crop 256]

输出为可读表格, 同时写入 JSON 便于后续比较。
"""
import argparse
import json
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _read_image(path, crop):
    from PIL import Image
    import torchvision.transforms.functional as TF
    img = TF.to_tensor(Image.open(path).convert("RGB"))
    if crop:
        _, h, w = img.shape
        top, left = (h - crop) // 2, (w - crop) // 2
        img = img[:, top:top + crop, left:left + crop]
    return img


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--config", required=True)
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/test/low-light")
    p.add_argument("--gt_root", default="/home/a1005/yzy/dataset/SDSD/test/GT")
    p.add_argument("--n_seqs", type=int, default=10)
    p.add_argument("--crop", type=int, default=256)
    p.add_argument("--device", default="cuda")
    p.add_argument("--out_json", default=None)
    args = p.parse_args()

    import yaml
    from models.golf_v7r import GolfNet_v8
    from utils.io import load_state_dict_compat
    from utils.metrics import tensor_psnr, tensor_ssim

    cfg = yaml.safe_load(open(args.config))
    mk = dict(cfg.get("model", {}) or {})
    mk.pop("arch", None)
    dev = args.device if torch.cuda.is_available() else "cpu"
    model = GolfNet_v8(**mk).to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    load_state_dict_compat(model, ck["model_state_dict"])

    if getattr(model, "sgrf", None) is None:
        raise SystemExit("本脚本只适用于 SGRFv8 / SGRFv9 (fusion_mode 为 sgrf* )")
    is_v9 = model.sgrf.__class__.__name__ == "SGRFv9"
    print(f"RESULT loaded epoch {ck.get('epoch')} fusion={model.sgrf.__class__.__name__}",
          flush=True)

    try:
        import lpips
        lpips_fn = lpips.LPIPS(net="vgg", verbose=False).to(dev).eval()
    except Exception:
        lpips_fn = None
        print("WARNING: lpips 不可用, 跳过 LPIPS", flush=True)

    names = sorted(os.listdir(args.seq_root),
                   key=lambda x: int(x.replace("pair", "")) if x.replace("pair", "").isdigit() else 0)
    seqs = names[:args.n_seqs]

    # ---------- 门控与限幅统计 (v9 专有) ----------
    gate_stats = {k: [] for k in ("mean", "std", "p05", "p50", "p95", "sat", "cv")}
    gate_by_stage = {s: {k: [] for k in gate_stats} for s in "NLM"}
    clamp_stats = {s: {k: [] for k in ("rms_pre", "rms_post", "absmax_pre",
                                      "absmax_post", "frac_compress")}
                   for s in "NLM"}
    # ---------- 反事实 ----------
    conds = ["full", "no_N", "no_L", "no_M"]
    cf = {c: {"psnr": [], "ssim": [], "lpips": []} for c in conds}

    # 采集一次前向中的门控与限幅中间量
    captured = {}
    if is_v9:
        sg = model.sgrf
        orig_forward = sg.forward

        def probing_forward(*a, **kw):
            res = orig_forward(*a, **kw)
            return res

        # 钩住门控模块, 记录输出
        for name in "NLM":
            def mk_hook(nm):
                def hook(mod, inp, out):
                    g = out.detach()
                    bucket = gate_by_stage[nm]
                    bucket["mean"].append(g.mean().item())
                    bucket["std"].append(g.std().item())
                    q = torch.quantile(g.flatten().float(),
                                       torch.tensor([0.05, 0.5, 0.95], device=g.device))
                    bucket["p05"].append(q[0].item())
                    bucket["p50"].append(q[1].item())
                    bucket["p95"].append(q[2].item())
                    bucket["sat"].append(((g > 0.9) | (g < 0.1)).float().mean().item())
                    bucket["cv"].append((g.std() / (g.mean() + 1e-8)).item())
                return hook
            sg.gate[name].register_forward_hook(mk_hook(name))

    with torch.no_grad():
        for seq in seqs:
            low_dir = os.path.join(args.seq_root, seq)
            gt_dir = os.path.join(args.gt_root, seq)
            if not os.path.isdir(low_dir) or not os.path.isdir(gt_dir):
                continue
            lows = sorted(os.listdir(low_dir))
            gts = sorted(os.listdir(gt_dir))
            if not lows or not gts:
                continue
            c = len(lows) // 2
            idxs = [min(max(c + o, 0), len(lows) - 1) for o in (-2, -1, 0, 1, 2)]
            clip = torch.stack([_read_image(os.path.join(low_dir, lows[i]), args.crop)
                                for i in idxs], 0).to(dev).unsqueeze(0)
            gt = _read_image(os.path.join(gt_dir, gts[min(c, len(gts) - 1)]),
                             args.crop).to(dev).unsqueeze(0)

            for cond in conds:
                zs = None if cond == "full" else cond[-1]
                out = model(clip, zero_stage=zs)["final"]
                cf[cond]["psnr"].append(tensor_psnr(out, gt))
                cf[cond]["ssim"].append(tensor_ssim(out, gt))
                if lpips_fn is not None:
                    cf[cond]["lpips"].append(
                        lpips_fn(out * 2 - 1, gt * 2 - 1).item())

    print(f"RESULT {len(seqs)} sequences, crop={args.crop}", flush=True)

    # ---------- 输出 ----------
    out = {"epoch": ck.get("epoch"), "fusion": model.sgrf.__class__.__name__}

    if is_v9:
        print("\n门控健康度 (每个位置一个门值, 统计自单个中心裁剪):", flush=True)
        hdr = f"{'段':<4}{'均值':>8}{'标准差':>9}{'变异系数':>10}{'p05':>8}{'p50':>8}{'p95':>8}{'饱和率':>9}"
        print(hdr, flush=True)
        gs = {}
        for s in "NLM":
            b = gate_by_stage[s]
            m = {k: float(np.mean(v)) for k, v in b.items()}
            gs[s] = m
            print(f"{s:<4}{m['mean']:>8.3f}{m['std']:>9.3f}{m['cv']:>10.3f}"
                  f"{m['p05']:>8.3f}{m['p50']:>8.3f}{m['p95']:>8.3f}{m['sat']:>9.3f}",
                  flush=True)
        out["gates"] = gs
        print("\n说明: 变异系数 = 标准差/均值。若接近 0, 门在全图近乎常数, 没有做空间选择。",
              flush=True)

        amps = model.sgrf._amp_all().detach().tolist()
        print(f"\n阶段幅度: N={amps[model.sgrf.stage_order.index('N')]:.3f} "
              f"L={amps[model.sgrf.stage_order.index('L')]:.3f} "
              f"M={amps[model.sgrf.stage_order.index('M')]:.3f}", flush=True)
        out["stage_amp"] = {s: amps[model.sgrf.stage_order.index(s)] for s in "NLM"}

    print("\n阶段反事实 (每次只关闭一段):", flush=True)
    print(f"{'条件':<8}{'PSNR':>9}{'SSIM':>9}{'LPIPS':>9}{'ΔPSNR':>9}", flush=True)
    base = float(np.mean(cf["full"]["psnr"]))
    cf_out = {}
    for cond in conds:
        ps = float(np.mean(cf[cond]["psnr"]))
        ss = float(np.mean(cf[cond]["ssim"]))
        lp = float(np.mean(cf[cond]["lpips"])) if cf[cond]["lpips"] else float("nan")
        cf_out[cond] = {"psnr": ps, "ssim": ss, "lpips": lp,
                        "delta_psnr": ps - base}
        print(f"{cond:<8}{ps:>9.3f}{ss:>9.4f}{lp:>9.4f}{ps - base:>+9.3f}", flush=True)
    out["counterfactual"] = cf_out
    print("\n注意: 关闭前序阶段会改变后续阶段的输入分布, 这一项只反映依赖程度, "
          "不是因果证明。", flush=True)

    if args.out_json:
        os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
        with open(args.out_json, "w") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)
        print(f"saved -> {args.out_json}", flush=True)


if __name__ == "__main__":
    main()
