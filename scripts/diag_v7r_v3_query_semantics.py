#!/usr/bin/env python3
"""§6.3 三路 Query 语义分工诊断 (对应 docs/v7/03-v7r-v3-design.md §6.3)。

背景: TripleQueryTCA 用三路查询 Q_N/L/M 共享 KV, 声称分别对应
噪声/光照/运动。但 §5.5.6 已发现三路 RGB 输出余弦 0.99+。本脚本直接检查
三路「查询图」与「注意力输出图」是否在空间上确实不同, 并用受控退化
(合成噪声/压暗/位移) 检验三路响应能否分离。

RWKVSpatialHead 是【递推 WKV】而非 softmax 注意力, 没有可画的注意力矩阵,
故以「通道均值绝对激活」作为空间响应 proxy。

输出:
  <out_dir>/<seq>_maps.png      每序列: 输入/GT + 三路 Q 能量 + 三路 attn 能量
  <out_dir>/summary.json        路径间相关系数 + 受控退化的响应变化

用法:
    PYTHONPATH=/home/a1005/25/TFS-Net python scripts/diag_v7r_v3_query_semantics.py \
        --ckpt outputs/golf_v7r_v3_pospair/best.pth
"""
import argparse
import glob
import json
import os

import numpy as np
import torch
import torch.nn.functional as Fn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

from models.golf_v7r import GolfNet_v7r_v3


def read_image(path):
    a = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(a).permute(2, 0, 1)


def build_clip(seq_root, seq, num_frames=5, center_frac=0.5):
    lq = sorted(glob.glob(os.path.join(seq_root, seq, "*")))
    ci = int(len(lq) * center_frac)
    ci = min(max(ci, num_frames // 2), len(lq) - 1 - num_frames // 2)
    inds = [ci + o for o in range(-(num_frames // 2), num_frames // 2 + 1)]
    return torch.stack([read_image(lq[j]) for j in inds], 0)


def energy_map(x):
    """[B,C,H,W] -> [H,W] 通道均值绝对激活, 归一化到 [0,1]。"""
    m = x[0].abs().mean(0)
    return (m / (m.max() + 1e-8)).cpu().numpy()


def flatten(x):
    return x[0].flatten(1).cpu()


def corr(a, b):
    a = a - a.mean()
    b = b - b.mean()
    denom = (a.norm() * b.norm()).item() + 1e-8
    return float((a * b).sum().item() / denom)


class Capture:
    """注册 forward hook, 抓取 query_N/L/M、attn_N/L/M 输出与共享 KV。"""

    def __init__(self, tca):
        self.store = {}
        self.handles = []
        self.tca = tca
        targets = {
            "Q_N": tca.query_N, "Q_L": tca.query_L, "Q_M": tca.query_M,
            "A_N": tca.attn_N, "A_L": tca.attn_L, "A_M": tca.attn_M,
            "KV": tca.kv_proj,
        }
        for name, mod in targets.items():
            self.handles.append(mod.register_forward_hook(self._mk(name)))
        self.handles.append(tca.register_forward_pre_hook(self._pre))

    def _pre(self, mod, inp):
        # inp = (feat_spatial, feats_seq, rwkv_ctx)
        self.store["feats_seq"] = inp[1].detach()
        feats = inp[1]
        with torch.no_grad():
            self.store["ctx_mean_energy"] = mod._ctx_mean(feats).abs().mean().item()
            self.store["ctx_smooth_energy"] = mod._ctx_smooth(feats).abs().mean().item()
            self.store["ctx_diff_energy"] = mod._ctx_diff(feats).abs().mean().item()

    def _mk(self, name):
        def hook(mod, inp, out):
            self.store[name] = out.detach()
        return hook

    def clear(self):
        self.store = {}

    def remove(self):
        for h in self.handles:
            h.remove()


def analyze(model, clip, dev, capture):
    capture.clear()
    with torch.no_grad():
        out = model(clip.unsqueeze(0).to(dev))
    s = capture.store
    res = {}
    for key in ["Q", "A"]:
        n, l, m = s[f"{key}_N"], s[f"{key}_L"], s[f"{key}_M"]
        res[f"{key}_corr_NL"] = corr(flatten(n), flatten(l))
        res[f"{key}_corr_NM"] = corr(flatten(n), flatten(m))
        res[f"{key}_corr_LM"] = corr(flatten(l), flatten(m))
    res["F_out_cos_NL"] = corr(flatten(out["branch_N"]), flatten(out["branch_L"]))
    res["F_out_cos_NM"] = corr(flatten(out["branch_N"]), flatten(out["branch_M"]))
    res["F_out_cos_LM"] = corr(flatten(out["branch_L"]), flatten(out["branch_M"]))
    return out, s, res


def degradation_response(model, clip, dev, capture):
    """受控退化: 对同一 clip 注入 噪声/压暗/位移, 看三路 |Q|/|attn| 能量与 KV 统计量
    相对无退化基线的变化。若三路真的分工, 每种退化应主要由对应路响应。"""
    base = clip.clone()
    T = base.shape[0]
    ci = T // 2

    variants = {}
    # 噪声: 中心帧加高斯噪声 (i.i.d.)
    v = base.clone()
    g = torch.randn_like(v) * 0.05
    v[ci] = (v[ci] + g[ci]).clamp(0, 1)
    variants["noise"] = v
    # 压暗: 全序列 gamma 压暗
    variants["dark"] = base.clone().pow(2.0)
    # 位移: 邻帧相对中心平移 (运动), 中心不动
    v = base.clone()
    shift = 4
    for t in range(T):
        if t != ci:
            v[t] = torch.roll(base[t], shifts=(shift, shift), dims=(1, 2))
    variants["motion"] = v

    def stats(variant):
        capture.clear()
        with torch.no_grad():
            model(variant.unsqueeze(0).to(dev))
        s = capture.store
        out = {
            "absQ": {k: float(s[f"Q_{k}"].abs().mean().item()) for k in "NLM"},
            "absAttn": {k: float(s[f"A_{k}"].abs().mean().item()) for k in "NLM"},
            "absKV": float(s["KV"].abs().mean().item()),
            "ctx_mean": s["ctx_mean_energy"],
            "ctx_smooth": s["ctx_smooth_energy"],
            "ctx_diff": s["ctx_diff_energy"],
        }
        return out

    e_base = stats(base)
    table = {"baseline": e_base}
    for name, var in variants.items():
        e = stats(var)
        row = {}
        for grp in ["absQ", "absAttn"]:
            row[grp] = {k: e[grp][k] / (e_base[grp][k] + 1e-8) for k in "NLM"}
        row["absKV"] = e["absKV"] / (e_base["absKV"] + 1e-8)
        for k in ["ctx_mean", "ctx_smooth", "ctx_diff"]:
            row[k] = e[k] / (e_base[k] + 1e-8)
        table[name] = row
    return table


def make_figure(clip, out, s, seq, out_path):
    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    ci = clip.shape[0] // 2
    axes[0, 0].imshow(clip[ci].permute(1, 2, 0).numpy())
    axes[0, 0].set_title("LQ center")
    axes[1, 0].imshow(out["final"][0].permute(1, 2, 0).detach().cpu().numpy().clip(0, 1))
    axes[1, 0].set_title("predicted")
    for j, k in enumerate("NLM"):
        axes[0, j + 1].imshow(energy_map(s[f"Q_{k}"]), cmap="inferno")
        axes[0, j + 1].set_title(f"|Q_{k}| energy")
        axes[1, j + 1].imshow(energy_map(s[f"A_{k}"]), cmap="inferno")
        axes[1, j + 1].set_title(f"|attn {k}| energy")
    for ax in axes.ravel():
        ax.axis("off")
    fig.suptitle(f"SDSD {seq} - triple query / attention response", fontsize=13)
    fig.tight_layout()
    fig.savefig(out_path, dpi=110)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default="outputs/golf_v7r_v3_pospair/best.pth")
    p.add_argument("--seq_root", default="/home/a1005/yzy/dataset/SDSD/indoor/input")
    p.add_argument("--seqs", nargs="+", default=["pair50", "pair45", "pair19"])
    p.add_argument("--out_dir", default="outputs/golf_v7r_v3_query_semantics")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = GolfNet_v7r_v3().to(dev).eval()
    ck = torch.load(args.ckpt, map_location=dev, weights_only=False)
    model.load_state_dict(ck["model_state_dict"])
    print(f"RESULT loaded epoch {ck.get('epoch')} from {args.ckpt}")

    capture = Capture(model.triple_query_tca)
    summary = {}

    for seq in args.seqs:
        clip = build_clip(args.seq_root, seq)
        out, s, res = analyze(model, clip, dev, capture)
        print(f"RESULT [{seq}] Q corr NL/NM/LM = "
              f"{res['Q_corr_NL']:.3f} / {res['Q_corr_NM']:.3f} / {res['Q_corr_LM']:.3f}")
        print(f"RESULT [{seq}] attn corr NL/NM/LM = "
              f"{res['A_corr_NL']:.3f} / {res['A_corr_NM']:.3f} / {res['A_corr_LM']:.3f}")
        print(f"RESULT [{seq}] F_out cos NL/NM/LM = "
              f"{res['F_out_cos_NL']:.3f} / {res['F_out_cos_NM']:.3f} / {res['F_out_cos_LM']:.3f}")
        deg = degradation_response(model, clip, dev, capture)
        for name in ["noise", "dark", "motion"]:
            row = deg[name]
            print(f"RESULT [{seq}] {name:6s} ratio absQ N/L/M="
                  f"{row['absQ']['N']:.3f}/{row['absQ']['L']:.3f}/{row['absQ']['M']:.3f} "
                  f"absAttn N/L/M="
                  f"{row['absAttn']['N']:.3f}/{row['absAttn']['L']:.3f}/{row['absAttn']['M']:.3f} "
                  f"KV={row['absKV']:.3f} ctx(diff)={row['ctx_diff']:.3f}")
        make_figure(clip, out, s, seq, os.path.join(args.out_dir, f"{seq}_maps.png"))
        summary[seq] = {"corr": res, "degradation_ratio": deg}

    capture.remove()
    with open(os.path.join(args.out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)
    print(f"RESULT wrote {os.path.join(args.out_dir, 'summary.json')} "
          f"and {len(args.seqs)} map figures")


if __name__ == "__main__":
    main()
