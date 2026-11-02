#!/usr/bin/env python3
"""跨版本 ERF 对比: v8.6 / v8.8 / v8.8N 的 EncoderRWKVBlock 与 QueryRWKVBlock。

生成一张 5 行 (v8.6-enc, v8.6-query, v8.8-enc, v8.8-query, v8.8N-query)
× 3 列 (Q 路 / KV 路 / Spatial Mix 总) 的对比图, 并输出定量表与 KV 路方向能量。

复用 `diag_erf.py` 的 erf_maps (同一 size/n/seed 保证与既有结果可比)。
探针口径与单版本版一致: "Spatial Mix 输出" = Channel Mix 的输入张量
(v8.8 起 cm_shift 位于两者之间, 探针截的是 shift 之后 —— 与"Channel Mix 输入"
的定义一致, 三个版本统一)。

用法 (CPU, GPU 被训练占用时):
    PYTHONPATH=. python scripts/diag_erf_compare.py --size 64 --n 16
"""
import argparse
import os
import sys

import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ.setdefault("MPLBACKEND", "Agg")

import matplotlib.pyplot as plt  # noqa: E402

from diag_erf import erf_maps, _force_cpu_wkv, locality_metrics  # noqa: E402

# (版本, checkpoint, 配置, 取哪些头)
SPECS = [
    ("v8.6", "outputs/v8_6_30ep/best.pth",
     "configs/v8_5/v8_6_30ep.yaml", ("encoder", "query")),
    ("v8.8", "outputs/v8_8_b_30ep/best.pth",
     "configs/v8_5/v8_8_b_30ep.yaml", ("encoder", "query")),
    ("v8.8N", "outputs/v8_8n_30ep/best.pth",
     "configs/v8_5/v8_8n_30ep.yaml", ("query",)),
]
PATHS = ("q", "kv", "spatial")
PATH_NAMES = {"q": "Q path", "kv": "KV path", "spatial": "Spatial Mix (total)"}
BLOCK_NAMES = {"encoder": "EncoderRWKVBlock", "query": "QueryRWKVBlock"}


def load_heads(ckpt, config, device):
    import yaml
    from models.golf_v7r import GolfNet_v8
    from utils.io import load_state_dict_compat
    mk = dict(yaml.safe_load(open(config)).get("model", {}) or {})
    mk.pop("arch", None)
    model = GolfNet_v8(**mk).to(device).eval()
    ck = torch.load(ckpt, map_location=device, weights_only=False)
    try:
        load_state_dict_compat(model, ck["model_state_dict"])
    except RuntimeError:
        model.load_state_dict(ck["model_state_dict"], strict=False)
    heads = {"encoder": model.encoder.scale1_blocks[0].head,
             "query": model.tca.attn_N}
    return heads, ck.get("epoch")


def dir_energy(m):
    h, w = m.shape
    cy, cx = h // 2, w // 2
    yy, xx = np.mgrid[0:h, 0:w]
    total = m.sum() + 1e-20
    return {"row": float(m[cy, :].sum() / total),
            "col": float(m[:, cx].sum() / total),
            "diag": float(m[(xx - yy) == 0].sum() / total),
            "adiag": float(m[(xx + yy) == (h - 1)].sum() / total)}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--n", type=int, default=16)
    p.add_argument("--out_dir", default="outputs/erf_compare")
    p.add_argument("--device", default="cpu")
    args = p.parse_args()
    os.makedirs(args.out_dir, exist_ok=True)
    torch.set_num_threads(16)

    rows = []          # (行标签, maps dict, 备注)
    for name, ck, cfg, which in SPECS:
        heads, ep = load_heads(ck, cfg, args.device)
        for w in which:
            h = heads[w]
            _force_cpu_wkv(h)
            maps = erf_maps(h, args.size, args.n, args.device)
            rows.append((f"{name} {BLOCK_NAMES[w]}", maps, f"ep{ep}"))
        del heads

    # ---- 定量表 ----
    print(f"RESULT size={args.size}, n={args.n}", flush=True)
    print(f"{'block':<28}{'path':<10}{'E(r<=4)':>10}{'R90(px)':>9}", flush=True)
    for label, maps, tag in rows:
        for pn in PATHS:
            f4, r90 = locality_metrics(maps[pn])
            print(f"{label+' ('+tag+')':<28}{pn:<10}{f4:>10.1%}{r90:>9.0f}",
                  flush=True)

    # ---- KV 路方向能量 (对照"近均匀"基线: 均匀期望每线 1.56% @64) ----
    print(f"\nKV path directional energy (uniform expectation 1.56%/line):",
          flush=True)
    print(f"{'block':<28}{'row':>7}{'col':>7}{'diag':>7}{'adiag':>7}", flush=True)
    for label, maps, tag in rows:
        d = dir_energy(maps["kv"])
        print(f"{label:<28}{d['row']:>7.1%}{d['col']:>7.1%}"
              f"{d['diag']:>7.1%}{d['adiag']:>7.1%}", flush=True)

    # ---- 对比图 ----
    n_rows = len(rows)
    fig, axes = plt.subplots(n_rows, 3, figsize=(13, 3.7 * n_rows))
    for i, (label, maps, tag) in enumerate(rows):
        for j, pn in enumerate(PATHS):
            m = maps[pn]
            disp = np.log10(m / (m.max() + 1e-20) + 1e-6)
            ax = axes[i, j]
            im = ax.imshow(disp, cmap="magma", vmin=-6, vmax=0)
            f4, r90 = locality_metrics(m)
            ax.set_title(f"{label} ({tag})\n{PATH_NAMES[pn]}  "
                         f"E(r<=4) {f4:.1%} / R90 {r90:.0f}px", fontsize=9)
            ax.axis("off")
            fig.colorbar(im, ax=ax, fraction=0.046)
    fig.suptitle("ERF comparison across versions (log scale, |grad|)", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    out = os.path.join(args.out_dir, "erf_compare.png")
    fig.savefig(out, dpi=125)
    plt.close(fig)
    print(f"\nsaved -> {out}", flush=True)


if __name__ == "__main__":
    main()
