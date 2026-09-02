#!/usr/bin/env python3
# 温度折线图 — 从 temp_log.csv 生成, 标注断电/重启间隙 (崩溃前温度分析用)
# 用法: python scripts/plot_temps.py [--out xxx.png]

import argparse
import csv
import os
from datetime import datetime

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

plt.rcParams["font.sans-serif"] = ["Noto Sans CJK SC", "Noto Sans CJK JP", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

ROOT = "/home/a1005/25/TFS-Net"
CSV_PATH = os.path.join(ROOT, "scripts/monitor_data/temp_log.csv")


def load():
    rows = []
    with open(CSV_PATH) as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "ts": int(r["ts"]),
                    "dt": datetime.fromtimestamp(int(r["ts"])),
                    "pkg": float(r["pkg"]) if r.get("pkg") else None,
                    "core16": float(r["core16"]) if r.get("core16") else None,
                    "core20": float(r["core20"]) if r.get("core20") else None,
                    "gpu_temp": float(r["gpu_temp"]) if r.get("gpu_temp") else None,
                    "gpu_power": float(r["gpu_power"]) if r.get("gpu_power") else None,
                    "gpu_util": float(r["gpu_util"]) if r.get("gpu_util") else None,
                })
            except (ValueError, KeyError):
                continue
    return sorted(rows, key=lambda x: x["ts"])


def series(rows, key):
    xs = [r["dt"] for r in rows if r[key] is not None]
    ys = [r[key] for r in rows if r[key] is not None]
    return xs, ys


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=os.path.join(ROOT, "scripts/monitor_data/temp_chart.png"))
    args = ap.parse_args()

    rows = load()
    if len(rows) < 2:
        print("数据不足 2 行, 跳过绘图")
        return

    fig, axes = plt.subplots(3, 1, figsize=(14, 10), sharex=True)

    # ── 面板 1: CPU 温度 ──
    ax = axes[0]
    for key, label, color in [("pkg", "Package", "#d62728"),
                              ("core16", "Core16(E)", "#1f77b4"),
                              ("core20", "Core20(E)", "#2ca02c")]:
        xs, ys = series(rows, key)
        if xs:
            ax.plot(xs, ys, label=label, color=color, linewidth=1.2)
    ax.axhline(80, color="orange", ls="--", lw=0.8, label="high 80°C")
    ax.axhline(100, color="red", ls="--", lw=0.8, label="crit 100°C")
    ax.set_ylabel("CPU °C")
    ax.set_title(f"温度监控  ({rows[0]['dt']:%m-%d %H:%M} ~ {rows[-1]['dt']:%m-%d %H:%M}, "
                 f"{len(rows)} 样本, 间隔5s)")
    ax.legend(loc="upper right", fontsize=8, ncol=5)
    ax.grid(alpha=0.3)

    # ── 面板 2: GPU 温度/利用率 ──
    ax = axes[1]
    xs, ys = series(rows, "gpu_temp")
    if xs:
        ax.plot(xs, ys, color="#d62728", label="GPU temp")
    ax2 = ax.twinx()
    xs, ys = series(rows, "gpu_util")
    if xs:
        ax2.plot(xs, ys, color="#9467bd", alpha=0.7, label="GPU util%")
    ax.set_ylabel("GPU °C")
    ax2.set_ylabel("util %")
    h1, l1 = ax.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax.legend(h1 + h2, l1 + l2, loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── 面板 3: GPU 功耗 ──
    ax = axes[2]
    xs, ys = series(rows, "gpu_power")
    if xs:
        ax.plot(xs, ys, color="#8c564b", label="GPU W")
    ax.set_ylabel("W")
    ax.set_xlabel("时间")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── 断电/重启间隙标注 (相邻样本间隔 > 120s) ──
    gaps = []
    for a, b in zip(rows, rows[1:]):
        if b["ts"] - a["ts"] > 120:
            gaps.append((a, b))
    for a, b in gaps:
        for ax_ in axes:
            ax_.axvline(b["dt"], color="red", ls=":", lw=1.5, alpha=0.8)
        axes[0].annotate(
            f"断电 {(b['ts']-a['ts'])//60}min\n前温度 pkg={a['pkg']}°C",
            xy=(b["dt"], 50), fontsize=8, color="red", ha="left",
            xytext=(8, 0), textcoords="offset points")

    axes[0].xaxis.set_major_formatter(mdates.DateFormatter("%m-%d %H:%M"))
    plt.setp(axes[2].get_xticklabels(), rotation=20, ha="right")
    fig.tight_layout()
    fig.savefig(args.out, dpi=110)
    print(f"已保存 {args.out}  (含 {len(gaps)} 个断电间隙标注)")


if __name__ == "__main__":
    main()
