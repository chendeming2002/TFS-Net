#!/usr/bin/env python3
"""高频 GPU 功耗采样器 — 捕捉 5s 采样看不到的瞬态尖峰。

动机 (2026-10-09 两次硬掉电):
  两次断电都是**瞬间失电** (无 panic / 无 vmcore / 无 MCE / 无热告警), 第二次断电时 CPU 仅 59°C
  → 排除过热与内核崩溃, 指向**供电侧** (PSU 保护 / 瞬态过流)。
  RTX 4090 的瞬时功率尖峰可达平均值的数倍且只持续微秒~毫秒, `temp_logger` 的 5s 采样
  完全看不到。本脚本以 ~10Hz 采样 GPU 功耗/时钟/温度, 记录峰值, 用于判断是否存在
  大幅瞬态 (若存在, 降低 `nvidia-smi -pl` 是标准缓解手段)。

用法:
    PYTHONPATH=. python scripts/gpu_power_watch.py [--interval 0.1] [--out outputs/gpu_power.csv]
    nohup ... &          # 与训练并行运行, 事后分析峰值
"""
import argparse
import csv
import os
import subprocess
import time
from datetime import datetime


QUERY = "power.draw,power.draw.average,clocks.sm,temperature.gpu,utilization.gpu,memory.used"


def sample(interval):
    """用 nvidia-smi 批量采样一次 (每次 ~100ms 开销, 故 interval>=0.1 才有意义)。"""
    try:
        out = subprocess.run(
            ["nvidia-smi", f"--query-gpu={QUERY}", "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5,
        )
        return out.stdout.strip()
    except Exception as exc:
        return f"ERROR,{exc}"


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--interval", type=float, default=0.1, help="采样间隔秒 (默认 0.1 = 10Hz)")
    p.add_argument("--out", default="outputs/gpu_power.csv")
    p.add_argument("--duration", type=float, default=0, help="运行时长秒; 0=无限")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    new = not os.path.exists(args.out)
    t0 = time.time()
    peak = 0.0
    n = 0
    with open(args.out, "a", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["unix_ts", "wall", "gpu_power_W", "gpu_power_avg_W", "sm_clk_MHz",
                        "gpu_temp_C", "gpu_util_pct", "mem_used_MiB"])
        print(f"[watch] interval={args.interval}s → {args.out}", flush=True)
        while True:
            raw = sample(args.interval)
            if raw.startswith("ERROR"):
                print(f"[watch] {raw}", flush=True)
            else:
                now = time.time()
                parts = [x.strip() for x in raw.split(",")]
                try:
                    pdraw = float(parts[1])
                except (ValueError, IndexError):
                    pdraw = None
                if pdraw is not None and pdraw > peak:
                    peak = pdraw
                    print(f"[watch] peak {peak:.1f}W @ {datetime.now().strftime('%H:%M:%S')}", flush=True)
                w.writerow([f"{now:.3f}", datetime.now().strftime("%F %T")] + parts)
                f.flush()
                n += 1
            if n % 600 == 0 and n:
                f.flush()
            if args.duration and (time.time() - t0) >= args.duration:
                break
            # 补偿 nvidia-smi 自身调用耗时, 使采样率稳定
            time.sleep(max(0.0, args.interval - (time.time() - now if 'now' in dir() else 0)))
    print(f"[watch] done: {n} samples, peak {peak:.1f}W", flush=True)


if __name__ == "__main__":
    main()
