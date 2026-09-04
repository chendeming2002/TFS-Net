#!/usr/bin/env python3
# 温度记录器 — 每 5 秒采样 CPU/GPU/训练进度, 追加 CSV, 定期自动生成折线图
# 用法: setsid nohup python scripts/temp_logger.py >/dev/null 2>&1 &
# 数据: scripts/monitor_data/temp_log.csv   图: scripts/monitor_data/temp_chart.png

import csv
import os
import re
import subprocess
import sys
import time
from datetime import datetime

ROOT = "/home/a1005/25/TFS-Net"
LOG_DIR = os.path.join(ROOT, "scripts/monitor_data")
CSV_PATH = os.path.join(LOG_DIR, "temp_log.csv")
CHART_PATH = os.path.join(LOG_DIR, "temp_chart.png")
PLOT_SCRIPT = os.path.join(ROOT, "scripts/plot_temps.py")
TRAIN_LOGS = [
    os.path.join(ROOT, "experiments/rwkv_only_v3/outputs_ablation_A/train.log"),
    os.path.join(ROOT, "experiments/rwkv_only_v3/outputs_notca/train.log"),
]

SAMPLE_INTERVAL = 5       # 秒
PLOT_EVERY = 60           # 每 60 个样本 (~5 分钟) 重画图

FIELDS = ["ts", "time", "pkg", "core16", "core20", "nvme", "pl1",
          "gpu_temp", "gpu_power", "gpu_util", "train_tag"]

SENSORS_RE = {
    "pkg": re.compile(r"^Package id 0:\s*\+(\d+\.?\d*)"),
    "core16": re.compile(r"^Core 16:\s*\+(\d+\.?\d*)"),
    "core20": re.compile(r"^Core 20:\s*\+(\d+\.?\d*)"),
}
STEP_RE = re.compile(r"step (\d+)/(\d+)")
EPOCH_RE = re.compile(r"Epoch (\d+)/(\d+) train")

RAPL_PL1 = "/sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw"
NVME_TEMP = "/sys/class/hwmon/hwmon1/temp2_input"


def read_sensors():
    try:
        out = subprocess.run(["sensors"], capture_output=True, text=True, timeout=5).stdout
    except Exception:
        return {}
    vals = {}
    for line in out.splitlines():
        for key, rx in SENSORS_RE.items():
            m = rx.match(line)
            if m and key not in vals:
                vals[key] = m.group(1)
    return vals


def read_gpu():
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=temperature.gpu,power.draw,utilization.gpu",
             "--format=csv,noheader,nounits"],
            capture_output=True, text=True, timeout=5).stdout.strip()
        t, p, u = [x.strip() for x in out.split(",")]
        return {"gpu_temp": t, "gpu_power": p.rstrip(" W"), "gpu_util": u.rstrip(" %")}
    except Exception:
        return {}


def read_train():
    """取最近一条训练日志标记 (优先当前活跃实验)。"""
    for path in TRAIN_LOGS:
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 2048))
                tail = f.read().decode("utf-8", errors="ignore")
            m = EPOCH_RE.search(tail) or STEP_RE.search(tail)
            if m:
                if "Epoch" in m.re.pattern:
                    return f"ep{m.group(1)}"
                return f"ep? s{m.group(1)}"
        except Exception:
            continue
    return ""


def read_pl1_nvme():
    out = {}
    try:
        with open(RAPL_PL1) as f:
            out["pl1"] = str(int(f.read().strip()) // 1000000)  # W
    except Exception:
        out["pl1"] = ""
    try:
        with open(NVME_TEMP) as f:
            out["nvme"] = str(int(f.read().strip()) // 1000)  # °C
    except Exception:
        out["nvme"] = ""
    return out


def main():
    os.makedirs(LOG_DIR, exist_ok=True)
    new_file = not os.path.exists(CSV_PATH)
    f = open(CSV_PATH, "a", newline="")
    writer = csv.DictWriter(f, fieldnames=FIELDS)
    if new_file:
        writer.writeheader()

    n = 0
    while True:
        row = {"ts": int(time.time()), "time": datetime.now().strftime("%H:%M:%S")}
        row.update(read_sensors())
        row.update(read_pl1_nvme())
        row.update(read_gpu())
        row["train_tag"] = read_train()
        try:
            writer.writerow(row)
            f.flush()
            n += 1
            if n % 10 == 0:
                os.fsync(f.fileno())  # 断电防丢: 每10样本强制落盘
        except Exception:
            pass
        if n % PLOT_EVERY == 0:
            try:
                subprocess.run(
                    [sys.executable, PLOT_SCRIPT, "--out", CHART_PATH],
                    capture_output=True, timeout=60)
            except Exception:
                pass
        time.sleep(SAMPLE_INTERVAL)


if __name__ == "__main__":
    main()
