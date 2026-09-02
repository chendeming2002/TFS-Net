#!/bin/bash
# 训练+温度 一体监视 (每 5 秒刷新)
# 用法: watch -n 5 bash scripts/monitor.sh   或直接 bash scripts/monitor.sh

LOG=/home/a1005/25/TFS-Net/experiments/rwkv_only_v3/outputs_ablation_A/train.log

echo "══════════ 训练进度 ══════════"
# 最近 3 条 step/Epoch/Val
grep -a "step\|Epoch\|Val:" "$LOG" | tail -3

# 速度 (最近两条 step 的时间差)
S1=$(grep -a "step" "$LOG" | tail -2 | head -1)
S2=$(grep -a "step" "$LOG" | tail -1)
python3 -c "
import re, sys
from datetime import datetime
def parse(s):
    m = re.search(r'(\S+ \S+) - INFO - step (\d+)/(\d+)', s)
    if not m: return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f'), int(m.group(2))
a, b = parse('''$S1'''), parse('''$S2''')
if a and b and b[1] > a[1]:
    dt = (b[0]-a[0]).total_seconds()
    print(f'速度: {(b[1]-a[1])/dt:.2f} it/s')
"

echo ""
echo "══════════ CPU 温度 ══════════"
sensors 2>/dev/null | grep -E "Package id 0|Core 16|Core 20"

echo ""
echo "══════════ GPU ══════════"
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader | awk -F', ' '{printf "利用率 %s  功耗 %s  温度 %s  显存 %s\n", $1, $2, $3, $4}'

echo ""
echo "══════════ 功耗墙 ══════════"
PL1=$(cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw 2>/dev/null || echo 0)
echo "CPU PL1=${PL1:0:-6}W (目标95W)  GPU墙=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

echo ""
echo "══════════ 温度记录器 ══════════"
if pgrep -f temp_logger >/dev/null; then
  N=$(wc -l < /home/a1005/25/TFS-Net/scripts/monitor_data/temp_log.csv 2>/dev/null)
  echo "状态: 运行中  累计样本: ${N:-0}"
else
  echo "状态: 未运行!  启动: setsid nohup python scripts/temp_logger.py &"
fi
echo "折线图: scripts/monitor_data/temp_chart.png (每5分钟自动更新)"
echo "手动出图: python scripts/plot_temps.py"

echo ""
echo "提示: Ctrl+C 退出 | watch -n 5 bash scripts/monitor.sh 可自动刷新"
