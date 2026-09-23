#!/bin/bash
# 训练+温度 一体监视 (每 5 秒刷新)
# 用法: watch -n 5 bash scripts/monitor.sh   或直接 bash scripts/monitor.sh

# 自动发现活跃训练日志: 取最近修改的 train.log (覆盖完整模型 outputs/* 与概念模型消融)
LOG=$(ls -t \
      /home/a1005/25/TFS-Net/outputs/*/train.log \
      /home/a1005/25/TFS-Net/experiments/rwkv_only_v3/outputs_ablation*/train.log \
      /home/a1005/25/TFS-Net/experiments/rwkv_only_v3/outputs_notca/train.log \
      /home/a1005/25/TFS-Net/experiments/rwkv_only_v3/outputs/train.log \
      2>/dev/null | head -1)
LOG=${LOG:-/home/a1005/25/TFS-Net/outputs/sdsd_f11_simple/train.log}
TAG=$(basename "$(dirname "$LOG")")

echo "══════════ 训练进度 [$TAG] ══════════"
# 最近 3 条 step/Epoch/Val
grep -a "step\|Epoch\|Val:" "$LOG" | tail -3

echo ""
echo "══════════ 验证指标 ══════════"
# val 与 pair45 双指标 (Golf-R3 R3-D)
VAL=$(grep -a "Val stats:" "$LOG" | tail -1)
P45=$(grep -a "Pair45 stats:" "$LOG" | tail -1)
if [ -n "$VAL" ]; then echo "$VAL" | sed 's/.*Val stats:/  val  :/'; else echo "  val  : (未到验证 epoch)"; fi
if [ -n "$P45" ]; then echo "$P45" | sed 's/.*Pair45 stats:/  pair45:/'; else echo "  pair45: (未到验证 epoch)"; fi

echo ""
echo "══════════ 诊断指标 ══════════"
# conf_map / FiLM / 时序损失 (若日志含 diag 行)
DIAG=$(grep -a "diag:" "$LOG" | tail -1)
if [ -n "$DIAG" ]; then
  echo "  $DIAG" | sed 's/.*diag:/diag:/'
  # R4-NaN-fix: conv1_max 预警 (>1500 危险, >1800 切 bf16)
  C1MAX=$(echo "$DIAG" | grep -oE "conv1_max=[0-9.]+" | grep -oE "[0-9.]+")
  if [ -n "$C1MAX" ]; then
    python3 -c "
v=float('$C1MAX')
if v>1800: print(f'  ⚠️  conv1_max={v:.0f} 极危险! 立即切换 bf16')
elif v>1500: print(f'  ⚠️  conv1_max={v:.0f} 危险, 接近 fp16 溢出阈值')
elif v>800:  print(f'  ⚡  conv1_max={v:.0f} 偏高, 持续观察')
else:        print(f'  ✓  conv1_max={v:.0f} 健康')
" 2>/dev/null
  fi
fi
LAST=$(grep -a "step" "$LOG" | tail -1)
LTEMP=$(echo "$LAST" | grep -oE "temp=[0-9.]+" | head -1)
if [ -n "$LTEMP" ]; then echo "  $LTEMP  (R4: 真时序一致性)"; fi
LDIV=$(echo "$LAST" | grep -oE "div=-?[0-9.]+" | head -1)
if [ -n "$LDIV" ]; then echo "  $LDIV"; fi

# NaN 统计 (当前 epoch)
NAN_CNT=$(grep -a "Skipping non-finite" "$LOG" | tail -100 | wc -l)
if [ "$NAN_CNT" -gt 0 ]; then
  echo "  ⚠️  近100条日志中 NaN skip: $NAN_CNT 次"
else
  echo "  ✓  近100条日志无 NaN"
fi

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
