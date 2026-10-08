#!/bin/bash
# Golf v7r-v3 训练监视 (含温度 + GPU + 指标历史)
# 用法: watch -n 5 bash scripts/monitor_golf_v7r_v3.sh   或直接 bash scripts/monitor_golf_v7r_v3.sh
# 参照 R4/monitor.sh 的一体化监视风格, 适配 v7r 日志格式

LOG=/home/a1005/25/TFS-Net/outputs/golf_v7r_v3_pospair/train.log
OUT_DIR=/home/a1005/25/TFS-Net/outputs/golf_v7r_v3_pospair

if [ ! -f "$LOG" ]; then
  echo "══════════ 等待日志 [$LOG] ══════════"
fi

echo "══════════ 进程 ══════════"
pgrep -af "train_golf_v7r_v3" || echo "  (无训练进程!)"

echo ""
echo "══════════ 训练进度 (最近 5 step) ══════════"
grep -a "Step \|Epoch " "$LOG" 2>/dev/null | tail -5

echo ""
echo "══════════ MatrixRWKV 门控诊断 (方案 B) ══════════"
GATE=$(grep -a "gate=" "$LOG" 2>/dev/null | tail -1)
if [ -n "$GATE" ]; then
  echo "  $GATE" | sed 's/.*| /  /'
  # 提取 gate 值并判读
  GV=$(echo "$GATE" | grep -oE "gate=[0-9.]+" | grep -oE "[0-9.]+")
  if [ -n "$GV" ]; then
    /home/a1005/anaconda3/envs/ptorch/bin/python -c "
v=float('$GV')
if v>0.5: print(f'  ✓ gate={v:.3f} MatrixRWKV 强参与 KV 调制')
elif v>0.1: print(f'  ✓ gate={v:.3f} MatrixRWKV 已激活')
elif v>0.01: print(f'  ⚡ gate={v:.3f} 弱激活, 持续观察')
else: print(f'  ⚠️  gate={v:.3f} 近零, MatrixRWKV 参与度低')
" 2>/dev/null
  fi
else
  echo "  (暂无 gate 诊断, 等待首个 log_interval)"
fi

echo ""
echo "══════════ 验证指标历史 ══════════"
if grep -aq "Validation - PSNR" "$LOG" 2>/dev/null; then
  grep -a "Validation - PSNR" "$LOG" | tail -8 | sed 's/.*Validation - /  /'
else
  echo "  (未到验证 epoch)"
fi

echo ""
echo "══════════ 当前 Epoch 耗时 ══════════"
grep -a "done in" "$LOG" 2>/dev/null | tail -5 | sed 's/.*Epoch /  Epoch /'

echo ""
echo "══════════ 检查点 ══════════"
ls -lht "$OUT_DIR"/*.pth 2>/dev/null | awk '{printf "  %-14s %s\n", $5, $9}' || echo "  (暂无)"

echo ""
echo "══════════ 配置 ══════════"
echo "  v7r-v3: 三路Query × 共享统计KV + PixelTemporal + MatrixRWKV-6"
echo "  baseline: Golf R2 PSNR=20.14 | v7 ep10=19.42 | v7r-v2 ep10=18.10"

echo ""
echo "══════════ 速度 ══════════"
grep -a "Step " "$LOG" 2>/dev/null | tail -2 > /tmp/v7r_v3_speed.txt
/home/a1005/anaconda3/envs/ptorch/bin/python -c "
import re
lines = open('/tmp/v7r_v3_speed.txt').read().splitlines()
def parse(s):
    m = re.search(r'(\S+ \S+) - INFO -\s+Step (\d+)/(\d+)', s)
    if not m: return None
    return datetime.strptime(m.group(1), '%Y-%m-%d %H:%M:%S,%f'), int(m.group(2)), int(m.group(3))
from datetime import datetime
p = [parse(l) for l in lines]
p = [x for x in p if x]
if len(p) == 2 and p[1][1] > p[0][1]:
    dt = (p[1][0]-p[0][0]).total_seconds()
    print(f'  {(p[1][1]-p[0][1])/dt:.2f} step/s   (进度 {p[1][1]}/{p[1][2]})')
else:
    print('  (数据不足)')
" 2>/dev/null

echo ""
echo "══════════ CPU 温度 ══════════"
sensors 2>/dev/null | grep -E "Package id 0|Core 16|Core 20"

echo ""
echo "══════════ GPU ══════════"
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader | awk -F', ' '{printf "  利用率 %s  功耗 %s  温度 %s  显存 %s\n", $1, $2, $3, $4}'

echo ""
echo "══════════ 功耗墙 ══════════"
PL1=$(cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw 2>/dev/null || echo 0)
echo "  CPU PL1=${PL1:0:-6}W  GPU墙=$(nvidia-smi --query-gpu=power.limit --format=csv,noheader)"

echo ""
echo "══════════ NaN 监控 ══════════"
NAN_CNT=$(grep -aci "nan\|nan" "$LOG" 2>/dev/null)
NAN_CNT=${NAN_CNT:-0}
echo "  NaN/Inf 日志行数: $NAN_CNT"

echo ""
echo "提示: Ctrl+C 退出 | watch -n 5 bash scripts/monitor_golf_v7r_v3.sh 自动刷新"
