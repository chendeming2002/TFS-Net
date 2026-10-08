#!/bin/bash
# Phase A 探针监视终端
# 用法: watch -n 5 bash scripts/monitor_phase_a.sh [run_name]
#   run_name: m | l | holdout | all (默认 m)
# 示例: watch -n 5 bash scripts/monitor_phase_a.sh m

RUN=${1:-m}
ROOT=/home/a1005/25/TFS-Net

case "$RUN" in
  all)
    # 三条链全部扫描
    for r in m l holdout; do
      LOG="$ROOT/outputs/golf_v7r_v3_phaseA_$r/train.log"
      echo "══ phaseA_$r ══"
      if [ -f "$LOG" ]; then
        grep -a "Validation - PSNR\|Step \|done in" "$LOG" | tail -3 | sed 's/.*Epoch /  Epoch /; s/.*Step /  Step /; s/.*Validation - /  ✓ /'
      else
        echo "  (未启动)"
      fi
    done
    echo ""
    echo "══ 串行链进程 ══"
    pgrep -af "train_golf_v7r_v3" | grep -v grep || echo "  (无训练进程)"
    echo ""
    echo "══ GPU ══"
    nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used \
      --format=csv,noheader | awk -F', ' '{printf "  利用率 %s  功耗 %s  温度 %s  显存 %s\n",$1,$2,$3,$4}'
    exit 0
    ;;
  m|l|holdout) ;;
  *) echo "未知 run: $RUN (m|l|holdout|all)"; exit 1 ;;
esac

LOG="$ROOT/outputs/golf_v7r_v3_phaseA_$RUN/train.log"
OUT_DIR="$ROOT/outputs/golf_v7r_v3_phaseA_$RUN"

echo "══════════ phaseA_$RUN 探针监视 ══════════"
echo ""

echo "══ 进程 ══"
pgrep -af "phaseA_$RUN" | grep -v grep || pgrep -af "train_golf_v7r_v3" | grep -v grep || echo "  (无训练进程!)"

echo ""
echo "══ 训练进度 (最近 5 step/epoch) ══"
grep -a "Step \|Epoch " "$LOG" 2>/dev/null | tail -5 || echo "  (等待日志...)"

echo ""
echo "══ MatrixRWKV 门控 ══"
GATE=$(grep -a "gate=" "$LOG" 2>/dev/null | tail -1)
if [ -n "$GATE" ]; then
  echo "  $GATE" | sed 's/.*| /  /'
  GV=$(echo "$GATE" | grep -oE "gate=[0-9.]+" | grep -oE "[0-9.]+")
  if [ -n "$GV" ]; then
    /home/a1005/anaconda3/envs/ptorch/bin/python -c "
v=float('$GV')
if v>0.5: print(f'  ✓ gate={v:.3f} 强参与')
elif v>0.1: print(f'  ✓ gate={v:.3f} 已激活')
elif v>0.01: print(f'  ⚡ gate={v:.3f} 弱激活')
else: print(f'  ⚠  gate={v:.3f} 近零')
" 2>/dev/null
  fi
else
  echo "  (暂无 gate, 等待 step 100)"
fi

echo ""
echo "══ 验证指标历史 ══"
if grep -aq "Validation - PSNR" "$LOG" 2>/dev/null; then
  grep -a "Validation - PSNR" "$LOG" | tail -8 | sed 's/.*Validation - /  /'
  # 参照: pospair_quick ep5 = 21.92 dB
  LAST_PSNR=$(grep -a "Validation - PSNR" "$LOG" | tail -1 | grep -oE "PSNR: [0-9.]+" | grep -oE "[0-9.]+")
  if [ -n "$LAST_PSNR" ]; then
    /home/a1005/anaconda3/envs/ptorch/bin/python -c "
p=float('$LAST_PSNR'); ref=21.92; thr=21.87
diff=p-ref
sym='✅' if p>=thr else '❌'
print(f'  {sym} 当前 {p:.2f} vs 参照 {ref} dB (Δ{diff:+.2f}, 阈值 ≥{thr})')
" 2>/dev/null
  fi
else
  echo "  (未到验证 epoch, val_interval=5)"
fi

echo ""
echo "══ Epoch 耗时 ══"
grep -a "done in" "$LOG" 2>/dev/null | tail -3 | sed 's/.*Epoch /  Epoch /'

echo ""
echo "══ 检查点 ══"
ls -lht "$OUT_DIR"/*.pth 2>/dev/null | awk '{printf "  %-14s %s\n",$5,$9}' || echo "  (暂无)"

echo ""
echo "══ 速度 ══"
grep -a "Step " "$LOG" 2>/dev/null | tail -2 > /tmp/probe_speed.txt
/home/a1005/anaconda3/envs/ptorch/bin/python -c "
import re
from datetime import datetime
lines=open('/tmp/probe_speed.txt').read().splitlines()
def parse(s):
    m=re.search(r'(\S+ \S+) - INFO -\s+Step (\d+)/(\d+)',s)
    if not m: return None
    return datetime.strptime(m.group(1),'%Y-%m-%d %H:%M:%S,%f'),int(m.group(2)),int(m.group(3))
p=[parse(l) for l in lines]; p=[x for x in p if x]
if len(p)==2 and p[1][1]>p[0][1]:
    dt=(p[1][0]-p[0][0]).total_seconds()
    rate=(p[1][1]-p[0][1])/dt
    total=p[1][2]; done=p[1][1]; remain=total-done
    eta_ep=remain/rate/3600
    print(f'  {rate:.2f} step/s  进度 {done}/{total}  本 epoch 剩余 ≈{eta_ep:.1f}h')
else:
    print('  (数据不足)')
" 2>/dev/null

echo ""
echo "══ GPU ══"
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used \
  --format=csv,noheader | awk -F', ' '{printf "  利用率 %s  功耗 %s  温度 %s  显存 %s\n",$1,$2,$3,$4}'

echo ""
echo "══ NaN 监控 ══"
# 注意: 不能用 "inf" (会匹配每行的 INFO), 也不能大小写不敏感
NAN_CNT=$(grep -acE "\bnan\b|\bNaN\b|\bInf\b|\b-inf\b|Loss: nan" "$LOG" 2>/dev/null); echo "  NaN/Inf 行数: ${NAN_CNT:-0}"

echo ""
echo "══ §6.14 参照阈值 ══"
echo "  PSNR ≥ 21.87  |  Y_M↔Y_N cos ≤ 0.9767  |  Y_L↔Y_N cos ≤ 0.9775"
echo "  motion absAttn_M ≥ 1.5(最大)  |  dark absAttn_L ≥ 2.0(最大, H-l) / ≥ 1.3 边界"
echo ""
echo "提示: watch -n 5 bash scripts/monitor_phase_a.sh $RUN  |  all = 三探针速览"
