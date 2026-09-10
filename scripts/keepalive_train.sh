#!/bin/bash
# 通用训练 keepalive — 断点续训直到目标 epoch (断电重启后需手动重跑本脚本)
# 用法: nohup bash scripts/keepalive_train.sh <config路径> <output_dir> <目标epoch> [额外train.py参数...] &

CONFIG=${1:?用法: keepalive_train.sh <config> <output_dir> <target_epoch> [extra args...]}
OUTDIR=${2:?缺少 output_dir}
TARGET=${3:?缺少目标 epoch}
shift 3
EXTRA_ARGS="$@"

ROOT=/home/a1005/25/TFS-Net
LOG=/tmp/keepalive_$(basename $OUTDIR).log

while true; do
  RESUME=""
  if [ -f "$OUTDIR/latest.pth" ]; then
    RESUME="--resume $OUTDIR/latest.pth"
  fi
  echo "[$(date '+%F %T')] 启动 config=$CONFIG resume=${RESUME:+yes}" >> "$LOG"
  cd $ROOT
  taskset -c 16-23 /home/a1005/anaconda3/envs/ptorch/bin/python -u train.py \
      --config "$CONFIG" $RESUME $EXTRA_ARGS >> "$LOG" 2>&1
  CODE=$?
  echo "[$(date '+%F %T')] 训练退出 code=$CODE, 60 秒后检查" >> "$LOG"

  DONE=$(/home/a1005/anaconda3/envs/ptorch/bin/python -c "
import torch
try:
    c = torch.load('$OUTDIR/latest.pth', map_location='cpu', weights_only=False)
    e = c['epoch'] if isinstance(c['epoch'], int) else c['epoch'].item()
    print(1 if e >= $TARGET else 0)
except Exception as ex: print(0)" 2>/dev/null)
  if [ "$DONE" = "1" ]; then
    echo "[$(date '+%F %T')] 已达 $TARGET epochs, keepalive 退出" >> "$LOG"
    break
  fi
  sleep 60
done
