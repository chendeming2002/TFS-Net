#!/bin/bash
# 训练 keepalive 包装器 — 进程崩溃自动断点续训 (断电重启后需手动跑一次本脚本)
# 用法: nohup bash experiments/rwkv_only_v3/keepalive_ablation.sh A &

ABLATION=${1:?用法: keepalive_ablation.sh <A|B|C|ABC|none>}
CFG=/tmp/ablation_${ABLATION}.yaml
CKPT_DIR=/home/a1005/25/TFS-Net/experiments/rwkv_only_v3/outputs_ablation_${ABLATION}
LOG=/tmp/keepalive_${ABLATION}.log

while true; do
  RESUME=""
  if [ -f "$CKPT_DIR/latest.pth" ]; then
    RESUME="--resume $CKPT_DIR/latest.pth"
  fi
  echo "[$(date '+%F %T')] 启动训练 ablation=$ABLATION resume=${RESUME:+yes}" >> "$LOG"
  bash /home/a1005/25/TFS-Net/experiments/rwkv_only_v3/launch_ablation.sh "$ABLATION" \
      $( [ -n "$RESUME" ] && echo "$CKPT_DIR/latest.pth" ) >> "$LOG" 2>&1
  CODE=$?
  echo "[$(date '+%F %T')] 训练退出 code=$CODE, 60 秒后自动续训 (Ctrl+C 或 kill 本进程停止)" >> "$LOG"
  # 正常完成 (code=0) 也重试 — train 脚本跑完 total_epochs 自然退出, 重启会 resume 到最后 epoch 继续空转
  # 检查是否已完成: 若最新 checkpoint epoch >= 目标, 退出
  DONE=$(/home/a1005/anaconda3/envs/ptorch/bin/python -c "
import torch
try:
    c = torch.load('$CKPT_DIR/latest.pth', map_location='cpu', weights_only=False)
    print(1 if c['epoch'] >= 20 else 0)
except: print(0)" 2>/dev/null)
  if [ "$DONE" = "1" ]; then
    echo "[$(date '+%F %T')] 已达 20 epochs, keepalive 退出" >> "$LOG"
    break
  fi
  sleep 60
done
