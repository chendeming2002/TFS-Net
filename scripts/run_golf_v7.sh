#!/bin/bash
# Golf v7 Training Launch Script

set -e

PYTHON=/home/a1005/anaconda3/envs/ptorch/bin/python
SCRIPT=/home/a1005/25/TFS-Net/train_golf_v7.py
CONFIG=/home/a1005/25/TFS-Net/configs/golf_v7.yaml
LOG_DIR=/home/a1005/25/TFS-Net/outputs/golf_v7

mkdir -p "$LOG_DIR"

echo "[$(date)] Starting Golf v7 training..."
echo "Config: $CONFIG"
echo "Output: $LOG_DIR"

cd /home/a1005/25/TFS-Net

# 使用 keepalive 包装以实现自动恢复
bash scripts/keepalive_train.sh \
    "$PYTHON" "$SCRIPT" --config "$CONFIG" \
    2>&1 | tee "$LOG_DIR/train.log"
