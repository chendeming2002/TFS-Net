#!/bin/bash
# Golf v7 Training Monitor

LOG_FILE=/home/a1005/25/TFS-Net/outputs/golf_v7/train.log
OUT_DIR=/home/a1005/25/TFS-Net/outputs/golf_v7

echo "========================================"
echo "Golf v7 Training Monitor"
echo "========================================"
echo ""

# Check process
echo "--- Process Status ---"
pgrep -a python | grep train_golf_v7 || echo "No training process found"
echo ""

# GPU status
echo "--- GPU Status ---"
nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu,temperature.gpu --format=csv,noheader,nounits
echo ""

# Latest training log (last 30 lines)
echo "--- Latest Training Log (last 30 lines) ---"
if [ -f "$LOG_FILE" ]; then
    tail -30 "$LOG_FILE"
else
    echo "Log file not found: $LOG_FILE"
fi
echo ""

# Extract validation results
echo "--- Validation Results ---"
if [ -f "$LOG_FILE" ]; then
    grep "Val stats:" "$LOG_FILE" | tail -5
else
    echo "No validation results yet"
fi
echo ""

# Checkpoints
echo "--- Checkpoints ---"
ls -lh "$OUT_DIR"/*.pth 2>/dev/null || echo "No checkpoints yet"
echo ""

echo "========================================"
echo "Monitor completed at $(date)"
echo "========================================"
