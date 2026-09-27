#!/bin/bash
# Golf v7 训练启动脚本 - 使用 screen 保持会话

cd /home/a1005/25/TFS-Net

SESSION_NAME="golf_v7_train"

# 检查是否已有同名 session
if screen -list | grep -q "$SESSION_NAME"; then
    echo "Session '$SESSION_NAME' already exists. Attach with:"
    echo "  screen -r $SESSION_NAME"
    exit 1
fi

# 创建新 screen session 并启动训练
screen -dmS "$SESSION_NAME" bash -c "\
    source /home/a1005/anaconda3/bin/activate ptorch && \
    python train_golf_v7_simple.py --config configs/golf_v7_bs1.yaml 2>&1 | tee outputs/golf_v7/train.log
"

echo "✓ Training started in screen session: $SESSION_NAME"
echo ""
echo "Commands:"
echo "  Attach to session:  screen -r $SESSION_NAME"
echo "  Detach from session: Ctrl+A then D"
echo "  Monitor log:        tail -f outputs/golf_v7/train.log"
echo "  Kill session:       screen -X -S $SESSION_NAME quit"
