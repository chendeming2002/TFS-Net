#!/usr/bin/env bash
# 训练完成通知器 — 配合 agent 的后台任务机制使用。
#
# 背景 (2026-10-09): 此前"监控"靠在单个回合内 `sleep` 后查一次, 回合结束就无人看守,
# 导致 `phaseA_holdout` 20:39 跑完、27 分钟后才被发现。
# 正确做法: 把本脚本作为**后台任务**启动, 它会阻塞直到训练退出并打印结果摘要;
# 后台任务退出时宿主会主动通知 agent, 无需轮询。
#
# 用法:
#   bash scripts/wait_for_training.sh <out_dir> [pid]
# 例 (先起训练, 再把本脚本作为后台任务挂上):
#   nohup python train_golf_v7r_v3.py --config <cfg> --stop_epoch 5 > <out>/train.log 2>&1 &
#   bash scripts/wait_for_training.sh outputs/golf_v7r_v3_phaseB_x $!
#
# 若不传 pid, 则自动匹配 train_golf_v7r_v3.py 进程。

set -u

OUT_DIR="${1:?用法: wait_for_training.sh <out_dir> [pid]}"
PID="${2:-}"
LOG="$OUT_DIR/train.log"

if [ -z "$PID" ]; then
    PID="$(pgrep -f 'train_golf_v7r_v3.py' | head -1 || true)"
fi

if [ -z "$PID" ]; then
    echo "[wait] 未找到训练进程; 直接检查日志是否已完成"
else
    echo "[wait] 等待 pid=$PID 退出 (日志: $LOG)"
    # 不能用 `wait`(非子进程), 用 /proc 存在性轮询; 间隔 60s 对 CPU 无感
    while [ -d "/proc/$PID" ]; do
        sleep 60
    done
    echo "[wait] pid=$PID 已退出于 $(date '+%F %T')"
fi

echo ""
echo "════════ 训练结果摘要 ════════"
if [ -f "$LOG" ]; then
    echo "--- epoch / 验证记录 ---"
    grep -aE 'Epoch [0-9]+ done|Validation|New best|Training completed|stop_epoch|Resumed' "$LOG" || true
    echo ""
    echo "--- 日志尾部 ---"
    tail -3 "$LOG"
    echo ""
    echo "--- 异常检查 (NaN / Error / OOM) ---"
    if grep -aciE 'nan|error|out of memory|traceback' "$LOG" | grep -qv '^0$'; then
        grep -aiE 'nan|error|out of memory|traceback' "$LOG" | tail -5
    else
        echo "  无"
    fi
else
    echo "  ⚠️ 日志不存在: $LOG"
fi

echo ""
echo "--- 检查点 ---"
ls -la --time-style=+'%m-%d %H:%M' "$OUT_DIR"/*.pth 2>/dev/null || echo "  无 .pth"

echo ""
echo "--- GPU 当前状态 ---"
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader 2>/dev/null || true

echo "════════ TRAINING WATCH DONE ════════"
