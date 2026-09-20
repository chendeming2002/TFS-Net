#!/bin/bash
# Golf-R3 训练监控脚本
# 用法: bash scripts/monitor_golf_r3.sh [output_dir]

OUTPUT_DIR="${1:-outputs/golf_r3_s1}"
LOG_FILE="$OUTPUT_DIR/train.log"

if [ ! -f "$LOG_FILE" ]; then
    echo "❌ 日志文件不存在: $LOG_FILE"
    exit 1
fi

echo "======================================"
echo "Golf-R3 训练监控"
echo "======================================"
echo "日志: $LOG_FILE"
echo ""

# 1. 训练进度
echo "[1/6] 训练进度"
LAST_EPOCH=$(grep -a "Epoch.*/" "$LOG_FILE" | tail -1 | grep -oP "Epoch \K\d+")
TOTAL_EPOCH=$(grep -a "Epoch.*/" "$LOG_FILE" | tail -1 | grep -oP "/ \K\d+")
if [ -n "$LAST_EPOCH" ]; then
    echo "  当前 epoch: $LAST_EPOCH / $TOTAL_EPOCH"
    PROGRESS=$((LAST_EPOCH * 100 / TOTAL_EPOCH))
    echo "  进度: $PROGRESS%"
else
    echo "  未找到进度信息"
fi
echo ""

# 2. 最新训练指标
echo "[2/6] 最新训练指标 (最近 5 个 epoch)"
grep -a "Train stats:" "$LOG_FILE" | tail -5
echo ""

# 3. 验证指标
echo "[3/6] 验证指标 (val)"
grep -a "Val stats:" "$LOG_FILE" | tail -5
echo ""

# 4. pair45 指标
echo "[4/6] pair45 指标 (运动泛化)"
PAIR45_STATS=$(grep -a "Pair45 stats:" "$LOG_FILE" | tail -5)
if [ -n "$PAIR45_STATS" ]; then
    echo "$PAIR45_STATS"
else
    echo "  未找到 pair45 统计（可能未到验证 epoch）"
fi
echo ""

# 5. 关键诊断指标 (conf_map, warp_t, FiLM)
echo "[5/6] 关键诊断指标"
echo "  conf_map 统计:"
grep -a "conf_map" "$LOG_FILE" | tail -3 || echo "    未找到"
echo ""
echo "  warp_t 贡献:"
grep -a "warp_t" "$LOG_FILE" | tail -3 || echo "    未找到"
echo ""
echo "  FiLM gamma 统计:"
grep -a "film_gamma" "$LOG_FILE" | tail -3 || echo "    未找到"
echo ""

# 6. 最近警告/错误
echo "[6/6] 最近警告/错误"
WARNINGS=$(grep -a -i "warning\|error" "$LOG_FILE" | tail -5)
if [ -n "$WARNINGS" ]; then
    echo "$WARNINGS"
else
    echo "  ✅ 无警告/错误"
fi
echo ""

# 7. 训练速度估算
echo "======================================"
echo "训练速度估算"
echo "======================================"
if [ -n "$LAST_EPOCH" ] && [ -n "$TOTAL_EPOCH" ]; then
    REMAINING=$((TOTAL_EPOCH - LAST_EPOCH))
    # 假设 30 min/epoch
    HOURS=$((REMAINING * 30 / 60))
    echo "剩余 epoch: $REMAINING"
    echo "预计剩余时间: ~${HOURS}h (假设 30 min/epoch)"
fi

echo ""
echo "======================================"
echo "实时监控命令:"
echo "  tail -f $LOG_FILE"
echo "======================================"
