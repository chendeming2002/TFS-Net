#!/bin/bash
# 训练热安全启动器 — E-core 部署 + 热治理提醒
# 用法: bash experiments/rwkv_only_v3/launch_ablation.sh <A|B|C|ABC|none> [resume_ckpt]
#
# 部署层 (按序生效):
#   1. root 侧: scripts/thermal_limit.sh (RAPL PL1=95W + GPU 350W, 重启后需重跑)
#   2. 进程侧: taskset -c 16-23 (主进程+workers 钉 E-core, 实测 package 74°C 稳态)
#   3. 数据侧: 时序分桶采样 + uint8 短路 + LRU 缓存 (已内置 train_ablation.py)

set -e
ABLATION=${1:?用法: launch_ablation.sh <A|B|C|ABC|none> [resume_ckpt]}
RESUME=${2:-}

# 热治理检查 (PL1 应为 95000000)
PL1=$(cat /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw 2>/dev/null || echo 0)
if [ "$PL1" != "95000000" ]; then
  echo "[警告] RAPL PL1=${PL1:0:-6}W != 95W, 请先执行: sudo bash scripts/thermal_limit.sh"
  echo "[警告] 未设功耗墙时继续运行有热关机风险, 5 秒后继续..."
  sleep 5
fi

cd /home/a1005/25/TFS-Net
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

CFG=/home/a1005/25/TFS-Net/experiments/rwkv_only_v3/config_ablation_${ABLATION}.yaml
if [ ! -f "$CFG" ]; then
  echo "缺少配置 $CFG"
  exit 1
fi

RESUME_ARG=""
if [ -n "$RESUME" ]; then
  RESUME_ARG="--resume $RESUME"
fi

exec taskset -c 16-23 /home/a1005/anaconda3/envs/ptorch/bin/python -u \
  experiments/rwkv_only_v3/train_ablation.py \
  --config "$CFG" --ablation "$ABLATION" $RESUME_ARG
