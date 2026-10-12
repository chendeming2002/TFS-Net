#!/usr/bin/env bash
# 等待交错多 seed 双臂链结束, 然后打印五个 run 的验证 PSNR。
#
# 用 PID 等待 (kill -0), 不做模式匹配: pgrep -f 会匹配到自己的命令行,
# 造成等待脚本永久空转 (本项目已踩过三次)。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PID=${1:?用法: wait_chain_done.sh <chain_pid>}

echo "等待链进程 $PID @ $(date '+%F %T')"
while kill -0 "$PID" 2>/dev/null; do sleep 60; done

echo "===== 链结束 @ $(date '+%F %T') ====="
for d in outputs/multiseed_ref_softmax_runs/seed0 \
         outputs/multiseed_ref_softmax_runs/seed1 \
         outputs/multiseed_ref_softmax_runs/seed2 \
         outputs/multiseed_decorr_runs/seed0 \
         outputs/multiseed_decorr_runs/seed1; do
    if [ -f "$d/train.log" ]; then
        psnr=$(grep -oE "Validation - PSNR: [0-9.]+" "$d/train.log" | tail -1)
        echo "  $d: ${psnr:-无验证记录}"
    fi
done
