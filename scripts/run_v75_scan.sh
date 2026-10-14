#!/usr/bin/env bash
# v7.5 第 2 段: 三个关键结构变量的 5ep 扫描。
#
# 等第 1 段探针 (Phase F) 结束后自动接续, 依次跑 S1 / S2 / S3。
# 每个 run 5ep ≈ 6.8 h, 三个约 20 h。
#
# 等待方式: PID + kill -0, 不做命令行模式匹配 —— pgrep/pkill -f 会匹配到自身 (已踩四次)。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
STAGE1_PID=${1:?用法: run_v75_scan.sh <stage1_pid>}
LOG=outputs/v7_5_scan/scan.log
mkdir -p outputs/v7_5_scan

log() { echo "[v75] $*"; }

{
log "start @ $(date '+%F %T'), 等待第 1 段探针 pid=$STAGE1_PID"
while kill -0 "$STAGE1_PID" 2>/dev/null; do sleep 60; done
log "第 1 段结束 @ $(date '+%F %T')"
s1_psnr=$(grep -oE 'Validation - PSNR: [0-9.]+' \
    outputs/golf_v7r_v3_phaseF_band_targets/train.log | tail -1)
log "  第 1 段 $s1_psnr (仅供参考, 不作判据)"

for name in S1_share_high S2_gate_on S3_kernel_15_5; do
    out="outputs/v7_5_scan/$name"
    log ""
    log "===== $name @ $(date '+%F %T') ====="
    mkdir -p "$out"

    free_gb=$(python3 -c "import os; s=os.statvfs('$out'); print(s.f_bavail*s.f_frsize/1e9)")
    if python3 -c "exit(0 if float('$free_gb') >= 2.0 else 1)"; then
        log "  磁盘余量 ${free_gb}GB ≥ 2GB, 继续"
    else
        log "  ⚠️ 磁盘余量 ${free_gb}GB < 2GB, 中止后续"
        break
    fi

    # 落一份配置副本, 保证诊断脚本能按训练时的配置重建模型
    cp "configs/v7_5/$name.yaml" "$out/_cfg.yaml"

    PYTHONPATH=. $PY train_golf_v7r_v3.py \
        --config "configs/v7_5/$name.yaml" --stop_epoch 5 \
        > "$out/train.log" 2>&1
    rc=$?
    log "  EXIT=$rc @ $(date '+%F %T')"
    if [ -f "$out/best.pth" ]; then
        psnr=$(grep -oE 'PSNR: [0-9.]+' "$out/train.log" | tail -1)
        log "  ✅ best.pth saved  $psnr"
    else
        log "  ⚠️ best.pth 未生成"
    fi
done

log ""
log "===== 扫描完成 @ $(date '+%F %T') ====="
log "下一步 (判据 1/2/3, 逐个 run 跑):"
for name in S1_share_high S2_gate_on S3_kernel_15_5; do
    log "  PYTHONPATH=. python scripts/diag_v7r_v3_branches.py \\"
    log "      --ckpt outputs/v7_5_scan/$name/best.pth --config outputs/v7_5_scan/$name/_cfg.yaml"
done
log ""
log "对照基线 (#1 段):"
log "  --ckpt outputs/golf_v7r_v3_phaseF_band_targets/best.pth \\"
log "  --config outputs/golf_v7r_v3_phaseF_band_targets/_cfg.yaml"
} >> "$LOG" 2>&1
