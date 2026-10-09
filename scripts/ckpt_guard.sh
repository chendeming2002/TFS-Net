#!/usr/bin/env bash
# 检查点守护: 监视训练输出的 latest.pth, 每次更新后复制一份带 epoch 标记的备份。
#
# 动机 (2026-10-09 断电事故): 训练脚本原本只在 epoch 结束落盘 latest.pth, epoch 中途断电
# 会丢掉整个 epoch (holdout 当时 8015 step 全废, 因 ep1 未跑完 = 零检查点)。事后已给
# train_golf_v7r_v3.py 增加 train.ckpt_interval (每 N step 落盘), 但对**已在运行的进程**
# 无法生效。本守护作为不改动运行中进程的外部保险。
#
# 用法:
#   bash scripts/ckpt_guard.sh <out_dir> [interval_sec]
# 例:
#   nohup bash scripts/ckpt_guard.sh outputs/golf_v7r_v3_phaseA_holdout 60 \
#       > outputs/golf_v7r_v3_phaseA_holdout/ckpt_guard.log 2>&1 &

set -u

OUT_DIR="${1:?用法: ckpt_guard.sh <out_dir> [interval_sec]}"
INTERVAL="${2:-60}"
SRC="$OUT_DIR/latest.pth"
BACKUP_DIR="$OUT_DIR/ckpt_backups"

mkdir -p "$BACKUP_DIR"
echo "[guard] watching $SRC every ${INTERVAL}s → $BACKUP_DIR"

last_sum=""
while true; do
    if [ -f "$SRC" ]; then
        # 用 mtime+size 判变化, 避免每次都拷 (拷贝 43MB 有 I/O 成本)
        cur="$(stat -c '%Y-%s' "$SRC" 2>/dev/null || true)"
        if [ -n "$cur" ] && [ "$cur" != "$last_sum" ]; then
            # 等 3s 让写盘落定, 再读 epoch (避免拷到半截文件)
            sleep 3
            epoch="$(/home/a1005/anaconda3/envs/ptorch/bin/python - "$SRC" <<'PY' 2>/dev/null || echo unknown
import sys, torch
try:
    ck = torch.load(sys.argv[1], map_location='cpu', weights_only=False)
    print(ck.get('epoch', 'unknown'))
except Exception:
    print('unknown')
PY
)"
            ts="$(date +%Y%m%d-%H%M%S)"
            dst="$BACKUP_DIR/latest_ep${epoch}_${ts}.pth"
            if cp "$SRC" "$dst" 2>/dev/null; then
                echo "[guard] $(date '+%F %T') backed up ep${epoch} → $(basename "$dst")"
                # 只保留最近 3 份, 防止磁盘堆积
                ls -1t "$BACKUP_DIR"/latest_ep*.pth 2>/dev/null | tail -n +4 | while read -r old; do
                    rm -f "$old" && echo "[guard] pruned $(basename "$old")"
                done
            else
                echo "[guard] $(date '+%F %T') copy failed (file busy?)"
            fi
            last_sum="$cur"
        fi
    fi
    sleep "$INTERVAL"
done
