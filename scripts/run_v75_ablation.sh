#!/usr/bin/env bash
# v7.5 第 2 段收尾: 对四个 run 逐个做分支诊断与消融 (判据 1/2/3/4)。
#
# 取消 40ep 训练段后, 消融实验是本次定稿唯一的判据 4 来源, 因此覆盖全部四个 run ——
# 它们之间的差异 (频带划分方式、残差门) 正好影响"某路是否携带大块直流分量",
# 而那是判据 4 数值的主要干扰, 四个一起看才能把"分工"与"直流分量"区分开。
#
# 单卡须串行 (并发会 OOM 且拖慢)。等待方式用 PID + kill -0, 不做命令行模式匹配。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
SCAN_PID=${1:?用法: run_v75_ablation.sh <scan_chain_pid>}
LOG=outputs/v7_5_scan/ablation/ablation.log
mkdir -p outputs/v7_5_scan/ablation

log() { echo "[abl] $*"; }

# run 列表: 名称 目录
RUNS=(
  "stage1:outputs/golf_v7r_v3_phaseF_band_targets"
  "S1:outputs/v7_5_scan/S1_share_high"
  "S2:outputs/v7_5_scan/S2_gate_on"
  "S3:outputs/v7_5_scan/S3_kernel_15_5"
)

{
log "start @ $(date '+%F %T'), 等待第 2 段扫描链 pid=$SCAN_PID"
while kill -0 "$SCAN_PID" 2>/dev/null; do sleep 60; done
log "扫描链结束 @ $(date '+%F %T')"
log ""

for entry in "${RUNS[@]}"; do
    name=${entry%%:*}
    dir=${entry##*:}

    log "===== $name ($dir) @ $(date '+%F %T') ====="
    if [ ! -f "$dir/best.pth" ]; then
        log "  ⚠️ 无 best.pth, 跳过"
        continue
    fi
    if [ ! -f "$dir/_cfg.yaml" ]; then
        log "  ⚠️ 无 _cfg.yaml (诊断脚本必须按训练配置重建模型), 跳过"
        continue
    fi

    # 判据 1/2/3: 三路形态、输出余弦、单独 PSNR
    log "  --- 分支诊断 (判据 1/2/3) ---"
    PYTHONPATH=. $PY -u scripts/diag_v7r_v3_branches.py \
        --ckpt "$dir/best.pth" --config "$dir/_cfg.yaml" \
        > "outputs/v7_5_scan/ablation/${name}_diag.log" 2>&1
    log "  EXIT=$?"
    grep -E "out cos|scale_|fusion gamma" \
        "outputs/v7_5_scan/ablation/${name}_diag.log" | sed 's/^/    /'

    # 判据 4: 去掉任一路后指标是否明显变差
    log "  --- 分支消融 (判据 4) ---"
    PYTHONPATH=. $PY -u scripts/diag_v7r_v3_branch_ablation.py \
        --ckpt "$dir/best.pth" --config "$dir/_cfg.yaml" --n_seqs 24 \
        > "outputs/v7_5_scan/ablation/${name}_ablation.log" 2>&1
    log "  EXIT=$?"
    grep "^RESULT minus" "outputs/v7_5_scan/ablation/${name}_ablation.log" | sed 's/^/    /'
    log ""
done

log "===== 全部完成 @ $(date '+%F %T') ====="
log "判读要点:"
log "  判据 4 单独看会被误导 —— 塌成常数的分支消融代价同样大。"
log "  只有判据 1/2/3 同时成立时, 才可把消融代价解释为分工。"
} >> "$LOG" 2>&1
