#!/usr/bin/env bash
# v8 判据读出: 对两个 v8 run 各跑 分支诊断(判据1/2/3) + 分支消融(判据4)。
# 参照 = outputs/golf_v7r_v3_phaseF_band_targets (已在 v7.5 消融日志中, 不重复跑)。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
OUT=outputs/v8_readout
mkdir -p "$OUT"

log() { echo "[v8abl] $*"; }

{
log "start @ $(date '+%F %T')"

for entry in "branchM_fix:outputs/v8_branchM_fix" "residual_omnishift:outputs/v8_phase1_residual_omnishift"; do
    name=${entry%%:*}
    dir=${entry##*:}
    # 配置文件: 优先用 run 目录内的 _cfg.yaml, 否则回落到 configs/v8/
    if [ -f "$dir/_cfg.yaml" ]; then cfg="$dir/_cfg.yaml"
    else cfg="configs/v8/phase1_residual_omnishift.yaml"; fi

    log ""
    log "===== $name ($dir) @ $(date '+%F %T') ====="
    log "  cfg: $cfg"
    if [ ! -f "$dir/best.pth" ]; then
        log "  ⚠️ 无 best.pth, 跳过"
        continue
    fi

    log "  --- 分支诊断 (判据 1/2/3) ---"
    PYTHONPATH=. $PY scripts/diag_v7r_v3_branches.py \
        --ckpt "$dir/best.pth" --config "$cfg" 2>&1 | grep -aE 'RESULT|Traceback|Error'
    log "  EXIT=$?"

    log "  --- 分支消融 (判据 4) ---"
    PYTHONPATH=. $PY scripts/diag_v7r_v3_branch_ablation.py \
        --ckpt "$dir/best.pth" --config "$cfg" --n_seqs 24 2>&1 | grep -aE 'RESULT|Traceback|Error'
    log "  EXIT=$?"
done

log ""
log "===== 全部完成 @ $(date '+%F %T') ====="
} >> "$OUT/v8_ablation.log" 2>&1
