#!/usr/bin/env bash
# 用【修正后的】消融口径重算全部加性融合 run 的判据 4 (2026-10-16)。
# 背景: 原脚本对 additive 融合错误地归一化剩余权重, 制造 ~8 dB 假衰减 (§6.20)。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
OUT=outputs/v8_readout
mkdir -p "$OUT"

{
echo "[fixabl] start @ $(date '+%F %T')"
for entry in \
  "phaseF:outputs/golf_v7r_v3_phaseF_band_targets:outputs/golf_v7r_v3_phaseF_band_targets/_cfg.yaml" \
  "S1:outputs/v7_5_scan/S1_share_high:outputs/v7_5_scan/S1_share_high/_cfg.yaml" \
  "S2:outputs/v7_5_scan/S2_gate_on:outputs/v7_5_scan/S2_gate_on/_cfg.yaml" \
  "S3:outputs/v7_5_scan/S3_kernel_15_5:outputs/v7_5_scan/S3_kernel_15_5/_cfg.yaml" \
  "v8_branchM_fix:outputs/v8_branchM_fix:outputs/v8_branchM_fix/_cfg.yaml" \
  "v8_residual:outputs/v8_phase1_residual_omnishift:configs/v8/phase1_residual_omnishift.yaml" ; do
    name=$(echo "$entry" | cut -d: -f1)
    dir=$(echo "$entry" | cut -d: -f2)
    cfg=$(echo "$entry" | cut -d: -f3)
    echo ""
    echo "[fixabl] ===== $name ====="
    [ -f "$dir/best.pth" ] || { echo "[fixabl]   ⚠️ 无 best.pth"; continue; }
    PYTHONPATH=. $PY scripts/diag_v7r_v3_branch_ablation.py \
        --ckpt "$dir/best.pth" --config "$cfg" --n_seqs 24 2>&1 | grep -aE 'RESULT|Traceback|Error'
done
echo ""
echo "[fixabl] 全部完成 @ $(date '+%F %T')"
} >> "$OUT/ablation_fixed.log" 2>&1
