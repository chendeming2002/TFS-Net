#!/usr/bin/env bash
# Phase E 机制判据全套读出 (链日志指定的三条命令 × 双臂 seed)
set -u
cd /home/a1005/25/TFS-Net
PY=/home/a1005/anaconda3/envs/ptorch/bin/python

for arm in targets shared; do
  for sd in 0 1 2; do
    D=outputs/multiseed_phaseE_$arm/seed$sd
    [ -f "$D/best.pth" ] || continue
    echo "===== [$arm s$sd] diag_branches @ $(date +%H:%M:%S) ====="
    PYTHONPATH=. $PY scripts/diag_v7r_v3_branches.py --ckpt "$D/best.pth" --config "$D/_cfg.yaml" \
        2>&1 | grep -aE 'RESULT|Traceback|Error'
  done
done

echo "===== [treated s0] ablation @ $(date +%H:%M:%S) ====="
PYTHONPATH=. $PY scripts/diag_v7r_v3_branch_ablation.py \
    --ckpt outputs/multiseed_phaseE_targets/seed0/best.pth \
    --config outputs/multiseed_phaseE_targets/seed0/_cfg.yaml --n_seqs 24

echo "===== [control s0] ablation @ $(date +%H:%M:%S) ====="
PYTHONPATH=. $PY scripts/diag_v7r_v3_branch_ablation.py \
    --ckpt outputs/multiseed_phaseE_shared/seed0/best.pth \
    --config outputs/multiseed_phaseE_shared/seed0/_cfg.yaml --n_seqs 24

echo "===== 双臂配对评估 3v3 @ $(date +%H:%M:%S) ====="
PYTHONPATH=. $PY scripts/eval_multiseed_arms.py \
    --control outputs/multiseed_phaseE_shared --control_seeds 0 1 2 \
    --treated outputs/multiseed_phaseE_targets --treated_seeds 0 1 2 \
    --out_json outputs/phaseE/per_seq_psnr.json

echo "===== ALL READOUT DONE @ $(date +%H:%M:%S) ====="
