#!/usr/bin/env bash
# Phase A 三点探针串行启动脚本
# 用法: bash scripts/run_phase_a_probes.sh
# 日志: outputs/golf_v7r_v3_phaseA_{m,l,holdout}/train.log
#       outputs/probe_all.log

set -e
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
ROOT=/home/a1005/25/TFS-Net

mkdir -p "$ROOT/outputs/golf_v7r_v3_phaseA_m"
mkdir -p "$ROOT/outputs/golf_v7r_v3_phaseA_l"
mkdir -p "$ROOT/outputs/golf_v7r_v3_phaseA_holdout"

cd "$ROOT"

echo "[probe] starting phaseA_m at $(date)"
$PY train_golf_v7r_v3.py \
    --config configs/golf_v7r_v3_phaseA_m.yaml \
    --stop_epoch 5 \
    > outputs/golf_v7r_v3_phaseA_m/train.log 2>&1
echo "[probe] phaseA_m done at $(date)"

echo "[probe] starting phaseA_l at $(date)"
$PY train_golf_v7r_v3.py \
    --config configs/golf_v7r_v3_phaseA_l.yaml \
    --stop_epoch 5 \
    > outputs/golf_v7r_v3_phaseA_l/train.log 2>&1
echo "[probe] phaseA_l done at $(date)"

echo "[probe] starting phaseA_holdout at $(date)"
$PY train_golf_v7r_v3.py \
    --config configs/golf_v7r_v3_phaseA_holdout.yaml \
    --stop_epoch 5 \
    > outputs/golf_v7r_v3_phaseA_holdout/train.log 2>&1
echo "[probe] phaseA_holdout done at $(date)"

echo "[probe] all three probes complete at $(date)"
