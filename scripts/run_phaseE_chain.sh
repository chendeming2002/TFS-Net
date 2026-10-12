#!/usr/bin/env bash
# Phase E 双臂串行链: 分支专属监督目标 vs 共享目标 (各 3 seed)。
#
# 唯一变量: loss.branch_specific_targets
#   treated: true   三路各回归 noise/illum/motion 三个不同目标
#   control: false  三路都回归同一张 GT (历史行为)
#
# 为什么对照臂要重跑而不复用 multiseed_ref_softmax_runs:
#   那组 run 跑在数据集加入 branch_targets 参数之前的代码上。开关虽默认关闭、返回
#   结构不变, 但为让 A/B 落在同一代码版本与同一数据加载路径上, 对照臂必须重跑
#   (§6.14 的教训: 跨代码版本比较不可信)。
#
# 交错顺序 (treated/control 交替): 避免"先跑完一臂再跑另一臂"时, 机器状态漂移
#   (温度、磁盘碎片、其他进程) 被整体归因到臂上。
#
# 等待方式用 PID (kill -0), 不做命令行模式匹配 —— pgrep -f 会匹配到自己 (已踩三次)。
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
LOG=outputs/phaseE/chain.log
TREATED=outputs/multiseed_phaseE_targets
CONTROL=outputs/multiseed_phaseE_shared
mkdir -p outputs/phaseE "$TREATED" "$CONTROL"

log() { echo "[phaseE] $*"; }

{
log "start @ $(date '+%F %T')"

for task in "treated 0" "control 0" "treated 1" "control 1" "treated 2" "control 2"; do
    arm=${task%% *}
    sd=${task##* }
    if [ "$arm" = treated ]; then
        base_cfg=configs/golf_v7r_v3_phaseE_branch_targets.yaml
        out_dir="$TREATED/seed$sd"
    else
        base_cfg=configs/golf_v7r_v3_phaseE_shared_ctl.yaml
        out_dir="$CONTROL/seed$sd"
    fi

    log ""
    log "===== $arm seed$sd @ $(date '+%F %T') ====="
    log "  out: $out_dir"
    mkdir -p "$out_dir"

    free_gb=$(python3 -c "import os; s=os.statvfs('$out_dir'); print(s.f_bavail*s.f_frsize/1e9)")
    if python3 -c "exit(0 if float('$free_gb') >= 2.0 else 1)"; then
        log "  磁盘余量 ${free_gb}GB ≥ 2GB, 继续"
    else
        log "  ⚠️  磁盘余量 ${free_gb}GB < 2GB, 中止后续"
        break
    fi

    cfg_path="$out_dir/_cfg.yaml"
    PYTHONPATH=. $PY - << PY
import yaml
c = yaml.safe_load(open('$base_cfg'))
c['seed'] = $sd
c['output_dir'] = '$out_dir'
yaml.safe_dump(c, open('$cfg_path', 'w'), allow_unicode=True, sort_keys=False)
PY

    PYTHONPATH=. $PY train_golf_v7r_v3.py --config "$cfg_path" --stop_epoch 5 \
        > "$out_dir/train.log" 2>&1
    rc=$?
    log "  EXIT=$rc @ $(date '+%F %T')"
    if [ -f "$out_dir/best.pth" ]; then
        psnr=$(grep -oE 'PSNR: [0-9.]+' "$out_dir/train.log" | tail -1)
        log "  ✅ best.pth saved  $psnr"
    else
        log "  ⚠️  best.pth 未生成"
    fi
done

log ""
log "===== 全部完成 @ $(date '+%F %T') ====="
log "汇总 (验证 PSNR 仅供参考, 本机 5ep seed 间方差达 3.2 dB, 判据见配置头部):"
for arm in treated control; do
    if [ "$arm" = treated ]; then d=$TREATED; else d=$CONTROL; fi
    for sd in 0 1 2; do
        f="$d/seed$sd/train.log"
        if [ -f "$f" ]; then
            psnr=$(grep -oE 'Validation - PSNR: [0-9.]+' "$f" | tail -1)
            log "  $arm seed$sd: ${psnr:-无记录}"
        fi
    done
done
log ""
log "下一步 (机制判据, 必须跑):"
log "  1) 三路输出余弦与分支形态:"
log "     PYTHONPATH=. python scripts/diag_v7r_v3_branches.py \\"
log "       --ckpt $TREATED/seed0/best.pth --config $TREATED/seed0/_cfg.yaml"
log "  2) 分工判据 (去掉任一路后指标应明显变差):"
log "     PYTHONPATH=. python scripts/diag_v7r_v3_branch_ablation.py \\"
log "       --ckpt $TREATED/seed0/best.pth --config $TREATED/seed0/_cfg.yaml --n_seqs 24"
log "  3) 双臂配对评估:"
log "     PYTHONPATH=. python scripts/eval_multiseed_arms.py \\"
log "       --control $CONTROL --control_seeds 0 1 2 \\"
log "       --treated $TREATED --treated_seeds 0 1 2 \\"
log "       --out_json outputs/phaseE/per_seq_psnr.json"
} >> "$LOG" 2>&1
