#!/usr/bin/env bash
# 交错双臂续接脚本: 等两个并行进程结束后, 接着跑剩余 4 个 run。
#
# 背景: interleave.sh 原设计是"control s0 完成后停掉旧链 → 串行交错"。
# 但 kill 只停了 bash 父进程, python 子进程(324034)以孤儿形式继续跑 control s1。
# 同时 interleave.sh 已启动了 decorr s0 (324216)。
# 结果: control s1 与 decorr s0 并行共享 GPU (总 23.3GB, 速率各慢 ~45%)。
#
# 修正方案 (2026-10-11):
# 不停两个进程 (它们写入目录正确, 进度完好), 而是:
# 1) 等两者都结束;
# 2) 按【跳过 control 1】的序列串行跑剩余: decorr1 → control2 → decorr2
# 3) 最后汇总两臂结果。
#
# 判据 (用 PID 等待, 不用 pgrep -f 避免自匹配):
set -u
cd /home/a1005/25/TFS-Net || exit 1
PY=/home/a1005/anaconda3/envs/ptorch/bin/python
LOG=outputs/readout_phaseA/interleave_resume.log
CTRL=outputs/multiseed_ref_softmax_runs
DEC=outputs/multiseed_decorr_runs

log() { echo "[resume] $*"; }

{
log "start @ $(date '+%F %T')"
log "等待并行进程 control-s1(324034) 和 decorr-s0(324216) 结束..."

# 用 PID 等待 (kill -0 检测存活, 不做模式匹配)
for pid in 324034 324216; do
    while kill -0 "$pid" 2>/dev/null; do sleep 60; done
    log "pid $pid 已退出 @ $(date '+%F %T')"
done
log "两个并行进程全部结束"

# 验证两者的 best.pth 是否已生成
for d in "$CTRL/seed1" "$DEC/seed0"; do
    if [ -f "$d/best.pth" ]; then
        psnr=$(grep -oE 'PSNR: [0-9.]+' "$d/train.log" 2>/dev/null | tail -1)
        log "  ✅ $d/best.pth  $psnr"
    else
        log "  ⚠️  $d/best.pth 未生成, 继续仍可跑后续"
    fi
done

# 接续序列: 跳过 control s1 (已在跑), 继续剩余
for t in "decorr 1" "control 2" "decorr 2"; do
    arm=${t%% *}; sd=${t##* }
    if [ "$arm" = decorr ]; then
        base_cfg=configs/golf_v7r_v3_phaseC_decorr.yaml
        out_dir="$DEC/seed$sd"
    else
        base_cfg=configs/golf_v7r_v3_phaseB_softmax_ctl.yaml
        out_dir="$CTRL/seed$sd"
    fi
    log ""
    log "===== $arm seed$sd @ $(date '+%F %T') ====="
    log "  out: $out_dir"
    mkdir -p "$out_dir"
    # 磁盘保护
    free_gb=$(python3 -c "import os; s=os.statvfs('$out_dir'); print(s.f_bavail*s.f_frsize/1e9)")
    if python3 -c "exit(0 if float('$free_gb') >= 2.0 else 1)"; then
        log "  磁盘余量 ${free_gb}GB ≥ 2GB, 继续"
    else
        log "  ⚠️  磁盘余量 ${free_gb}GB < 2GB, 中止"
        break
    fi
    # 生成临时配置
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
        psnr=$(grep -oE 'PSNR: [0-9.]+' "$out_dir/train.log" 2>/dev/null | tail -1)
        log "  ✅ best.pth saved  $psnr"
    else
        log "  ⚠️  best.pth 未生成"
    fi
done

log ""
log "===== 全部完成 @ $(date '+%F %T') ====="
log "汇总:"
for arm in control decorr; do
    if [ "$arm" = control ]; then d=$CTRL; else d=$DEC; fi
    for sd in 0 1 2; do
        f="$d/seed$sd/train.log"
        if [ -f "$f" ]; then
            psnr=$(grep -oE 'Validation - PSNR: [0-9.]+' "$f" | tail -1)
            log "  $arm seed$sd: $psnr"
        fi
    done
done
} >> "$LOG" 2>&1
