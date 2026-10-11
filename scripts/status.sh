#!/usr/bin/env bash
# 一键查看当前实验链状态 (只读, 不改动任何东西)。
#
# 动机: 本会话多次出现"以为任务在跑、其实早已退出/从未启动"或反之。
# 用固定入口查询, 避免每次临时拼命令 (也曾因临时命令里的字面量被 pgrep 自匹配而误判)。
#
# 用法: bash scripts/status.sh

set -u
cd "$(dirname "$0")/.." || exit 1

echo "════════ 实验链状态 $(date '+%F %T') ════════"

echo ""
echo "── 1. 运行中的训练/评估进程 ──"
found=0
for pat in train_golf_v7r_v3.py multiseed_baseline.py eval_checkpoint.py; do
    while read -r pid etime; do
        [ -z "${pid:-}" ] && continue
        found=1
        echo "  pid=$pid  已运行 $etime  $pat"
    done < <(ps -eo pid,etime,args | awk -v p="$pat" \
        '$0 ~ p && $0 !~ /awk|status\.sh|run_chain|queue_decorr/ {print $1, $2}')
done
[ $found -eq 0 ] && echo "  (无)"

echo ""
echo "── 2. 等待器 / 编排器 ──"
ps -eo pid,args | grep -E 'queue_decorr|run_chain' | grep -v grep | cut -c1-110 || echo "  (无)"

echo ""
echo "── 3. GPU ──"
nvidia-smi --query-gpu=utilization.gpu,power.draw,temperature.gpu,memory.used --format=csv,noheader 2>/dev/null || echo "  (不可用)"

echo ""
echo "── 4. 磁盘 ──"
df -h /home/a1005 | tail -1

echo ""
echo "── 5. 各实验链日志尾部 ──"
for f in outputs/readout_phaseA/multiseed_chain.log \
         outputs/readout_phaseA/phaseC_queue.log \
         outputs/readout_phaseA/phaseB_eval.log; do
    if [ -f "$f" ]; then
        echo "  ── $f ──"
        tail -3 "$f" | sed 's/^/    /'
    fi
done

echo ""
echo "── 6. 已完成的 run 检查点 ──"
for d in outputs/golf_v7r_v3_phaseB_floored outputs/golf_v7r_v3_phaseB_softmax_ctl; do
    [ -f "$d/best.pth" ] && echo "  ✅ $d/best.pth"
done
for d in outputs/multiseed_ref_softmax_runs/seed* outputs/multiseed_decorr_runs/seed*; do
    [ -f "$d/best.pth" ] && echo "  ✅ $d/best.pth"
done

echo ""
echo "── 7. 结果 JSON ──"
for f in outputs/multiseed_ref_softmax.json outputs/multiseed_decorr.json; do
    [ -f "$f" ] && echo "  ✅ $f" || echo "  ·  $f (未生成)"
done

echo "════════════════════════════════════════"
