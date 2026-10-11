#!/usr/bin/env bash
# 串行任务编排器 —— 避免「pgrep -f 自匹配」陷阱的正规做法。
#
# 动机 (2026-10-10/11 三次踩坑记录):
#   用 `while pgrep -f '<某模式>' ...; do sleep; done` 等待前置任务时, 若**本脚本自己的
#   命令行里包含该模式** (因为模式是以字面量写在脚本里的), pgrep 会匹配到自己 →
#   死循环 (实测: `multiseed_queue.log` 保持 0 字节, 任务从未启动)。
#   前两次同类事故: `pkill -f 'ckpt_guard.sh outputs/...'` 自杀; `pgrep -f` 匹配自身。
#
# 正确做法:
#   1. **用 PID 等待**, 不做模式匹配 —— bash 的 `wait <pid>` 只对本 shell 的子进程有效,
#      跨进程用 `kill -0 <pid>` 判存活;
#   2. 或让**上游任务自己**触发下一阶段 (本脚本采用: 顺序调用, 不用轮询)。
#
# 用法:
#   nohup bash scripts/run_chain.sh <out_log> <cmd1...> -- <cmd2...> -- ... &
# 例:
#   nohup bash scripts/run_chain.sh outputs/chain.log \
#       "python a.py" -- "python b.py" &

set -u

LOG="${1:?用法: run_chain.sh <out_log> <cmd> [-- <cmd> ...]}"
shift

mkdir -p "$(dirname "$LOG")"
{
    echo "[chain] start $(date '+%F %T')"
    # 用 -- 分隔多段命令
    IFS=$'\n'
    for cmd in "$@"; do
        [ "$cmd" = "--" ] && continue
        echo ""
        echo "[chain] ================ $(date '+%F %T') ================"
        echo "[chain] RUN: $cmd"
        echo "[chain] ===================================================="
        # 不用 set -e: 单段失败也要继续下一段 (便于一次拿到全部结果),
        # 但记录退出码供事后判断。
        eval "$cmd"
        rc=$?
        echo "[chain] EXIT=$rc  ($(date '+%F %T'))"
        if [ $rc -ne 0 ]; then
            echo "[chain] ⚠️ 上一条命令失败 (exit=$rc), 继续执行后续命令"
        fi
    done
    echo ""
    echo "[chain] all done $(date '+%F %T')"
} >> "$LOG" 2>&1
