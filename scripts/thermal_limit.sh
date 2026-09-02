#!/bin/bash
# CPU/GPU 热治理配置 — 一次性 root 脚本 (重启后需重跑)
# 用法: sudo bash /home/a1005/25/TFS-Net/scripts/thermal_limit.sh
#
# 作用:
#   1. CPU 封装功耗墙: PL1(持续)=95W, PL2(短时)=119W
#      — 直接封顶发热总量, 任何负载分布下 package 温度可控
#   2. GPU 功耗墙: 450W -> 350W (安全余量, 当前训练只用 ~155W 无性能损失)
#
# 这些是运行时设置, 重启失效。训练 launch 脚本会检测并提醒。

set -e

RAPL=/sys/class/powercap/intel-rapl:0

echo "== CPU RAPL 功耗墙 =="
OLD_PL1=$(cat $RAPL/constraint_0_power_limit_uw)
OLD_PL2=$(cat $RAPL/constraint_1_power_limit_uw)
echo "当前: PL1(长期)=$((OLD_PL1/1000000))W  PL2(短时)=$((OLD_PL2/1000000))W"

echo 95000000  > $RAPL/constraint_0_power_limit_uw
echo 119000000 > $RAPL/constraint_1_power_limit_uw

NEW_PL1=$(cat $RAPL/constraint_0_power_limit_uw)
NEW_PL2=$(cat $RAPL/constraint_1_power_limit_uw)
echo "已设: PL1(长期)=${NEW_PL1:0:-6}W  PL2(短时)=${NEW_PL2:0:-6}W"

echo ""
echo "== GPU 功耗墙 =="
nvidia-smi -pl 350
nvidia-smi --query-gpu=power.limit --format=csv,noheader

echo ""
echo "完成。训练进程无需重启, 立即生效。"
