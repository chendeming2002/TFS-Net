#!/bin/bash
# CPU/GPU 功耗墙配置 — 一次性 root 脚本 (重启后需重跑)
# 用法: sudo bash /home/a1005/25/TFS-Net/scripts/thermal_limit.sh
#       sudo PL1_W=65 PL2_W=80 DISABLE_TURBO=1 bash scripts/thermal_limit.sh   # 更激进
#
# 作用:
#   1. CPU 封装功耗墙: PL1(持续)=95W, PL2(短时)=119W
#   2. 可选关闭 Turbo (DISABLE_TURBO=1) — 消除单核瞬时冲顶
#   3. GPU 功耗墙: 450W -> 350W
#
# ⚠️ 2026-10-09 结论更新 (重要, 推翻早期的"过热"假设):
#   当日发生两次瞬间失电 (11:19 / 15:31)。事后证据链表明**与过热无关**:
#     - 第 2 次断电时 CPU 仅 59°C、GPU 62°C (当时 PL1=65W 生效中, 已降温 40°C);
#     - 无 vmcore / 无 pstore / 无 MCE / 无 thermal 告警 / 无 shutdown 序列。
#   → 判定为**供电侧瞬间失电** (PSU 保护或老化最可能, 见 docs §6.15)。
#   **因此本脚本的功耗墙对断电无预防作用；默认值已恢复原状, 不再下调。**
#
#   实测代价 (2026-10-09): PL1=65W + no_turbo=1 会把训练拖慢 18%
#   (0.54 → 0.93 s/step)。所以"降频换稳定性"在这里是净亏损 —— 断电不是热导致的,
#   降频既防不住断电、又实实在在损失吞吐。故**不建议**再应用激进参数。
#
#   保留本脚本仅用于: (a) 确实需要压温时手动下调; (b) 断电后需手动恢复默认时参考。
#
# 这些是运行时设置, 重启失效。

set -e

PL1_W=${PL1_W:-95}
PL2_W=${PL2_W:-119}
DISABLE_TURBO=${DISABLE_TURBO:-0}

RAPL=/sys/class/powercap/intel-rapl:0

echo "== CPU RAPL 功耗墙 =="
OLD_PL1=$(cat $RAPL/constraint_0_power_limit_uw)
OLD_PL2=$(cat $RAPL/constraint_1_power_limit_uw)
echo "当前: PL1(长期)=$((OLD_PL1/1000000))W  PL2(短时)=$((OLD_PL2/1000000))W"

echo $((PL1_W * 1000000)) > $RAPL/constraint_0_power_limit_uw
echo $((PL2_W * 1000000)) > $RAPL/constraint_1_power_limit_uw

NEW_PL1=$(cat $RAPL/constraint_0_power_limit_uw)
NEW_PL2=$(cat $RAPL/constraint_1_power_limit_uw)
echo "已设: PL1(长期)=$((NEW_PL1/1000000))W  PL2(短时)=$((NEW_PL2/1000000))W"

echo ""
echo "== CPU Turbo =="
TURBO=/sys/devices/system/cpu/intel_pstate/no_turbo
if [ -w "$TURBO" ]; then
    if [ "$DISABLE_TURBO" = "1" ]; then
        echo 1 > $TURBO
        echo "已关闭 Turbo (no_turbo=1)"
    else
        echo 0 > $TURBO
        echo "Turbo 保持开启 (no_turbo=0)"
    fi
    echo "当前 no_turbo=$(cat $TURBO)"
else
    echo "跳过: $TURBO 不可写 (intel_pstate 未启用)"
fi

echo ""
echo "== GPU 功耗墙 =="
nvidia-smi -pl 350
nvidia-smi --query-gpu=power.limit --format=csv,noheader

echo ""
echo "== 生效后温度 (5s 后采样) =="
sleep 5
sensors 2>/dev/null | grep -E 'Package id 0|Core 20' || true

echo ""
echo "完成。训练进程无需重启, 立即生效。"
echo "恢复默认 (不限制 CPU, GPU 450W):"
echo "  sudo bash -c 'echo 4095000000 > /sys/class/powercap/intel-rapl:0/constraint_0_power_limit_uw;"
echo "                echo 4095000000 > /sys/class/powercap/intel-rapl:0/constraint_1_power_limit_uw;"
echo "                echo 0 > /sys/devices/system/cpu/intel_pstate/no_turbo; nvidia-smi -pl 450'"
