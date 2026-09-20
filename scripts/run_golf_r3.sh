#!/bin/bash
# Golf-R3 训练启动脚本
# 用法: bash scripts/run_golf_r3.sh

set -e

CONFIG="configs/golf_r3.yaml"
OUTPUT_DIR="outputs/golf_r3_s1"

echo "======================================"
echo "Golf-R3 S1 启动前检查"
echo "======================================"

# 1. 语法检查
echo "[1/5] 语法检查..."
python3 -m py_compile models/golf_r3/*.py train_golf_r3.py || {
    echo "❌ 语法错误，请检查代码"
    exit 1
}
echo "✅ 语法检查通过"

# 2. 配置验证
echo "[2/5] 配置验证..."
python3 -c "
import yaml
cfg = yaml.safe_load(open('$CONFIG'))
assert cfg['model']['use_f3_film'] == True, 'F3 FiLM 未开启'
assert cfg['model']['use_hires_warp'] == True, 'hires warp 未开启'
assert cfg['loss']['temp_threshold'] == 1.2, 'temp_threshold 错误'
assert 'pair45_input_root' in cfg['dataset'], 'pair45 路径缺失'
print('✅ 配置验证通过')
" || {
    echo "❌ 配置错误"
    exit 1
}

# 3. 模型实例化测试
echo "[3/5] 模型实例化测试..."
python3 -c "
import torch
import sys
sys.path.insert(0, '.')
from models.golf_r3 import GolfNet_R3
from models.golf_r3.loss import GolfLoss

m = GolfNet_R3(use_f3_film=True, use_hires_warp=True, tca_channels=128, n_blocks=6)
x = torch.randn(1, 5, 3, 256, 256)
out = m(x)
assert 'res_t' in out, '输出缺少 res_t'
assert out['res_t'].shape == (1, 3, 256, 256), f'输出尺寸错误: {out[\"res_t\"].shape}'

loss_fn = GolfLoss(temp_threshold=1.2)
print(f'✅ 模型实例化成功, 输出={out[\"res_t\"].shape}')
" || {
    echo "❌ 模型实例化失败"
    exit 1
}

# 4. pair45 路径检查
echo "[4/5] pair45 数据集检查..."
PAIR45_INPUT="/home/a1005/yzy/dataset/SDSD/test/low-light/pair45"
PAIR45_GT="/home/a1005/yzy/dataset/SDSD/test/GT/pair45"

if [ ! -d "$PAIR45_INPUT" ] || [ ! -d "$PAIR45_GT" ]; then
    echo "⚠️  pair45 路径不存在，将跳过 pair45 验证"
else
    N_INPUT=$(ls $PAIR45_INPUT/*.png 2>/dev/null | wc -l)
    N_GT=$(ls $PAIR45_GT/*.png 2>/dev/null | wc -l)
    if [ $N_INPUT -lt 30 ] || [ $N_GT -lt 30 ]; then
        echo "⚠️  pair45 帧数不足 (input=$N_INPUT, gt=$N_GT)，建议检查"
    else
        echo "✅ pair45 数据集检查通过 (input=$N_INPUT, gt=$N_GT)"
    fi
fi

# 5. 输出目录准备
echo "[5/5] 输出目录准备..."
mkdir -p "$OUTPUT_DIR"
echo "✅ 输出目录: $OUTPUT_DIR"

echo ""
echo "======================================"
echo "所有检查通过，启动训练"
echo "======================================"
echo "配置文件: $CONFIG"
echo "输出目录: $OUTPUT_DIR"
echo "预计时长: 60 epoch × 30 min ≈ 30h"
echo ""
echo "按 Ctrl+C 可随时停止训练"
echo "======================================"
echo ""

# 启动训练
python3 train_golf_r3.py --config "$CONFIG" 2>&1 | tee "$OUTPUT_DIR/train.log"
