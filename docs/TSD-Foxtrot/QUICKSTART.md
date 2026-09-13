# Foxtrot 快速开始指南

## 环境验证

```bash
cd /home/a1005/25/TFS-Net
/home/a1005/anaconda3/envs/ptorch/bin/python -c "
from models.foxtrot import TSDNet
import torch
model = TSDNet(num_frames=5).cuda()
x = torch.randn(1, 5, 3, 64, 64).cuda()
out = model(x)
print('Environment OK, output shape:', out['O_t'].shape)
"
```

预期输出：
```
Environment OK, output shape: torch.Size([1, 3, 64, 64])
```

## 快速训练

### 1. 检查数据集路径

编辑 `configs/foxtrot.yaml`，确认路径正确：
```yaml
dataset:
  train_input_root: /home/a1005/yzy/dataset/SDSD/indoor/input
  train_target_root: /home/a1005/yzy/dataset/SDSD/indoor/GT
  val_input_root: /home/a1005/yzy/dataset/SDSD/test/low-light
  val_target_root: /home/a1005/yzy/dataset/SDSD/test/GT
```

### 2. 启动训练（标准）

```bash
cd /home/a1005/25/TFS-Net
/home/a1005/anaconda3/envs/ptorch/bin/python train_foxtrot.py \
    --config configs/foxtrot.yaml
```

### 3. 启动训练（Keepalive，推荐）

防止断电/崩溃的自动重启：

```bash
cd /home/a1005/25/TFS-Net
bash scripts/keepalive_train.sh \
    configs/foxtrot.yaml \
    outputs/foxtrot_r1 \
    60
```

参数说明：
- 参数1: 配置文件路径
- 参数2: 输出目录（从 config 读取）
- 参数3: 目标 epoch 数

### 4. 监控训练

**方法1：实时日志**
```bash
tail -f outputs/foxtrot_r1/train.log
```

**方法2：集成监控（温度+训练）**
```bash
bash scripts/monitor.sh
```
自动检测活跃训练日志 + 系统温度，5秒刷新。

**方法3：TensorBoard**
```bash
tensorboard --logdir outputs/foxtrot_r1/tensorboard --port 6006
```

## 训练阶段说明

### Warmup (Epoch 1-5)
- 学习率从 0 线性增长到 `lr` (默认 0.0008)
- 模型逐步适应数据分布
- 正交损失开始约束三查询特征空间

**预期现象**：
- `L_ortho` 从 ~3.0 逐渐降至 ~0.5
- `L_final` 快速下降（~2.0 → ~0.5）
- 三分支损失 (L_N/L_L/L_M) 趋同

### 主训练阶段 (Epoch 6-55)
- 学习率 Cosine 衰减
- 三分支协同优化
- Fusion 网络学习自适应权重

**预期现象**：
- Val PSNR 稳步上升（目标 18-20 dB @ep30）
- Fusion weights 开始分化（不同区域偏好不同分支）
- `L_ortho` 稳定在 0.2-0.5

### 收敛阶段 (Epoch 56-60)
- 学习率降至最小值（~0.0001）
- 微调细节
- 验证最优 checkpoint

## 预期性能基线

基于 TFS-Net Delta Flight11 经验推测：

| Metric | Warmup (ep5) | Mid (ep30) | Final (ep60) |
|--------|--------------|------------|--------------|
| Val PSNR | 14-16 dB | 18-20 dB | 20-21 dB |
| Val SSIM | 0.60-0.65 | 0.70-0.75 | 0.75-0.80 |
| Train Loss | 0.8-1.0 | 0.3-0.5 | 0.2-0.3 |
| L_ortho | 2.0-3.0 | 0.3-0.5 | 0.2-0.3 |

**参照点**：
- Flight11 S1: ep10=16.70, ep30=18.72 (老损失函数)
- TBC1B (概念模型): ep40=19.99

**目标**：Foxtrot ep60 ≥ 20.5 dB (超越 Flight11 基线)

## 硬件要求

- **GPU**: RTX 3090 / 4090 (24GB)
- **显存占用**: 
  - batch_size=2, 256×256: ~8GB
  - batch_size=4, 256×256: ~14GB
- **训练时间**: 
  - 1 epoch ≈ 12-15 min (RTX 4090)
  - 60 epochs ≈ 12-15 小时

**优化建议**：
- 如果显存不足，降低 `batch_size` 并增加 `grad_accum_steps`
- 如果 CPU 瓶颈，增加 `num_workers` (当前 2)

## 常见问题

### Q1: Loss 出现 NaN

**原因**：
- 正交损失权重过大
- 学习率过高
- 梯度爆炸

**解决**：
```yaml
loss:
  lambda_ortho: 0.005  # 从 0.01 降低
train:
  grad_clip: 0.5       # 启用梯度裁剪
  lr: 0.0005           # 降低学习率
```

### Q2: Val PSNR 不增长

**原因**：
- 三分支信息冗余（正交约束不足）
- Fusion 权重退化为均匀分布
- 数据增强不足

**诊断**：
```python
# 检查 fusion_weights 分化程度
out = model(x, return_intermediate=True)
weights = out['fusion_weights']  # (B, 3, H, W)
print('Weight std:', weights.std(dim=1).mean())  # 应 > 0.1
```

**解决**：
- 增加 `lambda_ortho` 到 0.02
- 检查 AdaptiveFusion 是否学习到有效特征

### Q3: 显存溢出 (OOM)

**解决**：
```yaml
train:
  batch_size: 1           # 降低 batch size
  grad_accum_steps: 16    # 增加累积步数保持等效 batch
dataset:
  crop_size: 192          # 降低裁剪尺寸
```

### Q4: 训练速度慢

**优化**：
```yaml
dataset:
  num_workers: 4          # 增加数据加载线程
train:
  amp: true               # 确保混合精度开启
```

**检查瓶颈**：
```bash
# 监控 GPU 利用率
watch -n 1 nvidia-smi
# 应保持 90%+ 利用率
```

## 消融实验

### 测试 TCA-RWKV vs LocalTCA

创建配置 `configs/foxtrot_ablation.yaml`：
```yaml
model:
  tca_type: local  # 替换为 LocalWindowAlignment
  # ... 其他配置同 foxtrot.yaml
```

### 测试正交约束影响

```yaml
loss:
  lambda_ortho: 0.0  # 禁用正交约束
```

### 测试单查询 baseline

修改 `tca_rwkv.py`，将三查询合并为单查询。

## 结果分析

### 查看训练曲线

```bash
grep "Val:" outputs/foxtrot_r1/train.log | tail -10
```

### 提取最优模型

```bash
# best.pth 是验证集上 PSNR 最高的 checkpoint
cp outputs/foxtrot_r1/checkpoints/best.pth \
   models/pretrained/foxtrot_best.pth
```

### Pair45 推理测试

```bash
python scripts/inference_pair45.py \
    --model foxtrot \
    --checkpoint outputs/foxtrot_r1/checkpoints/best.pth \
    --input /path/to/pair45/input \
    --output outputs/foxtrot_r1/pair45_pred
```

## 下一步

1. **完整训练**：运行 60 epochs，验证 PSNR ≥ 20.5
2. **消融实验**：验证 TCA-RWKV 和正交约束的贡献
3. **对比 Flight11**：在 pair45 上对比增强效果
4. **可视化**：
   - Fusion weights 热图（哪些区域偏好哪个分支）
   - 方差图 vs 噪声区域对应
   - 光流可视化（运动检测准确性）
5. **迁移测试**：在 LOL/SID 等其他数据集验证泛化性

## 技术支持

- **实现文档**: `docs/TSD-Foxtrot/IMPLEMENTATION.md`
- **设计文档**: `docs/TSD-Foxtrot/TSD-Foxtrot.md`
- **源码**: `models/foxtrot/`
- **训练脚本**: `train_foxtrot.py`

---

**最后更新**: 2026-09-13  
**状态**: ✅ 实现完成，等待完整训练验证
