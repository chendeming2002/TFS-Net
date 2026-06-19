"""分析模型参数分布和梯度流特性 (v3.2)"""
import torch
from models import TFSNet
from losses.losses import TFSNetLoss

model = TFSNet(
    in_channels=3,
    level_channels=[32, 64, 96, 128],
    fused_channels=64
)

print("=" * 70)
print("TFS-Net v3.1 参数量分析")
print("=" * 70)

total = 0
module_params = {}
for name, p in model.named_parameters():
    parts = name.split('.')
    top_module = parts[0]
    if top_module not in module_params:
        module_params[top_module] = 0
    module_params[top_module] += p.numel()
    total += p.numel()

print(f"\n{'模块':<20} {'参数量':>12} {'占比':>8}")
print("-" * 44)
for mod, count in sorted(module_params.items(), key=lambda x: -x[1]):
    print(f"  {mod:<18} {count:>10,} {count/total*100:>7.1f}%")
print("-" * 44)
print(f"  {'总计':<18} {total:>10,} {'100.0%':>8}")
print(f"\n  总参数量: {total/1e6:.3f}M")

# 更细粒度分析
print("\n" + "=" * 70)
print("子模块参数量")
print("=" * 70)
for top_name in ['encoder', 'tfsi', 'sace', 'ifpn', 'ndpn', 'mrpn', 'igrf']:
    top_mod = getattr(model, top_name)
    mod_total = sum(p.numel() for p in top_mod.parameters())
    print(f"\n  [{top_name}] total = {mod_total:,}")
    sub_dict = {}
    for n, p in top_mod.named_parameters():
        sub = n.split('.')[0]
        if sub not in sub_dict:
            sub_dict[sub] = 0
        sub_dict[sub] += p.numel()
    for sub, cnt in sorted(sub_dict.items(), key=lambda x: -x[1]):
        print(f"    {sub:<25} {cnt:>8,}")

# 梯度流分析：检测初始输出
print("\n" + "=" * 70)
print("初始前向传播分析")
print("=" * 70)
model.eval()
x = torch.randn(1, 5, 3, 64, 64)
with torch.no_grad():
    out = model(x)

for k, v in out.items():
    if isinstance(v, torch.Tensor) and v.ndim >= 2:
        print(f"  {k:<18} shape={str(list(v.shape)):<20} mean={v.mean():.4f}  std={v.std():.4f}  range=[{v.min():.3f}, {v.max():.3f}]")

# 检查s_*的初始分布
print(f"\n  s_illum 初始均值: {out['s_illum'].mean():.4f} (理想≈0.5)")
print(f"  s_noise 初始均值: {out['s_noise'].mean():.4f} (理想≈0.5)")

# 梯度流测试 - 使用 v3.2 完整 loss
print("\n" + "=" * 70)
print("梯度流测试 (v3.2 TFSNetLoss with aux + Charbonnier + SSIM)")
print("=" * 70)
model.train()
x = torch.randn(1, 5, 3, 64, 64)
out = model(x)
target = torch.randn(1, 3, 64, 64).clamp(0, 1)

criterion = TFSNetLoss(
    use_freq_loss=True,
    perceptual_pretrained=False,
    lambda_perc=0.1,
    lambda_freq=0.1,
    lambda_illum=0.05,
    lambda_ssim=0.2,
    lambda_aux=0.2,
    fused_channels=64,
)
loss, loss_dict = criterion(out, target)
loss.backward()

print(f"\n  Loss breakdown:")
for k, v in loss_dict.items():
    print(f"    {k:<15}: {v.item():.4f}")

grad_norms = {}
for name, p in model.named_parameters():
    if p.grad is not None:
        parts = name.split('.')
        top = parts[0]
        if top not in grad_norms:
            grad_norms[top] = []
        grad_norms[top].append(p.grad.norm().item())

print(f"\n  {'模块':<15} {'平均梯度范数':>15} {'最大梯度范数':>15}")
print("  " + "-" * 48)
for mod in ['encoder', 'tfsi', 'sace', 'ifpn', 'ndpn', 'mrpn', 'igrf']:
    if mod in grad_norms:
        norms = grad_norms[mod]
        avg_norm = sum(norms) / len(norms)
        max_norm = max(norms)
        print(f"  {mod:<15} {avg_norm:>15.6f} {max_norm:>15.6f}")
