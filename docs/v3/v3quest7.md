# TFS-Net v3 — 第四轮审查反馈与补全指令

---

## 一、审查结论

| 模块 | 结论 | 备注 |
|------|------|------|
| B.7 TFSNet 完整版 | ✅ 通过 | PyramidEncoder / IGRF / TFSI API 全部正确使用 |
| B.8 TFSNetLoss | ✅ 通过 | 损失结构、频域 FFT、边缘感知平滑均正确 |
| B.9 config YAML | ✅ 通过 | 与 v3quest6 设计一致 |
| B.9 `__init__.py` (modules/models/losses) | ✅ 通过 | 导入注册正确 |
| B.9 `train.py` | ❌ P0 + 不完整 | import 路径错误 + 只提供到 import 区就截断了 |
| B.9 `infer.py` | ❌ 完全缺失 | v3answer7 中未包含 infer.py 的任何内容 |

### B.7 详细审查（通过）

- `PyramidEncoder(in_channels, level_channels, fused_channels)` ✅
- `encoder(x, return_coarse=True)` → `(feats, coarse_feats)` tuple ✅
- `feats` 已是全分辨率 `(B, T, 48, H, W)`，无上采样 ✅
- `IGRF(channels=fused_channels, out_channels=in_channels)` ✅
- IFPN/NDPN/MRPN 调用参数与 v3answer6 实现一致 ✅
- 返回 dict 键名完整 ✅

**一个依赖提醒**（非 bug）：
```python
shared_lff = self.tfsi.freq_branch.lff if share_lff else None
```
此行要求 FrequencyBranch 内部已创建 `self.lff` 属性。当前实际 `tfsi.py` 中 FrequencyBranch 只有 `self._placeholder = nn.Identity()`。当 B.1 LFF 代码写入 `tfsi.py` 时，需确保 FrequencyBranch 将 LFF 实例保存为 `self.lff`。

### B.8 详细审查（通过）

- `torch.fft.rfft2(pred, norm='ortho')` 正确 ✅
- `_edge_aware_smooth` 梯度形状匹配 ✅
- `PerceptualLoss` 复用现有类 ✅
- loss_dict 键名与 train.py 对应 ✅

---

## 二、P0 修正：train.py import 路径错误

你写的：
```python
from datasets.sdsd import SDSDDataset    # ❌ 错误路径
```

实际项目结构：
- 文件是 `datasets/sdsd_dataset.py`（不是 `datasets/sdsd.py`）
- `datasets/__init__.py` 内容为 `from .sdsd_dataset import SDSDDataset`

正确写法：
```python
from datasets import SDSDDataset         # ✅
```

---

## 三、B.9 补全任务：train.py 完整版

你的 v3answer7 中 train.py 只写到了 import 区（约第 33 行）就截断了。请提供**完整可运行**的 train.py。

### 实际 train.py 当前结构（264 行）

以下是要点摘要，帮助你理解需要改什么：

```python
# === 现有 import 区（约 L1-33）===
from datasets import SDSDDataset           # 注意：不是 datasets.sdsd
from losses import MINSLoss
from models import MINSNet
from utils.io import save_checkpoint
from utils.inference import tiled_forward
from utils.metrics import tensor_psnr, tensor_ssim
from utils.misc import AverageMeter, create_logger, seed_everything

# === 需要修改的函数 ===

def build_model(cfg, device):              # L87-94
    # 当前: MINSNet(..., window_size=cfg["model"]["mins_window_size"])
    # 改为: TFSNet(in_channels, level_channels, fused_channels)
    # TFSNet 没有 window_size 参数

def build_loss(cfg, device):               # L97-105
    # 当前: MINSLoss(lambda_pix, lambda_ssim, lambda_perc, lambda_tv, ...)
    # 改为: TFSNetLoss(use_freq_loss, perceptual_pretrained, lambda_perc, lambda_freq, lambda_illum)

def train_one_epoch(...):                  # L108-160
    # 当前 meter: loss_total, loss_pix, loss_ssim, loss_perc, loss_prior
    # 新 meter:  loss_total, loss_pix, loss_freq, loss_perc, loss_illum
    # 需要对应更新 meter 名称和日志输出

# === 以下函数无需修改 ===
# parse_args(), load_config(), build_dataloaders(), validate(), main()
# 这些函数的逻辑不依赖模型/损失类型，保持原样即可
```

### 要求

1. 保持 `build_dataloaders()`、`validate()`、`main()` 完全不变
2. 修改 `build_model()` 使用 TFSNet（注意 TFSNet 无 `window_size` 参数）
3. 修改 `build_loss()` 使用 TFSNetLoss
4. 修改 `train_one_epoch()` 中的 AverageMeter 键名，匹配 TFSNetLoss 的 loss_dict：
   - `loss_total`, `loss_pix`, `loss_freq`, `loss_perc`, `loss_illum`
5. 更新日志格式字符串
6. import 区使用 `from datasets import SDSDDataset`（**不是** `from datasets.sdsd`）

---

## 四、B.9 补全任务：infer.py 完整版

你的 v3answer7 完全没包含 infer.py。请提供完整修改后的 infer.py。

### 实际 infer.py 当前结构（109 行）

```python
# === 需要修改的部分 ===

# L26 import:
from models import MINSNet    # 改为 TFSNet

# L73-78 模型构造:
model = MINSNet(
    in_channels=cfg["model"]["in_channels"],
    level_channels=tuple(cfg["model"]["level_channels"]),
    fused_channels=cfg["model"]["fused_channels"],
    window_size=cfg["model"]["mins_window_size"],   # TFSNet 无此参数
).to(device)

# 改为:
model = TFSNet(
    in_channels=cfg["model"]["in_channels"],
    level_channels=tuple(cfg["model"]["level_channels"]),
    fused_channels=cfg["model"]["fused_channels"],
).to(device)

# === 以下部分无需修改 ===
# parse_args(), load_config(), read_image(), list_sequences(), gather_clip()
# main() 中的推理循环（tiled_forward 已兼容 dict 返回，自动取 ["res_t"]）
```

### 兼容性说明

`utils/inference.py` 中的 `tiled_forward()` 已有：
```python
return model(clip)["res_t"]    # L30
tile_pred = model(tile)["res_t"]    # L47
```
TFSNet 返回 dict 含 `"res_t"` 键，完全兼容，**无需修改 inference.py**。

---

## 五、最终输出要求

请在下一轮答复中提供以下三个**完整可运行**文件（不是 diff，是完整代码）：

1. **`train.py`** — 基于实际 264 行结构修改
2. **`infer.py`** — 基于实际 109 行结构修改
3. **`configs/sdsd_stage1.yaml`** — 确认最终版（你 v3answer7 中的版本已正确，再输出一遍即可）

### 约束

- 保留 v1 (MINSNet) 的向后兼容性不是必须的，可以直接替换为 TFSNet
- 保持现有代码风格（try/except tqdm fallback、`tqdm` 兼容层等）
- 不要改动 `build_dataloaders()`、`validate()` 等无关函数
- `from datasets import SDSDDataset`（再次强调路径）

---

## 六、后续 Phase C 预告

B.9 补全后，Phase B 编码阶段即全部完成。下一步 Phase C 将是：

1. **C.1 模块文件落盘**：将 Claude 各轮输出的 `.py` 文件（lff.py, sace.py, ifpn.py, ndpn.py, mrpn.py）写入实际项目目录
2. **C.2 tfsi.py 更新**：将 LFF 集成到 FrequencyBranch（替换零张量占位），确保 `self.lff` 属性存在
3. **C.3 端到端 smoke test**：`python train.py --config configs/sdsd_stage1.yaml --smoke`
4. **C.4 参数量/FLOPs 统计**
