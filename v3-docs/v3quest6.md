# TFS-Net v3 — B.7 修正 + B.8/B.9 实现指令

> **日期**：2026-06-12
> **前置**：v3answer6.md（你的 B.3-B.7 实现）

---

## 一、B.4/B.5/B.6 审查结果

| 模块 | 状态 | 说明 |
|------|------|------|
| B.3 offset reshape 修正 | ✅ 正确 | `view(B,G,2,K,H,W).permute(0,1,3,2,4,5)` |
| B.4 IFPN IllumExtract | ✅ 正确 | groups=4, n_fea_middle=32 合理 |
| B.4 IFPN 主类 | ✅ 正确 | imgs_down 可选参数设计合理 |
| B.5 NDPN | ✅ 正确 | SNR 估计 + 双因素权重 + 中心帧 s_snr 互补设计合理 |
| B.6 MRPN | ✅ 正确 | 残差降权 sigmoid(-logits) + 中心帧固定权重 1.0 合理 |

**B.7 TFSNet 整合 — 发现 3 个 P0 级接口错误**（见下方 §二）。

---

## 二、B.7 的 3 个 P0 错误（必须修正）

### P0-1: PyramidEncoder 构造函数参数错误

```python
# 你的代码（错误）
self.encoder = PyramidEncoder(
    in_channels=in_channels,
    base_channels=base_channels,  # ← 不存在此参数
    c1=c1, c2=c2, c3=c3,          # ← 不存在这些参数
)

# 实际 PyramidEncoder 签名
class PyramidEncoder(nn.Module):
    def __init__(self, in_channels=3, level_channels=(32, 64, 96), fused_channels=48):
```

**修正**：
```python
self.encoder = PyramidEncoder(
    in_channels=in_channels,
    level_channels=level_channels,
    fused_channels=fused_channels,
)
```

### P0-2: PyramidEncoder.forward() 返回值解析错误

```python
# 你的代码（错误）— 把返回值当 dict 处理
enc_out = self.encoder(x)
if isinstance(enc_out, dict):
    feat_full = enc_out.get("feat2", ...)   # ← 完全错误
    coarse_feats = enc_out.get("feat3", ...)
```

**实际 forward() 签名**：
```python
def forward(self, x, return_coarse=False):
    """
    Args:
        x             : (B, T, C, H, W) 多帧序列
        return_coarse : 是否同时返回最粗尺度特征

    Returns:
        return_coarse=False: feats (B, T, C_f=48, H, W)  ← 全分辨率融合特征！
        return_coarse=True : (feats, coarse_feats)
                             feats        : (B, T, 48, H, W)
                             coarse_feats : (B, T, 96, H/4, W/4)
    """
```

**修正**：
```python
feats, coarse_feats = self.encoder(x, return_coarse=True)
# feats:        (B, T, 48, H, W)  ← 已经是全分辨率，无需上采样！
# coarse_feats: (B, T, 96, H/4, W/4)
```

**关键**：encoder 的 fused 输出已经是 **(B, T, 48, H, W) 全分辨率**，不需要任何上采样。

### P0-3: IGRF 构造函数参数错误

```python
# 你的代码（错误）
self.igrf = IGRF(channels=fused_channels, img_channels=in_channels)  # ← img_channels 不存在

# 实际 IGRF 签名
class IGRF(nn.Module):
    def __init__(self, channels: int = 48, out_channels: int = 3):
```

**修正**：
```python
self.igrf = IGRF(channels=fused_channels, out_channels=in_channels)
```

---

## 三、B.7 修正后的 TFSNet.__init__ 和 forward（仅展示修正点）

### __init__ 修正

```python
class TFSNet(nn.Module):
    def __init__(
        self,
        in_channels: int = 3,
        level_channels: tuple = (32, 64, 96),
        fused_channels: int = 48,
        eps: float = 1e-6,
        n_groups: int = 4,
        kernel_size: int = 3,
        share_lff: bool = True,
    ):
        super().__init__()
        c1, c2, c3 = level_channels

        # Stage 0: 编码器（使用正确参数名）
        self.encoder = PyramidEncoder(
            in_channels=in_channels,
            level_channels=level_channels,
            fused_channels=fused_channels,
        )

        # Stage 1: TFSI（channels=fused_channels=48）
        self.tfsi = TFSI(channels=fused_channels, fused_channels=fused_channels, eps=eps)

        # Stage 2: SACE（与 TFSI 共享 LFF）
        shared_lff = self.tfsi.freq_branch.lff if share_lff else None
        self.sace = SACE(
            channels=fused_channels, n_groups=n_groups, kernel_size=kernel_size,
            use_optimized=True, lff_module=shared_lff,
        )

        # Stage 3: 三源恢复分支
        self.ifpn = IFPN(fused_channels=fused_channels, coarse_channels=c3, img_channels=in_channels)
        self.ndpn = NDPN(channels=fused_channels)
        self.mrpn = MRPN(channels=fused_channels)

        # Stage 4: IGRF（使用正确参数名）
        self.igrf = IGRF(channels=fused_channels, out_channels=in_channels)
```

### forward 修正

```python
def forward(self, x: torch.Tensor) -> Dict[str, torch.Tensor]:
    B, T, C_in, H, W = x.shape
    center_idx = T // 2

    # ── Stage 0: 编码（正确的 API 调用）──
    feats, coarse_feats = self.encoder(x, return_coarse=True)
    # feats:        (B, T, 48, H, W)  全分辨率融合特征
    # coarse_feats: (B, T, 96, H/4, W/4)

    # ── Stage 1: TFSI ──
    tfsi_out = self.tfsi(feats)
    F_fused = tfsi_out["F_fused"]
    s_illum = tfsi_out["s_illum"]
    s_noise = tfsi_out["s_noise"]
    s_motion = tfsi_out["s_motion"]
    sigma_t = tfsi_out["sigma_t"]

    # ── Stage 2: SACE ──
    sace_out = self.sace(feats, tfsi_out)
    mu_t_clean = sace_out["mu_t_clean"]
    F_aligned_list = sace_out["F_aligned_list"]

    # ── Stage 3: 三源恢复 ──
    image_center = x[:, center_idx]  # (B, 3, H, W)

    # 下采样图像到 coarse 分辨率
    _, _, _, hc, wc = coarse_feats.shape
    image_down = F.interpolate(image_center, size=(hc, wc), mode='bicubic', align_corners=False)
    imgs_down = F.interpolate(
        x.view(B * T, C_in, H, W), size=(hc, wc), mode='bicubic', align_corners=False
    ).view(B, T, C_in, hc, wc)

    F_t_L = coarse_feats[:, center_idx]  # (B, 96, hc, wc)

    ifpn_out = self.ifpn(image_down, F_t_L, s_illum, feats, coarse_feats, center_idx, imgs_down)
    ndpn_out = self.ndpn(feats, F_aligned_list, mu_t_clean, sigma_t, s_noise, center_idx)
    mrpn_out = self.mrpn(feats, F_aligned_list, s_motion, center_idx)

    # ── Stage 4: IGRF ──
    igrf_out = self.igrf(
        f_t_base=F_fused,
        f_illum_out=ifpn_out["f_illum_out"],
        f_noise_out=ndpn_out["f_noise_out"],
        f_motion_out=mrpn_out["f_motion_out"],
        s_illum=s_illum, s_noise=s_noise, s_motion=s_motion,
        image_center=image_center,
    )

    return {
        "res_t": igrf_out["res_t"],
        "delta": igrf_out["delta"],
        "f_fused_igrf": igrf_out["f_fused"],
        "s_illum": s_illum, "s_noise": s_noise, "s_motion": s_motion,
        "L_t": ifpn_out["L_t"], "L_ref": ifpn_out["L_ref"],
        "attn_maps": sace_out["attn_maps"],
        "mu_t_clean": mu_t_clean,
        "s_snr": ndpn_out["s_snr"],
        "motion_weights": mrpn_out["motion_weights"],
        "tfsi_out": tfsi_out,
    }
```

---

## 四、B.8: `losses/losses.py` 新增 TFSNetLoss

在现有 losses.py 末尾追加（不修改 MINSLoss 和现有代码）：

### 设计要求（来自 TFSv3-result.md §5）

```python
class TFSNetLoss(nn.Module):
    """
    TFS-Net v3 损失函数（首版 use_temporal=False）

    包含:
        L_recon    = L1(pred, target) + λ_freq * L_freq(pred, target)
        L_perc     = PerceptualLoss(pred, target)  [复用现有 PerceptualLoss]
        L_illum    = 光照强度空间平滑正则

    权重:
        λ_perc = 0.1, λ_freq = 0.1, λ_illum = 0.01
    """

    def __init__(self, use_temporal=False, use_freq_loss=True, perceptual_pretrained=False):
        # 复用现有 PerceptualLoss
        self.perceptual = PerceptualLoss(pretrained=perceptual_pretrained)
        self.use_temporal = use_temporal
        self.use_freq_loss = use_freq_loss
        self.lambda_perc = 0.1
        self.lambda_freq = 0.1
        self.lambda_illum = 0.01

    def forward(self, outputs, target):
        """
        Args:
            outputs: dict from TFSNet.forward()
                - res_t     : (B, 3, H, W) 增强结果
                - s_illum   : (B, 1, H, W) 光照强度
            target:  (B, 3, H, W) GT

        Returns:
            (loss_total, loss_dict)
        """
        pred = outputs["res_t"]
        s_illum = outputs["s_illum"]

        # (1) 空间重建
        L_pix = F.l1_loss(pred, target)

        # (2) 频域重建（可选）
        L_freq = pred.new_tensor(0.0)
        if self.use_freq_loss:
            fft_pred = torch.fft.rfft2(pred, norm='ortho')
            fft_gt = torch.fft.rfft2(target, norm='ortho')
            L_freq = F.l1_loss(fft_pred.abs(), fft_gt.abs())

        L_recon = L_pix + self.lambda_freq * L_freq

        # (3) 感知损失
        L_perc = self.perceptual(pred, target)

        # (4) 光照平滑正则（仅对 s_illum）
        grad_s_x = (s_illum[:, :, :, 1:] - s_illum[:, :, :, :-1]).abs()
        grad_s_y = (s_illum[:, :, 1:, :] - s_illum[:, :, :-1, :]).abs()
        grad_i_x = (target[:, :, :, 1:] - target[:, :, :, :-1]).abs().mean(dim=1, keepdim=True)
        grad_i_y = (target[:, :, 1:, :] - target[:, :, :-1, :]).abs().mean(dim=1, keepdim=True)
        L_illum = (grad_s_x * torch.exp(-grad_i_x)).mean() + \
                  (grad_s_y * torch.exp(-grad_i_y)).mean()

        L_total = L_recon + self.lambda_perc * L_perc + self.lambda_illum * L_illum

        return L_total, {
            "loss_total": L_total.detach(),
            "loss_pix": L_pix.detach(),
            "loss_freq": L_freq.detach(),
            "loss_perc": L_perc.detach(),
            "loss_illum": L_illum.detach(),
        }
```

---

## 五、B.9: 配置与脚本适配

### 5.1 新版 `configs/sdsd_stage1.yaml`

```yaml
seed: 42
output_dir: outputs/sdsd_stage1
dataset:
  train_input_root: F:/DatasetDL/SDSD/indoor/input
  train_target_root: F:/DatasetDL/SDSD/indoor/GT
  val_input_root: F:/DatasetDL/SDSD/test/low-light
  val_target_root: F:/DatasetDL/SDSD/test/GT
  window_size: 5
  crop_size: 256
  num_workers: 4
model:
  type: TFSNet                     # 新增: 模型类型标识
  in_channels: 3
  level_channels: [32, 64, 96]
  fused_channels: 48
  # 移除: mins_window_size (v1 专用)
train:
  batch_size: 2
  epochs: 200
  lr: 0.0002
  weight_decay: 0.0001
  amp: false
  log_interval: 50
  val_interval: 5
loss:
  type: TFSNetLoss                 # 新增: 损失类型标识
  use_freq_loss: true
  perceptual_pretrained: false
eval:
  tile_size: 256
  tile_overlap: 32
  amp: false
```

### 5.2 `train.py` 修改清单

**修改 1** — import 区（约 L27-29）：
```python
# 旧
from models import MINSNet
from losses import MINSLoss

# 改为
from models import TFSNet
from losses.losses import TFSNetLoss
```

**修改 2** — `build_model()` 函数（L87-94）：
```python
def build_model(cfg, device):
    model_cfg = cfg["model"]
    model = TFSNet(
        in_channels=model_cfg["in_channels"],
        level_channels=tuple(model_cfg["level_channels"]),
        fused_channels=model_cfg["fused_channels"],
    )
    return model.to(device)
```

**修改 3** — `build_loss()` 函数（L97-105）：
```python
def build_loss(cfg, device):
    loss_cfg = cfg["loss"]
    criterion = TFSNetLoss(
        use_freq_loss=loss_cfg.get("use_freq_loss", True),
        perceptual_pretrained=loss_cfg.get("perceptual_pretrained", False),
    )
    return criterion.to(device)
```

**修改 4** — `train_one_epoch()` 中的 loss_dict 键名（L135-139）：
```python
# 旧键名: loss_pix, loss_ssim, loss_perc, loss_prior
# 新键名: loss_pix, loss_freq, loss_perc, loss_illum
# 需要对应更新 AverageMeter 和日志
```

### 5.3 `infer.py` 修改清单

**修改 1** — import 区（L26）：
```python
# 旧
from models import MINSNet

# 改为
from models import TFSNet
```

**修改 2** — 模型构造（L73-78）：
```python
model = TFSNet(
    in_channels=cfg["model"]["in_channels"],
    level_channels=tuple(cfg["model"]["level_channels"]),
    fused_channels=cfg["model"]["fused_channels"],
).to(device)
```

### 5.4 `models/modules/__init__.py` 更新

```python
from .encoder import PyramidEncoder
from .mins import MINSBlock
from .ispn import ISPN
from .mspn import MSPN
from .reconstruction import FinalReconstruction
# v3 新增模块
from .tfsi import TFSI
from .igrf import IGRF
from .lff import RadialBasisFilter, LFFFeatureAdapter
from .sace import SACE, DeformableCrossAttention
from .ifpn import IFPN, IllumExtract
from .ndpn import NDPN
from .mrpn import MRPN
```

### 5.5 `losses/__init__.py` 更新

```python
from .losses import MINSLoss, TFSNetLoss, PerceptualLoss, ssim_map
```

---

## 六、验证清单

请实现 B.8/B.9 后，提供以下验证：

1. **端到端 shape 测试**（修正后的 B.7 + B.8 损失）：
```python
from models import TFSNet
from losses.losses import TFSNetLoss

net = TFSNet()
x = torch.randn(2, 5, 3, 256, 256)
out = net(x)
assert out["res_t"].shape == (2, 3, 256, 256)

criterion = TFSNetLoss()
target = torch.randn(2, 3, 256, 256)
loss, loss_dict = criterion(out, target)
loss.backward()
```

2. **参数量统计**（目标 < 2M）

3. **完整的 train.py / infer.py 修改后代码**（不只是 diff，要完整可运行版本）
