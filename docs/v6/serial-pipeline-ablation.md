# 串行管线三退化分支消融实验（FastDVDNet → CDVD-TSP → StableLLVE）

> 对应用户方案：以 TSD-Net 的三源退化建模为指导，用三个**单任务预训练模型**串行处理三种退化：
> - **D** = FastDVDNet（CVPR 2020 / TIP，视频去噪，对应 noise 分支）
> - **DB** = CDVD-TSP（CVPR 2020，视频去模糊，对应 motion 分支）
> - **B** = StableLLVE（CVPR 2021，低光视频增强，对应 illumination 分支）
>
> 本文记录 **2³ 消融矩阵 + 顺序消融 + σ 敏感性 + 负贡献诊断** 的完整结果。
>
> 文档状态：实验已跑完，结果已归档。
> 实验日期：2026-09-11 · 数据：SDSD test/pair19 · 服务器环境：conda `CMKDK`（PyTorch 1.10.0 + CUDA 11.8 + CuPy 12.3）

---

## 一、结论摘要（TL;DR）

| # | 结论 | 证据 |
|:--|:--|:--|
| 1 | **光照分支 B（StableLLVE）是唯一的主贡献者** | B-only：LPIPS 0.170 / PSNR 16.27 / SSIM 0.748；去掉 B 后 LPIPS 退化到 0.50（D+DB） |
| 2 | **去噪分支 D（FastDVDNet）在本实验域上为负贡献** | 完整管线 0.3029 vs 去掉 D 的 DB+B 0.1584（LPIPS，↓48%）；D 在暗光帧上产生**棋盘伪影（checkerboard）**，被后续增亮放大 |
| 3 | **去模糊分支 DB（CDVD-TSP）贡献微弱但一致为正** | 在 B 前加入 DB：LPIPS 0.170→0.158、SSIM 0.748→0.765、NIQE 6.44→6.24（均为最优） |
| 4 | **最优组合是 DB→B，而非完整 D→DB→B** | 9 组配置中 DB+B 在 LPIPS / SSIM / NIQE 三项均为最优 |
| 5 | 顺序变体 DB→D→B 与原顺序 D→DB→B 相当（LPIPS 0.3040 vs 0.3029），但两者都远差于 DB→B | 说明问题不在顺序，而在 D 分支本身 |
| 6 | σ 敏感性：假定的噪声越大结果越差，最优趋近"不显式去噪" | σ=5 → LPIPS 0.1946；σ=50 → 0.3773（D+B 配置，单调递增） |

**一句话总结**：在 SDSD 暗光域上，off-the-shelf 的串行级联存在**域不匹配 + 误差放大**；主管线顺序（D→DB→B）虽然符合 TSD-Net 的 S1→S2→S3 物理顺序，但完整三支路反而不如"去模糊 + 增亮"两支路，这为 TSD-Net 强调的**联合建模/耦合优化**提供了反向证据。

---

## 二、实验设置

### 2.1 数据

| 项 | 值 |
|:--|:--|
| 数据源 | `/home/a1005/yzy/dataset/SDSD/test/low-light/pair19`（输入）；`/home/a1005/yzy/dataset/SDSD/test/GT/pair19`（参考） |
| 帧数 | 前 11 帧（`0156.png`–`0166.png`） |
| 分辨率 | 原始 1920×1080 → 下采样至 **960×540**（`--scale 0.5`，裁剪到 4 的倍数，满足 FastDVDNet / CDVD-TSP 要求） |
| 帧对齐 | SDSD 的双相机录制使 LL 与 GT 的文件名起始编号不同（LL 从 0156 起、GT 从 0161 起），**按时间索引对齐**（第 i 帧 ↔ 第 i 帧，两序列长度相同） |

### 2.2 模型与权重

| 分支 | 模型 | 权重文件 | 论文 | 任务 |
|:--|:--|:--|:--|:--|
| D | FastDVDNet（U-Net 时空，5 帧滑窗，无光流） | `reference_repos/fastdvdnet/model.pth` | CVPR 2020 Oral | 视频去噪（AWGN σ∈[5,55]） |
| DB | CDVD-TSP（PWC-Net 光流 + 时序清晰度先验 + 级联重建） | `reference_repos/CDVD-TSP/pretrain_models/CDVD_TSP_DVD_Convergent.pt` | CVPR 2020 | 视频去模糊（DVD 数据集） |
| B | StableLLVE（U-Net，单帧推理） | `reference_repos/StableLLVE/checkpoint.pth` | CVPR 2021 | 低光增强（合成干净数据训练） |

> 权重来源：CDVD-TSP 权重来自官方 Google Drive（`pretrain_models-20260716T054351Z-1-001.zip`，已解压到 `reference_repos/CDVD-TSP/pretrain_models/`）；FastDVDNet / StableLLVE 权重随仓库自带。

### 2.3 指标

| 指标 | 实现 | 方向 |
|:--|:--|:--|
| LPIPS | `lpips` 包，AlexNet backbone（v0.1） | ↓ 越低越好 |
| PSNR | `skimage.metrics.peak_signal_noise_ratio`，data_range=255 | ↑ |
| SSIM | `skimage.metrics.structural_similarity`，channel_axis=2 | ↑ |
| NIQE | BasicSR/HVI-CIDNet 实现（`reference_repos/HVI-CIDNet/loss/niqe_utils.py`）+ 官方 pristine 参数（`niqe_pris_params.npz`），BGR→Y 通道，block 96×96 | ↓ |

**参考值**：GT 自身 NIQE = **5.371 ± 0.122**（同一 11 帧，960×540）；输入暗光帧 NIQE = 7.433 ± 0.152。

### 2.4 为保证复现所做的兼容性修复（记录在案）

| 问题 | 修复位置 | 说明 |
|:--|:--|:--|
| FastDVDNet `utils.py` 导入 `skimage.measure.simple_metrics` 失败 | `reference_repos/fastdvdnet/utils.py` | fallback 到 `skimage.metrics.peak_signal_noise_ratio`；`tensorboardX` 改为可选 |
| CDVD-TSP `cupy.util.memoize`（CuPy <10 API） | `reference_repos/CDVD-TSP/code/model/correlation.py` | 改为 `cupy._util.memoize` |
| CuPy 12 下 `Stream` 类 | 同上 | 改为惰性获取 `torch.cuda.current_stream().cuda_stream` |
| `_FunctionCorrelation` 对非连续张量 assert 失败（新 PyTorch） | 同上 | 将两个输入改为 `.contiguous()`（替换两行 assert） |
| CDVD-TSP `utils` 包与项目 `utils` 冲突 | `reference_repos/CDVD-TSP/code/utils/__init__.py` | 新增空 `__init__.py` 使目录成为包 |
| StableLLVE `model.py` 与 CDVD-TSP `model/` 包名冲突 | `pipeline_full.py` | 用 `importlib` 按文件路径加载 UNet |

---

## 三、消融矩阵定义

保留因果顺序（D 只能出现在 DB、B 之前；DB 只能出现在 B 之前），消融项为 2³ 全部子集，外加一个顺序变体：

| 配置 | 流程 | 说明 |
|:--|:--|:--|
| `Input` | 无处理 | 基线（暗光输入） |
| `D` | FastDVDNet | 仅去噪 |
| `DB` | CDVD-TSP | 仅去模糊 |
| `B` | StableLLVE | 仅增亮 |
| `D+DB` | FastDVDNet → CDVD-TSP | 去噪 + 去模糊 |
| `D+B` | FastDVDNet → StableLLVE | 去噪 + 增亮 |
| `DB+B` | CDVD-TSP → StableLLVE | 去模糊 + 增亮 |
| `D+DB+B` | FastDVDNet → CDVD-TSP → StableLLVE | **完整主管线**（对应 TSD-Net S1→S2→S3） |
| `DB+D+B` | CDVD-TSP → FastDVDNet → StableLLVE | 顺序变体（先去模糊再去噪） |

---

## 四、主结果

**数据**：SDSD pair19 前 11 帧，960×540，FastDVDNet σ=25（0–255）；GT 按索引对齐。
**格式**：均值 ± 标准差（跨 11 帧）。

| 配置 | LPIPS ↓ | PSNR ↑ | SSIM ↑ | NIQE ↓ |
|:--|--:|--:|--:|--:|
| Input（无处理） | 0.3938 ± 0.001 | 6.56 ± 0.03 | 0.2248 ± 0.002 | 7.433 ± 0.152 |
| D only | 0.4966 ± 0.003 | 6.41 ± 0.03 | 0.1708 ± 0.001 | 10.617 ± 0.343 |
| DB only | 0.3999 ± 0.002 | 6.53 ± 0.03 | 0.2168 ± 0.001 | 6.695 ± 0.180 |
| B only | 0.1703 ± 0.004 | 16.27 ± 0.03 | 0.7476 ± 0.002 | 6.443 ± 0.147 |
| D+DB | 0.4984 ± 0.003 | 6.39 ± 0.03 | 0.1795 ± 0.001 | 10.117 ± 0.270 |
| D+B | 0.3102 ± 0.004 | 15.70 ± 0.06 | 0.7167 ± 0.005 | 8.420 ± 0.209 |
| **DB+B** | **0.1584 ± 0.002** | **16.31 ± 0.04** | **0.7645 ± 0.002** | **6.237 ± 0.157** |
| D+DB+B（完整） | 0.3029 ± 0.004 | 15.79 ± 0.06 | 0.7436 ± 0.006 | 8.454 ± 0.125 |
| DB+D+B（变体） | 0.3040 ± 0.004 | 15.62 ± 0.06 | 0.7191 ± 0.006 | 7.835 ± 0.376 |
| *GT（参考）* | — | — | — | *5.371 ± 0.122* |

> 最优值以 **加粗** 标出；`D+DB+B` 与 `DB+D+B` 的差异在误差范围内，说明"先 D 后 DB"与"先 DB 后 D"在本实验中没有本质区别。

**逐帧明细**：`outputs/pipeline_ablation/metrics_per_frame.csv`
**汇总数据**：`outputs/pipeline_ablation/metrics_summary.{csv,json}`

![ablation bars](../../outputs/pipeline_ablation/ablation_bars.png)

---

## 五、分析

### 5.1 单分支贡献（相对 Input）

| 分支 | ΔLPIPS | ΔPSNR | ΔSSIM | ΔNIQE |
|:--|--:|--:|--:|--:|
| D only | **+0.1028（变差）** | −0.15 | −0.054 | **+3.184（变差）** |
| DB only | +0.0061（略差） | −0.03 | −0.008 | **−0.738（改善）** |
| B only | **−0.2236（大幅改善）** | **+9.71** | **+0.523** | **−0.990（改善）** |

- **B 分支单独就完成几乎全部可见质量提升**（亮度域对齐 GT 是 PSNR/SSIM/LPIPS 大幅提升的主因）。
- **D 分支单独在四项指标上全部变差**，其中 NIQE 从 7.43 恶化到 10.62，说明 FastDVDNet 在极暗输入上引入了强烈的非自然结构（见 §5.4）。
- DB 单独在参考指标上几乎不变（暗光域下 CDVD-TSP 的去模糊收益被亮度差异掩盖），但 NIQE 明显改善（6.70）。

### 5.2 留一分析（leave-one-out，从完整管线中移除一个分支）

| 移除的分支 | LPIPS 变化 | PSNR 变化 | SSIM 变化 | NIQE 变化 | 贡献判定 |
|:--|:--|:--|:--|:--|:--|
| 移除 D（→DB+B） | 0.3029 → **0.1584（−0.1445）** | +0.52 | +0.021 | −2.22 | **负贡献** |
| 移除 DB（→D+B） | 0.3029 → 0.3102（+0.0073） | −0.09 | −0.027 | −0.03 | 正贡献（微弱） |
| 移除 B（→D+DB） | 0.3029 → 0.4984（+0.1955） | −9.40 | −0.564 | +1.66 | **主正贡献** |

**结论**：主管线中，**去掉 D 反而全面提升**；DB 有微弱正贡献；B 是决定性分支。

### 5.3 顺序消融

`D+DB+B`（0.3029 / 15.79 / 0.7436 / 8.454） vs `DB+D+B`（0.3040 / 15.62 / 0.7191 / 7.835）：

- 参考指标（LPIPS/PSNR/SSIM）两者相差在 ±0.01/±0.17/±0.025 以内；
- 无参考指标 NIQE 上"先 DB 后 D"略好（7.835 vs 8.454）；
- 两种顺序**都远差于不含 D 的 DB+B**（0.1584 / 6.237）。

**结论**：D 分支的负贡献与放置顺序无关，源于该模型在暗光域的域不匹配。

### 5.4 去噪分支负贡献诊断：棋盘伪影 + 域不匹配

**(a) 现象**。对 `D(dark)` 做 gamma 提亮（γ=0.45）后可见明显**棋盘格伪影**，残差图（×20 增强）呈现规则的高频网格；同样模型作用在正常曝光 GT 帧上时该伪影弱得多。

| 输入域 | σ=5 归一化 Nyquist 峰值能量 | σ=25 归一化 Nyquist 峰值能量 |
|:--|--:|--:|
| 暗光帧 | 179.3 | 208.5 |
| 正常光 GT 帧 | 154.4 | 158.5 |

（峰值能量 = 以 (Nyquist, Nyquist) 为中心的 FFT 幅值均值 / 高频带均值的比值，仅作相对比较。）

**(b) 原因候选**：
1. **领域不匹配**：FastDVDNet 在 DAVIS 正常曝光视频 + AWGN 上训练；SDSD 暗光帧信号均值仅 0.08（[0,1] 区间），噪声-信号比极高，Poisson-Gaussian 噪声与 AWGN 也不匹配。
2. **架构性棋盘伪影**：FastDVDNet 的 `UpBlock` 使用 `PixelShuffle(2)`，在低信噪比区域会产生周期性栅格；增亮阶段（B）将其放大。
3. **过平滑**：暗光帧经 σ=25 去噪后均值从 0.0793 降到 0.0715（−10%），细节被当作噪声移除，B 无法恢复。

![checker diagnosis](../../outputs/pipeline_ablation/checker_diagnosis.png)

**(c) σ 敏感性扫描**（固定模型，仅改变假定噪声水平；对比 `D(σ)` 与 `D(σ)+B`）：

| σ (0–255) | D(σ) LPIPS | D(σ) NIQE | D(σ)+B LPIPS | D(σ)+B PSNR | D(σ)+B SSIM | D(σ)+B NIQE |
|--:|--:|--:|--:|--:|--:|--:|
| 5 | 0.4284 | 7.303 | **0.1946** | **16.34** | **0.7708** | 6.386 |
| 10 | 0.4461 | 8.605 | 0.2338 | 16.27 | 0.7588 | 7.323 |
| 15 | 0.4692 | 9.121 | 0.2660 | 16.15 | 0.7454 | 7.264 |
| 25 | 0.4966 | 10.617 | 0.3102 | 15.70 | 0.7167 | 8.420 |
| 50 | 0.5456 | 17.147 | 0.3773 | 15.02 | 0.6768 | 8.169 |

- **单调趋势**：σ 越小越好，最优趋近"不显式去噪"（σ→0 时 D≈Identity，退化为 B-only / DB+B）。
- 即使 σ=5（FastDVDNet 训练下限），`D(5)+B`（LPIPS 0.1946）仍差于 `B only`（0.1703）与 `DB+B`（0.1584）；仅在 PSNR/SSIM/NIQE 上略优于 B-only，呈现**感知指标与保真指标的分歧**（去噪压平了噪声、提高了像素级指标，但引入的伪影破坏了 LPIPS 感知质量）。

![sigma sweep](../../outputs/pipeline_ablation/sigma_sweep.png)

### 5.5 最佳帧可视化（完整管线 LPIPS 最优帧 #1 / `0157.png`）

![ablation grid](../../outputs/pipeline_ablation/ablation_grid_best_frame001.png)

（2×5 网格：第一行 Input / D / DB / B / D+DB，第二行 D+B / DB+B / Full / Variant / GT；大图见 `outputs/pipeline_ablation/ablation_grid_best_frame001.png`。）

---

## 六、复现命令与产物清单

### 6.1 复现命令

```bash
cd /home/a1005/25/TFS-Net

# 三支路消融矩阵（9 配置）
/home/a1005/anaconda3/envs/CMKDK/bin/python pipeline_ablation.py \
    --input /home/a1005/yzy/dataset/SDSD/test/low-light/pair19 \
    --gt_dir /home/a1005/yzy/dataset/SDSD/test/GT/pair19 \
    --output outputs/pipeline_ablation \
    --num_frames 11 --noise_sigma 25 --scale 0.5 --gpu 0

# σ 敏感性扫描
/home/a1005/anaconda3/envs/CMKDK/bin/python pipeline_sigma_sweep.py \
    --input /home/a1005/yzy/dataset/SDSD/test/low-light/pair19 \
    --gt_dir /home/a1005/yzy/dataset/SDSD/test/GT/pair19 \
    --output outputs/pipeline_ablation \
    --num_frames 11 --scale 0.5 --gpu 0
```

> 注：`pipeline_ablation.py` 中 σ 参数名为 `--noise_sigma`；示例见 `--help`。

### 6.2 产物清单

| 文件 | 内容 |
|:--|:--|
| `outputs/pipeline_ablation/metrics_summary.csv` | 9 配置 × 4 指标（均值±标准差） |
| `outputs/pipeline_ablation/metrics_per_frame.csv` | 9 配置 × 11 帧逐帧指标 |
| `outputs/pipeline_ablation/metrics_summary.json` | 汇总 JSON |
| `outputs/pipeline_ablation/sigma_sweep.csv` | σ 扫描逐帧指标 |
| `outputs/pipeline_ablation/frames/{Input,D,DB,B,D_DB,D_B,DB_B,D_DB_B,DB_D_B}/` | 各配置输出帧（960×540 PNG） |
| `outputs/pipeline_ablation/ablation_bars.png` | 9 配置 × 4 指标柱状图 |
| `outputs/pipeline_ablation/ablation_grid_best_frame001.png` | 最佳帧 2×5 对比网格 |
| `outputs/pipeline_ablation/checker_diagnosis.png` | 棋盘伪影诊断（暗光 vs GT 输入） |
| `outputs/pipeline_ablation/crop_detail_0157.png` | 细节放大对比（gamma 提亮） |
| `outputs/pipeline_ablation/sigma_sweep.png` | σ 扫描曲线 |
| `pipeline_ablation.py` / `pipeline_sigma_sweep.py` | 实验脚本 |

---

## 七、结论与对 TSD-Net 的启示

1. **串行 off-the-shelf 级联存在域不匹配与误差累积**。去噪模型（在正常曝光 AWGN 上训练）直接用于暗光域会引入结构性伪影，并被后续增亮模块放大——这与仓库文档中引用的 VSRELL（CVPR 2026）观点一致：*串行结构中光照与噪声不联合建模会导致误差传播与累积*。
2. **单一任务模型的"单任务"假设在真实数据上不成立**。StableLLVE 官方说明其权重在合成干净数据上训练（"may be unsuitable for noisy data"）；本实验从定量上显示：若要串行工作，去噪分支必须先做**域适配（fine-tune / 噪声模型匹配）**，否则宁可不做。
3. **去模糊分支的收益依赖下游增亮**：DB-only 在暗光域几乎不可见，但在 B 之后体现出稳定的小幅提升（LPIPS −0.012、SSIM +0.017、NIQE −0.21）。这与 TSD-Net 将 motion 处理置于光照处理之前的顺序一致。
4. **对 TSD-Net 的启示**：端到端联合建模（NDPN/MCPN/ISPN 耦合）相对于"三个独立预训练模型串行"的优势在于可以避免跨域/跨阶段的误差放大。本实验可作为论文中**"serial pipeline baseline"** 的定量证据。

---

## 八、局限

1. **数据规模**：仅 SDSD test/pair19 的 11 帧（960×540），结论的统计强度有限；建议后续在 SDSD 全部 10 个 test pair + 更大分辨率上复验。
2. **单 σ 主实验**：主矩阵固定 σ=25，σ 扫描仅覆盖 D / D+B；DB 不受 σ 影响（无噪声参数）。
3. **GT 对齐假设**：按时间索引对齐（LL 0156+i ↔ GT 0161+i）；该假设基于 SDSD 双相机同长序列惯例，未见官方对齐脚本。
4. **NIQE 实现**：采用 BasicSR/HVI-CIDNet 复现 + 官方参数，非 MATLAB 原版；绝对值不可与 MATLAB 结果直接比较，但配置间相对比较有效。
5. **DB 权重选择**：使用 `CDVD_TSP_DVD_Convergent.pt`（DVD 域），未与 `GOPRO` 版本对比；SDSD 为室内手持视频，更接近 DVD 域。
6. **伪影归因**：棋盘伪影同时存在"域不匹配"与"PixelShuffle 架构"两个候选因素；本文通过 GT 输入对照与 σ 扫描给出间接证据，未做消融（如替换上采样层）分离两者。

---

## 附：本文相关文件（引用）

- 主管线脚本：`pipeline_full.py`
- 消融脚本：`pipeline_ablation.py`、`pipeline_sigma_sweep.py`
- 参考仓库：`reference_repos/fastdvdnet/`、`reference_repos/CDVD-TSP/`、`reference_repos/StableLLVE/`
- NIQE 实现：`reference_repos/HVI-CIDNet/loss/niqe_utils.py` + `niqe_pris_params.npz`
