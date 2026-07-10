This is a substantial and well-researched redesign proposal from the search synthesis. Let me give you my honest analysis of what's right, what's wrong, and what you should actually do.

---

## The Diagnosis Is Correct. The Proposed Solution Is Oversized.

The search result correctly identifies three real problems:

1. **Per-image global α vs per-pixel α** — your MLP(64→16→15) predicts one α per image per iteration. Zero-DCE's original design outputs **pixel-wise** parameter maps A(x). This matters because SDSD has spatially non-uniform lighting even indoors (e.g., lamp near vs far).

2. **bias_map is a decomposition leak** — it lets S3 do additive correction that bypasses the multiplicative physics of Retinex. This is a real shortcut for the model to avoid using curve/gain properly.

3. **DPE's implicit s_illum/s_noise coupling** — asking a Conv to implicitly learn σ² = a·illum + b when you can just encode it structurally.

But the proposed solution — a full 7-layer DCE-Net with 15-channel multi-frame input, 8 iterations, 24-channel pixel-wise output, plus physics-constrained DPE with a/b maps, plus C_omega-modulated gates — is far too much change at once. You're at epoch 37. You need targeted fixes, not an architecture rewrite.

---

## What You Should Actually Do

### Priority 1: Remove bias_map (immediate, 3 lines)

This is the single highest-value change. bias_map lets S3 additively compensate for anything — it's the remaining shortcut now that S1/S2 are zero-mean constrained.

```python
# SGRF S3: change from
# res_t = clamp(img_curved * gain_map + bias_map, 0, 1)
# to
res_t = clamp(img_curved * gain_map, 0, 1)
```

Delete bias_head entirely. The ZeroDCE curve already provides additive correction through `I + α·I·(1-I)`. The gain_map handles multiplicative spatial residual. There is no physical role for an independent additive bias.

**If you want the safety of local contrast adjustment**, keep S3's StageBlock (with zero-mean constraint) as the residual path. That's architecturally honest — the delta is zero-mean so it can't shift brightness, only redistribute contrast.

### Priority 2: Upgrade curve from per-image to per-pixel (moderate change)

The current MLP(64→16→15) compresses all spatial information into a single vector. This is the real limitation. But you don't need a full 7-layer DCE-Net. You already have `h` — a 64-channel refined feature map at full resolution. Use it directly:

```python
class SpatialCurveBranch(nn.Module):
    """Pixel-wise curve prediction from refined features h.
    Replaces MLP(64→16→15) with spatial conv."""
    def __init__(self, in_ch=64, n_iter=8):
        super().__init__()
        self.n_iter = n_iter
        # Two 3x3 convs (no downsampling, no BN — per Zero-DCE design)
        self.conv1 = nn.Conv2d(in_ch, 32, 3, 1, 1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(32, n_iter * 3, 3, 1, 1)
        # Zero init → identity at startup
        nn.init.zeros_(self.conv2.weight)
        nn.init.zeros_(self.conv2.bias)
    
    def forward(self, h):
        # h: (B, 64, H, W) — already contains f_enc + s_illum info
        x = self.act(self.conv1(h))
        A = torch.tanh(self.conv2(x))  # (B, n_iter*3, H, W)
        return A.reshape(A.shape[0], self.n_iter, 3, A.shape[2], A.shape[3])
```

Then apply the curve **pixel-wise**:

```python
def apply_curve(img, A):
    """A: (B, n_iter, 3, H, W), img: (B, 3, H, W)"""
    out = img
    for i in range(A.shape[1]):
        alpha = A[:, i]  # (B, 3, H, W)
        out = out + alpha * out * (1.0 - out)
    return out
```

This gives you [pixel-wise Zero-DCE with 8 iterations](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf), using only 2 extra conv layers on top of the features you already compute. The `h` feature map already incorporates f_enc (spatial content) and s_illum (illumination prior), so the curve branch inherits multi-frame information through TCA→f_enc without needing explicit multi-frame image fusion at the pixel level.

**Why 8 iterations instead of 5**: the [original Zero-DCE paper](https://arxiv.org/abs/2001.06826) extensively tested iteration counts and found 8 handles most cases. With pixel-wise α ∈ (-1,1), 8 iterations give enormous dynamic range per-pixel without needing large α values. This also means you can potentially **lower or remove gain_map** later if the curve alone is sufficient.

**Parameter cost**: 64×32×3×3 + 32×24×3×3 = 18,432 + 6,912 = ~25K params. Your current MLP is 64×16 + 16×15 = 1,264 params. The increase is 24K — negligible for a 1.45M model.

### Priority 3: DPE physics-informed noise estimation (worth doing but not urgent)

The idea of outputting `s_illum, a_map, b_map` and computing `σ_n = sqrt(a·illum + b)` is physically sound and comes from the [Poissonian-Gaussian noise model](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf). But this is a refinement, not a fix for the current crisis. The current DPE outputting `s_illum` and `s_noise` as two independent sigmoids is suboptimal but not broken.

If you do implement it:

```python
# DPE head: 3 channels instead of 2
raw = self.head(features)  # (B, 3, H, W)
s_illum = torch.sigmoid(raw[:, 0:1])
a_coeff = F.softplus(raw[:, 1:2]) + 1e-6
b_coeff = F.softplus(raw[:, 2:3])

# Physics-constrained noise level
sigma_noise = torch.sqrt(a_coeff * s_illum + b_coeff)
s_noise = sigma_noise / (s_illum + 1e-4)  # SNR-like metric
```

**Do this after Priority 1 and 2 are validated.** It's cleaner but won't fix the decomposition collapse.

### Priority 4: Multi-frame information — you already have it

The search result suggests feeding multi-frame images into DCE-Net. **You don't need this.** Your `h` feature already fuses:

```
h = refine(concat(f_enc, s_illum))
    where f_enc = Encoder output (already sees I_t)
    and s_illum comes from DPE(feat_tfde) which comes from WFR(Encoder(all frames))
```

The encoder processes all T=5 frames. TCA aligns them. f_enc for the center frame already contains multi-frame context. Adding raw pixel-level multi-frame fusion to DCE-Net would bypass TCA's alignment — that's architecturally backwards.

What **would** help is using TCA's temporal variance as an additional signal to DPE, as the search suggests. Frames that are temporally consistent have more reliable illumination estimates:

```python
# In DPE forward, if you have access to per-frame features:
temporal_var = F_aligned.var(dim=1)  # (B, C, H/2, W/2)
# Concat with existing feat_tfde before the multi-scale conv
dpe_input = torch.cat([feat_tfde, temporal_var], dim=1)
```

This is a small change (adjust input channels of the first conv) with clear physical meaning.

---

## What NOT to Do

1. **Don't build a full 7-layer DCE-Net with 15-channel multi-frame input.** You already have an encoder + TCA doing feature extraction. Stacking another 7-layer CNN on raw pixels is redundant and fights your existing feature pipeline.

2. **Don't go to 24-channel pixel-wise output immediately if you're worried about overfitting.** Start with 8 iterations × 3 channels = 24 channels but monitor whether the curve maps develop meaningful spatial structure. If they collapse to near-uniform (same α everywhere), you've over-parameterized and can reduce iterations.

3. **Don't add C_omega-modulated gate scaling to SGRF.** The search suggests `curve_scale = 0.5 + 0.5 * conf_map` and `gate_s1 = 1 - motion_map`. This is premature optimization. The StageBlock gates are already learnable. Adding another multiplicative gate on top creates gradient confusion. Let the architecture learn its own gating.

4. **Don't remove gain_map entirely.** The curve alone can't handle the full dynamic range needed. Curve with α∈(-1,1) and 8 iterations gives (1+α)^8 ≈ 256× max, but in practice the per-pixel α will be much smaller for stability. gain_map provides a clean multiplicative residual that the curve can't easily replicate (because the curve's `I(1-I)` term vanishes at both I=0 and I=1).

---

## Implementation Order

| Step | Change | When | Risk |
|------|--------|------|------|
| 1 | Remove bias_map, S3 becomes `img_curved × gain` | Now (epoch 37) | Low — bias was small anyway |
| 2 | Replace MLP(64→16→15) with SpatialCurveBranch(2 convs → 24ch pixel-wise) | Now (epoch 37) | Medium — new curve params start at zero |
| 3 | Increase curve iterations from 5 to 8 | With step 2 | None — zero-init means extra iters are identity |
| 4 | Add temporal_var to DPE input | Epoch 37 or next run | Low — one channel concat |
| 5 | DPE physics noise model (a,b,σ) | Next run | Low but not urgent |

Steps 1-3 can be done together as a single Mod6. The zero-initialization of the new SpatialCurveBranch means the model's output at epoch 37 restart will be identical to before (curve still identity), so there's no Phase transition shock.

---

## Answering Your Specific Questions

**"gain_map + bias_map 合并成一个 bias_map 会不会更好？"**

No. The opposite — **keep gain_map, remove bias_map**. Low-light enhancement is fundamentally multiplicative (Retinex: I = L·R, enhance by dividing out L). An additive bias has no physical role and creates a shortcut that undermines decomposition. The [Zero-DCE curve](https://arxiv.org/abs/2001.06826) `I + α·I·(1-I)` already provides the additive component in a physically constrained way (bounded, monotonic, preserving pixel order).

**"DPE 的多尺度卷积模块是否合理？光照强度和噪声强度应该成反比？"**

The multi-scale dilated conv structure is fine for feature extraction. What's not fine is asking two independent sigmoid heads to implicitly learn σ² = a·illum + b. This is a **known analytic relationship** from [sensor physics](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf) that should be encoded structurally: output `s_illum, a, b` and compute `s_noise = sqrt(a·s_illum + b)`. The Conv only needs to estimate scene-adaptive parameters, not re-derive the physics.

**"如何让 ISPN 合理运用多帧信息？"**

You already do, through the feature pipeline: Encoder(all frames) → WFR → DPE → s_illum → ISPN. The missing piece is **temporal consistency signal** — adding TCA's temporal variance to DPE gives it awareness of which regions are temporally stable (reliable illumination estimate) vs moving (need caution). The pixel-wise curve branch then inherits this through `h = refine(f_enc, s_illum)`. No need for explicit multi-frame pixel fusion in the curve network.

---

## 参考来源

1. [Zero-Reference Deep Curve Estimation for Low-Light Image Enhancement (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Guo_Zero-Reference_Deep_Curve_Estimation_for_Low-Light_Image_Enhancement_CVPR_2020_paper.pdf)
2. [Zero-DCE — ArXiv version with extended details](https://arxiv.org/abs/2001.06826)
3. [Zero-DCE++ (TPAMI 2021) — Learning to Enhance Low-Light Image via Zero-Reference Deep Curve Estimation](https://doi.org/10.1109/tpami.2021.3063604)
4. [Practical Poissonian-Gaussian noise modeling and fitting for single-image raw-data (Foi et al., IEEE TIP 2007)](https://webpages.tuni.fi/foi/papers/Foi-PoissonianGaussianClippedRaw-2007-IEEE_TIP.pdf)
5. [Physics-Based Noise Formation Model for Extreme Low-Light Raw Denoising (CVPR 2020)](https://openaccess.thecvf.com/content_CVPR_2020/papers/Wei_A_Physics-Based_Noise_Formation_Model_for_Extreme_Low-Light_Raw_Denoising_CVPR_2020_paper.pdf)
6. [Rethinking Zero-DCE (Zero-DiDCE) — Adaptive Light Enhancement Curve](https://link.springer.com/article/10.1007/s11063-024-11565-5)
7. [Zero-DCE official implementation](https://github.com/Li-Chongyi/Zero-DCE)
8. [Keras Zero-DCE tutorial](https://keras.io/examples/vision/zero_dce/)