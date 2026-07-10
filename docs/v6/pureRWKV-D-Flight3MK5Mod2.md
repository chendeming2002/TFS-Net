Let me look at what the data is actually telling you, because I think the immediate reaction of "wait for Phase 2 to fix it" is wrong.

---

## The core problem is not Phase 1.5 being too short. The core problem is that the stages have swapped roles.

Look at the SGRF pipeline on that diagnostic frame:

```
Input:     mean=0.044  (dark, correct)
S1 out:    mean=0.056  (barely changed — NDPN did almost nothing)
S2 out:    mean=0.671  (12× brightness jump — "deblur" is doing brightening)
S2+Curve:  mean=0.840  (curve pushes further — correct behavior for curve)
S3 out:    mean=0.261  (gain=0.50 pulls it back down — ISPN is dimming)
```

The architecture was designed for:
- S1: remove noise (small pixel changes, similar brightness)
- S2: fix motion (spatial shifts, similar brightness)  
- S2.5: curve brightens globally
- S3: gain/bias does spatial fine adjustment (gain > 1, brightening)

What actually happened:
- S1: nothing
- S2: does all the brightening (12×)
- S2.5: brightens more (correct direction, wrong magnitude because S2 already overshot)
- S3: gain < 1, actively dimming to compensate for S2 overshoot

The physical ordering constraint from Section 2.1 of your architecture document — "denoise before brighten to avoid noise amplification" — is being violated in practice. S2's StageBlock learned to brighten because that's the easiest way to reduce pixel loss during Phase 1.5 unlock. The gradient path of least resistance went through S2's gate, not through ISPN's curve/gain.

---

## Why this happened

During Phase 1.5 (epoch 10-30), NDPN and MCPN outputs were linearly unlocked from 0 to 100%. The SGRF StageBlocks simultaneously had their zero-gates learning to open. But the StageBlock in S2 has a direct residual connection to the image:

```
img_s2 = img_s1 + delta_2 * gamma_gate
```

When `delta_2` is large and positive (which MCPN's features can easily produce), this directly adds brightness. The loss immediately rewards this because the image is dark and GT is bright. Meanwhile, the intended brightening path (curve → gain) is:

```
img_s2 → curve(img_s2, α) → img_curved × gain + bias → res_t
```

This is a longer gradient path with more nonlinearities. The optimizer found the shortcut: push brightness through S2's gate instead.

The consequence is that ISPN's gain_map learned to be a corrective brake (gain < 1) rather than the primary brightener (gain > 1). The curve then overshoots on top of S2's overshoot, and gain has to pull everything back. The whole pipeline is fighting itself.

---

## Why ifpn (C_Ω diagonal) is degrading

The ifpn loss going from 3.0 to 6.36 is not just "TCA alignment getting worse." It's a direct consequence of the role swap.

When S2's StageBlock is doing brightening instead of motion compensation, the gradient flowing back through MCPN into TCA says "give me features that increase brightness" rather than "give me features that align frames." TCA's correspondence matrix C_Ω responds by becoming less diagonal-dominant — not because it's detecting more motion, but because off-diagonal entries allow mixing of neighbor frame brightness into the center frame features, which S2's gate then exploits for brightening.

So C_Ω degradation is a symptom, not a cause. The cause is S2 doing the wrong job.

---

## Why MCPN γ is stuck at 0.011

MCPN γ isn't growing because MCPN doesn't need to grow. S2's StageBlock gate is already open and doing the heavy lifting (brightening). MCPN's output features are just being used as a brightness carrier, not as motion-compensated features. The gamma anti-collapse regularizer keeps γ above 0.005, so it reads as "alive" in the logs, but it's not functionally contributing motion information.

Meanwhile NDPN γ at 0.029 is slightly higher because S1's StageBlock hasn't found as strong a shortcut — the delta from S1 is small (0.044 → 0.056), so NDPN's features pass through mostly unused.

---

## What you should actually do

The fundamental issue is that **StageBlock residual connections are a lower-resistance path for brightening than the ISPN curve/gain path**. You need to either close that shortcut or make the intended path easier.

### Intervention 1: Clamp StageBlock output to prevent brightness shifts

The StageBlock delta should not change mean brightness. It should only change spatial patterns (denoise, deblur). Add a zero-mean constraint:

```python
delta = conv_block(features) * gamma_gate
delta = delta - delta.mean(dim=[-2,-1], keepdim=True)  # force zero-mean
img_out = img_in + delta
```

This is a hard architectural constraint that says "S1 and S2 can rearrange pixels but cannot shift overall brightness." All brightness change must come through curve and gain. This directly enforces the physical ordering from Section 2.1.

This is the single most impactful change you can make right now.

### Intervention 2: If zero-mean feels too aggressive, use a soft penalty

```python
L_brightness_leak = (delta.mean(dim=[-2,-1]) ** 2).mean()
```

Add this for S1 and S2 deltas with weight 1.0-5.0. Penalizes any mean brightness shift from the StageBlocks. Softer than hard clamping but still redirects brightening gradient to ISPN.

### Intervention 3: Don't wait for Phase 2

The role swap happened during Phase 1.5 and is now baked into the weights. Phase 2 adding perceptual loss and frequency loss won't fix it — those losses also reward brightness correctness and will continue to flow through the S2 shortcut. More training on a broken decomposition just deepens the local minimum.

If you intervene now (epoch 37), you have 53 epochs of Phase 2 remaining for the corrected architecture to converge. If you wait until epoch 60 to confirm the problem, you'll need to restart.

---

## Answering the diagnostic probe question from before

You don't even need to run the 5-configuration ablation anymore. The SGRF stage tracking already answered the question:

| Branch | Designed role | Actual role | Evidence |
|--------|--------------|-------------|----------|
| NDPN → S1 | Denoise | Near-identity | mean 0.044→0.056 |
| MCPN → S2 | Deblur | **Primary brightener** | mean 0.056→0.671 |
| Curve | Global brighten | **Secondary brightener (overshoot)** | mean 0.671→0.840 |
| ISPN gain | Spatial brighten | **Brightness brake** | gain=0.50, dimming |

The decomposition has collapsed, but not in the way I predicted. It didn't collapse into "ISPN does everything." It collapsed into "S2 does everything and ISPN compensates." The three-source decomposition is not functioning.

---

## Priority order

1. **Now (epoch 37)**: Add zero-mean constraint on S1/S2 StageBlock deltas. This is a 5-line code change.
2. **Epoch 38-40**: Monitor whether gain_map mean rises above 1.0 (it should, once S2 stops stealing the brightening job).
3. **Epoch 40**: Check if curve_α increases (it should absorb the brightening that S2 can no longer do).
4. **Epoch 50**: Run the 5-configuration ablation to verify decomposition health.

Don't wait. The data is already telling you what's wrong.