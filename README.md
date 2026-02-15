
[![arXiv](https://img.shields.io/badge/arXiv-2511.09180-b31b1b.svg)](https://arxiv.org/abs/2511.09180)


## **FSampler for ComfyUI — Fast Skips via Epsilon Extrapolation**

FSampler is a training‑free, sampler‑agnostic acceleration layer for diffusion sampling that reduces model calls by predicting each step’s epsilon  
(noise) from recent real calls and feeding it into the existing integrator. It provides fixed history modes (h2/h3/h4) and an aggressive adaptive   
mode that, per step, builds two predictions (e.g., h3 vs h2), compares their predicted next states to get a relative error, and skips the model call
when that error is below a hardcoded tolerance. Predicted epsilons are validated and scaled by a universal learning stabilizer L, skips are bounded 
by guard rails, and the sampler math (Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS) is unchanged.

## FSampler Changelog:
## 2026-01-01

## v1.5.0
### Sigma-Aware Extrapolation
- New `sigma_aware` toggle (FSampler Advanced / FSampler Select) — uses actual sigma coordinates for Lagrange interpolation
- Improves skip prediction accuracy with non-uniform schedulers (bong_tangent, karras, exponential, beta, etc.)
- Enabled by default in the simple FSampler node

### Denoised Extrapolation
- New `extrapolate_denoised` toggle — extrapolates the model's denoised (clean image) output instead of epsilon
- Denoised converges smoothly toward the final image, can produce more stable skip predictions
- Enabled by default in the simple FSampler node
- Composes with sigma_aware (orthogonal: sigma_aware controls weights, denoised controls what quantity is extrapolated)


### New Node: FSampler Select
- Outputs `SAMPLER` + `SIGMAS` for use with ComfyUI's **SamplerCustom** (KSamplerCustom) node
- Lets you plug FSampler's skip-aware sampling into the standard modular workflow alongside any guider (CFGGuider, DualCFGGuider, BasicGuider, etc.)
- Wire `sampler` → SamplerCustom's sampler slot, `sigmas` → SamplerCustom's sigmas slot
- All FSampler skip/adaptive controls available; model/conditioning/cfg/noise handled by SamplerCustom

### Live Preview
- FSampler and FSampler Advanced now show live image previews during sampling (Latent2RGB, auto-enabled)

### Explicit Skip Indices with Predictor Selection
-  Manual step selection with extrapolation method control (Good for low step  count workflows)
- Take precise control over which steps to skip and how predictions are made using the `skip_indices` parameter (available in FSampler Advanced node).

## Overview
- Training‑free acceleration that skips full model calls using predicted epsilon (noise) from recent REAL steps.
- Works with existing samplers: Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS, RES_Multistep.
- Stability via a universal learning stabilizer L and strict validators; clear per‑step diagnostics.
- Since all equations are deterministic, running high skips will still produce very similar results as if ran with no skips menaing you can generate alot more test quicker before using a single seed for production.
- Very simple extrapolation
---

## IMPORTANT NOTE

**Important Compatibility Information:**
- The RES family samplers will not produce 1:1 parity with the official KSampler or ClownShark KSampler implementations- despite having 1:1 code
- Even ComfyUI vs ClownShark produce different results, due to environment variables and implementation details
- Simple samplers like Euler will have 1:1 parity across implementations
---

- Open/enlarge the picture below and note how generations change with the more predictions and steps between them. We dont see as much quality loss but rather the direction of where the model goes. Thats not to say there isnt any quality loss but instead this method creates more variations in the image.
- All tests were done using comfy cache to prevent time distortions and create a fairer test. This means that model loading time i sthe same for each generation. If you do tests please do the same.

![article fsampler.jpg](article%20fsampler.jpg)


## Installation

Method 1: Git Clone
```bash
cd ComfyUI/custom_nodes
git clone https://github.com/obisin/comfyui-FSampler
# Restart ComfyUI
```

Method 2: Manual Download
- Download the repository as ZIP
- Extract to `ComfyUI/custom_nodes/comfyui-FSampler/`
- Restart ComfyUI

## Highlights
- Fixed modes: h2 (~25%), h3 (~16%), h4 (~12%) NFE reduction with parity on standard configs.
- Adaptive mode: aggressive gate; can reach 40–60%+ reduction on smooth runs while preserving quality.

## Skip Modes
- none: baseline (no skipping)
- hN/sK: s=history used for predictor, s=steps/calls before skip
  - h2/s2..s5: linear predictor; common picks h2/s2 (~24%) or h2/s3 (~20%+)
  - h3/s3..s5: Richardson; common picks h3/s3 (~16%) or h3/s4 (~12%+)
  - h4/s4..s5: cubic; conservative, quality-sensitive; typically h4/s4
- adaptive: aggressive skip gate using two predictors (h3 vs h2) in predicted‑state space

## Usage (ComfyUI)

- For quick usage start with the Fsampler rather than the FSampler Advanced as the simpler version only need noise and adaption mode to operate.
- Swap with your normal KSampler node.

1. Add the **FSampler** node (or **FSampler Advanced** for more control)
2. Choose your **sampler** and **scheduler** as usual
3. Set **skip_mode**:
   - `none` — baseline (no skipping, use this first to validate)
   - `h2` — conservative, ~20-30% speedup (recommended starting point)
   - `h3` — more conservative, ~16% speedup
   - `h4` — very conservative, ~12% speedup
   - `adaptive` — aggressive, 40-60%+ speedup (may degrade on tough configs)
4. Adjust **protect_first_steps** / **protect_last_steps** if needed (defaults are usually fine)

### FSampler Select (Modular / KSamplerCustom)

For advanced workflows using ComfyUI's modular sampling system:

1. Add **FSampler Select** — configure your sampler, scheduler, steps, and skip settings
2. Add **SamplerCustom** (ComfyUI's KSamplerCustom node)
3. Wire FSampler Select's `sampler` output → SamplerCustom's `sampler` input
4. Wire FSampler Select's `sigmas` output → SamplerCustom's `sigmas` input
5. Wire your model, positive/negative conditioning, cfg, and latent into SamplerCustom as normal

This lets you use FSampler with any guider (CFGGuider, DualCFGGuider, BasicGuider) and the rest of the modular sampling ecosystem.

## Extrapolation Settings (v1.5.0)

FSampler Advanced and FSampler Select expose two new boolean toggles that can help reduce prediction drift on skipped steps. Both are enabled by default in the simple FSampler node.

### Sigma-Aware Extrapolation (`sigma_aware`)

Standard extrapolation assumes steps are uniformly spaced. Many schedulers (bong_tangent, karras, exponential, beta, etc.) have highly non-uniform spacing — dense in some regions, sparse in others. With uniform assumptions, the Lagrange basis weights can be wrong and predictions may drift.

When `sigma_aware` is enabled, FSampler uses the actual sigma values as coordinates for Lagrange interpolation. This can give more accurate polynomial weights when scheduler spacing is non-uniform.

- **Best for:** non-uniform schedulers (bong_tangent, karras, exponential, beta)
- **No effect on:** uniform schedulers (normal) — weights are identical either way
- **Default:** OFF in Advanced/Select, ON in simple FSampler

### Denoised Extrapolation (`extrapolate_denoised`)

Standard FSampler extrapolates **epsilon** (noise prediction) = `denoised - x`. Epsilon depends on both the model's output AND the changing noisy latent `x`, which can introduce variability from both sources.

When `extrapolate_denoised` is enabled, FSampler extrapolates the model's **denoised** output directly — its best guess at the clean image. Denoised tends to converge more smoothly toward the final image as noise decreases, which can make it a more stable quantity to extrapolate and may reduce prediction drift. The predicted denoised is then converted back to epsilon for the sampler.

- **Orthogonal to sigma_aware:** sigma_aware controls interpolation weights, denoised controls what quantity is extrapolated. Both can be used together.
- **Default:** OFF in Advanced/Select, ON in simple FSampler

### Recommended Combinations

| Setting | Best For |
|---------|----------|
| Both OFF | Baseline comparison, uniform schedulers with simple samplers |
| `sigma_aware` only | Non-uniform schedulers where epsilon extrapolation already works well |
| `extrapolate_denoised` only | Any scheduler where skip quality needs improvement |
| Both ON (simple FSampler default) | Generally good results, especially with non-uniform schedulers |

## Quality & Safety
- Validators: finite checks, magnitude clamp vs history, cosine vs last REAL epsilon.
- Learning stabilizer L: scales predicted epsilon by 1/L on skipped steps; updates on REAL steps only.
- Diagnostics: per‑step timing + concise line showing σ targets, h/weights (where relevant), epsilon norms, x_rms, and [RISK].


### Adaptive Skip Modes

All samplers now support four adaptive modes for intelligent step skipping:

#### `none`
- Standard operation without adaptive corrections
- SKIP steps use cached predictions directly

#### `learning`
- Learns a stabilization factor `L` from REAL steps
- Scales cached predictions by `1/L` on SKIP steps
- L is learned via exponential moving average and clamped to [0.5, 2.0]
- Helps adapt to model behavior across the denoising trajectory

#### `grad_est`
- Applies gradient estimation correction on SKIP steps only
- Adds directional correction based on gradient differences
- No L scaling applied
- Best for maintaining prediction directionality

#### `learn+grad_est`
- **Combines both approaches** for maximum adaptivity
- Applies L scaling AND gradient correction on SKIP steps
- Recommended for best quality when using aggressive skip patterns

### Explicit Skip Indices with Predictor Selection

**New Feature**: Manual step selection with extrapolation method control

Take precise control over which steps to skip and how predictions are made using the `skip_indices` parameter (available in FSampler Advanced node).

#### Input Format

Write configs like: `"h2, 3, 4, 7, 9"` or `"h4, 10, 12"` or simply `"3, 6, 8"`

**Components:**
- **Predictor token** (optional): `h2`, `h3`, or `h4` - specifies the extrapolation method
- **Step indices**: Comma or space-separated integers (0-based)
- **Empty/whitespace**: `""` or `"[]"` - ignored, uses other skip settings

#### Automatic Fallback Ladder

FSampler adapts when history is insufficient:

```
h4 requested → uses h4 if ≥4 history, else h3 if ≥3, else h2 if ≥2, else cancels skip
h3 requested → uses h3 if ≥3 history, else h2 if ≥2, else cancels skip
h2 requested → uses h2 if ≥2 history, else cancels skip
```
**Never falls below h2** - minimum 2 REAL epsilon history required for any skip.

#### Index Rules and Constraints

**Always Enforced:**
- Steps **0 and 1 never skipped** (need initial history)
- Final step **can be skipped** (explicit indices bypass protect windows)

#### Example with Explanation

**Low-step generation (5 steps):**
```
skip_indices = "4"
```
- Skip only step 4 using h2 (linear)
- Steps 0,1,2,3 are REAL → saves ~20% compute


#### Why Explicit Indices Matter

**For Low Step Counts (4-8 steps):**
- Modern models excel at 4-6 step generation
- Each step has **outsized impact** on final quality
- Early steps establish composition, final steps add detail
- Manual control preserves the most critical steps


**For Medium Step Counts (10-20 steps):**
```
Steps: 15
skip_indices = "h2, 5, 7, 9, 11, 13"  # Skip ~40% with better extrapolation
```

## Notes
- h2/h3/h4 are conservative and deterministic; adaptive is aggressive and may show degradation on tough configs — validators and L minimize artifacts.
- Protect first/last windows guard early/late critical regions.
- Anchors and max consecutive skips are internal to adaptive to bound drift.

## Issues
- Please incude the verbose output for the run so I can see what the calculations are doing and diagnose the problem
- Not all schedulers and samplr combos will produce results- some will produce nonsense on some models as is the case without skipping

---

## FAQ

**Q: Does this work with LoRAs/ControlNet/IP-Adapter?**
A: Yes! FSampler sits between the scheduler and sampler, so it's transparent to conditioning.

**Q: Will this work on SDXL Turbo / LCM?**
A: Potentially, but low-step models (<10 steps) won't benefit much since there's less history to extrapolate from.

**Q: Can I use this with custom schedulers?**
A: Yes, FSampler works with any scheduler that produces sigma values.

**Q: I'm getting artifacts/weird images**
A: Try these in order:
1. Use `skip_mode=none` first to verify baseline quality
2. Switch to `h2` or `h3` (more conservative than adaptive)
3. Increase `protect_first_steps` and `protect_last_steps`
4. Some sampler+scheduler combos produce nonsense even without skipping — try different combinations

**Q: How does this compare to other speedup methods?**
A: FSampler is complementary to:
- **Distillation** (LCM, Turbo): Use both together
- **Quantization**: Use both together
- **Dynamic CFG**: Use both together
- FSampler specifically reduces *sampling steps*, not model inference cost

---

## ALL TESTERS WELCOME! THANKS!!1
