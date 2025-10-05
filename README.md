**FSampler for ComfyUI — Fast Skips via Epsilon Extrapolation**

FSampler is a training‑free, sampler‑agnostic acceleration layer for diffusion sampling that reduces model calls by predicting each step’s epsilon  
(noise) from recent real calls and feeding it into the existing integrator. It provides fixed history modes (h2/h3/h4) and an aggressive adaptive   
mode that, per step, builds two predictions (e.g., h3 vs h2), compares their predicted next states to get a relative error, and skips the model call
when that error is below a hardcoded tolerance. Predicted epsilons are validated and scaled by a universal learning stabilizer L, skips are bounded 
by guard rails, and the sampler math (Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS) is unchanged.

Note:
- currently only tested on flux, wan2.2 and qwen- happy for anyone to test and give feedback- I will test on otehrs later. 
- The longer a single run on one model the better. Split model like Wan2.2 will see less benefit due to lower step count per model as this leads to less history to predict future values.

Overview
- Training‑free acceleration that skips full model calls using predicted epsilon (noise) from recent REAL steps.
- Works with existing samplers: Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS, RES_Multistep.
- Stability via a universal learning stabilizer L and strict validators; clear per‑step diagnostics.


![article fsampler.jpg](article%20fsampler.jpg)


Installation

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

Highlights
- Fixed modes: h2 (~25%), h3 (~16%), h4 (~12%) NFE reduction with parity on standard configs.
- Adaptive mode: aggressive gate; can reach 40–60%+ reduction on smooth runs while preserving quality.
- Example run (Euler, 35 steps): 15 model calls, 20 skipped → 57.1% reduction with decent image quality.

Skip Modes
- none: baseline (no skipping)
- hN/sK: N=history used for predictor, K=calls before skip
  - h2/s2..s5: linear predictor; common picks h2/s2 (~24%) or h2/s3 (~20%+)
  - h3/s3..s5: Richardson; common picks h3/s3 (~16%) or h3/s4 (~12%+)
  - h4/s4..s5: cubic; conservative, quality-sensitive; typically h4/s4
- adaptive: aggressive skip gate using two predictors (h3 vs h2) in predicted‑state space

Usage (ComfyUI)
- Node: "FSampler" or  "Advanced"(if you want more control)
- Choose `sampler` and `scheduler`, set `steps`, `cfg`, `protect_first_steps`, `protect_last_steps`.
- Choose `skip_mode`: `none | h2/s2..s6 | h3/s3..s6 | h4/s4..s6 | adaptive`.
- For validation, start with `skip_mode=none`, then try `h2/s2`, then `adaptive`.

Quality & Safety
- Validators: finite checks, magnitude clamp vs history, cosine vs last REAL epsilon.
- Learning stabilizer L: scales predicted epsilon by 1/L on skipped steps; updates on REAL steps only.
- Diagnostics: per‑step timing + concise line showing σ targets, h/weights (where relevant), epsilon norms, x_rms, and [RISK].

Visual Gallery (Placeholders)
- Flux
  - See  Iamge above

- Wan22
  - Baseline: images/wan22/baseline.png
  - FSampler none: images/wan22/fsampler_none.png
  - FSampler h2: images/wan22/fsampler_h2.png
  - FSampler adaptive: images/wan22/fsampler_adaptive.png

- Qwen
  - Baseline: images/qwen/baseline.png
  - FSampler none: images/qwen/fsampler_none.png
  - FSampler h2: images/qwen/fsampler_h2.png
  - FSampler adaptive: images/qwen/fsampler_adaptive.png


Notes
- h2/h3/h4 are conservative and deterministic; adaptive is aggressive and may show degradation on tough configs — validators and L minimize artifacts.
- Protect first/last windows guard early/late critical regions.
- Anchors and max consecutive skips are internal to adaptive to bound drift.

Issues
- Please incude the verbose output for the run so I can see what the calculations are doing and diagnose the problem
- Not all schedulers and samplr combos will produce results- some will produce nonsense on some models as is the case without skipping

ALL TESTERS WELCOME! THANKS!!1
