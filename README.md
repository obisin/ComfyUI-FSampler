## **FSampler for ComfyUI — Fast Skips via Epsilon Extrapolation**

FSampler is a training‑free, sampler‑agnostic acceleration layer for diffusion sampling that reduces model calls by predicting each step’s epsilon  
(noise) from recent real calls and feeding it into the existing integrator. It provides fixed history modes (h2/h3/h4) and an aggressive adaptive   
mode that, per step, builds two predictions (e.g., h3 vs h2), compares their predicted next states to get a relative error, and skips the model call
when that error is below a hardcoded tolerance. Predicted epsilons are validated and scaled by a universal learning stabilizer L, skips are bounded 
by guard rails, and the sampler math (Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS) is unchanged.

## Note:
- currently only tested on flux, wan2.2 and qwen- happy for anyone to test and give feedback- I will test on otehrs later. 
- Testing done on a 2080ti with loras and f8 and f16 models. 
- The longer a single run on one model the better. Split model like Wan2.2 will see less benefit due to lower step count per model as this leads to less history to predict future values.
- Runs in place of your regular KSampler node.

## Overview
- Training‑free acceleration that skips full model calls using predicted epsilon (noise) from recent REAL steps.
- Works with existing samplers: Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS, RES_Multistep.
- Stability via a universal learning stabilizer L and strict validators; clear per‑step diagnostics.
- Since all equations are deterministic, running high skips will still produce very similar results as if ran with no skips menaing you can generate alot more test quicker before using a single seed for production.
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

## Quality & Safety
- Validators: finite checks, magnitude clamp vs history, cosine vs last REAL epsilon.
- Learning stabilizer L: scales predicted epsilon by 1/L on skipped steps; updates on REAL steps only.
- Diagnostics: per‑step timing + concise line showing σ targets, h/weights (where relevant), epsilon norms, x_rms, and [RISK].

## Visual Gallery (Placeholders)
- Flux
  - See  Image above

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

[![ko-fi](https://ko-fi.com/img/githubbutton_sm.svg)](https://ko-fi.com/G2G61M09C8)

<img width="3000" height="3000" alt="qr-code" src="https://github.com/user-attachments/assets/50949a09-1cc3-4f85-9b8b-b3273f8dc8e5" />



