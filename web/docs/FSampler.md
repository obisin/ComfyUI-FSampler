# FSampler - Fast Diffusion Sampling via Epsilon Extrapolation

FSampler is a training-free, sampler-agnostic acceleration layer for diffusion sampling that reduces model calls by predicting each step's epsilon (noise) from recent real calls and feeding it into the existing integrator.

## NOTE

**Important Compatibility Information:**
- The RES family samplers will not produce 1:1 parity with the official KSampler or ClownShark KSampler implementations- despite having 1:1 code
- Even ComfyUI vs ClownShark produce different results, due to environment variables and implementation details
- Simple samplers like Euler will have 1:1 parity across implementations

## Features

- **Training-Free Acceleration**: Skips full model calls using predicted epsilon from recent steps
- **Sampler Agnostic**: Works with Euler, RES 2M/2S, DDIM, DPM++ 2M/2S, LMS, and more
- **Multiple Skip Modes**:
  - Fixed modes (h2/h3/h4): Conservative, deterministic speedup
  - Adaptive mode: Aggressive skipping for 40-60%+ speedup
  - Skip indices: Manually pick which steps to skip (useful for low step counts)
- **Built-in Stability**: Universal learning stabilizer and validators prevent artifacts

## Skip Modes

- **none**: baseline (no skipping)
- **hN/sK**: h=history used for predictor, s=steps/calls before skip
  - **h2/s2..s5**: linear predictor; common picks h2/s2 (~24%) or h2/s3 (~20%+)
  - **h3/s3..s5**: Richardson; common picks h3/s3 (~16%) or h3/s4 (~12%+)
  - **h4/s4..s5**: cubic; conservative, quality-sensitive; typically h4/s4
- **adaptive**: aggressive skip gate using two predictors (h3 vs h2) in predicted-state space

### Skip Indices
- For low step count workflows, use skip indices to manually pick which steps to skip
- Gives you precise control over the sampling process

## Usage

**For quick usage start with FSampler (simple) rather than FSampler Advanced** - the simple version only needs noise and skip mode to operate. Swap with your normal KSampler node.

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

- **Validators**: finite checks, magnitude clamp vs history, cosine vs last REAL epsilon
- **Learning stabilizer L**: scales predicted epsilon by 1/L on skipped steps; updates on REAL steps only
- **Diagnostics**: per-step timing + concise line showing σ targets, h/weights (where relevant), epsilon norms, x_rms, and [RISK]

## Notes

- h2/h3/h4 are conservative and deterministic; adaptive is aggressive and may show degradation on tough configs — validators and L minimize artifacts
- Protect first/last windows guard early/late critical regions
- Anchors and max consecutive skips are internal to adaptive to bound drift
- Works with LoRAs, ControlNet, IP-Adapter
- Since all equations are deterministic, running high skips will still produce very similar results as if ran with no skips meaning you can generate a lot more tests quicker before using a single seed for production

## Troubleshooting

### Getting artifacts or weird images?
1. Use `skip_mode=none` to verify baseline quality
2. Switch to `h2` or `h3` (more conservative than adaptive)
3. Increase `protect_first_steps` and `protect_last_steps`
4. Some sampler+scheduler combos produce issues even without skipping

### Not seeing speedup?
- FSampler needs history to extrapolate - works best with 10+ steps

## FAQ

**Q: Does this work with LoRAs/ControlNet/IP-Adapter?**
A: Yes! FSampler sits between the scheduler and sampler, so it's transparent to conditioning.

**Q: Will this work on SDXL Turbo / LCM?**
A: Potentially, but low-step models (<10 steps) won't benefit much since there's less history to extrapolate from. Use explicit skip indices for precise control with low step counts.

**Q: Can I use this with custom schedulers?**
A: Yes, FSampler works with any scheduler that produces sigma values.

**Q: How does this compare to other speedup methods?**
A: FSampler is complementary to:
- **Distillation** (LCM, Turbo): Use both together
- **Quantization**: Use both together
- **Dynamic CFG**: Use both together
- FSampler specifically reduces *sampling steps*, not model inference cost

## Tested Models

- Flux (tested on 2080ti with LoRAs, f8 and f16 models)
- Wan2.2
- Qwen

Testing and feedback welcome on other models!
