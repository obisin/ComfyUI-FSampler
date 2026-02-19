"""FSampler nodes for ComfyUI."""
import time
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
import latent_preview
from comfy.cli_args import args, LatentPreviewMethod

def _make_safe_preview_callback(model_or_patcher, total_steps, x0_output_dict=None):
    """Create a preview callback that respects ComfyUI's preview setting."""
    pbar = comfy.utils.ProgressBar(total_steps)

    if args.preview_method == LatentPreviewMethod.NoPreviews:
        def safe_callback(step, x0, x, total):
            pbar.update_absolute(step + 1, total)
        return safe_callback

    inner_cb = latent_preview.prepare_callback(model_or_patcher, total_steps, x0_output_dict)

    def safe_callback(step, x0, x, total):
        try:
            inner_cb(step, x0, x, total)
        except Exception:
            pbar.update_absolute(step + 1, total)
    return safe_callback
from .comfy_copy import k_diffusion_sampling
from .sampling.fibonacci_scheduler import get_fsampler_sigmas, FSAMPLER_SCHEDULERS
from .sampling.engine import sample_fsampler, create_fsampler_ksampler


# Available schedulers
FSAMPLER_AVAILABLE_SCHEDULERS = [
    # Standard ComfyUI schedulers
    "simple",
    "normal",
    "sgm_uniform",
    "ddim_uniform",
    "beta",
    "linear_quadratic",
    "karras",
    "exponential",
    "polyexponential",
    "vp",
    "laplace",
    "kl_optimal",
    # res4lyf custom
    "beta57",
    # res4lyf tangent variants (internal defaults)
    "bong_tangent",
    "bong_tangent_2",
    "bong_tangent_2_simple",
    "constant",
    # FSampler custom schedulers
    "fibonacci",
    "fibonacci_rev"
]


def get_sigma_schedule(model, scheduler_name, num_steps):
    """Get sigma schedule using model-aware ranges.
    """
    try:
        import comfy.samplers as _samplers
        model_sampling = model.get_model_object("model_sampling")
        if scheduler_name in FSAMPLER_SCHEDULERS:
            sigma_min = float(model_sampling.sigma_min)
            sigma_max = float(model_sampling.sigma_max)
            return get_fsampler_sigmas(scheduler_name, num_steps, sigma_min, sigma_max)
        else:
            return _samplers.calculate_sigmas(model_sampling, scheduler_name, num_steps)
    except Exception:
        # Fallback to k-diffusion helpers with generic bounds
        return k_diffusion_sampling.get_sigmas(scheduler_name, num_steps, 0.03, 14.6)


class FSamplerAdvanced:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 100.0, "step": 0.05}),
                "scheduler": (FSAMPLER_AVAILABLE_SCHEDULERS, {"default": "simple"}),
                "sampler": (["euler", "res_2m", "res_2s", "ddim", "dpmpp_2m", "dpmpp_2s", "lms", "res_multistep", "res_multistep_ancestral", "heun", "gradient_estimation"], {
                    "default": "euler",
                    "tooltip": "Sampling method"
                }),
                "protect_first_steps": ("INT", {
                    "default": 2,
                    "min": 0,
                    "max": 20,
                    "tooltip": "Number of initial steps never skipped (warmup)."
                }),
                "protect_last_steps": ("INT", {
                    "default": 2,
                    "min": 1,
                    "max": 100,
                    "tooltip": "Number of final steps never skipped (quality safeguard)."
                }),
                "adaptive_mode": (["none", "learning", "grad_est", "learn+grad_est"], {
                    "default": "none",
                    "tooltip": "Adaptive skip corrections: none=off; learning=EMA L stabilizer scales predicted epsilon on SKIP (smoothed by smoothing_beta); grad_est=gradient-estimation correction on SKIP only (directional, clamped to ≤25% of step); learn+grad_est=apply both (L scaling + grad correction)."
                }),
                "smoothing_beta": ("FLOAT", {
                    "default": 0.9990,
                    "min": 0.0,
                    "max": 0.9999,
                    "step": 0.0001,
                    "tooltip": "EMA smoothing for learning mode (0.9=balanced, 0.99-0.999=high stability, 0.9999=extreme dampening). Set to 0.0 for instant reaction. Only used when adaptive_mode is learning."
                }),
                "skip_mode": ([
                    "none",
                    # History 2 (linear): K in 2..6
                    "h2/s2", "h2/s3", "h2/s4", "h2/s5",
                    # History 3 (Richardson): K in 3..6
                    "h3/s3", "h3/s4", "h3/s5",
                    # History 4 (cubic): K in 4..6
                    "h4/s4", "h4/s5",
                    "adaptive"
                ], {
                    "default": "none",
                    "tooltip": "Skipping: hN/sK with N=history (2=linear,3=Richardson,4=cubic) and K=calls before skip. Supported: h2/s2..s6, h3/s3..s6, h4/s4..s6."
                }),
                "skip_indices": ("STRING", {"default": "", "multiline": False, "tooltip": "Explicit Skip Mode: indices to skip, e.g. 'h2, 3, 4, 7'. First hN selects predictor (defaults to h2). Indices are 0-based after slicing (start/end), 0/1 never skipped; final step may be. When non-empty, this overrides and nullifies other skip/adaptive/history controls."}),
                "anchor_interval": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Absolute cadence: force a REAL call on every Nth step index counted from the end of the protected warmup (protect_first_steps). Set 0 to disable anchors. (adaptive only)"}),
                "max_consecutive_skips": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "Local cap: maximum back-to-back skips allowed; resets immediately after any REAL call. (adaptive only)"}),
                # KSampler (Advanced) compatibility controls
                "start_at_step": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "Start denoising at this step index (−1 = start from 0)."}),
                "end_at_step": ("INT", {"default": -1, "min": -1, "max": 10000, "tooltip": "End denoising at this step index inclusive (−1 = use full schedule). When set and less than last step, forces final sigma to 0."}),
                "add_noise": ("FLOAT", {
                    "default": 0.0,
                    "min": 0.0,
                    "max": 1.0,
                    "step": 0.01,
                    "tooltip": "Add per-step stochastic noise. 0.0 = no noise (deterministic). 1.0 = maximum ancestral noise for the step. Values in between scale the amount of new noise injected each step."
                }),
                "noise_type": (["whitened", "gaussian"], {
                    "default": "whitened",
                    "tooltip": "Noise sampler for add_noise: 'whitened' (res4lyf style: normalize to unit variance each step) or 'gaussian' (official KSampler style: raw randn). Ratio still controls σ_up."
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Verbose debug logging (per-step sampler details). Timing is always shown."}),
                # Place control at the bottom as requested
                "no_grad": ("BOOLEAN", {"default": True, "tooltip": "Run sampling under torch.no_grad (Comfy parity). Disable only for debugging/experiments."}),
                "official_comfy": ("BOOLEAN", {"default": True, "tooltip": "When enabled, use algorithms that mirror official Comfy samplers/schedulers; when disabled, use res4lyf or other variants."}),
                "sigma_aware": ("BOOLEAN", {"default": False, "tooltip": "Use actual sigma coordinates for extrapolation instead of assuming uniform step spacing. May improve prediction accuracy with non-uniform schedulers (karras, bong_tangent, exponential, etc.)."}),
                "extrapolate_denoised": ("BOOLEAN", {"default": False, "tooltip": "Extrapolate the model's denoised output instead of epsilon. Denoised converges smoothly toward the clean image, potentially improving skip predictions."}),
            },
        }

    RETURN_TYPES = ("LATENT", "STRING")
    RETURN_NAMES = ("samples", "metadata")
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               scheduler, sampler, adaptive_mode, smoothing_beta, skip_mode, skip_indices, anchor_interval, max_consecutive_skips, protect_first_steps, protect_last_steps, start_at_step, end_at_step, add_noise, noise_type, denoise, verbose, no_grad, official_comfy, sigma_aware=False, extrapolate_denoised=False):
        # Build sigma schedule with a factory to mirror KSampler (Advanced) denoise semantics
        def _build_sigmas(n_steps: int):
            if scheduler == "bong_tangent":
                # Route based on official_comfy flag
                if official_comfy:
                    from .comfy_copy.official_schedulers import get_bong_tangent_sigmas_official
                    s = get_bong_tangent_sigmas_official(model, n_steps)
                else:
                    from .comfy_copy.res4lyf_schedulers import get_bong_tangent_sigmas
                    s = get_bong_tangent_sigmas(model, n_steps)
                    # Ensure trailing zero for Comfy parity
                    try:
                        if float(s[-1]) != 0.0:
                            import torch as _torch
                            s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                    except Exception:
                        pass
                return s
            elif scheduler == "bong_tangent_2":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_sigmas
                s = get_bong_tangent_2_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "bong_tangent_2_simple":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_simple_sigmas
                s = get_bong_tangent_2_simple_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "constant":
                from .comfy_copy.res4lyf_schedulers import get_constant_sigmas
                s = get_constant_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            else:
                return get_sigma_schedule(model, scheduler, n_steps)

        # Compute sigmas with KSampler-compatible denoise behavior
        if denoise < 1.0:
            if denoise <= 0.0:
                return (latent_image,)
            # new_steps mirrors KSampler's int(steps/denoise)
            total_steps_for_schedule = int(steps / denoise)
            if total_steps_for_schedule <= 0:
                return (latent_image,)
            sigmas_full = _build_sigmas(total_steps_for_schedule)
            # Take the last (steps + 1) sigmas
            if len(sigmas_full) >= (steps + 1):
                sigmas = sigmas_full[-(steps + 1):]
            else:
                sigmas = sigmas_full
        else:
            sigmas = _build_sigmas(steps)

        if len(sigmas) == 0:
            return (latent_image,)

        # Apply end_at_step slicing (inclusive) similar to KSampler Advanced
        if end_at_step is not None and end_at_step >= 0:
            if end_at_step < (len(sigmas) - 1):
                sigmas = sigmas[:end_at_step + 1]
                # Force final sigma to 0 to end denoising at this step
                if len(sigmas) > 0:
                    sigmas[-1] = 0.0

        # Apply start_at_step slicing similar to KSampler Advanced
        if start_at_step is not None and start_at_step >= 0:
            if start_at_step < (len(sigmas) - 1):
                sigmas = sigmas[start_at_step:]
            else:
                # Nothing to do if starting beyond final denoise; return input latent
                return (latent_image,)

        # Prepare latent (ensure correct channels/dimensions for model)
        latent = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])

        # Generate noise
        torch.manual_seed(seed)
        noise = torch.randn_like(latent)

        # Preview callback (live image preview during sampling)
        total_steps = max(0, len(sigmas) - 1)
        preview_callback = _make_safe_preview_callback(model, total_steps)

        timestamp_start = time.time()

        samples = sample_fsampler(
            model_patcher=model,
            noise=noise,
            sigmas=sigmas,
            positive_conditioning=positive,
            negative_conditioning=negative,
            cfg_scale=cfg,
            latent_image=latent,
            sampler=sampler,
            adaptive_mode=adaptive_mode,
            smoothing_beta=smoothing_beta,
            skip_mode=skip_mode,
            skip_indices=skip_indices,
            add_noise_ratio=add_noise,
            add_noise_type=noise_type,
            scheduler=scheduler,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            denoise=denoise,
            debug=bool(verbose),
            callback=preview_callback,
            protect_first_steps=protect_first_steps,
            protect_last_steps=protect_last_steps,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
            use_no_grad=bool(no_grad),
            official_comfy=bool(official_comfy),
            seed=seed,
            timestamp_start=timestamp_start,
            sigma_aware=bool(sigma_aware),
            extrapolate_denoised=bool(extrapolate_denoised),
        )

        # Unpack metadata if returned (when verbose=True)
        if isinstance(samples, tuple):
            samples, metadata = samples
        else:
            metadata = {}

        # Convert metadata to JSON string
        import json
        metadata_json = json.dumps(metadata) if metadata else ""

        return ({"samples": samples}, metadata_json)


class FSampler:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL",),
                "positive": ("CONDITIONING",),
                "negative": ("CONDITIONING",),
                "latent_image": ("LATENT",),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "cfg": ("FLOAT", {"default": 1.00, "min": 0.00, "max": 100.00, "step": 0.05}),
                "scheduler": (FSAMPLER_AVAILABLE_SCHEDULERS, {"default": "simple"}),
                "sampler": (["euler", "res_2m", "res_2s", "ddim", "dpmpp_2m", "dpmpp_2s", "lms", "res_multistep", "res_multistep_ancestral", "heun", "gradient_estimation"], {"default": "euler"}),
                "skip_mode": ([
                    "none",
                    # History 2 (linear): K in 2..5
                    "h2/s2", "h2/s3", "h2/s4", "h2/s5",
                    # History 3 (Richardson): K in 3..5
                    "h3/s3", "h3/s4", "h3/s5",
                    # History 4 (cubic): K in 4..5
                    "h4/s4", "h4/s5",
                    "adaptive"
                ], {"default": "none"}),
                "add_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.05, "tooltip": "Ancestral noise ratio (eta). 0.0 = ODE (no noise), >0 enables ancestral noise with dynamic sigma_up each step. Noise type is fixed to gaussian in this simple node."}),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Denoise fraction (KSampler-style). 1.0 = full schedule; <1.0 samples the last portion of the schedule only."}),
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               scheduler, sampler, skip_mode, add_noise, denoise, verbose):
        # Build sigmas (denoise hardcoded to 1.0)
        def _build_sigmas(n_steps: int):
            if scheduler == "bong_tangent":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_sigmas
                s = get_bong_tangent_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "bong_tangent_2":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_sigmas
                s = get_bong_tangent_2_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "bong_tangent_2_simple":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_simple_sigmas
                s = get_bong_tangent_2_simple_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "constant":
                from .comfy_copy.res4lyf_schedulers import get_constant_sigmas
                s = get_constant_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        import torch as _torch
                        s = _torch.cat([s, _torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            else:
                return get_sigma_schedule(model, scheduler, n_steps)

        # Compute sigmas with KSampler-compatible denoise behavior
        if denoise < 1.0:
            if denoise <= 0.0:
                return (latent_image,)
            total_steps_for_schedule = int(steps / denoise)
            if total_steps_for_schedule <= 0:
                return (latent_image,)
            sigmas_full = _build_sigmas(total_steps_for_schedule)
            # Take the last (steps + 1) sigmas
            if len(sigmas_full) >= (steps + 1):
                sigmas = sigmas_full[-(steps + 1):]
            else:
                sigmas = sigmas_full
        else:
            sigmas = _build_sigmas(steps)
        if len(sigmas) == 0:
            return (latent_image,)

        # Prepare latent and noise
        latent = comfy.sample.fix_empty_latent_channels(model, latent_image["samples"])

        torch.manual_seed(seed)
        noise = torch.randn_like(latent)

        # Simple defaults
        adaptive_mode = "learning"
        smoothing_beta = 0.9999
        protect_first_steps = 2
        protect_last_steps = 3
        add_noise_ratio = float(add_noise)
        add_noise_type = "gaussian"
        denoise = float(denoise)
        anchor_interval = 4
        max_consecutive_skips = 4

        # Preview callback (live image preview during sampling)
        total_steps = max(0, len(sigmas) - 1)
        preview_callback = _make_safe_preview_callback(model, total_steps)

        samples = sample_fsampler(
            model_patcher=model,
            noise=noise,
            sigmas=sigmas,
            positive_conditioning=positive,
            negative_conditioning=negative,
            cfg_scale=cfg,
            latent_image=latent,
            sampler=sampler,
            adaptive_mode=adaptive_mode,
            smoothing_beta=smoothing_beta,
            skip_mode=skip_mode,
            add_noise_ratio=add_noise_ratio,
            add_noise_type=add_noise_type,
            scheduler=scheduler,
            start_at_step=None,
            end_at_step=None,
            denoise=denoise,
            debug=bool(verbose),
            callback=preview_callback,
            protect_first_steps=protect_first_steps,
            protect_last_steps=protect_last_steps,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
            use_no_grad=True,  # default to Comfy parity in simple node
            official_comfy=True,
            sigma_aware=True,
            extrapolate_denoised=True,
        )

        # Unpack metadata if returned (sample_fsampler always returns tuple)
        if isinstance(samples, tuple):
            samples, metadata = samples

        return ({"samples": samples},)


class FSamplerSelect:
    """Outputs SAMPLER + SIGMAS for use with KSamplerCustom (SamplerCustom).

    Plug SAMPLER into KSamplerCustom's 'sampler' input and SIGMAS into 'sigmas'.
    KSamplerCustom handles model, conditioning, cfg, noise, and latent.
    """

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": ("MODEL", {"tooltip": "Model (needed to compute sigma schedule)."}),
                "steps": ("INT", {"default": 25, "min": 1, "max": 150}),
                "scheduler": (FSAMPLER_AVAILABLE_SCHEDULERS, {"default": "simple"}),
                "sampler": (["euler", "res_2m", "res_2s", "ddim", "dpmpp_2m", "dpmpp_2s", "lms", "res_multistep", "res_multistep_ancestral", "heun", "gradient_estimation"], {
                    "default": "euler",
                    "tooltip": "Sampling method"
                }),
                "denoise": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "skip_mode": ([
                    "none",
                    "h2/s2", "h2/s3", "h2/s4", "h2/s5",
                    "h3/s3", "h3/s4", "h3/s5",
                    "h4/s4", "h4/s5",
                    "adaptive"
                ], {
                    "default": "none",
                    "tooltip": "Skipping: hN/sK with N=history (2=linear,3=Richardson,4=cubic) and K=calls before skip."
                }),
                "skip_indices": ("STRING", {"default": "", "multiline": False, "tooltip": "Explicit skip indices (e.g. 'h2, 3, 4, 7'). Overrides skip_mode when non-empty."}),
                "adaptive_mode": (["none", "learning", "grad_est", "learn+grad_est"], {
                    "default": "none",
                    "tooltip": "Adaptive skip corrections: none=off; learning=EMA L stabilizer; grad_est=gradient-estimation correction; learn+grad_est=both."
                }),
                "smoothing_beta": ("FLOAT", {
                    "default": 0.9990,
                    "min": 0.0,
                    "max": 0.9999,
                    "step": 0.0001,
                    "tooltip": "EMA smoothing for learning mode."
                }),
                "protect_first_steps": ("INT", {"default": 2, "min": 0, "max": 20, "tooltip": "Initial steps never skipped (warmup)."}),
                "protect_last_steps": ("INT", {"default": 2, "min": 1, "max": 100, "tooltip": "Final steps never skipped."}),
                "anchor_interval": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Force REAL call every Nth step (adaptive only). 0=disable."}),
                "max_consecutive_skips": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "Max back-to-back skips (adaptive only)."}),
                "add_noise": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01, "tooltip": "Per-step stochastic noise ratio. 0=deterministic."}),
                "noise_type": (["whitened", "gaussian"], {"default": "whitened", "tooltip": "Noise sampling method for add_noise."}),
                "verbose": ("BOOLEAN", {"default": False, "tooltip": "Verbose debug logging."}),
                "no_grad": ("BOOLEAN", {"default": True, "tooltip": "Run under torch.no_grad (Comfy parity)."}),
                "official_comfy": ("BOOLEAN", {"default": True, "tooltip": "Use official Comfy algorithm variants."}),
                "sigma_aware": ("BOOLEAN", {"default": False, "tooltip": "Use actual sigma coordinates for extrapolation instead of assuming uniform step spacing. May improve prediction accuracy with non-uniform schedulers (karras, bong_tangent, exponential, etc.)."}),
                "extrapolate_denoised": ("BOOLEAN", {"default": False, "tooltip": "Extrapolate the model's denoised output instead of epsilon. Denoised converges smoothly toward the clean image, potentially improving skip predictions."}),
            }
        }

    RETURN_TYPES = ("SAMPLER", "SIGMAS")
    RETURN_NAMES = ("sampler", "sigmas")
    FUNCTION = "get_sampler"
    CATEGORY = "sampling/custom_sampling"

    def get_sampler(self, model, steps, scheduler, sampler, denoise,
                    skip_mode, skip_indices, adaptive_mode, smoothing_beta,
                    protect_first_steps, protect_last_steps,
                    anchor_interval, max_consecutive_skips,
                    add_noise, noise_type, verbose, no_grad, official_comfy,
                    sigma_aware=False, extrapolate_denoised=False):

        # --- Build sigmas ---
        def _build_sigmas(n_steps):
            if scheduler == "bong_tangent":
                if official_comfy:
                    from .comfy_copy.official_schedulers import get_bong_tangent_sigmas_official
                    s = get_bong_tangent_sigmas_official(model, n_steps)
                else:
                    from .comfy_copy.res4lyf_schedulers import get_bong_tangent_sigmas
                    s = get_bong_tangent_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        s = torch.cat([s, torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "bong_tangent_2":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_sigmas
                s = get_bong_tangent_2_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        s = torch.cat([s, torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "bong_tangent_2_simple":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_simple_sigmas
                s = get_bong_tangent_2_simple_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        s = torch.cat([s, torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            elif scheduler == "constant":
                from .comfy_copy.res4lyf_schedulers import get_constant_sigmas
                s = get_constant_sigmas(model, n_steps)
                try:
                    if float(s[-1]) != 0.0:
                        s = torch.cat([s, torch.tensor([0.0], dtype=s.dtype)])
                except Exception:
                    pass
                return s
            else:
                return get_sigma_schedule(model, scheduler, n_steps)

        if denoise < 1.0 and denoise > 0.0:
            total_steps_for_schedule = int(steps / denoise)
            if total_steps_for_schedule <= 0:
                total_steps_for_schedule = 1
            sigmas_full = _build_sigmas(total_steps_for_schedule)
            if len(sigmas_full) >= (steps + 1):
                sigmas = sigmas_full[-(steps + 1):]
            else:
                sigmas = sigmas_full
        else:
            sigmas = _build_sigmas(steps)

        # --- Build KSAMPLER ---
        timestamp_start = time.time()
        ksampler = create_fsampler_ksampler(
            sampler=sampler,
            adaptive_mode=adaptive_mode,
            smoothing_beta=smoothing_beta,
            skip_mode=skip_mode,
            skip_indices=skip_indices,
            add_noise_ratio=add_noise,
            add_noise_type=noise_type,
            scheduler=scheduler,
            denoise=denoise,
            debug=bool(verbose),
            protect_first_steps=protect_first_steps,
            protect_last_steps=protect_last_steps,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
            use_no_grad=bool(no_grad),
            official_comfy=bool(official_comfy),
            timestamp_start=timestamp_start,
            sigma_aware=bool(sigma_aware),
            extrapolate_denoised=bool(extrapolate_denoised),
        )

        return (ksampler, sigmas)


NODE_CLASS_MAPPINGS = {
    "FSamplerAdvanced": FSamplerAdvanced,
    "FSampler": FSampler,
    "FSamplerSelect": FSamplerSelect,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSamplerAdvanced": "FSampler Advanced",
    "FSampler": "FSampler",
    "FSamplerSelect": "FSampler Select",
}
