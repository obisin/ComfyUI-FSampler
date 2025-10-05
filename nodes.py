"""FSampler nodes for ComfyUI."""
import torch
import comfy.sample
import comfy.samplers
import comfy.utils
from .comfy_copy import k_diffusion_sampling
from .sampling.fibonacci_scheduler import get_fsampler_sigmas, FSAMPLER_SCHEDULERS
from .sampling.sampling_engine import sample_fsampler


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
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "scheduler": (FSAMPLER_AVAILABLE_SCHEDULERS, {"default": "simple"}),
                "sampler": (["euler", "res_2m", "res_2s", "ddim", "dpmpp_2m", "dpmpp_2s", "lms", "res_multistep"], {
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
                "adaptive_mode": (["none", "learning"], {
                    "default": "none",
                    "tooltip": "Weight adjustment: none=baseline weights only, learning=smoothed weight adaptation based on error history"
                }),
                "smoothing_beta": ("FLOAT", {
                    "default": 0.9,
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
                "anchor_interval": ("INT", {"default": 4, "min": 0, "max": 100, "tooltip": "Absolute cadence: force a REAL call on every Nth step index counted from the end of the protected warmup (protect_first_steps). Set 0 to disable anchors. (adaptive only)"}),
                "max_consecutive_skips": ("INT", {"default": 4, "min": 1, "max": 100, "tooltip": "Local cap: maximum back-to-back skips allowed; resets immediately after any REAL call. (adaptive only)"}),
                # NOTE: Adaptive tuning parameters (intentionally NOT exposed yet).
                # "adaptive_tol_rel": ("FLOAT", {"default": 0.10, "min": 0.0, "max": 1.0, "step": 0.001, "tooltip": "Adaptive gate tolerance (relative err in predicted state)."}),
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
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               scheduler, sampler, adaptive_mode, smoothing_beta, skip_mode, anchor_interval, max_consecutive_skips, protect_first_steps, protect_last_steps, start_at_step, end_at_step, add_noise, noise_type, denoise, verbose):
        # Build sigma schedule with a factory to mirror KSampler (Advanced) denoise semantics
        def _build_sigmas(n_steps: int):
            if scheduler == "bong_tangent":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_sigmas
                return get_bong_tangent_sigmas(model, n_steps)
            elif scheduler == "bong_tangent_2":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_sigmas
                return get_bong_tangent_2_sigmas(model, n_steps)
            elif scheduler == "bong_tangent_2_simple":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_simple_sigmas
                return get_bong_tangent_2_simple_sigmas(model, n_steps)
            elif scheduler == "constant":
                from .comfy_copy.res4lyf_schedulers import get_constant_sigmas
                return get_constant_sigmas(model, n_steps)
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

        # Progress bar
        total_steps = max(0, len(sigmas) - 1)
        progress_bar = comfy.utils.ProgressBar(total_steps)

        def progress_callback(i, denoised, x, total):
            try:
                if hasattr(progress_bar, 'update'):
                    progress_bar.update(1)
                elif hasattr(progress_bar, 'update_absolute'):
                    progress_bar.update_absolute(i + 1)
            except Exception:
                pass

        # Sample
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
            add_noise_ratio=add_noise,
            add_noise_type=noise_type,
            scheduler=scheduler,
            start_at_step=start_at_step,
            end_at_step=end_at_step,
            denoise=denoise,
            debug=bool(verbose),
            callback=progress_callback,
            protect_first_steps=protect_first_steps,
            protect_last_steps=protect_last_steps,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )


        return ({"samples": samples},)


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
                "cfg": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 100.0}),
                "scheduler": (FSAMPLER_AVAILABLE_SCHEDULERS, {"default": "fibonacci"}),
                "sampler": (["euler", "res_2m", "res_2s", "ddim", "dpmpp_2m", "dpmpp_2s", "lms", "res_multistep"], {"default": "euler"}),
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
                "verbose": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("LATENT",)
    FUNCTION = "sample"
    CATEGORY = "sampling/custom_sampling"

    def sample(self, model, positive, negative, latent_image, seed, steps, cfg,
               scheduler, sampler, skip_mode, verbose):
        # Build sigmas (denoise hardcoded to 1.0)
        def _build_sigmas(n_steps: int):
            if scheduler == "bong_tangent":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_sigmas
                return get_bong_tangent_sigmas(model, n_steps)
            elif scheduler == "bong_tangent_2":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_sigmas
                return get_bong_tangent_2_sigmas(model, n_steps)
            elif scheduler == "bong_tangent_2_simple":
                from .comfy_copy.res4lyf_schedulers import get_bong_tangent_2_simple_sigmas
                return get_bong_tangent_2_simple_sigmas(model, n_steps)
            elif scheduler == "constant":
                from .comfy_copy.res4lyf_schedulers import get_constant_sigmas
                return get_constant_sigmas(model, n_steps)
            else:
                return get_sigma_schedule(model, scheduler, n_steps)

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
        add_noise_ratio = 0.0
        add_noise_type = "gaussian"
        denoise = 1.0
        anchor_interval = 4
        max_consecutive_skips = 4

        # Progress bar
        total_steps = max(0, len(sigmas) - 1)
        progress_bar = comfy.utils.ProgressBar(total_steps)

        def progress_callback(i, denoised, x, total):
            try:
                if hasattr(progress_bar, 'update'):
                    progress_bar.update(1)
                elif hasattr(progress_bar, 'update_absolute'):
                    progress_bar.update_absolute(i + 1)
            except Exception:
                pass

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
            user_skip_indices=None,
            add_noise_ratio=add_noise_ratio,
            add_noise_type=add_noise_type,
            scheduler=scheduler,
            start_at_step=None,
            end_at_step=None,
            denoise=denoise,
            debug=bool(verbose),
            callback=progress_callback,
            protect_first_steps=protect_first_steps,
            protect_last_steps=protect_last_steps,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )

        return ({"samples": samples},)


NODE_CLASS_MAPPINGS = {
    "FSamplerAdvanced": FSamplerAdvanced,
    "FSampler": FSampler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FSamplerAdvanced": "FSampler Advanced",
    "FSampler": "FSampler",
}
