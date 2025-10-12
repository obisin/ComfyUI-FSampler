import time
import torch
from ..comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
import comfy.sample
from .log import print_step_timing
from .skip import _parse_hs_mode, parse_skip_indices_config
from .samplers.euler import sample_step_euler
from .samplers.res2m import sample_step_res_2m
from .samplers.res2s import sample_step_res_2s
from .samplers.ddim import sample_step_ddim
from .samplers.dpmpp_2m import sample_step_dpmpp_2m
from .samplers.dpmpp_2s import sample_step_dpmpp_2s
from .samplers.lms import sample_step_lms
from .samplers.res_multistep import sample_step_res_multistep
from .samplers.res_multistep_official import sample_step_res_multistep_official
from .samplers.res_multistep_ancestral import sample_step_res_multistep_ancestral
from .samplers.heun import sample_step_heun
from .samplers.gradient_estimation import sample_step_gradient_estimation


def sample_fsampler(model_patcher, noise, sigmas, positive_conditioning, negative_conditioning,
                    cfg_scale, latent_image, sampler="euler", adaptive_mode="none",
                    smoothing_beta=0.9, skip_mode="none", add_noise_ratio=0.0, add_noise_type="whitened",
                    scheduler=None, start_at_step=None, end_at_step=None, denoise=None,
                    debug=False, callback=None, protect_last_steps=4, protect_first_steps=2,
                    anchor_interval=None, max_consecutive_skips=None, use_no_grad=True,
                    official_comfy=False, skip_indices: str = ""):
    """Orchestrates sampling with pluggable samplers and shared skip/learning/timing.
    """

    def ksampler_function(model, x, sigmas, extra_args=None, callback=None, disable=None):
        # Allow runtime control of autograd to match official behavior
        import contextlib, torch as _torch
        ctx = _torch.no_grad() if use_no_grad else contextlib.nullcontext()

        with ctx:
            extra_args = {} if extra_args is None else extra_args
            s_in = x.new_ones([x.shape[0]])
            error_history = []          # RES2M denoised history (REAL + SKIPPED)
            epsilon_history = []        # REAL-only epsilon history for extrapolation + learning
            sigma_previous = None
            smoothed_error_ratio = 1.0  # For RES2M Phase-1 adaptive weights
            learning_ratio = 1.0        # Universal learning stabilizer

            # Normalize predictor selection using history from skip_mode (hN/sK or legacy aliases)
            _skip_mode_l = str(skip_mode).lower() if isinstance(skip_mode, str) else "none"
            hs = _parse_hs_mode(_skip_mode_l)
            if hs is not None:
                history_order, _ = hs
                if history_order >= 4:
                    predictor_type = "h4"
                elif history_order == 3:
                    predictor_type = "richardson"
                else:
                    predictor_type = "linear"
            else:
                if _skip_mode_l in ("h4", ):  # 4-point predictor
                    predictor_type = "h4"
                elif _skip_mode_l in ("h3", "richardson"):
                    predictor_type = "richardson"
                else:
                    predictor_type = "linear"

            skip_stats = {
                "total_steps": 0,
                "model_calls": 0,
                "skipped": 0,
                # Adaptive skip controller state
                "consecutive_skips": 0,
                "last_anchor_step": -1,
                # Explicit indices gating state
                "explicit_streak": False,
                "needed_learns": 2,
            }

            if debug:
                print(f"\n{'='*60}")
                print(f"FSampler Settings:")
                print(f"  sampler: {sampler}")
                if scheduler is not None:
                    print(f"  scheduler: {scheduler}")
                print(f"  adaptive_mode: {adaptive_mode}")
                print(f"  smoothing_beta: {smoothing_beta}")
            print(f"  skip_mode: {skip_mode}")
            print(f"  add_noise_ratio: {add_noise_ratio}")
            print(f"  noise_type: {add_noise_type}")
            print(f"  official_comfy: {official_comfy}")
            if skip_mode != "none":
                print(f"  protect_first_steps: {protect_first_steps}")
                print(f"  protect_last_steps: {protect_last_steps}")
            if denoise is not None:
                print(f"  denoise: {denoise}")
                if start_at_step is not None:
                    print(f"  start_at_step: {start_at_step}")
                if end_at_step is not None:
                    print(f"  end_at_step: {end_at_step}")
            print(f"  steps: {len(sigmas)-1}")
            try:
                import torch as _torch
                s = sigmas
                if isinstance(s, _torch.Tensor):
                    s0 = float(s[0]); sL = float(s[-1])
                    f3 = [float(v) for v in s[:3]]
                    l3 = [float(v) for v in s[-3:]]
                else:
                    s0 = float(s[0]); sL = float(s[-1])
                    f3 = [float(v) for v in s[:3]]
                    l3 = [float(v) for v in s[-3:]]
                print(f"  sigma_range: [{s0:.4f}, {sL:.4f}] len={len(s)}")
                print(f"  sigmas head: {[round(v,4) for v in f3]} tail: {[round(v,4) for v in l3]}")
            except Exception:
                pass
                print(f"{'='*60}\n")

            # Parse explicit skip indices early (indices bounded once total_steps is known)
            explicit_predictor, explicit_indices = parse_skip_indices_config(skip_indices or "")
            explicit_mode = len(explicit_indices) > 0

            res2m_prev_was_skipped = False
            res2m_noise_cooldown = 0
            res2m_prev_sigma_down = None
            resms_prev_sigma_down = None
            _t_start = time.time()
            total_steps = len(sigmas) - 1
            # Determine effective modes and bound explicit indices
            if explicit_mode:
                # Bound and filter per-total-steps; also ensure we never include 0/1
                explicit_indices = {i for i in explicit_indices if 0 <= i < total_steps and i >= 2}
                if len(explicit_indices) == 0:
                    explicit_mode = False
            effective_skip_mode = ("none" if explicit_mode else skip_mode)
            effective_adaptive_mode = ("none" if explicit_mode else adaptive_mode)
            if explicit_mode:
                # Override predictor used for learning/hints
                predictor_type = explicit_predictor
                if debug:
                    print(f"Explicit Skip Mode: ON")
                    print(f"  explicit_predictor: {explicit_predictor}")
                    print(f"  explicit_indices: {sorted(list(explicit_indices))}")
                    print(f"  disabled: skip_mode/adaptive/anchor/max_consecutive/protect_*")
            for step_index in range(total_steps):
                sigma_current = sigmas[step_index]
                sigma_next = sigmas[step_index + 1]

                print_step_timing(sampler, step_index, _t_start, total_steps)

                if sampler == "res_2m":
                    x, smoothed_error_ratio, learning_ratio, res2m_prev_was_skipped, res2m_noise_cooldown, res2m_prev_sigma_down = sample_step_res_2m(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        sigma_previous=sigma_previous,
                        s_in=s_in,
                        extra_args=extra_args,
                        error_history=error_history,
                        epsilon_history=epsilon_history,
                        prev_was_skipped=res2m_prev_was_skipped,
                        step_index=step_index,
                        total_steps=total_steps,
                        adaptive_mode=effective_adaptive_mode,
                        smoothing_beta=smoothing_beta,
                        smoothed_error_ratio=smoothed_error_ratio,
                        learning_ratio=learning_ratio,
                        predictor_type=predictor_type,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        noise_cooldown=res2m_noise_cooldown,
                        old_sigma_down=res2m_prev_sigma_down,
                    )
                elif sampler == "res_2s":
                    x, learning_ratio = sample_step_res_2s(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        debug=debug,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "euler":
                    x, learning_ratio = sample_step_euler(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "ddim":
                    x, learning_ratio = sample_step_ddim(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "dpmpp_2m":
                    x, learning_ratio = sample_step_dpmpp_2m(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        sigma_previous=sigma_previous,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "dpmpp_2s":
                    x, learning_ratio = sample_step_dpmpp_2s(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "lms":
                    x, learning_ratio = sample_step_lms(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        sigma_previous=sigma_previous,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "heun":
                    x, learning_ratio = sample_step_heun(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "gradient_estimation":
                    x, learning_ratio = sample_step_gradient_estimation(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        sigma_previous=sigma_previous,
                        s_in=s_in,
                        extra_args=extra_args,
                        epsilon_history=epsilon_history,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        step_index=step_index,
                        total_steps=total_steps,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        protect_last_steps=protect_last_steps,
                        debug=debug,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                elif sampler == "res_multistep":
                    # Choose official vs res4lyf implementation
                    if official_comfy:
                        try:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_official(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                                explicit_skip_indices=(explicit_indices if explicit_mode else None),
                                explicit_predictor=(explicit_predictor if explicit_mode else None),
                            )
                        except TypeError:
                            # Back-compat with older function signature
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_official(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                            )
                    else:
                        try:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                                explicit_skip_indices=(explicit_indices if explicit_mode else None),
                                explicit_predictor=(explicit_predictor if explicit_mode else None),
                            )
                        except TypeError:
                            # Back-compat with older function signature
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                            )
                elif sampler == "res_multistep_ancestral":
                    # Dedicated ancestral variant; always call the wrapper and continue
                    x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_ancestral(
                        model=model,
                        noisy_latent=x,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        sigma_previous=sigma_previous,
                        old_sigma_down=resms_prev_sigma_down,
                        s_in=s_in,
                        extra_args=extra_args,
                        error_history=error_history,
                        epsilon_history=epsilon_history,
                        step_index=step_index,
                        total_steps=total_steps,
                        learning_ratio=learning_ratio,
                        smoothing_beta=smoothing_beta,
                        predictor_type=predictor_type,
                        add_noise_ratio=add_noise_ratio,
                        add_noise_type=add_noise_type,
                        skip_mode=effective_skip_mode,
                        skip_stats=skip_stats,
                        debug=debug,
                        protect_last_steps=protect_last_steps,
                        protect_first_steps=protect_first_steps,
                        anchor_interval=anchor_interval,
                        max_consecutive_skips=max_consecutive_skips,
                        official_comfy=official_comfy,
                        adaptive_mode=effective_adaptive_mode,
                        explicit_skip_indices=(explicit_indices if explicit_mode else None),
                        explicit_predictor=(explicit_predictor if explicit_mode else None),
                    )
                    continue
                    if official_comfy:
                        try:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_official(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                                explicit_skip_indices=(explicit_indices if explicit_mode else None),
                                explicit_predictor=(explicit_predictor if explicit_mode else None),
                            )
                        except TypeError:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_official(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                            )
                    # else (removed duplicate)
                        try:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                                explicit_skip_indices=(explicit_indices if explicit_mode else None),
                                explicit_predictor=(explicit_predictor if explicit_mode else None),
                            )
                        except TypeError:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                            )
                        except TypeError:
                            # Back-compat with older function signature
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep_official(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                            )
                    else:
                        try:
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                                explicit_skip_indices=(explicit_indices if explicit_mode else None),
                                explicit_predictor=(explicit_predictor if explicit_mode else None),
                            )
                        except TypeError:
                            # Back-compat with older function signature
                            x, learning_ratio, resms_prev_sigma_down = sample_step_res_multistep(
                                model=model,
                                noisy_latent=x,
                                sigma_current=sigma_current,
                                sigma_next=sigma_next,
                                sigma_previous=sigma_previous,
                                old_sigma_down=resms_prev_sigma_down,
                                s_in=s_in,
                                extra_args=extra_args,
                                error_history=error_history,
                                epsilon_history=epsilon_history,
                                step_index=step_index,
                                total_steps=total_steps,
                                learning_ratio=learning_ratio,
                                smoothing_beta=smoothing_beta,
                                predictor_type=predictor_type,
                                add_noise_ratio=add_noise_ratio,
                                add_noise_type=add_noise_type,
                                skip_mode=effective_skip_mode,
                                skip_stats=skip_stats,
                                debug=debug,
                                protect_last_steps=protect_last_steps,
                                protect_first_steps=protect_first_steps,
                                anchor_interval=anchor_interval,
                                max_consecutive_skips=max_consecutive_skips,
                                adaptive_mode=effective_adaptive_mode,
                            )

                sigma_previous = sigma_current

                if callback is not None:
                    callback({'x': x, 'i': step_index, 'sigma': sigma_current, 'sigma_next': sigma_next, 'denoised': x})

            if debug and (effective_skip_mode != "none" or explicit_mode):
                total = skip_stats["total_steps"]
                called = skip_stats["model_calls"]
                skipped = skip_stats["skipped"]
                if total > 0:
                    skip_percent = (skipped / total) * 100
                    print(f"\n{'='*60}")
                    print(f"Skip Statistics:")
                    print(f"  Total steps: {total}")
                    print(f"  Model calls: {called}")
                    print(f"  Skipped: {skipped}")
                    print(f"  Reduction: {skip_percent:.1f}%")
                    print(f"{'='*60}\n")

            return x

    from comfy.samplers import KSAMPLER
    wrapped_sampler = KSAMPLER(ksampler_function)

    samples = comfy.sample.sample_custom(
        model=model_patcher,
        noise=noise,
        cfg=cfg_scale,
        sampler=wrapped_sampler,
        sigmas=sigmas,
        positive=positive_conditioning,
        negative=negative_conditioning,
        latent_image=latent_image,
        noise_mask=None,
        callback=callback,
        disable_pbar=False,
        seed=None
    )

    return samples
