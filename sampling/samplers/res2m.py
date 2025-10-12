import math
import torch
from ..phi_functions import phi_function
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..log import print_step_diag
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive


def sample_step_res_2m(model, noisy_latent, sigma_current, sigma_next, sigma_previous,
                       s_in, extra_args, error_history, epsilon_history, prev_was_skipped, step_index, total_steps,
                       adaptive_mode="none", smoothing_beta=0.9, smoothed_error_ratio=1.0,
                       learning_ratio=1.0, predictor_type="linear", add_noise_ratio=0.0, add_noise_type="whitened",
                       skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2,
                       anchor_interval=None, max_consecutive_skips=None,
                       noise_cooldown=0, old_sigma_down=None):
    x_0 = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    if skip_mode == "adaptive":
        should_skip, epsilon_current, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=x_0,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="res_2m",
            sigma_previous=sigma_previous,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(smoothed_error_ratio, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon_current = None

    was_skipped = False
    cooldown_next = max(0, int(noise_cooldown) - 1)
    if should_skip and skip_method is not None:
        if epsilon_current is None:
            if skip_method == "richardson":
                epsilon_current = extrapolate_epsilon_richardson(epsilon_history)
            elif skip_method == "h4":
                epsilon_current = extrapolate_epsilon_h4(epsilon_history)
            else:
                epsilon_current = extrapolate_epsilon_linear(epsilon_history)
        # Bad-skip safety: validation + absurd magnitude guard
        bad_skip = False
        prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
        ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon_current, prev_eps)
        if not ok:
            bad_skip = True
        else:
            try:
                if prev_norm is not None and prev_norm > 0 and hat_norm > 50.0 * prev_norm:
                    bad_skip = True
                    reason = 'too_large_rel'
            except Exception:
                pass
        if bad_skip:
            should_skip = False
            if debug:
                print(f"res_2m step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                epsilon_current = epsilon_current / max(learning_ratio, 1e-8)
            denoised = x_0 + epsilon_current
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
            if debug:
                if skip_mode == "adaptive":
                    rel = (meta.get("relative_error") if isinstance(meta, dict) else None)
                    print(f"res_2m step {step_index} [SKIPPED-adaptive]: err_rel={(rel if rel is not None else float('nan')):.4f}, L={learning_ratio:.4f}")
                else:
                    print(f"res_2m step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}")

    # Determine model type once
    try:
        is_rf = isinstance(model.inner_model.inner_model.model_sampling, _model_sampling.CONST)
    except Exception:
        is_rf = False

    # Track whether we must force an Euler fallback for this step
    force_euler_step = False
    euler_reason = None

    if not should_skip:
        denoised = model(noisy_latent, sigma_current * s_in, **extra_args)
        epsilon_current = denoised - x_0
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index
        # Model/epsilon validity guard
        prev_eps_for_guard = epsilon_history[-1] if len(epsilon_history) >= 1 else None
        den_finite = torch.isfinite(denoised).all() and torch.isfinite(epsilon_current).all()
        if not den_finite:
            force_euler_step = True
            euler_reason = "model_invalid"
            cooldown_next = 1
            if prev_eps_for_guard is not None:
                epsilon_current = prev_eps_for_guard
                denoised = x_0 + epsilon_current
            else:
                epsilon_current = torch.zeros_like(x_0)
                denoised = x_0
        else:
            try:
                if prev_eps_for_guard is not None:
                    prevn = torch.norm(prev_eps_for_guard).item()
                    currn = torch.norm(epsilon_current).item()
                    if math.isfinite(prevn) and prevn > 0 and (not math.isfinite(currn) or currn > 50.0 * prevn):
                        force_euler_step = True
                        euler_reason = "epsilon_jump"
                        cooldown_next = 1
                        epsilon_current = prev_eps_for_guard
                        denoised = x_0 + epsilon_current
            except Exception:
                pass

    # Ancestral: integrate to sigma_down when enabled; otherwise use sigma_next
    sigma_up = None
    alpha_ratio = None
    target_sigma = sigma_next
    # Effective eta: disable noise when cooldown is active
    add_noise_ratio_eff = add_noise_ratio if cooldown_next == 0 else 0.0
    if add_noise_ratio_eff > 0.0 and float(sigma_next) > 0.0:
        sigma_up, _s, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
            model, sigma_current, sigma_next, add_noise_ratio_eff, "hard"
        )
        target_sigma = sigma_down

    h = -torch.log(target_sigma / sigma_current)
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next
    is_final_step = (sigma_next_value == 0)

    if len(error_history) >= 1 and sigma_previous is not None and old_sigma_down is not None and not is_final_step:
        denoised_previous = error_history[-1]
        epsilon_previous = denoised_previous - x_0

        # Reference geometry: use previous target (old_sigma_down) for c2
        t = -torch.log(sigma_current)
        t_next = -torch.log(target_sigma)
        h = t_next - t
        t_prev = -torch.log(sigma_previous)
        t_old = -torch.log(old_sigma_down)
        h_prev = t - t_prev  # only for logs if needed
        # Small-step guard
        h_abs = float(torch.abs(h)) if torch.is_tensor(h) else abs(h)
        if h_abs < 1e-8 or force_euler_step:
            # Euler fallback for tiny steps
            b1 = torch.tensor(1.0, dtype=h.dtype if torch.is_tensor(h) else torch.float32, device=x_0.device)
            b2 = torch.tensor(0.0, dtype=h.dtype if torch.is_tensor(h) else torch.float32, device=x_0.device)
            smoothed_error_ratio_next = 1.0
            error_ratio = None
            # Euler update directly
            x = x_0 + h * epsilon_current
            # Optional noise add (only if we weren't invalid earlier)
            if not force_euler_step and add_noise_ratio_eff > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
                if add_noise_type == "whitened":
                    noise = torch.randn_like(x)
                    noise = (noise - noise.mean()) / (noise.std() + 1e-12)
                else:
                    noise = torch.randn_like(x)
                if alpha_ratio is not None and alpha_ratio is not True:
                    x_try = alpha_ratio * x + noise * sigma_up
                else:
                    x_try = x + noise * sigma_up
                if torch.isfinite(x_try).all():
                    x = x_try
            if debug and (force_euler_step or h_abs < 1e-8):
                reason = euler_reason if force_euler_step else "small_h"
                print(f"res_2m step {step_index} ({reason}): using Euler")
            # Skip the multistep branch entirely
            error_history.append(denoised)
            if len(error_history) > 2:
                error_history.pop(0)
            if not was_skipped:
                epsilon_history.append(epsilon_current)
            if debug:
                try:
                    x_rms = float(torch.sqrt(torch.mean(x**2)).item())
                except Exception:
                    x_rms = None
                print_step_diag(
                    sampler="res_2m",
                    step_index=step_index,
                    sigma_current=sigma_current,
                    sigma_next=sigma_next,
                    target_sigma=target_sigma,
                    sigma_up=sigma_up,
                    alpha_ratio=alpha_ratio,
                    h=h,
                    c2=None,
                    b1=None,
                    b2=None,
                    eps_norm=float(torch.norm(epsilon_current).item()) if torch.is_tensor(epsilon_current) else None,
                    eps_prev_norm=float(torch.norm(epsilon_history[-1]).item()) if len(epsilon_history) >= 1 else None,
                    x_rms=x_rms,
                    flags=(euler_reason or "small_h"),
                )
            return x, smoothed_error_ratio_next, learning_ratio, was_skipped, cooldown_next
        else:
            c2_val = (t_prev - t_old) / h
            c2 = c2_val.item() if torch.is_tensor(c2_val) else float(c2_val)
            phi_1 = phi_function(order=1, step_size=-h)
            phi_2 = phi_function(order=2, step_size=-h)
            # Protect against pathological c2
            if not math.isfinite(c2) or abs(c2) < 1e-12:
                c2 = 1.0
            b2_base = phi_2 / c2
            b1_base = phi_1 - b2_base

        if h_abs >= 1e-8 and adaptive_mode != "none" and not was_skipped:
            error_curr = torch.norm(epsilon_current).item()
            error_prev = torch.norm(epsilon_previous).item()
            error_ratio_tmp = error_curr / (error_prev + 1e-8)
            error_ratio = error_ratio_tmp if math.isfinite(error_ratio_tmp) else 1.0

            if adaptive_mode == "learning":
                smoothed_error_ratio_next = (smoothing_beta * smoothed_error_ratio + (1 - smoothing_beta) * error_ratio)
                # Clamp to keep numerically sane
                smoothed_error_ratio_next = max(1e-3, min(1e3, smoothed_error_ratio_next))
                adjustment = 1.0 / smoothed_error_ratio_next
                adjustment = max(0.5, min(2.0, adjustment))
            else:
                adjustment = 1.0
                smoothed_error_ratio_next = 1.0

            b1_adjusted = b1_base * adjustment
            b2_adjusted = b2_base / adjustment
            sum_adjusted = b1_adjusted + b2_adjusted
            sum_target = b1_base + b2_base
            scale = sum_target / sum_adjusted
            b1 = b1_adjusted * scale
            b2 = b2_adjusted * scale
            # Coefficients sanity guard
            if not torch.isfinite(b1).all() or not torch.isfinite(b2).all() or (abs(b1.item()) + abs(b2.item())) > 1e4:
                x = x_0 + h * epsilon_current
                cooldown_next = 1
                if debug:
                    print(f"res_2m step {step_index}: coeffs invalid → Euler fallback")
                error_history.append(denoised)
                if len(error_history) > 2:
                    error_history.pop(0)
                if not was_skipped:
                    epsilon_history.append(epsilon_current)
                return x, smoothed_error_ratio_next, learning_ratio, was_skipped, cooldown_next
        elif h_abs >= 1e-8 and adaptive_mode != "none" and was_skipped:
            b1 = b1_base
            b2 = b2_base
            adjustment = 1.0
            smoothed_error_ratio_next = smoothed_error_ratio
            error_ratio = None
        elif h_abs >= 1e-8:
            b1 = b1_base
            b2 = b2_base
            adjustment = 1.0
            smoothed_error_ratio_next = 1.0
            error_ratio = None

        # Compute integrator update and validate before noise
        x_pre = x_0 + h * (b1 * epsilon_current + b2 * epsilon_previous)
        if not torch.isfinite(x_pre).all():
            # Re-anchor to Euler with a safe epsilon
            eps_safe = epsilon_previous if was_skipped else epsilon_current
            x = x_0 + h * eps_safe
            cooldown_next = 1
            if debug:
                print(f"res_2m step {step_index}: integrator invalid → Euler fallback")
            allow_noise = False
        else:
            x = x_pre
            allow_noise = True

        # Grad-estimation correction (post-integrator Euler-space tweak) on SKIP
        if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est"):
            try:
                d_prev_ge = skip_stats.get("d_prev") if skip_stats is not None else None
            except Exception:
                d_prev_ge = None
            if d_prev_ge is not None:
                d_hat = -(epsilon_current) / (sigma_current + 1e-8)
                dbar = (2.0 - 1.0) * (d_hat - d_prev_ge)
                try:
                    ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    ratio = 0.0
                if ratio > 0.25:
                    dbar = dbar * (0.25 / ratio)
                dt = target_sigma - sigma_current
                x = x + dbar * dt

        # Ancestral noise add after integrator (ODE and RF)
        if allow_noise and add_noise_ratio_eff > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
            if add_noise_type == "whitened":
                noise = torch.randn_like(x)
                noise = (noise - noise.mean()) / (noise.std() + 1e-12)
            else:
                noise = torch.randn_like(x)
            if alpha_ratio is not None and alpha_ratio is not True:
                x = alpha_ratio * x + noise * sigma_up
            else:
                x = x + noise * sigma_up
            # If noise add produced invalid state, revert to pre-noise x
            if not torch.isfinite(x).all():
                x = x_pre
                cooldown_next = 1

        if debug:
            eps_prev_norm = torch.norm(epsilon_previous).item()
            eps_curr_norm = torch.norm(epsilon_current).item()
            if was_skipped and 'dbar' in locals() and isinstance(dbar, torch.Tensor):
                try:
                    d_hat_norm = float(torch.norm(-(epsilon_current) / (sigma_current + 1e-8)).item())
                    dbar_norm = float(torch.norm(dbar).item())
                    ratio_print = dbar_norm / (d_hat_norm + 1e-8)
                except Exception:
                    d_hat_norm = None; dbar_norm = 0.0; ratio_print = 0.0
                print(f"res_2m step {step_index} [SKIP-APPLY]: d_norm={(d_hat_norm if d_hat_norm is not None else float('nan')):.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio_print:.2f}, L={learning_ratio:.4f}, mode={adaptive_mode}")
            if adaptive_mode != "none":
                if error_ratio is None:
                    print(
                        f"res_2m step {step_index} [learning] [EXTRAPOLATED]: baseline φ-weights (adaptive error_ratio preserved); "
                        f"ε̂ scaled by 1/L={learning_ratio:.4f}; b1={b1.item():.4f}, b2={b2.item():.4f}"
                    )
            else:
                print(f"res_2m step {step_index}: eps_prev_norm={eps_prev_norm:.2f}, eps_curr_norm={eps_curr_norm:.2f}, c2={c2:.4f}, b1={b1.item():.4f}, b2={b2.item():.4f}")
            try:
                x_rms = float(torch.sqrt(torch.mean(x**2)).item())
            except Exception:
                x_rms = None
            print_step_diag(
                sampler="res_2m",
                step_index=step_index,
                sigma_current=sigma_current,
                sigma_next=sigma_next,
                target_sigma=target_sigma,
                sigma_up=sigma_up,
                alpha_ratio=alpha_ratio,
                h=h,
                c2=c2,
                b1=b1,
                b2=b2,
                eps_norm=eps_curr_norm,
                eps_prev_norm=eps_prev_norm,
                x_rms=x_rms,
                flags="",
            )
    else:
        if is_final_step:
            x = denoised
            if debug:
                print(f"res_2m step {step_index} (final step): using Euler")
            reason = "final"
        else:
            if prev_was_skipped:
                reason = "post-skip reanchor"
            elif sigma_previous is None or len(error_history) == 0:
                reason = "first step"
            else:
                reason = "no-history reanchor"
            x = x_0 + h * epsilon_current
            # Grad-estimation correction on Euler fallback during SKIP
            if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est"):
                try:
                    d_prev_ge = skip_stats.get("d_prev") if skip_stats is not None else None
                except Exception:
                    d_prev_ge = None
                if d_prev_ge is not None:
                    d_hat = -(epsilon_current) / (sigma_current + 1e-8)
                    dbar = (2.0 - 1.0) * (d_hat - d_prev_ge)
                    try:
                        ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                    except Exception:
                        ratio = 0.0
                    if ratio > 0.25:
                        dbar = dbar * (0.25 / ratio)
                    dt = target_sigma - sigma_current
                    x = x + dbar * dt
                    if debug:
                        try:
                            d_hat_norm = float(torch.norm(d_hat).item())
                            dbar_norm = float(torch.norm(dbar).item())
                        except Exception:
                            d_hat_norm = None; dbar_norm = 0.0
                        print(f"res_2m step {step_index} [SKIP-APPLY]: d_norm={(d_hat_norm if d_hat_norm is not None else float('nan')):.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio:.2f}, L={learning_ratio:.4f}, mode={adaptive_mode}")
            if add_noise_ratio_eff > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
                if add_noise_type == "whitened":
                    noise = torch.randn_like(x)
                    noise = (noise - noise.mean()) / (noise.std() + 1e-12)
                else:
                    noise = torch.randn_like(x)
                if alpha_ratio is not None and alpha_ratio is not True:
                    x = alpha_ratio * x + noise * sigma_up
                else:
                    x = x + noise * sigma_up
            if debug:
                print(f"res_2m step {step_index} ({reason}): using Euler")
        smoothed_error_ratio_next = 1.0
        if debug:
            try:
                x_rms = float(torch.sqrt(torch.mean(x**2)).item())
            except Exception:
                x_rms = None
            print_step_diag(
                sampler="res_2m",
                step_index=step_index,
                sigma_current=sigma_current,
                sigma_next=sigma_next,
                target_sigma=target_sigma,
                sigma_up=sigma_up,
                alpha_ratio=alpha_ratio,
                h=h,
                c2=None,
                b1=None,
                b2=None,
                eps_norm=float(torch.norm(epsilon_current).item()) if torch.is_tensor(epsilon_current) else None,
                eps_prev_norm=float(torch.norm(epsilon_history[-1]).item()) if len(epsilon_history) >= 1 else None,
                x_rms=x_rms,
                flags=reason,
            )

    error_history.append(denoised)
    if len(error_history) > 2:
        error_history.pop(0)

    if not was_skipped:
        epsilon_history.append(epsilon_current)
        # Update last REAL slope for grad_est
        try:
            d_real = -(epsilon_current) / (sigma_current + 1e-8)
            if skip_stats is not None:
                skip_stats["d_prev"] = d_real.detach()
        except Exception:
            if skip_stats is not None:
                skip_stats["d_prev"] = d_real if 'd_real' in locals() else None
        if len(epsilon_history) >= 3:
            if predictor_type == "h4":
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
            elif predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                num = torch.norm(epsilon_hat)
                den = torch.norm(epsilon_current) + 1e-8
                learn_obs = (num / den).item()
                if math.isfinite(learn_obs):
                    learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                    if learning_ratio < 0.5:
                        learning_ratio = 0.5
                    elif learning_ratio > 2.0:
                        learning_ratio = 2.0
                if debug:
                    if adaptive_mode != "none" and 'error_ratio' in locals() and error_ratio is not None:
                        print(
                            f"res_2m step {step_index} [learning] [REAL]: "
                            f"err_ratio={error_ratio:.4f}, adjust={adjustment:.4f}, b1={b1.item():.4f}({b1_base.item():.4f}), "
                            f"b2={b2.item():.4f}({b2_base.item():.4f}) | learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}"
                        )
                    elif adaptive_mode != "none":
                        print(f"res_2m step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    return x, smoothed_error_ratio_next, learning_ratio, was_skipped, cooldown_next, target_sigma
