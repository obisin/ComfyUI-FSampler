import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def _to_d(x, sigma, denoised):
    """Converts a denoiser output to a Karras ODE derivative."""
    # Broadcast-safe: sigma is a scalar tensor
    return (x - denoised) / (sigma + 1e-8)


def _get_ancestral_step(sigma_from, sigma_to, eta=1.):
    """Calculates sigma_down and sigma_up for ancestral sampling."""
    if not eta:
        return sigma_to, torch.zeros_like(sigma_to)
    sigma_up = min(sigma_to, eta * (sigma_to ** 2 * (sigma_from ** 2 - sigma_to ** 2) / sigma_from ** 2) ** 0.5)
    sigma_down = (sigma_to ** 2 - sigma_up ** 2) ** 0.5
    return sigma_down, sigma_up


def sample_step_res_multistep_official(
    model,
    noisy_latent,
    sigma_current,
    sigma_next,
    sigma_previous,
    old_sigma_down,
    s_in,
    extra_args,
    error_history,
    epsilon_history,
    step_index,
    total_steps,
    learning_ratio,
    smoothing_beta,
    predictor_type,
    add_noise_ratio=0.0,
    add_noise_type="whitened",
    skip_mode="none",
    skip_stats=None,
    debug=False,
    protect_last_steps=4,
    protect_first_steps=2,
    anchor_interval=None,
    max_consecutive_skips=None,
    adaptive_mode="none",
    explicit_skip_indices=None,
    explicit_predictor=None,
):
    """
    Official ComfyUI res_multistep implementation with FSampler skip integration.

    Based on official reference code using phi functions and exponential integrators
    from https://arxiv.org/pdf/2308.02157
    """
    x = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Phi functions (exponential integrators)
    sigma_fn = lambda t: t.neg().exp()
    t_fn = lambda sigma: sigma.log().neg()
    phi1_fn = lambda t: torch.expm1(t) / t
    phi2_fn = lambda t: (phi1_fn(t) - 1.0) / t

    # Explicit skip indices take precedence (ignores protect windows); apply streak gating
    was_skipped = False
    if explicit_skip_indices is not None and isinstance(explicit_skip_indices, set) and step_index in explicit_skip_indices:
        es = skip_stats.get("explicit_streak", False) if skip_stats is not None else False
        nl = skip_stats.get("needed_learns", 2) if skip_stats is not None else 2
        allowed_by_streak = es or (nl <= 0)
        if allowed_by_streak and len(epsilon_history) >= 2:
            pred = (explicit_predictor or "linear")
            if pred == "h4" and len(epsilon_history) >= 4:
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
            elif (pred in ("richardson", "h3")) and len(epsilon_history) >= 3:
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
            ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon_hat, prev_eps)
            if ok:
                if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                    epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
                denoised = x + epsilon_hat
                was_skipped = True
                if skip_stats is not None:
                    skip_stats["skipped"] = skip_stats.get("skipped", 0) + 1
                    skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
                    skip_stats["explicit_streak"] = True
                    skip_stats["needed_learns"] = 0
                if debug:
                    # Summary line consistent with Euler
                    try:
                        dt_val = float((sigma_next - sigma_current).item()) if hasattr(sigma_current, 'item') else float(sigma_next - sigma_current)
                    except Exception:
                        dt_val = float('nan')
                    print(f"res_multistep_official step {step_index} [SKIPPED-explicit-{pred if pred != 'richardson' else 'h3'}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                    try:
                        x_rms = float(torch.sqrt(torch.mean((denoised)**2)).item())
                    except Exception:
                        x_rms = None
                    print_step_diag(
                        sampler="res_multistep_official",
                        step_index=step_index,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        target_sigma=sigma_next,
                        sigma_up=None,
                        alpha_ratio=None,
                        h=None,
                        c2=None,
                        b1=None,
                        b2=None,
                        eps_norm=hat_norm,
                        eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
                        x_rms=x_rms,
                        flags=f"SKIPPED-explicit-{pred if pred != 'richardson' else 'h3'}",
                    )
            else:
                if debug:
                    print(f"res_multistep_official step {step_index}: explicit skip cancelled (ε̂ invalid: {reason})")
        else:
            if debug:
                reason = "need_two_learns_before_skip" if not (es or nl <= 0) else "insufficient_history"
                print(f"res_multistep_official step {step_index}: explicit skip gated ({reason})")

    # Decide skip (only if not explicitly skipped)
    if (not was_skipped) and skip_mode == "adaptive":
        should_skip, epsilon_hat, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=x,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="res_multistep",
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(
            1.0, step_index, total_steps, skip_mode, epsilon_history,
            protect_last_steps, protect_first_steps
        )
        epsilon_hat = None

    if (not was_skipped) and should_skip and skip_method is not None:
        # Extrapolate epsilon if not provided by adaptive
        if epsilon_hat is None:
            if skip_method == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            elif skip_method == "h4":
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)

        prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
        ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon_hat, prev_eps)
        if not ok:
            should_skip = False
            if debug:
                print(f"res_multistep_official step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            # Apply learning ratio correction (only for learning or learn+grad_est modes)
            if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            denoised = x + epsilon_hat
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
            if debug:
                if skip_mode == "adaptive":
                    rel = (meta.get("relative_error") if isinstance(meta, dict) else None)
                    print(f"res_multistep_official step {step_index} [SKIPPED-adaptive]: err_rel={(rel if rel is not None else float('nan')):.4f}, L={learning_ratio:.4f}")
                else:
                    print(f"res_multistep_official step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}")

    if not was_skipped and not should_skip:
        denoised = model(x, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index
            # Gating update: REAL call increments learns and may end explicit streak
            try:
                es = skip_stats.get("explicit_streak", False)
                nl = skip_stats.get("needed_learns", 2)
                if es:
                    skip_stats["explicit_streak"] = False
                    skip_stats["needed_learns"] = 1
                else:
                    skip_stats["needed_learns"] = max(0, int(nl) - 1)
            except Exception:
                pass

    # Ancestral step calculation
    sigma_down, sigma_up = _get_ancestral_step(sigma_current, sigma_next, eta=add_noise_ratio)

    # Store denoised for multistep history
    if not was_skipped:
        error_history.append(denoised)

    # Integration step
    if sigma_down == 0 or old_sigma_down is None or len(error_history) < 2:
        # First order Euler method
        d = _to_d(x, sigma_current, denoised)
        dt = sigma_down - sigma_current
        x = x + d * dt
        # Grad-estimation correction for SKIP steps (Euler-space)
        if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and skip_stats is not None:
            d_prev = skip_stats.get("d_prev")
            if d_prev is not None:
                d_hat = -(denoised - noisy_latent) / (sigma_current + 1e-8)
                dbar = (2.0 - 1.0) * (d_hat - d_prev)
                try:
                    ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    ratio = 0.0
                if ratio > 0.25:
                    dbar = dbar * (0.25 / ratio)
                x = x + dbar * dt
        # Grad-estimation correction for SKIP steps (Euler-space)
        if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and skip_stats is not None:
            d_prev = skip_stats.get("d_prev")
            if d_prev is not None:
                d_hat = -(denoised - noisy_latent) / (sigma_current + 1e-8)
                dbar = (2.0 - 1.0) * (d_hat - d_prev)
                try:
                    ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    ratio = 0.0
                if ratio > 0.25:
                    dbar = dbar * (0.25 / ratio)
                x = x + dbar * dt
    else:
        # Second order multistep method using phi functions
        t = t_fn(sigma_current)
        t_old = t_fn(old_sigma_down)
        t_next = t_fn(sigma_down)
        t_prev = t_fn(sigma_previous) if sigma_previous is not None else t

        h = t_next - t
        c2 = (t_prev - t_old) / (h + 1e-8)

        phi1_val = phi1_fn(-h)
        phi2_val = phi2_fn(-h)
        b1 = torch.nan_to_num(phi1_val - phi2_val / c2, nan=0.0)
        b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)

        old_denoised = error_history[-2] if len(error_history) >= 2 else denoised
        x = sigma_fn(h) * x + h * (b1 * denoised + b2 * old_denoised)
        # Post-integrator grad-estimation correction on SKIP
        if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and skip_stats is not None:
            d_prev = skip_stats.get("d_prev")
            if d_prev is not None:
                d_hat = -(denoised - noisy_latent) / (sigma_current + 1e-8)
                dbar = (2.0 - 1.0) * (d_hat - d_prev)
                try:
                    ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    ratio = 0.0
                if ratio > 0.25:
                    dbar = dbar * (0.25 / ratio)
                dt2 = sigma_down - sigma_current
                x = x + dbar * dt2
        # Grad-estimation correction for SKIP steps (post-integrator Euler-space tweak)
        if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and skip_stats is not None:
            d_prev = skip_stats.get("d_prev")
            if d_prev is not None:
                d_hat = -(denoised - noisy_latent) / (sigma_current + 1e-8)
                dbar = (2.0 - 1.0) * (d_hat - d_prev)
                try:
                    ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    ratio = 0.0
                if ratio > 0.25:
                    dbar = dbar * (0.25 / ratio)
                dt = sigma_down - sigma_current
                x = x + dbar * dt

    # Noise addition (ancestral)
    if float(sigma_next) > 0 and add_noise_ratio > 0:
        noise = torch.randn_like(x)
        if add_noise_type == "whitened":
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        x = x + noise * sigma_up

    # Update epsilon history and learning ratio
    if not was_skipped:
        epsilon = denoised - noisy_latent
        epsilon_history.append(epsilon)
        if len(epsilon_history) >= 3:
            if predictor_type == "h4":
                epsilon_pred = extrapolate_epsilon_h4(epsilon_history)
            elif predictor_type == "richardson":
                epsilon_pred = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_pred = extrapolate_epsilon_linear(epsilon_history)

            if epsilon_pred is not None:
                learn_obs = (torch.norm(epsilon_pred) / (torch.norm(epsilon) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                learning_ratio = max(0.5, min(2.0, learning_ratio))
        # Update last REAL slope for grad_est
        try:
            d_real = -(epsilon) / (sigma_current + 1e-8)
            if skip_stats is not None:
                skip_stats["d_prev"] = d_real.detach()
        except Exception:
            if skip_stats is not None:
                skip_stats["d_prev"] = d_real if 'd_real' in locals() else None

    if debug and not was_skipped:
        e_norm = torch.norm(epsilon).item() if not was_skipped else None
        try:
            dt_print = float((sigma_down - sigma_current).item()) if add_noise_ratio > 0 and 'sigma_down' in locals() else float((sigma_next - sigma_current).item()) if hasattr(sigma_current,'item') else float(sigma_next - sigma_current)
        except Exception:
            dt_print = float('nan')
        print(f"res_multistep_official step {step_index}: e_norm={e_norm:.2f}, dt={dt_print:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}, order={2 if old_sigma_down is not None and len(error_history) >= 2 else 1}")
    elif debug and was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and 'dbar' in locals() and isinstance(dbar, torch.Tensor):
        try:
            d_hat_norm = float(torch.norm(-(denoised - noisy_latent) / (sigma_current + 1e-8)).item())
            dbar_norm = float(torch.norm(dbar).item())
            ratio_print = dbar_norm / (d_hat_norm + 1e-8)
        except Exception:
            d_hat_norm = None; dbar_norm = 0.0; ratio_print = 0.0
        print(f"res_multistep_official step {step_index} [SKIP-APPLY]: d_norm={(d_hat_norm if d_hat_norm is not None else float('nan')):.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio_print:.2f}, L={learning_ratio:.4f}, mode={adaptive_mode}")

    return x, learning_ratio, sigma_down
