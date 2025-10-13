import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..noise import get_eps_step_official
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def sample_step_gradient_estimation(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                                    epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                                    step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2, anchor_interval=None, max_consecutive_skips=None, official_comfy=False,
                                    explicit_skip_indices=None, explicit_predictor=None, ge_gamma: float = 2.0):
    x = noisy_latent
    # Ensure commonly logged metrics are always defined
    x_rms = None

    if skip_stats is not None:
        skip_stats["total_steps"] = skip_stats.get("total_steps", 0) + 1

    # Final step guard: land on denoised
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else float(sigma_next)
    if abs(sigma_next_value) <= 1e-12:
        den = model(x, sigma_current * s_in, **extra_args)
        x = den
        eps_real = den - noisy_latent
        epsilon_history.append(eps_real)
        if skip_stats is not None:
            skip_stats["model_calls"] = skip_stats.get("model_calls", 0) + 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index
        if len(epsilon_history) >= 3:
            if predictor_type == "h4":
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
            elif predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps_real) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                learning_ratio = max(0.5, min(2.0, learning_ratio))
                if debug:
                    print(f"gradient_est step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        return x, learning_ratio

    # Target sigma and noise planning
    target_sigma = sigma_next
    sigma_up = None
    alpha_ratio = None
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0:
        if official_comfy:
            sigma_up, sigma_down = get_eps_step_official(sigma_current, sigma_next, eta=add_noise_ratio)
            target_sigma = sigma_down
            alpha_ratio = None
        else:
            sigma_up, _s, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
                model, sigma_current, sigma_next, add_noise_ratio, "hard"
            )
            target_sigma = sigma_down
    dt = target_sigma - sigma_current

    # d_prev from last REAL epsilon if available
    d_prev = None
    if sigma_previous is not None and len(epsilon_history) >= 1:
        d_prev = -(epsilon_history[-1]) / sigma_previous

    # Explicit skip indices take precedence
    if explicit_skip_indices is not None and isinstance(explicit_skip_indices, set) and step_index in explicit_skip_indices:
        es = skip_stats.get("explicit_streak", False) if skip_stats is not None else False
        nl = skip_stats.get("needed_learns", 2) if skip_stats is not None else 2
        allowed_by_streak = es or (nl <= 0)
        if allowed_by_streak and len(epsilon_history) >= 2:
            pred = (explicit_predictor or "linear")
            if pred == "h4" and len(epsilon_history) >= 4:
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
                tag = "explicit-h4"
            elif (pred in ("richardson", "h3")) and len(epsilon_history) >= 3:
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
                tag = "explicit-h3"
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
                tag = "explicit-h2"
            prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
            ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon_hat, prev_eps)
            if ok:
                if len(epsilon_history) >= 3:
                    epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
                d_hat = -(epsilon_hat) / sigma_current
                dbar_hat = (ge_gamma - 1.0) * (d_hat - d_prev) if d_prev is not None else 0.0
                # Clamp correction magnitude relative to base slope
                if isinstance(dbar_hat, torch.Tensor):
                    try:
                        _ratio = float(torch.norm(dbar_hat) / (torch.norm(d_hat) + 1e-8))
                    except Exception:
                        _ratio = 0.0
                    if _ratio > 0.25:
                        dbar_hat = dbar_hat * (0.25 / _ratio)
                x = x + (d_hat + (dbar_hat if isinstance(dbar_hat, torch.Tensor) else 0.0)) * dt
                if skip_stats is not None:
                    skip_stats["skipped"] = skip_stats.get("skipped", 0) + 1
                    skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
                    skip_stats["explicit_streak"] = True
                    skip_stats["needed_learns"] = 0
                if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
                    noise = torch.randn_like(x)
                    if add_noise_type == "whitened":
                        noise = (noise - noise.mean()) / (noise.std() + 1e-12)
                    if official_comfy or alpha_ratio is None or alpha_ratio is True:
                        x = x + noise * sigma_up
                    else:
                        x = alpha_ratio * x + noise * sigma_up
                if debug:
                    try:
                        x_rms = float(torch.sqrt(torch.mean(x**2)).item())
                    except Exception:
                        x_rms = None
                    print_step_diag(
                        sampler="gradient_estimation",
                        step_index=step_index,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        target_sigma=target_sigma,
                        sigma_up=sigma_up,
                        alpha_ratio=alpha_ratio,
                        h=dt,
                        c2=None,
                        b1=None,
                        b2=None,
                        eps_norm=hat_norm,
                        eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
                        x_rms=x_rms,
                        flags=f"SKIPPED-{tag}",
                    )
                return x, learning_ratio
            else:
                if debug:
                    print(f"gradient_est step {step_index}: explicit skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}")
        else:
            if debug:
                reason = "need_two_learns_before_skip" if not (es or nl <= 0) else "insufficient_history"
                print(f"gradient_est step {step_index}: explicit skip gated ({reason})")

    # Decide skip (non-explicit)
    if skip_mode == "adaptive":
        should_skip, epsilon_hat, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=x,
            sigma_current=sigma_current,
            sigma_next=target_sigma,
            sampler_kind="euler",
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon_hat = None

    if should_skip and skip_method is not None:
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
            if debug:
                print(f"gradient_est step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}")
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            d_hat = -(epsilon_hat) / sigma_current
            dbar_hat = (ge_gamma - 1.0) * (d_hat - d_prev) if d_prev is not None else 0.0
            # Clamp correction magnitude relative to base slope
            if isinstance(dbar_hat, torch.Tensor):
                try:
                    _ratio = float(torch.norm(dbar_hat) / (torch.norm(d_hat) + 1e-8))
                except Exception:
                    _ratio = 0.0
                if _ratio > 0.25:
                    dbar_hat = dbar_hat * (0.25 / _ratio)
                x = x + (d_hat + (dbar_hat if isinstance(dbar_hat, torch.Tensor) else 0.0)) * dt
                if skip_stats is not None:
                    skip_stats["skipped"] = skip_stats.get("skipped", 0) + 1
                    skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
                    skip_stats["explicit_streak"] = True
                    skip_stats["needed_learns"] = 0
                if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
                    noise = torch.randn_like(x)
                    if add_noise_type == "whitened":
                        noise = (noise - noise.mean()) / (noise.std() + 1e-12)
                    if official_comfy or alpha_ratio is None or alpha_ratio is True:
                        x = x + noise * sigma_up
                    else:
                        x = alpha_ratio * x + noise * sigma_up
                # Ensure x_rms is defined even if debug is False
                x_rms = None
                if debug:
                    # Summary line consistent with Euler
                    print(f"gradient_est step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={(dt.item() if hasattr(dt, 'item') else float(dt)):.4f}")
                    try:
                        x_rms = float(torch.sqrt(torch.mean(x**2)).item())
                    except Exception:
                        x_rms = None
                print_step_diag(
                    sampler="gradient_estimation",
                    step_index=step_index,
                    sigma_current=sigma_current,
                    sigma_next=sigma_next,
                    target_sigma=target_sigma,
                    sigma_up=sigma_up,
                    alpha_ratio=alpha_ratio,
                    h=dt,
                    c2=None,
                    b1=None,
                    b2=None,
                    eps_norm=hat_norm,
                    eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
                    x_rms=x_rms,
                    flags=f"SKIPPED-{skip_method}",
                )
            return x, learning_ratio

    # REAL Gradient Estimation step
    den = model(x, sigma_current * s_in, **extra_args)
    d = (x - den) / (sigma_current + 1e-8)
    x = x + d * dt
    if d_prev is not None:
        dbar = (ge_gamma - 1.0) * (d - d_prev)
        # Clamp REAL correction for stability
        try:
            _ratio_real = float(torch.norm(dbar) / (torch.norm(d) + 1e-8))
        except Exception:
            _ratio_real = 0.0
        if _ratio_real > 0.25:
            dbar = dbar * (0.25 / _ratio_real)
        x = x + dbar * dt

    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
        noise = torch.randn_like(x)
        if add_noise_type == "whitened":
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        if official_comfy or alpha_ratio is None or alpha_ratio is True:
            x = x + noise * sigma_up
        else:
            x = alpha_ratio * x + noise * sigma_up

    if skip_stats is not None:
        skip_stats["model_calls"] = skip_stats.get("model_calls", 0) + 1
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

    eps_real = den - noisy_latent
    epsilon_history.append(eps_real)
    if len(epsilon_history) >= 3:
        if predictor_type == "h4":
            epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
        elif predictor_type == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is not None:
            learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps_real) + 1e-8)).item()
            learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
            learning_ratio = max(0.5, min(2.0, learning_ratio))
            if debug:
                print(f"gradient_est step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    if debug:
        # Summary line consistent with Euler
        try:
            e_norm = float(torch.norm(eps_real).item())
            d_norm = float(torch.norm(d).item())
            dt_val = (dt.item() if hasattr(dt, 'item') else float(dt))
        except Exception:
            e_norm = float('nan'); d_norm = float('nan'); dt_val = float('nan')
        print(f"gradient_estimation step {step_index}: e_norm={e_norm:.2f}, d_norm={d_norm:.2f}, dt={dt_val:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        print_step_diag(
            sampler="gradient_estimation",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=target_sigma,
            sigma_up=sigma_up,
            alpha_ratio=alpha_ratio,
            h=dt,
            c2=None,
            b1=None,
            b2=None,
            eps_norm=float(torch.norm(eps_real).item()),
            eps_prev_norm=float(torch.norm(epsilon_history[-2]).item()) if len(epsilon_history) >= 2 else None,
            x_rms=x_rms,
            flags="",
        )
    # Optional SKIP diagnostics for grad-est
    try:
        if debug and 'd_hat' in locals():
            d_norm = float(torch.norm(d_hat).item())
            dbar_norm = float(torch.norm(dbar_hat).item()) if isinstance(dbar_hat, torch.Tensor) else 0.0
            ratio = dbar_norm / (d_norm + 1e-8)
        print(f"gradient_est step {step_index} [SKIP-APPLY]: d_norm={d_norm:.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio:.2f}, L={learning_ratio:.4f}, gamma={ge_gamma:.2f}")
    except Exception:
        pass

    return x, learning_ratio
