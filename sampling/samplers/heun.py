import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..noise import get_eps_step_official
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def sample_step_heun(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                     epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                     step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2, anchor_interval=None, max_consecutive_skips=None, official_comfy=False,
                     explicit_skip_indices=None, explicit_predictor=None, adaptive_mode="none"):
    x = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] = skip_stats.get("total_steps", 0) + 1

    # Final step guard: land on denoised (no noise)
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else float(sigma_next)
    if abs(sigma_next_value) <= 1e-12:
        denoised = model(x, sigma_current * s_in, **extra_args)
        x = denoised
        epsilon_real = denoised - noisy_latent
        epsilon_history.append(epsilon_real)
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
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon_real) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                learning_ratio = max(0.5, min(2.0, learning_ratio))
                if debug:
                    print(f"heun step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        if debug:
            print(f"heun step {step_index} (final step): landing on denoised")
        return x, learning_ratio

    # Target sigma and noise planning
    target_sigma = sigma_next
    sigma_up = None
    alpha_ratio = None
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0:
        if official_comfy:
            sigma_up, sigma_down = get_eps_step_official(sigma_current, sigma_next, eta=add_noise_ratio)
            target_sigma = sigma_down
        else:
            sigma_up, _s, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
                model, sigma_current, sigma_next, add_noise_ratio, "hard"
            )
            target_sigma = sigma_down

    # Explicit skip indices take precedence
    was_skipped = False
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
                if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                    epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
                dt = target_sigma - sigma_current
                d1 = -(epsilon_hat) / sigma_current
                d2 = -(epsilon_hat) / (target_sigma + 1e-8)
                # Base Heun emulation update
                slope = 0.5 * (d1 + d2)
                # Gradient-estimation correction (Euler-space)
                dbar = 0.0
                if skip_stats is not None and adaptive_mode in ("grad_est", "learn+grad_est"):
                    d_prev = skip_stats.get("d_prev")
                    if d_prev is not None:
                        d_hat = -(epsilon_hat) / sigma_current
                        dbar = (2.0 - 1.0) * (d_hat - d_prev)
                        try:
                            ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                        except Exception:
                            ratio = 0.0
                        if ratio > 0.25:
                            dbar = dbar * (0.25 / ratio)
                x = x + dt * (slope + (dbar if isinstance(dbar, torch.Tensor) else 0.0))
                if debug and isinstance(dbar, torch.Tensor):
                    try:
                        d_norm = float(torch.norm(d1).item())
                        dbar_norm = float(torch.norm(dbar).item())
                        ratio = dbar_norm / (d_norm + 1e-8)
                    except Exception:
                        d_norm = None; dbar_norm = 0.0; ratio = 0.0
                    print(f"heun step {step_index} [SKIP-APPLY]: d_norm={(d_norm if d_norm is not None else float('nan')):.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio:.2f}, L={learning_ratio:.4f}, mode={adaptive_mode}")
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
                        sampler="heun",
                        step_index=step_index,
                        sigma_current=sigma_current,
                        sigma_next=sigma_next,
                        target_sigma=target_sigma,
                        sigma_up=sigma_up,
                        alpha_ratio=alpha_ratio,
                        h=(target_sigma - sigma_current),
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
                    print(f"heun step {step_index}: explicit skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}")
        else:
            if debug:
                reason = "need_two_learns_before_skip" if not (es or nl <= 0) else "insufficient_history"
                print(f"heun step {step_index}: explicit skip gated ({reason})")

    # Non-explicit path: decide skip
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
            should_skip = False
            if debug:
                print(f"heun step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}")
        else:
            if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            dt = target_sigma - sigma_current
            d1 = -(epsilon_hat) / sigma_current
            d2 = -(epsilon_hat) / (target_sigma + 1e-8)
            slope = 0.5 * (d1 + d2)
            # Gradient-estimation correction (Euler-space)
            dbar = 0.0
            if skip_stats is not None and adaptive_mode in ("grad_est", "learn+grad_est"):
                d_prev = skip_stats.get("d_prev")
                if d_prev is not None:
                    d_hat = -(epsilon_hat) / sigma_current
                    dbar = (2.0 - 1.0) * (d_hat - d_prev)
                    try:
                        ratio = float(torch.norm(dbar) / (torch.norm(d_hat) + 1e-8))
                    except Exception:
                        ratio = 0.0
                    if ratio > 0.25:
                        dbar = dbar * (0.25 / ratio)
            x = x + dt * (slope + (dbar if isinstance(dbar, torch.Tensor) else 0.0))
            if debug:
                # Summary line consistent with Euler
                try:
                    dt_val = (dt.item() if hasattr(dt, 'item') else float(dt))
                except Exception:
                    dt_val = float('nan')
                print(f"heun step {step_index} [SKIPPED-explicit-{tag}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
            if debug and isinstance(dbar, torch.Tensor):
                try:
                    d_norm = float(torch.norm(d1).item())
                    dbar_norm = float(torch.norm(dbar).item())
                    ratio = dbar_norm / (d_norm + 1e-8)
                except Exception:
                    d_norm = None; dbar_norm = 0.0; ratio = 0.0
                print(f"heun step {step_index} [SKIP-APPLY]: d_norm={(d_norm if d_norm is not None else float('nan')):.2f}, dbar_norm={dbar_norm:.2f}, ratio={ratio:.2f}, L={learning_ratio:.4f}, mode={adaptive_mode}")
            if skip_stats is not None:
                skip_stats["skipped"] = skip_stats.get("skipped", 0) + 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
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
                    sampler="heun",
                    step_index=step_index,
                    sigma_current=sigma_current,
                    sigma_next=sigma_next,
                    target_sigma=target_sigma,
                    sigma_up=sigma_up,
                    alpha_ratio=alpha_ratio,
                    h=(target_sigma - sigma_current),
                    c2=None,
                    b1=None,
                    b2=None,
                    eps_norm=hat_norm,
                    eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
                    x_rms=x_rms,
                    flags=f"SKIPPED-{skip_method}",
                )
            return x, learning_ratio

    # REAL Heun step
    denoised1 = model(x, sigma_current * s_in, **extra_args)
    d1 = (x - denoised1) / (sigma_current + 1e-8)
    dt = target_sigma - sigma_current
    x2 = x + d1 * dt
    denoised2 = model(x2, target_sigma * s_in, **extra_args)
    d2 = (x2 - denoised2) / (target_sigma + 1e-8)
    dprime = 0.5 * (d1 + d2)
    x = x + dprime * dt

    # Noise after integrator when enabled
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
        noise = torch.randn_like(x)
        if add_noise_type == "whitened":
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        if official_comfy or alpha_ratio is None or alpha_ratio is True:
            x = x + noise * sigma_up
        else:
            x = alpha_ratio * x + noise * sigma_up

    if skip_stats is not None:
        skip_stats["model_calls"] = skip_stats.get("model_calls", 0) + 2
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
        # Update last REAL slope for grad_est
        try:
            skip_stats["d_prev"] = d1.detach()
        except Exception:
            skip_stats["d_prev"] = d1

    epsilon_real = denoised1 - noisy_latent
    epsilon_history.append(epsilon_real)
    if len(epsilon_history) >= 3:
        if predictor_type == "h4":
            epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
        elif predictor_type == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is not None:
            learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon_real) + 1e-8)).item()
            learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
            learning_ratio = max(0.5, min(2.0, learning_ratio))
            if debug:
                print(f"heun step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    if debug:
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        print_step_diag(
            sampler="heun",
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
            eps_norm=float(torch.norm(epsilon_real).item()),
            eps_prev_norm=float(torch.norm(epsilon_history[-2]).item()) if len(epsilon_history) >= 2 else None,
            x_rms=x_rms,
            flags="",
        )

    return x, learning_ratio
