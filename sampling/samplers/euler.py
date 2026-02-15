import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


from ..noise import get_eps_step_official


def sample_step_euler(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                      epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                      step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2, anchor_interval=None, max_consecutive_skips=None, official_comfy=False,
                      adaptive_mode="none", explicit_skip_indices=None, explicit_predictor=None):
    x = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    was_skipped = False

    # Explicit skip indices take precedence (ignores protect windows); apply streak gating
    if explicit_skip_indices is not None and isinstance(explicit_skip_indices, set) and step_index in explicit_skip_indices:
        es = skip_stats.get("explicit_streak", False) if skip_stats is not None else False
        nl = skip_stats.get("needed_learns", 2) if skip_stats is not None else 2
        allowed_by_streak = es or (nl <= 0)
        if allowed_by_streak and len(epsilon_history) >= 2:
            # predictor preference with fallback ladder
            pred = (explicit_predictor or "linear")
            if pred == "h4" and len(epsilon_history) >= 4:
                epsilon = extrapolate_epsilon_h4(epsilon_history)
                skip_method = "explicit-h4"
            elif (pred in ("richardson", "h3")) and len(epsilon_history) >= 3:
                epsilon = extrapolate_epsilon_richardson(epsilon_history)
                skip_method = "explicit-h3"
            else:
                epsilon = extrapolate_epsilon_linear(epsilon_history)
                skip_method = "explicit-h2"
            prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
            ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon, prev_eps)
            if ok:
                if len(epsilon_history) >= 3:
                    epsilon = epsilon / max(learning_ratio, 1e-8)
                denoised = x + epsilon
                was_skipped = True
                if skip_stats is not None:
                    skip_stats["skipped"] = skip_stats.get("skipped", 0) + 1
                    skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
                    skip_stats["explicit_streak"] = True
                    skip_stats["needed_learns"] = 0
                if debug:
                    dt_val = (sigma_next - sigma_current).item() if torch.is_tensor(sigma_next) else float(sigma_next - sigma_current)
                    print(f"euler step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                    try:
                        x_rms = float(torch.sqrt(torch.mean((denoised)**2)).item())
                    except Exception:
                        x_rms = None
                    print_step_diag(
                        sampler="euler",
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
                        flags=f"SKIPPED-{skip_method}",
                    )
            else:
                if debug:
                    print(f"euler step {step_index}: skip rejected by validate_epsilon_hat (reason={reason})")
        else:
            if debug:
                reason = "need_two_learns_before_skip" if not (es or nl <= 0) else "insufficient_history"
                print(f"euler step {step_index}: explicit skip gated ({reason})")

    # Decide skip (only if not explicitly skipped)
    if (not was_skipped) and skip_mode == "adaptive":
        should_skip, epsilon, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=x,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="euler",
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = (False, None) if was_skipped else should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon = None

    if (not was_skipped) and should_skip and skip_method is not None:
        if epsilon is None:
            if skip_method == "richardson":
                epsilon = extrapolate_epsilon_richardson(epsilon_history)
            elif skip_method == "h4":
                epsilon = extrapolate_epsilon_h4(epsilon_history)
            else:
                epsilon = extrapolate_epsilon_linear(epsilon_history)
        prev_eps = epsilon_history[-1] if len(epsilon_history) >= 1 else None
        ok, reason, hat_norm, prev_norm = validate_epsilon_hat(epsilon, prev_eps)
        if not ok:
            should_skip = False
            if debug:
                print(f"euler step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            # Apply L scaling only for learning or learn+grad_est modes
            if len(epsilon_history) >= 3 and adaptive_mode in ("learning", "learn+grad_est"):
                epsilon = epsilon / max(learning_ratio, 1e-8)
            denoised = x + epsilon
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
            if debug:
                dt_val = (sigma_next - sigma_current).item() if torch.is_tensor(sigma_next) else float(sigma_next - sigma_current)
                if skip_mode == "adaptive":
                    rel = (meta.get("relative_error") if isinstance(meta, dict) else None)
                    print(f"euler step {step_index} [SKIPPED-adaptive]: err_rel={(rel if rel is not None else float('nan')):.4f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                else:
                    print(f"euler step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                try:
                    x_rms = float(torch.sqrt(torch.mean((denoised)**2)).item())
                except Exception:
                    x_rms = None
                print_step_diag(
                    sampler="euler",
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
                    flags=f"SKIPPED-{skip_method}",
                )

    if not was_skipped and not should_skip:
        denoised = model(x, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index
            # Gating update: REAL call increments learns and may end explicit streak
            es = skip_stats.get("explicit_streak", False)
            nl = skip_stats.get("needed_learns", 2)
            if es:
                skip_stats["explicit_streak"] = False
                skip_stats["needed_learns"] = 1  # this REAL counts as first learn after streak end
            else:
                skip_stats["needed_learns"] = max(0, int(nl) - 1)

    # Karras ODE derivative at current sigma
    d = (x - denoised) / sigma_current
    # Gradient-estimation correction on skipped steps
    d_prev = None
    if skip_stats is not None and isinstance(skip_stats, dict):
        d_prev = skip_stats.get("d_prev")
    dbar = 0.0
    if was_skipped and adaptive_mode in ("grad_est", "learn+grad_est") and d_prev is not None:
        dbar = (2.0 - 1.0) * (d - d_prev)  # gamma=2.0
        # Clamp correction magnitude relative to d
        try:
            ratio = float(torch.norm(dbar) / (torch.norm(d) + 1e-8))
        except Exception:
            ratio = 0.0
        if ratio > 0.25:
            dbar = dbar * (0.25 / ratio)

    # If adding noise (ancestral), follow res4lyf: adjust target sigma to sigma_down and add noise via alpha_ratio/sigma_up
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and not was_skipped:
        if official_comfy:
            sigma_up, sigma_down = get_eps_step_official(sigma_current, sigma_next, eta=add_noise_ratio)
            dt = sigma_down - sigma_current
            x = x + d * dt
            noise = torch.randn_like(x)
            if add_noise_type == "whitened":
                noise = (noise - noise.mean()) / (noise.std() + 1e-12)
            x = x + noise * sigma_up
            alpha_ratio = None
        else:
            sigma_up, _sigma_for_calc, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
                model, sigma_current, sigma_next, add_noise_ratio, "hard"
            )
            dt = sigma_down - sigma_current
            x = x + d * dt
            # whitened Gaussian noise
            if add_noise_type == "whitened":
                noise = torch.randn_like(x)
                std = noise.std()
                noise = (noise - noise.mean()) / (std + 1e-12)
            else:  # gaussian
                noise = torch.randn_like(x)
            x = alpha_ratio * x + noise * sigma_up
    else:
        dt = sigma_next - sigma_current
        x = x + (d + (dbar if was_skipped else 0.0)) * dt
        sigma_up = None
        alpha_ratio = None
        sigma_down = None

    if not was_skipped:
        epsilon = denoised - noisy_latent
        epsilon_history.append(epsilon)
        if len(epsilon_history) >= 3:
            if predictor_type == "h4":
                epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
            elif predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                if learning_ratio < 0.5:
                    learning_ratio = 0.5
                elif learning_ratio > 2.0:
                    learning_ratio = 2.0
                # (no aggregation; keep original verbose-only behavior)
    # Update last REAL slope for grad_est modes
    if skip_stats is not None and not was_skipped:
        try:
            skip_stats["d_prev"] = d.detach()
        except Exception:
            skip_stats["d_prev"] = d

    if debug:
        d_norm = torch.norm(d).item()
        # Compute an epsilon norm for diagnostics regardless of branch
        try:
            e_norm = float(torch.norm(epsilon).item()) if 'epsilon' in locals() and isinstance(epsilon, torch.Tensor) else None
        except Exception:
            e_norm = None
        if not was_skipped:
            if len(epsilon_history) >= 3:
                print(f"euler step {step_index}: e_norm={(e_norm if e_norm is not None else float('nan')):.2f}, d_norm={d_norm:.2f}, dt={dt.item():.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
            else:
                print(f"euler step {step_index}: e_norm={(e_norm if e_norm is not None else float('nan')):.2f}, d_norm={d_norm:.2f}, dt={dt.item():.4f}")
        else:
            # Skipped with potential grad_est
            try:
                dbar_norm = float(torch.norm(dbar).item()) if isinstance(dbar, torch.Tensor) else float(dbar)
            except Exception:
                dbar_norm = 0.0
            print(f"euler step {step_index} [SKIP-APPLY]: d_norm={d_norm:.2f}, dbar_norm={dbar_norm:.2f}, mode={adaptive_mode}")
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        # Compute h in t-domain when possible; guard missing sigma_down
        try:
            target_sigma_print = sigma_down if ('sigma_down' in locals() and sigma_down is not None) else sigma_next
            h_val = -torch.log(target_sigma_print / sigma_current)
        except Exception:
            h_val = None
            target_sigma_print = sigma_next
        print_step_diag(
            sampler="euler",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=target_sigma_print,
            sigma_up=sigma_up,
            alpha_ratio=alpha_ratio,
            h=h_val,
            c2=None,
            b1=None,
            b2=None,
            eps_norm=e_norm,
            eps_prev_norm=float(torch.norm(epsilon_history[-2]).item()) if len(epsilon_history) >= 2 else None,
            x_rms=x_rms,
            flags="",
        )

    return x, learning_ratio
