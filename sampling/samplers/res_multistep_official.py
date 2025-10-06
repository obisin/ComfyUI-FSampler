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

    # Decide skip
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

    was_skipped = False
    if should_skip and skip_method is not None:
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
            # Apply learning ratio correction
            if len(epsilon_history) >= 3:
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

    if not should_skip:
        denoised = model(x, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index

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

    if debug and not was_skipped:
        e_norm = torch.norm(epsilon).item() if not was_skipped else None
        print(f"res_multistep_official step {step_index}: e_norm={e_norm:.2f}, L={learning_ratio:.4f}, order={2 if old_sigma_down is not None and len(error_history) >= 2 else 1}")

    return x, learning_ratio, sigma_down
