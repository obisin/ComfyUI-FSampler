import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def sample_step_euler(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                      epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                      step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2, anchor_interval=None, max_consecutive_skips=None):
    x = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Decide skip
    if skip_mode == "adaptive":
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
        should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon = None

    was_skipped = False
    if should_skip and skip_method is not None:
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
            if len(epsilon_history) >= 3:
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

    if not should_skip:
        denoised = model(x, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index

    # Karras ODE derivative at current sigma
    d = (x - denoised) / sigma_current

    # If adding noise (ancestral), follow res4lyf: adjust target sigma to sigma_down and add noise via alpha_ratio/sigma_up
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and not was_skipped:
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
        x = x + d * dt
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

    if debug and not was_skipped:
        d_norm = torch.norm(d).item()
        e_norm = torch.norm(epsilon).item()
        if len(epsilon_history) >= 3:
            print(f"euler step {step_index}: e_norm={e_norm:.2f}, d_norm={d_norm:.2f}, dt={dt.item():.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        else:
            print(f"euler step {step_index}: e_norm={e_norm:.2f}, d_norm={d_norm:.2f}, dt={dt.item():.4f}")
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        # Compute h in t-domain when possible
        try:
            target_sigma = sigma_down if sigma_down is not None else sigma_next
            h_val = -torch.log(target_sigma / sigma_current)
        except Exception:
            h_val = None
        print_step_diag(
            sampler="euler",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=target_sigma,
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
