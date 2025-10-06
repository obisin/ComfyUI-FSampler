import torch
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..noise import get_eps_step_official
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def sample_step_dpmpp_2m(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                         epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                         step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", skip_mode="none", skip_stats=None, debug=False, protect_last_steps=4, protect_first_steps=2, anchor_interval=None, max_consecutive_skips=None, official_comfy=False):
    x = noisy_latent
    if skip_stats is not None:
        skip_stats["total_steps"] += 1

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
            sampler_kind="dpmpp_2m",
            sigma_previous=sigma_previous,
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon_hat = None

    d_prev = None
    if sigma_previous is not None and len(epsilon_history) >= 1:
        eps_prev = epsilon_history[-1]
        d_prev = -(eps_prev) / sigma_previous

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
                print(f"dpmpp_2m step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            d_curr = -(epsilon_hat) / sigma_current
            dt = sigma_next - sigma_current
            if d_prev is not None:
                x = x + dt * (1.5 * d_curr - 0.5 * d_prev)
            else:
                x = x + dt * d_curr
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
            if debug:
                d_norm = torch.norm(d_curr).item()
                if skip_mode == "adaptive":
                    rel = (meta.get("relative_error") if isinstance(meta, dict) else None)
                    print(f"dpmpp_2m step {step_index} [SKIPPED-adaptive]: err_rel={(rel if rel is not None else float('nan')):.4f}, L={learning_ratio:.4f}")
                else:
                    print(f"dpmpp_2m step {step_index} [SKIPPED-{skip_method}]: d_norm={d_norm:.2f}, L={learning_ratio:.4f}")
                try:
                    x_rms = float(torch.sqrt(torch.mean(x**2)).item())
                except Exception:
                    x_rms = None
                print_step_diag(
                    sampler="dpmpp_2m",
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
                    eps_norm=float(torch.norm(epsilon_hat).item()) if epsilon_hat is not None else None,
                    eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
                    x_rms=x_rms,
                    flags=f"SKIPPED-{skip_method}",
                )
            return x, learning_ratio

    denoised = model(x, sigma_current * s_in, **extra_args)
    eps_curr = denoised - x
    d_curr = -eps_curr / sigma_current
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
    if d_prev is not None:
        x = x + dt * (1.5 * d_curr - 0.5 * d_prev)
    else:
        x = x + dt * d_curr
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and float(sigma_up) > 0.0:
        noise = torch.randn_like(x)
        if add_noise_type == "whitened":
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        if official_comfy or alpha_ratio is None or alpha_ratio is True:
            x = x + noise * sigma_up
        else:
            x = alpha_ratio * x + noise * sigma_up
    if skip_stats is not None:
        skip_stats["model_calls"] += 1
        skip_stats["consecutive_skips"] = 0
        skip_stats["last_anchor_step"] = step_index

    epsilon_history.append(eps_curr)
    if len(epsilon_history) >= 3:
        if predictor_type == "h4":
            epsilon_hat = extrapolate_epsilon_h4(epsilon_history)
        elif predictor_type == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is not None:
            learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps_curr) + 1e-8)).item()
            learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
            if learning_ratio < 0.5:
                learning_ratio = 0.5
            elif learning_ratio > 2.0:
                learning_ratio = 2.0
            if debug:
                print(f"dpmpp_2m step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    if debug and d_prev is not None:
        d_norm = torch.norm(d_curr).item()
        print(f"dpmpp_2m step {step_index}: d_norm={d_norm:.2f}, AB2")
    elif debug:
        d_norm = torch.norm(d_curr).item()
        print(f"dpmpp_2m step {step_index}: d_norm={d_norm:.2f}, Euler")
    if debug:
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        try:
            target_sigma = sigma_down if (add_noise_ratio > 0.0 and float(sigma_next) > 0.0) else sigma_next
            h_val = -torch.log(target_sigma / sigma_current)
        except Exception:
            h_val = None
        prev_eps = epsilon_history[-2] if len(epsilon_history) >= 2 else None
        print_step_diag(
            sampler="dpmpp_2m",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=(sigma_down if (add_noise_ratio > 0.0 and float(sigma_next) > 0.0) else sigma_next) if 'sigma_down' in locals() else sigma_next,
            sigma_up=(sigma_up if 'sigma_up' in locals() else None),
            alpha_ratio=(alpha_ratio if 'alpha_ratio' in locals() else None),
            h=h_val,
            c2=None,
            b1=None,
            b2=None,
            eps_norm=float(torch.norm(eps_curr).item()),
            eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
            x_rms=x_rms,
            flags="",
        )

    return x, learning_ratio
