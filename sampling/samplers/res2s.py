import torch
from ..phi_functions import phi_function
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def sample_step_res_2s(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                       epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                       step_index, total_steps, add_noise_ratio=0.0, add_noise_type="whitened", debug=False, skip_mode="none", protect_last_steps=4, protect_first_steps=2, skip_stats=None, anchor_interval=None, max_consecutive_skips=None):
    noisy_latent_at_step_start = noisy_latent

    if skip_mode == "adaptive":
        should_skip, epsilon_hat, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=noisy_latent,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="res_2s",
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
                print(f"res_2s step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            d = -(epsilon_hat) / sigma_current
            dt = sigma_next - sigma_current
            noisy_latent = noisy_latent + d * dt
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
            if debug:
                dt_val = (sigma_next - sigma_current).item() if torch.is_tensor(sigma_next) else float(sigma_next - sigma_current)
                if skip_mode == "adaptive":
                    rel = (meta.get("relative_error") if isinstance(meta, dict) else None)
                    print(f"res_2s step {step_index} [SKIPPED-adaptive]: err_rel={(rel if rel is not None else float('nan')):.4f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                else:
                    print(f"res_2s step {step_index} [SKIPPED-{skip_method}]: e_norm={hat_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")
                try:
                    x_rms = float(torch.sqrt(torch.mean(noisy_latent**2)).item())
                except Exception:
                    x_rms = None
                print_step_diag(
                    sampler="res_2s",
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
            return noisy_latent, learning_ratio

    # Ancestral: integrate to sigma_down when enabled; otherwise use sigma_next
    target_sigma = sigma_next
    sigma_up = None
    alpha_ratio = None
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0:
        sigma_up, _s, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
            model, sigma_current, sigma_next, add_noise_ratio, "hard"
        )
        target_sigma = sigma_down

    step_size = -torch.log(target_sigma / sigma_current)
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next
    is_final_step = (sigma_next_value == 0)

    if is_final_step:
        model_prediction = model(noisy_latent, sigma_current * s_in, **extra_args)
        noisy_latent = model_prediction
        if debug:
            print(f"res_2s step {step_index} (final step): using Euler")
        epsilon_real = model_prediction - noisy_latent_at_step_start
        epsilon_history.append(epsilon_real)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index
        if len(epsilon_history) >= 3:
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history) if predictor_type == "richardson" else extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon_real) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                if learning_ratio < 0.5:
                    learning_ratio = 0.5
                elif learning_ratio > 2.0:
                    learning_ratio = 2.0
                if debug:
                    print(f"res_2s step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        return noisy_latent, learning_ratio

    midpoint_fraction = 0.5
    phi_1_value = phi_function(order=1, step_size=-step_size)
    phi_2_value = phi_function(order=2, step_size=-step_size)
    weight_stage2 = phi_2_value / midpoint_fraction
    weight_stage1 = phi_1_value - weight_stage2
    phi_1_at_midpoint = phi_function(order=1, step_size=-step_size * midpoint_fraction)
    stage2_advance_weight = midpoint_fraction * phi_1_at_midpoint

    model_prediction_stage1 = model(noisy_latent, sigma_current * s_in, **extra_args)
    error_stage1 = -(noisy_latent_at_step_start - model_prediction_stage1)
    sigma_midpoint = torch.exp(-(-torch.log(sigma_current) + step_size * midpoint_fraction))
    noisy_latent_midpoint = noisy_latent_at_step_start + (step_size * stage2_advance_weight) * error_stage1
    model_prediction_stage2 = model(noisy_latent_midpoint, sigma_midpoint * s_in, **extra_args)
    error_stage2 = -(noisy_latent_at_step_start - model_prediction_stage2)

    noisy_latent = noisy_latent_at_step_start + step_size * (weight_stage1 * error_stage1 + weight_stage2 * error_stage2)

    # Ancestral noise add after integrator (works for ODE and RF)
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
        if add_noise_type == "whitened":
            noise = torch.randn_like(noisy_latent)
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        else:
            noise = torch.randn_like(noisy_latent)
        if alpha_ratio is not None and alpha_ratio is not True:
            noisy_latent = alpha_ratio * noisy_latent + noise * sigma_up
        else:
            noisy_latent = noisy_latent + noise * sigma_up

    if debug:
        stage1_norm = torch.norm(error_stage1).item()
        stage2_norm = torch.norm(error_stage2).item()
        print(f"res_2s step {step_index}: stage1_norm={stage1_norm:.2f}, stage2_norm={stage2_norm:.2f}, weight_s1={weight_stage1.item():.4f}, weight_s2={weight_stage2.item():.4f}")
        try:
            x_rms = float(torch.sqrt(torch.mean(noisy_latent**2)).item())
        except Exception:
            x_rms = None
        print_step_diag(
            sampler="res_2s",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=(target_sigma if 'target_sigma' in locals() else sigma_next),
            sigma_up=(sigma_up if 'sigma_up' in locals() else None),
            alpha_ratio=(alpha_ratio if 'alpha_ratio' in locals() else None),
            h=step_size,
            c2=None,
            b1=weight_stage1,
            b2=weight_stage2,
            eps_norm=stage1_norm,
            eps_prev_norm=float(torch.norm(epsilon_history[-2]).item()) if len(epsilon_history) >= 2 else None,
            x_rms=x_rms,
            flags="",
        )

    epsilon_real = error_stage1
    epsilon_history.append(epsilon_real)
    if skip_stats is not None:
        # REAL step taken; update anchor tracking and reset consecutive skips
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
            if learning_ratio < 0.5:
                learning_ratio = 0.5
            elif learning_ratio > 2.0:
                learning_ratio = 2.0
            if debug:
                print(f"res_2s step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    return noisy_latent, learning_ratio
