import math
import torch
from ..phi_functions import phi_function
from ...comfy_copy.res4lyf_sampling import get_res4lyf_step_with_model
from ..extrapolation import extrapolate_epsilon_linear, extrapolate_epsilon_richardson, extrapolate_epsilon_h4
from ..skip import should_skip_model_call, validate_epsilon_hat, decide_skip_adaptive
from ..log import print_step_diag


def _to_d(x, sigma, denoised):
    # Broadcast-safe: sigma is a scalar tensor; rely on PyTorch broadcasting
    return (x - denoised) / (sigma + 1e-8)


def sample_step_res_multistep(
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
    x_0 = noisy_latent

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
            x_current=x_0,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="res_multistep",
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon_hat = None
    was_skipped = False

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
        bad_skip = not ok
        if not bad_skip and prev_norm is not None and prev_norm > 0 and hat_norm > 50.0 * prev_norm:
            bad_skip = True
            reason = 'too_large_rel'

        if bad_skip:
            should_skip = False
            if debug:
                print(f"res_multistep step {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}, prev_norm={(prev_norm if prev_norm is not None else float('nan')):.2e}")
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            denoised = x_0 + epsilon_hat
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1
    
    if not should_skip:
        denoised = model(x_0, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index

    epsilon_current = denoised - x_0

    sigma_up = None
    alpha_ratio = None
    target_sigma = sigma_next
    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0:
        sigma_up, _s, sigma_down, alpha_ratio = get_res4lyf_step_with_model(
            model, sigma_current, sigma_next, add_noise_ratio, "hard"
        )
        target_sigma = sigma_down

    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next
    is_final_step = (sigma_next_value == 0)

    if is_final_step:
        x = denoised
        error_history.append(denoised)
        if len(error_history) > 2:
            error_history.pop(0)
        if not was_skipped:
            epsilon_history.append(epsilon_current)
            if len(epsilon_history) >= 3:
                epsilon_hat = (
                    extrapolate_epsilon_richardson(epsilon_history)
                    if predictor_type == "richardson"
                    else extrapolate_epsilon_linear(epsilon_history)
                )
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
            try:
                x_rms = float(torch.sqrt(torch.mean(x**2)).item())
            except Exception:
                x_rms = None
            print_step_diag(
                sampler="res_multistep",
                step_index=step_index,
                sigma_current=sigma_current,
                sigma_next=sigma_next,
                target_sigma=target_sigma,
                sigma_up=sigma_up,
                alpha_ratio=alpha_ratio,
                h=None,
                c2=None,
                b1=None,
                b2=None,
                eps_norm=float(torch.norm(epsilon_current).item()) if torch.is_tensor(epsilon_current) else None,
                eps_prev_norm=float(torch.norm(epsilon_history[-2]).item()) if len(epsilon_history) >= 2 else None,
                x_rms=x_rms,
                flags="final",
            )
        return x, learning_ratio, None

    if len(error_history) == 0 or sigma_previous is None or old_sigma_down is None or float(target_sigma) == 0.0:
        d = _to_d(x_0, sigma_current, denoised)
        dt = target_sigma - sigma_current
        x = x_0 + d * dt
    else:
        t = -torch.log(sigma_current)
        t_old = -torch.log(old_sigma_down)
        t_next = -torch.log(target_sigma)
        t_prev = -torch.log(sigma_previous)
        h = t_next - t

        h_abs = float(torch.abs(h)) if torch.is_tensor(h) else abs(h)
        if h_abs < 1e-8:
            d = _to_d(x_0, sigma_current, denoised)
            dt = target_sigma - sigma_current
            x = x_0 + d * dt
        else:
            c2 = (t_prev - t_old) / h
            phi1_val = phi_function(order=1, step_size=-h)
            phi2_val = phi_function(order=2, step_size=-h)
            b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
            b1 = torch.nan_to_num(phi1_val - b2, nan=0.0)
            scale = torch.exp(-h)
            x = scale * x_0 + h * (b1 * denoised + b2 * error_history[-1])

    if add_noise_ratio > 0.0 and float(sigma_next) > 0.0 and sigma_up is not None and float(sigma_up) > 0.0:
        if add_noise_type == "whitened":
            noise = torch.randn_like(x)
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        else:
            noise = torch.randn_like(x)
        if alpha_ratio is not None and alpha_ratio is not True:
            x = alpha_ratio * x + noise * sigma_up
        else:
            x = x + noise * sigma_up

    error_history.append(denoised)
    if len(error_history) > 2:
        error_history.pop(0)

    if not was_skipped:
        epsilon_history.append(epsilon_current)
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
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        try:
            h_val = -torch.log(target_sigma / sigma_current)
        except Exception:
            h_val = None
        prev_eps = epsilon_history[-2] if len(epsilon_history) >= 2 else None
        # Extract scalar c2 for printing if available
        c2_print = None
        try:
            c2_print = float(c2.item()) if hasattr(c2, 'item') else float(c2)
        except Exception:
            c2_print = None
        print_step_diag(
            sampler="res_multistep",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=target_sigma,
            sigma_up=sigma_up,
            alpha_ratio=alpha_ratio,
            h=h_val,
            c2=c2_print,
            b1=b1 if 'b1' in locals() else None,
            b2=b2 if 'b2' in locals() else None,
            eps_norm=float(torch.norm(epsilon_current).item()) if torch.is_tensor(epsilon_current) else None,
            eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
            x_rms=x_rms,
            flags=("SKIPPED" if was_skipped else ""),
        )
    return x, learning_ratio, target_sigma


def _phi12_official(h):
    # Compute phi1 and phi2 using stable closed forms from official Comfy
    # phi1(-h) = (exp(-h) - 1)/(-h) = expm1(-h)/(-h)
    # phi2(-h) = (phi1(-h) - 1)/(-h)
    e1 = torch.expm1(-h)
    phi1 = e1 / (-h)
    phi2 = (phi1 - 1.0) / (-h)
    return phi1, phi2


def _get_ancestral_step_eps(sigma, sigma_next, eta=0.0):
    # K-diffusion EPS formula (official-like): eta in [0,1]
    if eta is None or float(eta) <= 0.0:
        return torch.tensor(0.0, dtype=sigma.dtype, device=sigma.device), sigma_next
    # guard types
    s = sigma.to(torch.float64)
    sn = sigma_next.to(torch.float64)
    su = torch.clamp(eta * torch.sqrt(torch.clamp(sn**2 * (s**2 - sn**2) / torch.clamp(s**2, min=1e-12), min=0.0)), max=sn)
    sd = torch.sqrt(torch.clamp(sn**2 - su**2, min=0.0))
    return su.to(sigma.dtype), sd.to(sigma.dtype)


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
    x_0 = noisy_latent

    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Skip decision identical to res4lyf path
    if skip_mode == "adaptive":
        should_skip, epsilon_hat, meta = decide_skip_adaptive(
            epsilon_history=epsilon_history,
            step_index=step_index,
            total_steps=total_steps,
            protect_last_steps=protect_last_steps,
            protect_first_steps=protect_first_steps,
            skip_stats=skip_stats,
            x_current=x_0,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            sampler_kind="res_multistep",
            anchor_interval=anchor_interval,
            max_consecutive_skips=max_consecutive_skips,
        )
        skip_method = "adaptive"
    else:
        should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps, protect_first_steps)
        epsilon_hat = None
    was_skipped = False

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
        bad_skip = not ok
        if not bad_skip and prev_norm is not None and prev_norm > 0 and hat_norm > 50.0 * prev_norm:
            bad_skip = True
            reason = 'too_large_rel'

        if bad_skip:
            should_skip = False
            if debug:
                print(f"res_multistep(off) {step_index}: skip cancelled (ε̂ invalid: {reason}) hat_norm={hat_norm:.2e}")
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            denoised = x_0 + epsilon_hat
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
                skip_stats["consecutive_skips"] = skip_stats.get("consecutive_skips", 0) + 1

    if not should_skip:
        denoised = model(x_0, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1
            skip_stats["consecutive_skips"] = 0
            skip_stats["last_anchor_step"] = step_index

    epsilon_current = denoised - x_0

    # Official-like ancestral step (eta maps to add_noise_ratio)
    eta = add_noise_ratio if add_noise_ratio is not None else 0.0
    sigma_up, sigma_down = _get_ancestral_step_eps(sigma_current, sigma_next, eta=eta)
    target_sigma = sigma_down if float(eta) > 0.0 else sigma_next

    # Decide Euler vs Multistep
    need_euler = False
    if old_sigma_down is None or sigma_previous is None:
        need_euler = True
    if float(target_sigma) == 0.0:
        need_euler = True

    if need_euler:
        d = _to_d(x_0, sigma_current, denoised)
        dt = target_sigma - sigma_current
        x = x_0 + d * dt
    else:
        t = -torch.log(sigma_current)
        t_old = -torch.log(old_sigma_down)
        t_next = -torch.log(target_sigma)
        t_prev = -torch.log(sigma_previous)
        h = t_next - t
        phi1_val, phi2_val = _phi12_official(h)
        c2 = (t_prev - t_old) / h
        b2 = torch.nan_to_num(phi2_val / c2, nan=0.0)
        b1 = torch.nan_to_num(phi1_val - b2, nan=0.0)
        scale = torch.exp(-h)
        den_prev = error_history[-1] if len(error_history) > 0 else denoised
        x = scale * x_0 + h * (b1 * denoised + b2 * den_prev)

    # Add noise like official: x = x + noise * sigma_up (no alpha_ratio mixing)
    if float(eta) > 0.0 and float(sigma_next) > 0.0 and float(sigma_up) > 0.0:
        if add_noise_type == "whitened":
            noise = torch.randn_like(x)
            noise = (noise - noise.mean()) / (noise.std() + 1e-12)
        else:
            noise = torch.randn_like(x)
        x = x + noise * sigma_up

    # Histories and learning
    error_history.append(denoised)
    if len(error_history) > 2:
        error_history.pop(0)

    if not was_skipped:
        epsilon_history.append(epsilon_current)
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
        try:
            x_rms = float(torch.sqrt(torch.mean(x**2)).item())
        except Exception:
            x_rms = None
        try:
            h_val = -torch.log(target_sigma / sigma_current)
        except Exception:
            h_val = None
        prev_eps = epsilon_history[-2] if len(epsilon_history) >= 2 else None
        print_step_diag(
            sampler="res_multistep(off)",
            step_index=step_index,
            sigma_current=sigma_current,
            sigma_next=sigma_next,
            target_sigma=target_sigma,
            sigma_up=(sigma_up if float(eta) > 0.0 else None),
            alpha_ratio=None,
            h=h_val,
            c2=None,
            b1=None,
            b2=None,
            eps_norm=float(torch.norm(epsilon_current).item()) if torch.is_tensor(epsilon_current) else None,
            eps_prev_norm=float(torch.norm(prev_eps).item()) if prev_eps is not None else None,
            x_rms=x_rms,
            flags=("SKIPPED" if was_skipped else ""),
        )

    return x, learning_ratio, target_sigma
