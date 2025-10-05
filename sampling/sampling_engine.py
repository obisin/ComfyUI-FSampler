from .engine import sample_fsampler


def sample_step_euler(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                      epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                      step_index, total_steps, skip_mode="none", skip_stats=None, debug=False):
    """Standard Euler step using Karras ODE derivative formulation.

    Implements the standard k-diffusion Euler method:
    - Converts denoised to ODE derivative: d = (x - denoised) / sigma
    - Takes Euler step: x = x + d * dt, where dt = sigma_next - sigma_current

    Supports model call skipping via epsilon extrapolation.
    """
    x = noisy_latent

    # Update skip statistics
    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Check if we should skip the model call
    should_skip, skip_method = should_skip_model_call(
        1.0,  # error_ratio - Euler doesn't track this, use neutral value
        step_index,
        total_steps,
        skip_mode,
        epsilon_history
    )

    # Get denoised: either from model call or extrapolation
    was_skipped = False

    if should_skip and skip_method is not None:
        # SKIP: Use epsilon extrapolation
        if skip_method == "linear":
            epsilon = extrapolate_epsilon_linear(epsilon_history)
        elif skip_method == "richardson":
            epsilon = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon = None

        # Safety check: if extrapolation failed, fall back to model call
        if epsilon is None or torch.isnan(epsilon).any():
            should_skip = False
            if debug:
                print(f"euler step {step_index}: extrapolation failed, falling back to model call")

        if should_skip and epsilon is not None:
            # Successful skip - reconstruct denoised from extrapolated epsilon
            # Apply universal learning stabilizer if we have enough REAL history (>=3)
            if len(epsilon_history) >= 3:
                epsilon = epsilon / max(learning_ratio, 1e-8)
            denoised = x + epsilon
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
            if debug:
                e_norm = torch.norm(epsilon).item()
                dt_val = (sigma_next - sigma_current).item() if torch.is_tensor(sigma_next) else float(sigma_next - sigma_current)
                print(f"euler step {step_index} [SKIPPED-{skip_method}]: e_norm={e_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")

    if not should_skip:
        # CALL MODEL: Normal path
        denoised = model(x, sigma_current * s_in, **extra_args)
        if skip_stats is not None:
            skip_stats["model_calls"] += 1

    # Karras ODE derivative: d = (x - denoised) / sigma
    # This is the standard k-diffusion formulation
    d = (x - denoised) / sigma_current

    # Euler step in sigma space: x = x + d * dt
    dt = sigma_next - sigma_current
    x = x + d * dt

    # Store REAL epsilon for extrapolation/learning (append full history for this run)
    if not was_skipped:
        epsilon = denoised - noisy_latent
        epsilon_history.append(epsilon)
        # Universal learning update only when enough REAL history exists (>=3)
        if len(epsilon_history) >= 3:
            # Compute predictor-matched epsilon_hat from REAL history
            if predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon) + 1e-8)).item()
                # EMA update with smoothing_beta and clamp
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                # clamps (hidden constants)
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
    return x, learning_ratio


def sample_step_ddim(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                     epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                     step_index, total_steps, skip_mode="none", debug=False):
    """DDIM deterministic step (eta=0) with optional skipping.

    Formula: x_next = x0 + (sigma_next / sigma_current) * (x - x0), where x0 = denoised.
    On skips, use extrapolated epsilon_hat to form x0_hat = x + epsilon_hat_scaled.
    """
    x = noisy_latent

    # Decide skip
    should_skip, skip_method = should_skip_model_call(
        1.0, step_index, total_steps, skip_mode, epsilon_history
    )

    was_skipped = False
    if should_skip and skip_method is not None:
        # Predictor from REAL history
        if skip_method == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)

        if epsilon_hat is None or torch.isnan(epsilon_hat).any():
            should_skip = False
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)

            x0_hat = x + epsilon_hat
            scale = (sigma_next / sigma_current)
            x = x0_hat + scale * (x - x0_hat)
            was_skipped = True
            if debug:
                e_norm = torch.norm(epsilon_hat).item()
                print(f"ddim step {step_index} [SKIPPED-{skip_method}]: e_norm={e_norm:.2f}, L={learning_ratio:.4f}")

    if not should_skip:
        # REAL call
        denoised = model(x, sigma_current * s_in, **extra_args)
        # Update: x_next = denoised + (sigma_next / sigma_current) * (x - denoised)
        scale = (sigma_next / sigma_current)
        x = denoised + scale * (x - denoised)

        # Learning update (append REAL epsilon and update L if ≥3 REAL eps)
        epsilon_real = denoised - noisy_latent
        epsilon_history.append(epsilon_real)
        if len(epsilon_history) >= 3:
            if predictor_type == "richardson":
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
                    print(f"ddim step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    return x, learning_ratio


def sample_step_dpmpp_2m(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                         epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                         step_index, total_steps, skip_mode="none", skip_stats=None, debug=False):
    """DPM++ 2M (second-order multistep) with learning + skip.

    Update: x_next = x + dt * [ (3/2)·d_n − (1/2)·d_{n−1} ], with d = (x − denoised)/sigma.
    First step falls back to Euler.
    On skip, use epsilon_hat (scaled by 1/L) to form d_n.
    """
    x = noisy_latent

    # Count step
    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Skip decision
    should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history)

    d_prev = None
    if sigma_previous is not None and len(epsilon_history) >= 1:
        eps_prev = epsilon_history[-1]
        d_prev = -(eps_prev) / sigma_previous

    if should_skip and skip_method is not None:
        if skip_method == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)

        if epsilon_hat is None or torch.isnan(epsilon_hat).any():
            should_skip = False
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
            if debug:
                d_norm = torch.norm(d_curr).item()
                print(f"dpmpp_2m step {step_index} [SKIPPED-{skip_method}]: d_norm={d_norm:.2f}, L={learning_ratio:.4f}")
            return x, learning_ratio

    # REAL call
    denoised = model(x, sigma_current * s_in, **extra_args)
    eps_curr = denoised - x
    d_curr = -eps_curr / sigma_current
    dt = sigma_next - sigma_current
    if d_prev is not None:
        x = x + dt * (1.5 * d_curr - 0.5 * d_prev)
    else:
        x = x + dt * d_curr
    if skip_stats is not None:
        skip_stats["model_calls"] += 1

    # Learning update
    epsilon_history.append(eps_curr)
    if len(epsilon_history) >= 3:
        if predictor_type == "richardson":
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

    return x, learning_ratio


def sample_step_dpmpp_2s(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                         epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                         step_index, total_steps, skip_mode="none", debug=False):
    """DPM++ 2S (two-stage ODE) with learning + skip.

    Two real evaluations:
      d1 at (x, sigma_current), predictor x_pred = x + dt*d1
      d2 at (x_pred, sigma_next), corrector: x_next = x + dt*0.5*(d1 + d2)
    On skip, use Euler-like inter-step update with epsilon_hat.
    """
    x = noisy_latent

    # Final step: avoid division by zero at sigma_next ~ 0; land on denoised
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else float(sigma_next)
    if abs(sigma_next_value) <= 1e-8:
        den = model(x, sigma_current * s_in, **extra_args)
        x = den
        # Learning update (REAL)
        eps_real = den - noisy_latent
        epsilon_history.append(eps_real)
        if len(epsilon_history) >= 3:
            if predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps_real) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                if learning_ratio < 0.5:
                    learning_ratio = 0.5
                elif learning_ratio > 2.0:
                    learning_ratio = 2.0
                if debug:
                    print(f"dpmpp_2s step {step_index} (final) [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")
        if debug:
            print(f"dpmpp_2s step {step_index} (final step): landing on denoised")
        return x, learning_ratio

    # Skip decision
    should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history)

    if should_skip and skip_method is not None:
        if skip_method == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is None or torch.isnan(epsilon_hat).any():
            should_skip = False
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            d = -(epsilon_hat) / sigma_current
            dt = sigma_next - sigma_current
            x = x + dt * d
            if debug:
                e_norm = torch.norm(epsilon_hat).item()
                print(f"dpmpp_2s step {step_index} [SKIPPED-{skip_method}]: e_norm={e_norm:.2f}, L={learning_ratio:.4f}")
            return x, learning_ratio

    # REAL evaluations
    den1 = model(x, sigma_current * s_in, **extra_args)
    d1 = (x - den1) / sigma_current
    dt = sigma_next - sigma_current
    x_pred = x + dt * d1
    den2 = model(x_pred, sigma_next * s_in, **extra_args)
    d2 = (x_pred - den2) / sigma_next
    x = x + dt * 0.5 * (d1 + d2)

    # Learning update from stage-1 epsilon
    eps_real = den1 - noisy_latent
    epsilon_history.append(eps_real)
    if len(epsilon_history) >= 3:
        if predictor_type == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is not None:
            learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps_real) + 1e-8)).item()
            learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
            if learning_ratio < 0.5:
                learning_ratio = 0.5
            elif learning_ratio > 2.0:
                learning_ratio = 2.0
            if debug:
                print(f"dpmpp_2s step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    if debug:
        d1n = torch.norm(d1).item(); d2n = torch.norm(d2).item()
        print(f"dpmpp_2s step {step_index}: d1_norm={d1n:.2f}, d2_norm={d2n:.2f}")

    return x, learning_ratio


def _ab2_update(x, dt, d_curr, d_prev=None):
    if d_prev is not None:
        return x + dt * (1.5 * d_curr - 0.5 * d_prev)
    else:
        return x + dt * d_curr


def sample_step_lms(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                    epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                    step_index, total_steps, skip_mode="none", skip_stats=None, debug=False):
    """LMS (AB2 baseline) with learning + skip.

    d = (x - denoised)/sigma; x_next = x + dt * [ (3/2)·d_n − (1/2)·d_{n−1} ]
    """
    x = noisy_latent
    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    should_skip, skip_method = should_skip_model_call(1.0, step_index, total_steps, skip_mode, epsilon_history)

    d_prev = None
    if sigma_previous is not None and len(epsilon_history) >= 1:
        d_prev = -(epsilon_history[-1]) / sigma_previous

    if should_skip and skip_method is not None:
        epsilon_hat = extrapolate_epsilon_richardson(epsilon_history) if skip_method == "richardson" else extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is None or torch.isnan(epsilon_hat).any():
            should_skip = False
        else:
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)
            d_curr = -epsilon_hat / sigma_current
            dt = sigma_next - sigma_current
            x = _ab2_update(x, dt, d_curr, d_prev)
            if skip_stats is not None:
                skip_stats["skipped"] += 1
            if debug:
                d_norm = torch.norm(d_curr).item()
                print(f"lms step {step_index} [SKIPPED-{skip_method}]: d_norm={d_norm:.2f}, L={learning_ratio:.4f}")
            return x, learning_ratio

    # REAL call
    den = model(x, sigma_current * s_in, **extra_args)
    eps = den - x
    d_curr = -eps / sigma_current
    dt = sigma_next - sigma_current
    x = _ab2_update(x, dt, d_curr, d_prev)
    if skip_stats is not None:
        skip_stats["model_calls"] += 1

    # Learning
    epsilon_history.append(eps)
    if len(epsilon_history) >= 3:
        epsilon_hat = extrapolate_epsilon_richardson(epsilon_history) if predictor_type == "richardson" else extrapolate_epsilon_linear(epsilon_history)
        if epsilon_hat is not None:
            learn_obs = (torch.norm(epsilon_hat) / (torch.norm(eps) + 1e-8)).item()
            learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
            if learning_ratio < 0.5:
                learning_ratio = 0.5
            elif learning_ratio > 2.0:
                learning_ratio = 2.0
            if debug:
                print(f"lms step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    if debug:
        dn = torch.norm(d_curr).item()
        print(f"lms step {step_index}: d_norm={dn:.2f}{', AB2' if d_prev is not None else ', Euler'}")

    return x, learning_ratio


def sample_step_plms(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                     epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                     step_index, total_steps, skip_mode="none", skip_stats=None, debug=False):
    """PLMS (baseline AB2 for now) with learning + skip.

    Note: For a full PLMS (PNDM) 4-step, we'd need sigma history for prior steps.
    This baseline uses AB2 until we thread sigma history; still useful and consistent with LMS.
    """
    # For now, mirror LMS AB2 behavior
    return sample_step_lms(model, noisy_latent, sigma_current, sigma_next, sigma_previous, s_in, extra_args,
                           epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                           step_index, total_steps, skip_mode, skip_stats, debug)

# Rebind local names to refactored implementations (ensures imports take precedence)
from .samplers.euler import sample_step_euler as _euler_impl
from .samplers.res2m import sample_step_res_2m as _res2m_impl
from .samplers.res2s import sample_step_res_2s as _res2s_impl
from .samplers.ddim import sample_step_ddim as _ddim_impl
from .samplers.dpmpp_2m import sample_step_dpmpp_2m as _dpmpp2m_impl
from .samplers.dpmpp_2s import sample_step_dpmpp_2s as _dpmpp2s_impl
from .samplers.lms import sample_step_lms as _lms_impl

sample_step_euler = _euler_impl
sample_step_res_2m = _res2m_impl
sample_step_res_2s = _res2s_impl
sample_step_ddim = _ddim_impl
sample_step_dpmpp_2m = _dpmpp2m_impl
sample_step_dpmpp_2s = _dpmpp2s_impl
sample_step_lms = _lms_impl


def sample_step_res_2m(model, noisy_latent, sigma_current, sigma_next, sigma_previous,
                       s_in, extra_args, error_history, epsilon_history, prev_was_skipped, step_index, total_steps,
                       adaptive_mode="none", smoothing_beta=0.9, smoothed_error_ratio=1.0,
                       learning_ratio=1.0, predictor_type="linear",
                       skip_mode="none", skip_stats=None, debug=False):
    """res_2m: 2-multistep method using history from previous steps.

    Matches RES4LYF implementation:
    - Stores denoised predictions in history (not epsilon directly)
    - Recomputes epsilon from stored denoised each step
    - Uses c2 = (-h_prev / h) for multistep coefficients
    """
    x_0 = noisy_latent  # Starting point for this step

    # Update skip statistics
    if skip_stats is not None:
        skip_stats["total_steps"] += 1

    # Check if we should skip the model call
    should_skip, skip_method = should_skip_model_call(
        smoothed_error_ratio, step_index, total_steps, skip_mode, epsilon_history
    )

    # Get epsilon: either from model call or extrapolation
    was_skipped = False  # Track if this step used extrapolation

    if should_skip and skip_method is not None:
        # SKIP: Use extrapolation
        if skip_method == "linear":
            epsilon_current = extrapolate_epsilon_linear(epsilon_history)
        elif skip_method == "richardson":
            epsilon_current = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_current = None

        # Safety check: if extrapolation failed, fall back to model call
        if epsilon_current is None or torch.isnan(epsilon_current).any():
            should_skip = False
            if debug:
                print(f"res_2m step {step_index}: extrapolation failed, falling back to model call")

        if should_skip and epsilon_current is not None:
            # Successful skip - reconstruct denoised from extrapolated epsilon
            if len(epsilon_history) >= 3:
                epsilon_current = epsilon_current / max(learning_ratio, 1e-8)
            denoised = x_0 + epsilon_current
            was_skipped = True
            if skip_stats is not None:
                skip_stats["skipped"] += 1
            if debug:
                e_norm = torch.norm(epsilon_current).item()
                print(f"res_2m step {step_index} [SKIPPED-{skip_method}]: e_norm={e_norm:.2f}, L={learning_ratio:.4f}")

    if not should_skip:
        # CALL MODEL: Normal path
        denoised = model(noisy_latent, sigma_current * s_in, **extra_args)
        epsilon_current = denoised - x_0
        if skip_stats is not None:
            skip_stats["model_calls"] += 1

    # Step size in log space: h = -log(sigma_next / sigma_current)
    h = -torch.log(sigma_next / sigma_current)

    # Check if this is the final step (sigma_next = 0)
    # RES4LYF line 178: if sigma_next == 0
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next
    is_final_step = (sigma_next_value == 0)

    # Check if we have history and can use multistep
    # RES4LYF stores denoised in data_[] array, loads it as: eps_[1] = -(x_0 - data_[1])
    if len(error_history) >= 1 and sigma_previous is not None and not is_final_step:
        # Load previous denoised from history and compute epsilon from it
        # RES4LYF line 215: eps_[1] = -(x_0 - data_[1]) = data_[1] - x_0
        denoised_previous = error_history[-1]
        epsilon_previous = denoised_previous - x_0

        # Multistep coefficient: RES4LYF line 808: c2 = (-h_prev / h).item()
        h_prev = -torch.log(sigma_current / sigma_previous)
        c2 = (-h_prev / h).item()

        # Phi function weights: RES4LYF lines 889-890
        # b2 = φ(2)/c2, b1 = φ(1) - b2
        phi_1 = phi_function(order=1, step_size=-h)
        phi_2 = phi_function(order=2, step_size=-h)

        b2_base = phi_2 / c2
        b1_base = phi_1 - b2_base

        # Adaptive weight adjustment based on error ratio
        # IMPORTANT: Only calculate error_ratio on real model calls, not extrapolated epsilon
        if adaptive_mode != "none" and not was_skipped:
            # Calculate error ratio (only on real model calls)
            error_curr = torch.norm(epsilon_current).item()
            error_prev = torch.norm(epsilon_previous).item()
            error_ratio = error_curr / (error_prev + 1e-8)  # Avoid division by zero

            if adaptive_mode == "learning":
                # MODE 2: EMA smoothed adjustment (learned pattern)
                smoothed_error_ratio_next = (smoothing_beta * smoothed_error_ratio +
                                             (1 - smoothing_beta) * error_ratio)
                adjustment = 1.0 / smoothed_error_ratio_next
                adjustment = max(0.5, min(2.0, adjustment))  # Clamp to [0.5, 2.0]
            else:
                adjustment = 1.0
                smoothed_error_ratio_next = 1.0

            # Apply adjustment to weights
            b1_adjusted = b1_base * adjustment
            b2_adjusted = b2_base / adjustment

            # Normalize to preserve sum (maintains phi_1 constraint)
            sum_adjusted = b1_adjusted + b2_adjusted
            sum_target = b1_base + b2_base  # Should equal phi_1
            scale = sum_target / sum_adjusted

            b1 = b1_adjusted * scale
            b2 = b2_adjusted * scale
        elif adaptive_mode != "none" and was_skipped:
            # Skipped step: preserve previous smoothed_error_ratio, use baseline weights
            # Don't poison the adaptive system with extrapolated epsilon
            b1 = b1_base
            b2 = b2_base
            adjustment = 1.0
            smoothed_error_ratio_next = smoothed_error_ratio  # Preserve previous value
            error_ratio = None  # Mark as not calculated
        else:
            # No adaptation (baseline RES2M)
            b1 = b1_base
            b2 = b2_base
            adjustment = 1.0
            smoothed_error_ratio_next = 1.0
            error_ratio = None

        # Integration: RES4LYF line 364: x = x_0 + h * rk.b_k_sum(eps_, 0)
        # For 2-multistep: b = [b1, b2], eps_ = [eps_current, eps_previous]
        # So: b_k_sum = b1*eps_current + b2*eps_previous
        x = x_0 + h * (b1 * epsilon_current + b2 * epsilon_previous)

        if debug:
            eps_prev_norm = torch.norm(epsilon_previous).item()
            eps_curr_norm = torch.norm(epsilon_current).item()
            if adaptive_mode != "none":
                # Only print immediate EXTRAPOLATED case here; REAL case is printed after learning update
                if error_ratio is None:
                    print(
                        f"res_2m step {step_index} [learning] [EXTRAPOLATED]: "
                        f"baseline φ-weights (adaptive error_ratio preserved); ε̂ scaled by 1/L={learning_ratio:.4f}; "
                        f"b1={b1.item():.4f}, b2={b2.item():.4f}"
                    )
            else:
                print(f"res_2m step {step_index}: eps_prev_norm={eps_prev_norm:.2f}, eps_curr_norm={eps_curr_norm:.2f}, "
                      f"c2={c2:.4f}, b1={b1.item():.4f}, b2={b2.item():.4f}")
    else:
        # First step / post-skip reanchor / final step
        if is_final_step:
            # Final step: sigma_next = 0
            # Return denoised directly (Euler method for final step)
            # Note: Computing h = -log(0/sigma) would give infinity, causing NaN
            # Full DEIS final step would require porting get_deis_coeff_list() from res4lyf
            # For now, standard Euler works perfectly for the final step
            x = denoised
            if debug:
                print(f"res_2m step {step_index} (final step): using Euler")
        else:
            # Use standard Euler integration when we cannot form a valid previous step
            # Reason classification improves log clarity
            if prev_was_skipped:
                reason = "post-skip reanchor"
            elif sigma_previous is None or len(error_history) == 0:
                reason = "first step"
            else:
                reason = "no-history reanchor"
            x = x_0 + h * epsilon_current
            if debug:
                print(f"res_2m step {step_index} ({reason}): using Euler")

        # No adaptation on first/final steps
        smoothed_error_ratio_next = 1.0

    # Store denoised for NEXT step (include SKIPPED to preserve multistep continuity)
    error_history.append(denoised)
    if len(error_history) > 2:
        error_history.pop(0)

    # Store REAL epsilon only for extrapolation/learning; keep full history (no cap)
    if not was_skipped:
        epsilon_history.append(epsilon_current)
        # Universal learning update when enough REAL history exists (>=3)
        if len(epsilon_history) >= 3:
            if predictor_type == "richardson":
                epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
            else:
                epsilon_hat = extrapolate_epsilon_linear(epsilon_history)
            if epsilon_hat is not None:
                learn_obs = (torch.norm(epsilon_hat) / (torch.norm(epsilon_current) + 1e-8)).item()
                learning_ratio = smoothing_beta * learning_ratio + (1.0 - smoothing_beta) * learn_obs
                if learning_ratio < 0.5:
                    learning_ratio = 0.5
                elif learning_ratio > 2.0:
                    learning_ratio = 2.0
                if debug:
                    if adaptive_mode != "none" and 'error_ratio' in locals() and error_ratio is not None:
                        # Combined one-line print for learning mode on REAL step
                        print(
                            f"res_2m step {step_index} [learning] [REAL]: "
                            f"err_ratio={error_ratio:.4f}, adjust={adjustment:.4f}, "
                            f"b1={b1.item():.4f}({b1_base.item():.4f}), b2={b2.item():.4f}({b2_base.item():.4f})"
                            f" | learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}"
                        )
                    elif adaptive_mode != "none":
                        # If for any reason error_ratio wasn't available, still show learning update succinctly
                        print(f"res_2m step {step_index} [LEARN]: learn_obs={learn_obs:.4f}, L={learning_ratio:.4f}, beta={smoothing_beta}")

    return x, smoothed_error_ratio_next, learning_ratio, was_skipped


def sample_step_res_2s(model, noisy_latent, sigma_current, sigma_next, s_in, extra_args,
                       epsilon_history, learning_ratio, smoothing_beta, predictor_type,
                       step_index, total_steps, debug=False, skip_mode="none"):
    """res_2s: 2-stage exponential integrator (baseline, no skipping).

    - Stage 1: Evaluate at current sigma
    - Stage 2: Evaluate at midpoint sigma (geometric in log-sigma)
    - Combine with phi-based weights
    - Update universal learning ratio on REAL steps (epsilon_history REAL-only)
    """
    noisy_latent_at_step_start = noisy_latent

    # Inter-step skip support (baseline: Euler-like update with ε̂)
    should_skip, skip_method = should_skip_model_call(
        1.0,  # res_2s doesn't track error_ratio; adaptive uses bands but we'll pass 1.0
        step_index,
        total_steps,
        skip_mode,
        epsilon_history
    )
    # Note: should_skip_model_call internally checks first <2 and last 4 guards and history length.
    if should_skip and skip_method is not None:
        # Build epsilon_hat from REAL history
        if skip_method == "richardson":
            epsilon_hat = extrapolate_epsilon_richardson(epsilon_history)
        else:
            epsilon_hat = extrapolate_epsilon_linear(epsilon_history)

        # Fallback if missing/NaN
        if epsilon_hat is None or torch.isnan(epsilon_hat).any():
            should_skip = False
        else:
            # Scale by learning ratio if we have ≥3 REAL eps in history
            if len(epsilon_history) >= 3:
                epsilon_hat = epsilon_hat / max(learning_ratio, 1e-8)

            # Euler-like update using epsilon_hat
            d = -(epsilon_hat) / sigma_current
            dt = sigma_next - sigma_current
            noisy_latent = noisy_latent + d * dt

            if debug:
                e_norm = torch.norm(epsilon_hat).item()
                dt_val = (sigma_next - sigma_current).item() if torch.is_tensor(sigma_next) else float(sigma_next - sigma_current)
                print(f"res_2s step {step_index} [SKIPPED-{skip_method}]: e_norm={e_norm:.2f}, L={learning_ratio:.4f}, dt={dt_val:.4f}")

            return noisy_latent, learning_ratio

    # Step size in log space
    step_size = -torch.log(sigma_next / sigma_current)

    # Check if this is the final step (sigma_next = 0)
    # When sigma_next = 0, step_size → ∞, causing numerical issues
    # RES4LYF switches to ralston for final step; we use Euler for simplicity
    sigma_next_value = sigma_next.item() if torch.is_tensor(sigma_next) else sigma_next
    is_final_step = (sigma_next_value == 0)

    if is_final_step:
        # Final step: land on denoised directly (avoid infinite step size)
        model_prediction = model(noisy_latent, sigma_current * s_in, **extra_args)
        noisy_latent = model_prediction
        if debug:
            print(f"res_2s step {step_index} (final step): using Euler")
        # Learning update on REAL call
        epsilon_real = model_prediction - noisy_latent_at_step_start
        epsilon_history.append(epsilon_real)
        if len(epsilon_history) >= 3:
            if predictor_type == "richardson":
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

    midpoint_fraction = 0.5  # Evaluate at midpoint

    # Phi function weights for 2-stage method
    phi_1_value = phi_function(order=1, step_size=-step_size)
    phi_2_value = phi_function(order=2, step_size=-step_size)

    # Weights for final integration
    weight_stage2 = phi_2_value / midpoint_fraction
    weight_stage1 = phi_1_value - weight_stage2

    # Weight for advancing to stage 2
    phi_1_at_midpoint = phi_function(order=1, step_size=-step_size * midpoint_fraction)
    stage2_advance_weight = midpoint_fraction * phi_1_at_midpoint

    # Stage 1: Evaluate at current sigma
    model_prediction_stage1 = model(noisy_latent, sigma_current * s_in, **extra_args)
    error_stage1 = -(noisy_latent_at_step_start - model_prediction_stage1)  # epsilon at current sigma

    # Stage 2: Evaluate at midpoint sigma
    sigma_midpoint = torch.exp(-(-torch.log(sigma_current) + step_size * midpoint_fraction))
    noisy_latent_midpoint = noisy_latent_at_step_start + (step_size * stage2_advance_weight) * error_stage1

    model_prediction_stage2 = model(noisy_latent_midpoint, sigma_midpoint * s_in, **extra_args)
    error_stage2 = -(noisy_latent_at_step_start - model_prediction_stage2)  # epsilon at midpoint

    # Final integration with weighted stages
    noisy_latent = noisy_latent_at_step_start + step_size * (
        weight_stage1 * error_stage1 +
        weight_stage2 * error_stage2
    )

    if debug:
        stage1_norm = torch.norm(error_stage1).item()
        stage2_norm = torch.norm(error_stage2).item()
        print(f"res_2s step {step_index}: stage1_norm={stage1_norm:.2f}, stage2_norm={stage2_norm:.2f}, "
              f"weight_s1={weight_stage1.item():.4f}, weight_s2={weight_stage2.item():.4f}")

    # Learning update on REAL call (use epsilon at current sigma: error_stage1)
    epsilon_real = error_stage1
    epsilon_history.append(epsilon_real)
    if len(epsilon_history) >= 3:
        if predictor_type == "richardson":
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
