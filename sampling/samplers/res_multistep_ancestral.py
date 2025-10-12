import torch
from .res_multistep import sample_step_res_multistep


def sample_step_res_multistep_ancestral(
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
    official_comfy=False,
    explicit_skip_indices=None,
    explicit_predictor=None,
    adaptive_mode="none",
):
    """Dedicated ancestral RES multistep per-step wrapper.

    Uses the same integrator as res_multistep; ancestral behavior is governed by add_noise_ratio (eta).
    Returns (x, learning_ratio, prev_sigma_down_like) for engine chaining.
    """
    # Single canonical variant: always use the res_multistep ancestral path,
    # independent of official_comfy flag.
    return sample_step_res_multistep(
        model=model,
        noisy_latent=noisy_latent,
        sigma_current=sigma_current,
        sigma_next=sigma_next,
        sigma_previous=sigma_previous,
        old_sigma_down=old_sigma_down,
        s_in=s_in,
        extra_args=extra_args,
        error_history=error_history,
        epsilon_history=epsilon_history,
        step_index=step_index,
        total_steps=total_steps,
        learning_ratio=learning_ratio,
        smoothing_beta=smoothing_beta,
        predictor_type=predictor_type,
        add_noise_ratio=add_noise_ratio,
        add_noise_type=add_noise_type,
        skip_mode=skip_mode,
        skip_stats=skip_stats,
        debug=debug,
        protect_last_steps=protect_last_steps,
        protect_first_steps=protect_first_steps,
        anchor_interval=anchor_interval,
        max_consecutive_skips=max_consecutive_skips,
        adaptive_mode=adaptive_mode,
        explicit_skip_indices=(explicit_skip_indices if explicit_skip_indices is not None else None),
        explicit_predictor=(explicit_predictor if explicit_predictor is not None else None),
    )
