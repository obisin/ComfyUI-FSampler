L_MIN = 0.5
L_MAX = 2.0
SIGMA_ZERO_EPS = 1e-8


def update_learning_ratio(current_L: float, smoothing_beta: float, learn_obs: float) -> float:
    """EMA update for learning ratio with clamping.

    Args:
        current_L: current learning ratio value
        smoothing_beta: EMA smoothing factor (0.0–0.9999)
        learn_obs: observation ratio = ||epsilon_hat|| / (||epsilon_real|| + 1e-8)
    Returns:
        new learning ratio (clamped)
    """
    new_L = smoothing_beta * current_L + (1.0 - smoothing_beta) * learn_obs
    if new_L < L_MIN:
        new_L = L_MIN
    elif new_L > L_MAX:
        new_L = L_MAX
    return new_L


def scale_epsilon_hat(epsilon_hat, learning_ratio: float):
    """Scale predicted epsilon by 1/L (with tiny floor to avoid div-by-zero)."""
    import torch
    return epsilon_hat / max(learning_ratio, 1e-8)

