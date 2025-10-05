import torch


def extrapolate_epsilon_linear(epsilon_history):
    """Linear (2-point) epsilon extrapolation using last two REAL epsilons.

    Args:
        epsilon_history: list[Tensor] of REAL epsilons, oldest..newest
    Returns:
        Tensor or None
    """
    if len(epsilon_history) < 2:
        return None
    eps_prev = epsilon_history[-2]
    eps_curr = epsilon_history[-1]
    return eps_curr + (eps_curr - eps_prev)


def extrapolate_epsilon_richardson(epsilon_history):
    """Richardson (3-point) epsilon extrapolation using last three REAL epsilons.

    Args:
        epsilon_history: list[Tensor] of REAL epsilons, oldest..newest
    Returns:
        Tensor or None
    """
    if len(epsilon_history) < 3:
        return extrapolate_epsilon_linear(epsilon_history)
    eps_old = epsilon_history[-3]
    eps_prev = epsilon_history[-2]
    eps_curr = epsilon_history[-1]
    return 3 * eps_curr - 3 * eps_prev + eps_old


def extrapolate_epsilon_h4(epsilon_history):
    """4-point (cubic) epsilon extrapolation using last four REAL epsilons.

    Assumes uniform step spacing in the prediction index. Uses Lagrange
    coefficients for points at t = [-3, -2, -1, 0] to predict at t = 1:
        ε̂_{n+1} = -1·ε_{n-3} + 4·ε_{n-2} - 6·ε_{n-1} + 4·ε_{n}

    Falls back to 3-point when history is insufficient.
    """
    if len(epsilon_history) < 4:
        return extrapolate_epsilon_richardson(epsilon_history)
    eps_older = epsilon_history[-4]
    eps_old = epsilon_history[-3]
    eps_prev = epsilon_history[-2]
    eps_curr = epsilon_history[-1]
    return (-1.0) * eps_older + 4.0 * eps_old - 6.0 * eps_prev + 4.0 * eps_curr
