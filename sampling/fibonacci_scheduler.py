"""Fibonacci-based sigma schedulers for FSampler."""
import torch
import numpy as np


def fibonacci_sequence(num_steps):
    """Generate first num_steps Fibonacci numbers."""
    if num_steps <= 0:
        return []
    elif num_steps == 1:
        return [1]
    elif num_steps == 2:
        return [1, 1]

    fibs = [1, 1]
    for i in range(2, num_steps):
        fibs.append(fibs[-1] + fibs[-2])
    return fibs


def get_fsampler_sigmas(scheduler_name, num_steps, sigma_min, sigma_max):
    """Generate sigma schedule for FSampler custom schedulers.

    Args:
        scheduler_name: "fibonacci" or "fibonacci_rev"
        num_steps: Number of sampling steps
        sigma_min: Minimum sigma value (from model)
        sigma_max: Maximum sigma value (from model)

    Returns:
        Tensor of sigma values with shape (num_steps + 1,)
        Last value is always 0.0
    """
    if scheduler_name == "fibonacci":
        # Forward fibonacci: dense at high sigma (early steps), sparse at low sigma (late steps)
        fibs = fibonacci_sequence(num_steps)
        # Normalize to 0-1 range
        total = sum(fibs)
        cumulative = [sum(fibs[:i+1]) / total for i in range(num_steps)]
        # Map to log-sigma space
        log_min = np.log(sigma_min)
        log_max = np.log(sigma_max)
        sigmas = [np.exp(log_max - (log_max - log_min) * c) for c in cumulative]
        sigmas.append(0.0)
        return torch.FloatTensor(sigmas)

    elif scheduler_name == "fibonacci_rev":
        # Reverse fibonacci: sparse at high sigma (early steps), dense at low sigma (late steps)
        fibs = fibonacci_sequence(num_steps)
        fibs_rev = list(reversed(fibs))
        total = sum(fibs_rev)
        cumulative = [sum(fibs_rev[:i+1]) / total for i in range(num_steps)]
        log_min = np.log(sigma_min)
        log_max = np.log(sigma_max)
        sigmas = [np.exp(log_max - (log_max - log_min) * c) for c in cumulative]
        sigmas.append(0.0)
        return torch.FloatTensor(sigmas)

    else:
        raise ValueError(f"Unknown FSampler scheduler: {scheduler_name}")


FSAMPLER_SCHEDULERS = ["fibonacci", "fibonacci_rev"]
