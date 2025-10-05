# Scheduler functions copied from ComfyUI k_diffusion
# Original source: ComfyUI/comfy/k_diffusion/sampling.py

import math
import torch
from .utils import append_zero


def get_sigmas_karras(n, sigma_min, sigma_max, rho=7., device='cpu'):
    """Constructs the noise schedule of Karras et al. (2022)."""
    ramp = torch.linspace(0, 1, n, device=device)
    min_inv_rho = sigma_min ** (1 / rho)
    max_inv_rho = sigma_max ** (1 / rho)
    sigmas = (max_inv_rho + ramp * (min_inv_rho - max_inv_rho)) ** rho
    return append_zero(sigmas).to(device)


def get_sigmas_exponential(n, sigma_min, sigma_max, device='cpu'):
    """Constructs an exponential noise schedule."""
    sigmas = torch.linspace(math.log(sigma_max), math.log(sigma_min), n, device=device).exp()
    return append_zero(sigmas)


def get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=1., device='cpu'):
    """Constructs an polynomial in log sigma noise schedule."""
    ramp = torch.linspace(1, 0, n, device=device) ** rho
    sigmas = torch.exp(ramp * (math.log(sigma_max) - math.log(sigma_min)) + math.log(sigma_min))
    return append_zero(sigmas)


def get_sigmas_vp(n, beta_d=19.9, beta_min=0.1, eps_s=1e-3, device='cpu'):
    """Constructs a continuous VP noise schedule."""
    t = torch.linspace(1, eps_s, n, device=device)
    sigmas = torch.sqrt(torch.special.expm1(beta_d * t ** 2 / 2 + beta_min * t))
    return append_zero(sigmas)


def get_sigmas_laplace(n, sigma_min, sigma_max, mu=0., beta=0.5, device='cpu'):
    """Constructs the noise schedule proposed by Tiankai et al. (2024). """
    epsilon = 1e-5  # avoid log(0)
    x = torch.linspace(0, 1, n, device=device)
    clamp = lambda x: torch.clamp(x, min=sigma_min, max=sigma_max)
    lmb = mu - beta * torch.sign(0.5-x) * torch.log(1 - 2 * torch.abs(0.5-x) + epsilon)
    sigmas = clamp(torch.exp(lmb))
    return append_zero(sigmas)


# Standard scheduler names supported by this fallback module
STANDARD_SCHEDULERS = [
    # Implemented here
    "karras",            # Karras et al. (2022)
    "exponential",       # Uniform in log-sigma
    "polyexponential",   # Polynomial in log-sigma (rho)
    "vp",                # Continuous VP schedule
    "laplace",           # Laplace distribution schedule
    # Additional common ComfyUI schedulers (fallback approximations here)
    "normal",            # Uniform in sigma (fallback)
    "simple",            # Alias of normal (fallback)
    "sgm_uniform",       # Approximated by exponential (fallback)
    "ddim_uniform",      # Approximated by uniform in sigma (fallback)
    "beta",              # Approximated by VP (fallback)
    "linear_quadratic",  # Polyexponential with rho=2 (fallback)
    "kl_optimal",        # Alias of karras (fallback)
]

def get_sigmas_normal(n, sigma_min, sigma_max, device='cpu'):
    """Uniform spacing in sigma (descending). Fallback for 'normal'/'simple'."""
    sigmas = torch.linspace(float(sigma_max), float(sigma_min), n, device=device)
    return append_zero(sigmas).to(device)


def get_sigmas_simple(n, sigma_min, sigma_max, device='cpu'):
    """Alias of get_sigmas_normal (historically named 'simple')."""
    return get_sigmas_normal(n, sigma_min, sigma_max, device=device)


def get_sigmas_sgm_uniform(n, sigma_min, sigma_max, device='cpu'):
    """Approximate SGM-uniform by uniform in log-sigma (same as exponential)."""
    return get_sigmas_exponential(n, sigma_min, sigma_max, device=device)


def get_sigmas_ddim_uniform(n, sigma_min, sigma_max, device='cpu'):
    """Approximate DDIM-uniform by uniform sigma spacing (conservative fallback)."""
    return get_sigmas_normal(n, sigma_min, sigma_max, device=device)


def get_sigmas_beta(n, sigma_min, sigma_max, device='cpu'):
    """Approximate 'beta' schedule by VP schedule (commonly used for DDPM/VP)."""
    return get_sigmas_vp(n, device=device)


def get_sigmas_linear_quadratic(n, sigma_min, sigma_max, device='cpu'):
    """Piecewise-like curvature via polyexponential with rho=2 (quadratic in log-sigma)."""
    return get_sigmas_polyexponential(n, sigma_min, sigma_max, rho=2.0, device=device)


def get_sigmas_kl_optimal(n, sigma_min, sigma_max, device='cpu'):
    """Use Karras schedule as a strong KL-friendly default."""
    return get_sigmas_karras(n, sigma_min, sigma_max, device=device)


def get_sigmas(scheduler_name, steps, sigma_min=0.03, sigma_max=14.6, device='cpu'):
    """Get sigma schedule using specified scheduler.

    Currently implements the mathematical schedulers we've copied from ComfyUI.
    """
    if scheduler_name == "karras":
        return get_sigmas_karras(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "exponential":
        return get_sigmas_exponential(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "polyexponential":
        return get_sigmas_polyexponential(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "vp":
        return get_sigmas_vp(steps, device=device)
    elif scheduler_name == "laplace":
        return get_sigmas_laplace(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "normal":
        return get_sigmas_normal(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "simple":
        return get_sigmas_simple(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "sgm_uniform":
        return get_sigmas_sgm_uniform(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "ddim_uniform":
        return get_sigmas_ddim_uniform(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "beta":
        return get_sigmas_beta(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "linear_quadratic":
        return get_sigmas_linear_quadratic(steps, sigma_min, sigma_max, device=device)
    elif scheduler_name == "kl_optimal":
        return get_sigmas_kl_optimal(steps, sigma_min, sigma_max, device=device)
    else:
        raise ValueError(f"Scheduler '{scheduler_name}' not implemented in comfy_copy. "
                        f"Available schedulers: {STANDARD_SCHEDULERS}")
