import torch


def get_eps_step_official(sigma, sigma_next, eta=0.0):
    """Official-like EPS ancestral step.

    Returns (sigma_up, sigma_down). If eta<=0, sigma_up=0 and sigma_down=sigma_next.
    Accepts scalar tensors and preserves dtype/device.
    """
    if eta is None or float(eta) <= 0.0:
        # Ensure tensor outputs matching input dtype/device
        return torch.zeros_like(sigma), sigma_next
    s = sigma.to(torch.float64)
    sn = sigma_next.to(torch.float64)
    num = torch.clamp(sn**2 * (torch.clamp(s**2, min=1e-12) - sn**2) / torch.clamp(s**2, min=1e-12), min=0.0)
    su = torch.sqrt(num) * float(eta)
    su = torch.minimum(su, sn)
    sd = torch.sqrt(torch.clamp(sn**2 - su**2, min=0.0))
    return su.to(sigma.dtype), sd.to(sigma.dtype)

