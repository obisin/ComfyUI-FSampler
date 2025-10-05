import torch
import comfy.model_sampling
# functions copied from RES4LYF

def has_nested_attr(obj, attr_path: str) -> bool:
    parts = attr_path.split('.')
    for p in parts:
        if not hasattr(obj, p):
            return False
        obj = getattr(obj, p)
    return True


def get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max=1.0):
    if sigma_up >= sigma_next and sigma_next > 0:
        if eta >= 1:
            sigma_up = sigma_next * 0.9999
        else:
            sigma_up = sigma_next * eta

    sigma_signal = sigma_max - sigma_next
    sigma_residual = torch.sqrt(sigma_next**2 - sigma_up**2)

    alpha_ratio = sigma_signal + sigma_residual
    sigma_down = sigma_residual / alpha_ratio
    return alpha_ratio, sigma_up, sigma_down


def get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max=1.0):
    alpha_ratio = (1 - sigma_next) / (1 - sigma_down)
    sigma_up = (sigma_next ** 2 - sigma_down ** 2 * alpha_ratio ** 2) ** 0.5

    if sigma_up >= sigma_next:
        alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return alpha_ratio, sigma_up, sigma_down


def get_ancestral_step_RF_var(sigma, sigma_next, eta, sigma_max=1.0):
    dtype = sigma.dtype
    sigma, sigma_next = sigma.to(torch.float64), sigma_next.to(torch.float64)
    sigma_diff = (sigma - sigma_next).abs() + 1e-10
    sigma_up = torch.sqrt(sigma_diff).to(torch.float64) * eta
    sigma_down_num = (sigma_next**2 - sigma_up**2).to(torch.float64)
    sigma_down = torch.sqrt(sigma_down_num) / ((1 - sigma_next).to(torch.float64) + torch.sqrt(sigma_down_num).to(torch.float64))
    alpha_ratio = (1 - sigma_next).to(torch.float64) / (1 - sigma_down).to(torch.float64)
    return sigma_up.to(dtype), sigma_down.to(dtype), alpha_ratio.to(dtype)


def get_ancestral_step_RF_lorentzian(sigma, sigma_next, eta, sigma_max=1.0):
    dtype = sigma.dtype
    alpha = 1 / ((sigma.to(torch.float64))**2 + 1)
    sigma_up = eta * (1 - alpha) ** 0.5
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up.to(dtype), sigma_down.to(dtype), alpha_ratio.to(dtype)


def get_ancestral_step_EPS(sigma, sigma_next, eta=1.0):
    alpha_ratio = torch.full_like(sigma, 1.0)
    if not eta or not sigma_next:
        return torch.full_like(sigma, 0.0), sigma_next, alpha_ratio
    sigma_up = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)
    sigma_down = (sigma_next ** 2 - sigma_up ** 2) ** 0.5
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_sinusoidal(sigma_next, eta, sigma_max=1.0):
    sigma_up = eta * sigma_next * torch.sin(torch.pi * sigma_next) ** 2
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_softer(sigma, sigma_next, eta, sigma_max=1.0):
    sigma_down = sigma_next * torch.sqrt(1 - (eta**2 * (sigma**2 - sigma_next**2)) / sigma**2)
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_soft(sigma, sigma_next, eta, sigma_max=1.0):
    down_ratio = (1 - eta) + eta * (sigma_next / sigma)
    sigma_down = down_ratio * sigma_next
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_soft_linear(sigma, sigma_next, eta, sigma_max=1.0):
    sigma_down = sigma_next + eta * (sigma_next - sigma)
    if sigma_down < 0:
        return torch.full_like(sigma, 0.0), sigma_next, torch.full_like(sigma, 1.0)
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_down(sigma_down, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_exp(sigma, sigma_next, eta, sigma_max=1.0):
    h = -torch.log(sigma_next / sigma)
    sigma_up = sigma_next * (1 - (-2 * eta * h).exp()) ** 0.5
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_sqrd(sigma, sigma_next, eta, sigma_max=1.0):
    sigma_hat = sigma * (1 + eta)
    sigma_up = (sigma_hat ** 2 - sigma ** 2) ** 0.5
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_ancestral_step_RF_hard(sigma_next, eta, sigma_max=1.0):
    sigma_up = sigma_next * eta
    alpha_ratio, sigma_up, sigma_down = get_alpha_ratio_from_sigma_up(sigma_up, sigma_next, eta, sigma_max)
    return sigma_up, sigma_down, alpha_ratio


def get_vpsde_step_RF(sigma, sigma_next, eta, sigma_max=1.0):
    dt = sigma - sigma_next
    sigma_up = eta * sigma * dt**0.5
    alpha_ratio = 1 - dt * (eta**2 / 4) * (1 + sigma)
    sigma_down = sigma_next - (eta / 4) * sigma * (1 - sigma) * (sigma - sigma_next)
    return sigma_up, sigma_down, alpha_ratio


def get_fuckery_step_RF(sigma, sigma_next, eta, sigma_max=1.0):
    sigma_down = (1 - eta) * sigma_next
    sigma_up = torch.sqrt(sigma_next**2 - sigma_down**2)
    alpha_ratio = torch.ones_like(sigma_next)
    return sigma_up, sigma_down, alpha_ratio


def get_res4lyf_step_with_model(model, sigma, sigma_next, eta=0.0, noise_mode="hard"):
    su, sd, alpha_ratio = torch.zeros_like(sigma), sigma_next.clone(), torch.ones_like(sigma)

    if has_nested_attr(model, "inner_model.inner_model.model_sampling"):
        model_sampling = model.inner_model.inner_model.model_sampling
    elif has_nested_attr(model, "model.model_sampling"):
        model_sampling = model.model.model_sampling
    else:
        model_sampling = None

    if model_sampling is not None and isinstance(model_sampling, comfy.model_sampling.CONST):
        sigma_var = (-1 + torch.sqrt(1 + 4 * sigma)) / 2
        if noise_mode == "hard_var" and eta > 0.0 and sigma_next > sigma_var:
            su, sd, alpha_ratio = get_ancestral_step_RF_var(sigma, sigma_next, eta)
        else:
            if noise_mode == "soft":
                su, sd, alpha_ratio = get_ancestral_step_RF_soft(sigma, sigma_next, eta)
            elif noise_mode == "softer":
                su, sd, alpha_ratio = get_ancestral_step_RF_softer(sigma, sigma_next, eta)
            elif noise_mode == "hard_sq":
                su, sd, alpha_ratio = get_ancestral_step_RF_sqrd(sigma, sigma_next, eta)
            elif noise_mode == "sinusoidal":
                su, sd, alpha_ratio = get_ancestral_step_RF_sinusoidal(sigma_next, eta)
            elif noise_mode == "exp":
                su, sd, alpha_ratio = get_ancestral_step_RF_exp(sigma, sigma_next, eta)
            elif noise_mode == "soft-linear":
                su, sd, alpha_ratio = get_ancestral_step_RF_soft_linear(sigma, sigma_next, eta)
            elif noise_mode == "lorentzian":
                su, sd, alpha_ratio = get_ancestral_step_RF_lorentzian(sigma, sigma_next, eta)
            elif noise_mode == "vpsde":
                su, sd, alpha_ratio = get_vpsde_step_RF(sigma, sigma_next, eta)
            elif noise_mode == "fuckery":
                su, sd, alpha_ratio = get_fuckery_step_RF(sigma, sigma_next, eta)
            else:
                su, sd, alpha_ratio = get_ancestral_step_RF_hard(sigma_next, eta)
    else:
        alpha_ratio = torch.full_like(sigma, 1.0)
        if noise_mode == "hard_sq":
            sd = sigma_next
            sigma_hat = sigma * (1 + eta)
            su = (sigma_hat ** 2 - sigma ** 2) ** 0.5
            sigma = sigma_hat
        elif noise_mode == "hard":
            su = eta * sigma_next
            sd = (sigma_next ** 2 - su ** 2) ** 0.5
        elif noise_mode == "exp":
            h = -torch.log(sigma_next / sigma)
            su = sigma_next * (1 - (-2 * eta * h).exp()) ** 0.5
            sd = (sigma_next ** 2 - su ** 2) ** 0.5
        else:
            su = min(sigma_next, eta * (sigma_next ** 2 * (sigma ** 2 - sigma_next ** 2) / sigma ** 2) ** 0.5)

    su = torch.nan_to_num(su, 0.0)
    sd = torch.nan_to_num(sd, float(sigma_next))
    alpha_ratio = torch.nan_to_num(alpha_ratio, 1.0)

    return su, sigma, sd, alpha_ratio


NOISE_MODE_NAMES = [
    "none",
    "hard_sq",
    "hard",
    "lorentzian",
    "soft",
    "soft-linear",
    "softer",
    "eps",
    "sinusoidal",
    "exp",
    "vpsde",
]

