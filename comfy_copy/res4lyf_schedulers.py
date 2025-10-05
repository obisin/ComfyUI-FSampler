import math
import torch
# functions copied from RES4LYF

def _get_model_sigma_bounds(model):
    # Try the common ComfyUI paths to extract sigma_min/max
    sigma_min = 0.0291675
    sigma_max = 14.614642
    try:
        if hasattr(model, "get_model_object"):
            ms = model.get_model_object("model_sampling")
            sigma_min = float(ms.sigma_min)
            sigma_max = float(ms.sigma_max)
        elif hasattr(model, "inner_model") and hasattr(model.inner_model, "inner_model") and hasattr(model.inner_model.inner_model, "model_sampling"):
            ms = model.inner_model.inner_model.model_sampling
            sigma_min = float(ms.sigma_min)
            sigma_max = float(ms.sigma_max)
        elif hasattr(model, "model") and hasattr(model.model, "model_sampling"):
            ms = model.model.model_sampling
            sigma_min = float(ms.sigma_min)
            sigma_max = float(ms.sigma_max)
    except Exception:
        pass
    return sigma_min, sigma_max


def get_bong_tangent_sigmas(model, steps: int) -> torch.Tensor:
    # Defaults: from RES4LYF
    #   offset = steps (matches res4lyf tan default of 20 when steps=20)
    #   slope = 20.0
    #   start/end = model sigma max/min
    #   sgm = False, pad = True (Comfy expects trailing zero)
    sigma_min, sigma_max = _get_model_sigma_bounds(model)
    # Res4LYF tan_scheduler UI defaults:
    #   steps=20, offset=20, slope=20, start=20, end=20, sgm=False, pad=False
    # For practical use with arbitrary models, map start/end to model sigma bounds,
    # while keeping offset/slope/sgm/pad exactly as defaults.
    offset = 20.0
    slope = 20.0
    start = float(sigma_max)
    end = float(sigma_min)
    sgm = False
    pad = False

    def s_val(x):
        return ((2 / math.pi) * math.atan(-slope * (x - offset)) + 1) / 2

    local_steps = steps + 1 if sgm else steps
    smax = s_val(0)
    smin = s_val(local_steps - 1)
    srange = smax - smin if (smax - smin) != 0 else 1e-12
    sscale = start - end

    vals = [(((s_val(x) - smin) * (1.0 / srange)) * sscale + end) for x in range(local_steps)]
    if sgm and len(vals) > 0:
        vals = vals[:-1]

    sigmas = torch.tensor(vals, dtype=torch.float32)
    if pad:
        sigmas = torch.cat([sigmas, torch.tensor([0.0], dtype=sigmas.dtype)])
    return sigmas


def _tan_curve_list(steps: int, slope: float, pivot: float, start: float, end: float):
    # Replicate res4lyf tan mapping, scaled to [end, start]
    from math import atan, pi
    smax = ((2 / pi) * atan(-slope * (0 - pivot)) + 1) / 2
    smin = ((2 / pi) * atan(-slope * ((steps - 1) - pivot)) + 1) / 2
    srange = smax - smin if (smax - smin) != 0 else 1e-12
    sscale = start - end
    vals = [((((2 / pi) * atan(-slope * (x - pivot)) + 1) / 2) - smin) * (1.0 / srange) * sscale + end for x in range(steps)]
    return vals


def get_bong_tangent_2_sigmas(model, steps: int) -> torch.Tensor:
    # Mirror tan_scheduler_2stage with internal defaults; map start/middle/end to model bounds
    sigma_min, sigma_max = _get_model_sigma_bounds(model)
    steps_total = steps + 2
    midpoint = steps_total // 2
    stage_2_len = steps_total - midpoint
    stage_1_len = steps_total - stage_2_len

    # Defaults derived proportionally to res4lyf defaults: pivot_1=mid/2, pivot_2=mid + (stage_2_len/2)
    pivot_1 = stage_1_len // 2
    pivot_2 = midpoint + (stage_2_len // 2)
    slope_1 = 1.0
    slope_2 = 1.0
    start = float(sigma_max)
    end = float(sigma_min)
    # middle as geometric mean to fit log-sigma shape
    middle = float((sigma_max * sigma_min) ** 0.5)

    tan1 = _tan_curve_list(stage_1_len, slope_1, pivot_1, start, middle)
    tan2 = _tan_curve_list(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
    # Drop overlap of join
    tan1 = tan1[:-1]
    vals = tan1 + tan2
    return torch.tensor(vals, dtype=torch.float32)


def get_bong_tangent_2_simple_sigmas(model, steps: int) -> torch.Tensor:
    # Mirror tan_scheduler_2stage_simple with internal defaults
    sigma_min, sigma_max = _get_model_sigma_bounds(model)
    steps_total = steps + 2
    # Default fractional pivots and slopes (res4lyf defaults are 1.0, but scale meaning differs).
    pivot_1_frac = 0.5
    pivot_2_frac = 0.75
    slope_1 = 1.0 / max(steps_total / 40.0, 1e-6)
    slope_2 = 1.0 / max(steps_total / 40.0, 1e-6)

    midpoint = int((steps_total * pivot_1_frac + steps_total * pivot_2_frac) / 2)
    pivot_1 = int(steps_total * pivot_1_frac)
    pivot_2 = int(steps_total * pivot_2_frac)

    stage_2_len = steps_total - midpoint
    stage_1_len = steps_total - stage_2_len

    start = float(sigma_max)
    end = float(sigma_min)
    middle = float((sigma_max * sigma_min) ** 0.5)

    tan1 = _tan_curve_list(stage_1_len, slope_1, pivot_1, start, middle)
    tan2 = _tan_curve_list(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)
    tan1 = tan1[:-1]
    vals = tan1 + tan2
    return torch.tensor(vals, dtype=torch.float32)


def get_constant_sigmas(model, steps: int) -> torch.Tensor:
    # Mirror constant_scheduler defaults, mapping to model bounds
    sigma_min, sigma_max = _get_model_sigma_bounds(model)
    value_start = float(sigma_max)
    value_end = float(sigma_min)
    cutoff_percent = 1.0
    total = steps + 1
    cutoff_step = int(round(steps * cutoff_percent)) + 1
    first = torch.ones(total if cutoff_step > total else cutoff_step, dtype=torch.float32) * value_start
    rest_len = total - first.shape[0]
    if rest_len > 0:
        rest = torch.ones(rest_len, dtype=torch.float32) * value_end
        sigmas = torch.cat([first, rest])
    else:
        sigmas = first
    return sigmas
