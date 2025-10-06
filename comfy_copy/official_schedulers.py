"""
Official scheduler implementations copied from RES4LYF-main.
Self-contained to avoid import dependencies.
"""
import math
import torch


def get_bong_tangent_sigmas(steps, slope, pivot, start, end):
    """
    Core bong_tangent sigma calculation using arctangent curve.

    Copied from RES4LYF-main/sigmas.py:4065-4074
    """
    pi = math.pi
    atan = math.atan

    smax = ((2/pi)*atan(-slope*(0-pivot))+1)/2
    smin = ((2/pi)*atan(-slope*((steps-1)-pivot))+1)/2

    srange = smax-smin
    sscale = start - end

    sigmas = [  ( (((2/pi)*atan(-slope*(x-pivot))+1)/2) - smin) * (1/srange) * sscale + end    for x in range(steps)]

    return sigmas


def bong_tangent_scheduler(model_sampling, steps, start=1.0, middle=0.5, end=0.0, pivot_1=0.6, pivot_2=0.6, slope_1=0.2, slope_2=0.2, pad=False):
    """
    Two-stage bong_tangent scheduler with flexible pivot points.

    Copied from RES4LYF-main/sigmas.py:4076-4098

    This is actually a two-stage implementation (similar to bong_tangent_2):
    - Stage 1: from start to geometric mean middle
    - Stage 2: from middle to end
    - Uses separate pivots and slopes for each stage
    """
    steps += 2

    midpoint = int( (steps*pivot_1 + steps*pivot_2) / 2 )
    pivot_1 = int(steps * pivot_1)
    pivot_2 = int(steps * pivot_2)

    slope_1 = slope_1 / (steps/40)
    slope_2 = slope_2 / (steps/40)

    stage_2_len = steps - midpoint
    stage_1_len = steps - stage_2_len

    tan_sigmas_1 = get_bong_tangent_sigmas(stage_1_len, slope_1, pivot_1, start, middle)
    tan_sigmas_2 = get_bong_tangent_sigmas(stage_2_len, slope_2, pivot_2 - stage_1_len, middle, end)

    tan_sigmas_1 = tan_sigmas_1[:-1]
    if pad:
        tan_sigmas_2 = tan_sigmas_2+[0]

    tan_sigmas = torch.tensor(tan_sigmas_1 + tan_sigmas_2)

    return tan_sigmas


def get_bong_tangent_sigmas_official(model, steps):
    """
    Wrapper for official RES4LYF bong_tangent scheduler with default parameters.

    Parameters match RES4LYF-main/__init__.py registration:
    - pivot_1=0.6, pivot_2=0.6: Pivot points for two stages
    - slope_1=0.2, slope_2=0.2: Slope steepness for each stage
    - Two-stage curve with geometric mean middle

    Key difference from custom FSampler bong_tangent:
    - Official: Two-stage with flexible pivots, geometric mean middle
    - Custom: Single-stage with hardcoded offset=20, slope=20
    """
    try:
        model_sampling = model.get_model_object("model_sampling")
    except Exception:
        # Fallback if model doesn't support get_model_object
        model_sampling = None

    sigmas = bong_tangent_scheduler(
        model_sampling,
        steps,
        start=1.0,
        middle=0.5,  # Will be overridden by geometric mean in scheduler
        end=0.0,
        pivot_1=0.6,
        pivot_2=0.6,
        slope_1=0.2,
        slope_2=0.2,
        pad=False
    )

    # Ensure trailing zero for Comfy parity
    if float(sigmas[-1]) != 0.0:
        sigmas = torch.cat([sigmas, torch.tensor([0.0], dtype=sigmas.dtype)])

    return sigmas
