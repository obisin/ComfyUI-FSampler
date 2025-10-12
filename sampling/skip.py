import math
import torch
import re
from .extrapolation import (
    extrapolate_epsilon_linear,
    extrapolate_epsilon_richardson,
)


def _parse_hs_mode(skip_mode: str):
    """Parse decoupled history/stride mode strings like 'h2/s3'.

    Returns (history_order:int, skip_calls:int) if recognized and allowed,
    otherwise None. Allowed set (explicit, compact):
      - h2/sK for K in {2,3,4,5,6}
      - h3/sK for K in {3,4,5,6}
      - h4/sK for K in {4,5,6}
    (No h4/s2 or h4/s3 to avoid overly aggressive cadence for high-order.)
    """
    try:
        if not isinstance(skip_mode, str):
            return None
        m = skip_mode.strip().lower()
        if m == "h2":
            return (2, 2)
        if m == "h3":
            return (3, 3)
        if m == "h4":
            return (4, 4)
        # hN/sK form
        if "/" in m:
            left, right = m.split("/", 1)
            left = left.strip()
            right = right.strip()
            if left.startswith("h") and right.startswith("s") and len(left) > 1 and len(right) > 1:
                n_str = left[1:]
                k_str = right[1:]
                if all(ch.isdigit() for ch in n_str) and all(ch.isdigit() for ch in k_str):
                    n = int(n_str)
                    k = int(k_str)
                    allowed = set((2, kk) for kk in (2, 3, 4, 5, 6))
                    allowed |= set((3, kk) for kk in (3, 4, 5, 6))
                    allowed |= set((4, kk) for kk in (4, 5, 6))
                    if (n, k) in allowed:
                        return (n, k)
        return None
    except Exception:
        return None


def _normalize_skip_mode(skip_mode: str):
    """Map UI aliases to canonical internal modes.

    Canonical: none | linear | richardson | quad | adaptive
    Aliases: h2 -> linear, h3 -> richardson, h4 -> quad
    Legacy: linear, richardson, quad kept for back-compat
    """
    if not isinstance(skip_mode, str):
        return "none"
    m = skip_mode.lower()
    if m in ("h2", "linear"):
        return "linear"
    if m in ("h3", "richardson"):
        return "richardson"
    if m in ("h4", "quad"):
        return "h4"
    if m == "adaptive":
        return "adaptive"
    return "none"


def parse_skip_indices_config(text: str):
    """Parse explicit skip indices configuration string.

    Input examples:
      - "h2, 3, 4, 7, 9"
      - "3 6 8" (defaults to h2)
      - "h4, 10, 12" (first hN wins)

    Behavior:
      - Tokenize on commas and/or whitespace; case-insensitive.
      - First hN token wins among {h2,h3,h4}; default predictor is h2 when not specified.
      - Collect distinct integer tokens into a set; drop invalids.
      - Always filter out step indices 0 and 1 here; engine will further bound to total steps.

    Returns: (predictor: 'linear'|'richardson'|'h4', indices: set[int]).
             If no indices parsed, indices will be an empty set.
    """
    if not isinstance(text, str):
        return "linear", set()

    tokens = [t.strip() for t in re.split(r"[\s,]+", text.strip()) if t.strip()]
    predictor = None
    indices = set()

    for tok in tokens:
        tl = tok.lower()
        # predictor selection (first hN wins)
        if predictor is None and len(tl) >= 2 and tl[0] == 'h' and tl[1:].isdigit():
            n = int(tl[1:])
            if n >= 4:
                predictor = "h4"
            elif n == 3:
                predictor = "richardson"
            elif n == 2:
                predictor = "linear"
            continue
        # integer index
        if tl.lstrip("+-").isdigit():
            try:
                v = int(tl)
            except Exception:
                continue
            # Filter out 0 and 1 here
            if v >= 2:
                indices.add(v)

    if predictor is None:
        predictor = "linear"

    return predictor, indices


def should_skip_model_call(error_ratio, step_index, total_steps, skip_mode, epsilon_history, protect_last_steps=4, protect_first_steps=2):
    """Decide whether to skip the model call based on skip_mode pattern and history.

    This mirrors the existing logic used in sampling_engine.py, including first/last step
    protections and required history lengths for linear/richardson patterns.
    """
    # First, handle decoupled hN/sK modes if provided
    hs = _parse_hs_mode(skip_mode)

    # Never skip first few steps (need to build history)
    try:
        pfs = int(protect_first_steps)
    except Exception:
        pfs = 2
    if pfs < 0:
        pfs = 0

    # Never skip last few steps (critical for quality)
    try:
        pls = int(protect_last_steps)
    except Exception:
        pls = 4
    if pls < 1:
        pls = 1

    # Decoupled history/stride path
    if hs is not None:
        history_order, skip_calls = hs
        # Guard windows
        if step_index < pfs or step_index >= total_steps - pls:
            return False, None
        # Require sufficient REAL epsilon history
        if len(epsilon_history) < history_order:
            return False, None
        # Align first eligible skip to the later of warmup or required history
        anchor = max(pfs, history_order)
        # Pattern: Call×K, then Skip → cycle length = K+1, skip on last position
        cycle_len = int(skip_calls) + 1
        cycle_position = (step_index - anchor) % cycle_len
        if cycle_position == (cycle_len - 1):
            # Choose predictor by history order
            if history_order >= 4:
                return True, "h4"
            elif history_order == 3:
                return True, "richardson"
            else:
                return True, "linear"
        return False, None

    # Normalize legacy modes when not using hN/sK
    skip_mode = _normalize_skip_mode(skip_mode)

    # Never skip if mode is "none"
    if skip_mode == "none":
        return False, None

    # Guard windows for legacy modes
    if step_index < pfs:
        return False, None
    if step_index >= total_steps - pls:
        return False, None

    # Check if we have enough history
    if len(epsilon_history) < 2:
        return False, None

    if skip_mode == "h4":
        # 4-history mode: Call 4, Skip 1 (uses 4-point predictor on skip)
        anchor = max(pfs, 4)
        if step_index >= anchor:
            cycle_position = (step_index - anchor) % 5
            if cycle_position == 4:
                return True, "h4"
        return False, None

    elif skip_mode == "linear":
        # Call, Call, Skip cycle
        cycle_position = (step_index - pfs) % 3
        if cycle_position == 2:
            return True, "linear"
        return False, None

    elif skip_mode == "richardson":
        # Call, Call, Call, Skip cycle, needs 3 history
        if len(epsilon_history) < 3:
            return False, None
        anchor = pfs + 1  # shift so default pfs=2 behaves like previous (anchor=3)
        cycle_position = (step_index - anchor) % 4
        if cycle_position == 3:
            return True, "richardson"
        return False, None

    elif skip_mode == "adaptive":
        # Pattern-gated adaptive using error_ratio bands (legacy behavior)
        # Potential skip (every 3rd step after warmup) with tight band
        cycle_position = (step_index - 2) % 3
        if cycle_position == 2 and 0.97 <= error_ratio <= 1.03 and len(epsilon_history) >= 2:
            return True, "linear"
        # Every 4th step after more warmup with very tight band
        cycle_position = (step_index - 3) % 4
        if cycle_position == 3 and 0.99 <= error_ratio <= 1.01 and len(epsilon_history) >= 3:
            return True, "richardson"
        return False, None

    return False, None


def validate_epsilon_hat(eps_hat, prev_eps=None, min_abs=1e-8, min_rel=1e-6):
    """Validate extrapolated epsilon before using it for a skip step.

    Returns (ok, reason, hat_norm, prev_norm).
    Reasons: 'none', 'nan_inf', 'too_small_abs', 'too_small_rel'
    """
    prev_norm = None
    if eps_hat is None:
        return False, 'none', 0.0, prev_norm
    try:
        if torch.isnan(eps_hat).any() or torch.isinf(eps_hat).any():
            return False, 'nan_inf', float('nan'), None
        hat_norm = torch.norm(eps_hat).item()
    except Exception:
        return False, 'nan_inf', float('nan'), None

    if not math.isfinite(hat_norm):
        return False, 'nan_inf', hat_norm, None
    if hat_norm < min_abs:
        return False, 'too_small_abs', hat_norm, None

    if prev_eps is not None:
        try:
            prev_norm = torch.norm(prev_eps).item()
        except Exception:
            prev_norm = None
        if prev_norm is not None and prev_norm > 0 and hat_norm < (min_rel * prev_norm):
            return False, 'too_small_rel', hat_norm, prev_norm

    return True, '', hat_norm, prev_norm


def decide_skip_adaptive(
    epsilon_history,
    step_index,
    total_steps,
    protect_last_steps=4,
    protect_first_steps=2,
    tol_relative=0.10,
    anchor_interval=None,
    max_consecutive_skips=None,
    skip_stats=None,
    # Optional: predicted-state (x_next) gating context
    x_current=None,
    sigma_current=None,
    sigma_next=None,
    sampler_kind=None,
    sigma_previous=None,
):
    """Adaptive skip decision using a dual-order epsilon-space gate.

    - Builds two predictions from REAL epsilon history at the current step time:
      high order (richardson, h3) and lower order (linear, h2).
    - Computes a relative error in epsilon space: ||ε̂_hi - ε̂_lo|| / max(||ε̂_hi||, eps_rel).
    - Allows skipping when error is below tolerance and guard rails permit.

    Returns: (should_skip: bool, epsilon_hat: Tensor|None, meta: dict)
    """
    # Guard: skip disabled near start/end
    try:
        pfs = max(0, int(protect_first_steps))
    except Exception:
        pfs = 2
    try:
        pls = max(1, int(protect_last_steps))
    except Exception:
        pls = 4
    if step_index < pfs or step_index >= total_steps - pls:
        return False, None, {"reason": "protected_region"}

    # History requirement (need >=3 REAL eps for richardson)
    if len(epsilon_history) < 3:
        return False, None, {"reason": "insufficient_history"}

    # Defaults for aggressive profile if not provided
    if anchor_interval is None:
        anchor_interval = 4            # Absolute cadence: every Nth step index (offset by protect_first_steps)
    if max_consecutive_skips is None:
        max_consecutive_skips = 2 # Local cap on back-to-back skips

    # Local consecutive cap
    if isinstance(skip_stats, dict):
        consec = skip_stats.get("consecutive_skips", 0)
        if consec >= max_consecutive_skips:
            return False, None, {"reason": "max_consecutive"}

    # Absolute anchor cadence (does not reset on REAL calls)
    # Force REAL on every Nth index from the first eligible step (pfs offset), excluding protected tail
    if anchor_interval and anchor_interval > 0 and step_index >= pfs:
        try:
            offset_idx = step_index - pfs
            if (offset_idx % int(anchor_interval)) == 0:
                return False, None, {"reason": "anchor_abs"}
        except Exception:
            pass

    # Build predictions
    eps_hat_hi = extrapolate_epsilon_richardson(epsilon_history)
    eps_hat_lo = extrapolate_epsilon_linear(epsilon_history)
    if eps_hat_hi is None or eps_hat_lo is None:
        return False, None, {"reason": "predict_failed"}

    # Finite checks
    if (not torch.isfinite(eps_hat_hi).all()) or (not torch.isfinite(eps_hat_lo).all()):
        return False, None, {"reason": "non_finite"}

    # Prefer predicted-state (x_next) gating when context is provided; else epsilon-space fallback
    def _rms(t):
        try:
            return float(torch.sqrt(torch.mean((t.float()) ** 2)).item())
        except Exception:
            return float('inf')

    if x_current is not None and sigma_current is not None and sigma_next is not None and sampler_kind is not None:
        try:
            dt = sigma_next - sigma_current
            # Compute predicted next states for this sampler
            if sampler_kind in ("euler", "res_2s", "dpmpp_2s"):
                # Euler-like update
                d_hi = -eps_hat_hi / sigma_current
                d_lo = -eps_hat_lo / sigma_current
                x_next_hi = x_current + dt * d_hi
                x_next_lo = x_current + dt * d_lo
            elif sampler_kind == "ddim":
                # x_next = x0 + (sigma_next/sigma_current)*(x - x0), x0 = x + eps
                scale = (sigma_next / sigma_current)
                x0_hi = x_current + eps_hat_hi
                x0_lo = x_current + eps_hat_lo
                x_next_hi = x0_hi + scale * (x_current - x0_hi)
                x_next_lo = x0_lo + scale * (x_current - x0_lo)
            elif sampler_kind in ("dpmpp_2m", "lms"):
                # AB2-style with optional previous derivative from last REAL epsilon
                d_hi = -eps_hat_hi / sigma_current
                d_lo = -eps_hat_lo / sigma_current
                d_prev = None
                if sigma_previous is not None and len(epsilon_history) >= 1:
                    d_prev = -(epsilon_history[-1]) / sigma_previous
                if d_prev is not None:
                    x_next_hi = x_current + dt * (1.5 * d_hi - 0.5 * d_prev)
                    x_next_lo = x_current + dt * (1.5 * d_lo - 0.5 * d_prev)
                else:
                    x_next_hi = x_current + dt * d_hi
                    x_next_lo = x_current + dt * d_lo
            else:
                # Fallback to epsilon-space metric if sampler not supported
                x_next_hi = None
                x_next_lo = None

            if x_next_hi is not None and x_next_lo is not None:
                num = _rms(x_next_hi - x_next_lo)
                den = max(_rms(x_next_hi), 1e-6)
                rel_err = num / den
                meta = {"relative_error": rel_err, "hi_order": 3, "lo_order": 2, "space": "x_next"}
                if rel_err <= float(tol_relative):
                    return True, eps_hat_hi, meta
                return False, None, meta
        except Exception:
            # Fall through to epsilon-space
            pass

    # Epsilon-space fallback
    diff = eps_hat_hi - eps_hat_lo
    num = _rms(diff)
    den = max(_rms(eps_hat_hi), 1e-6)
    if not math.isfinite(num) or not math.isfinite(den) or den <= 0:
        return False, None, {"reason": "bad_metric"}
    rel_err = num / den
    meta = {"relative_error": rel_err, "hi_order": 3, "lo_order": 2, "space": "epsilon"}
    if rel_err <= float(tol_relative):
        return True, eps_hat_hi, meta
    return False, None, meta
