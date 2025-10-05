import time
import math


def _fmt_hms(seconds: float) -> str:
    s = int(max(seconds, 0))
    h = s // 3600
    m = (s % 3600) // 60
    ss = s % 60
    return f"{h:02d}:{m:02d}:{ss:02d}"


def print_step_timing(sampler_name: str, step_index: int, start_time: float, total_steps: int):
    """Standard per-step timing line printed at the start of each step."""
    steps_done = step_index
    elapsed = time.time() - start_time
    avg = elapsed / max(steps_done, 1) if steps_done > 0 else 0.0
    remaining = max(total_steps - steps_done, 0)
    eta = avg * remaining
    print(f"\n{sampler_name} step {step_index}: time elapsed {_fmt_hms(elapsed)} | time left {_fmt_hms(eta)}")


def _fmt(val, fmt: str = ".4g") -> str:
    try:
        if val is None:
            return "-"
        if hasattr(val, "item"):
            v = float(val.item())
        else:
            v = float(val)
        if not math.isfinite(v):
            return "nan"
        return f"{v:{fmt}}"
    except Exception:
        return "-"


def print_step_diag(
    sampler: str,
    step_index: int,
    sigma_current,
    sigma_next,
    *,
    target_sigma=None,
    sigma_up=None,
    alpha_ratio=None,
    h=None,
    c2=None,
    b1=None,
    b2=None,
    eps_norm=None,
    eps_prev_norm=None,
    x_rms=None,
    flags: str = "",
):
    """Compact per-step diagnostics used when debug/verbose is enabled.

    Accepts whatever fields the caller has; missing ones are shown as '-'.
    Safe to call from any sampler/model; prints a single concise line.
    """
    parts = [
        f"{sampler} diag {step_index}:",
        f"σ={_fmt(sigma_current)}→{_fmt(sigma_next)}",
    ]
    if target_sigma is not None:
        parts.append(f"tgt={_fmt(target_sigma)}")
    if h is not None:
        parts.append(f"h={_fmt(h)}")
    if c2 is not None:
        parts.append(f"c2={_fmt(c2)}")
    if b1 is not None or b2 is not None:
        parts.append(f"b1={_fmt(b1)} b2={_fmt(b2)}")
    if sigma_up is not None:
        parts.append(f"up={_fmt(sigma_up)}")
        try:
            sn = float(sigma_next.item()) if hasattr(sigma_next, "item") else float(sigma_next)
            su = float(sigma_up.item()) if hasattr(sigma_up, "item") else float(sigma_up)
            if sn != 0:
                parts.append(f"up/next={su/sn:.2f}")
        except Exception:
            pass
    if alpha_ratio is not None:
        parts.append(f"α={_fmt(alpha_ratio)}")
    if eps_norm is not None:
        if eps_prev_norm is not None:
            parts.append(f"|ε|={_fmt(eps_norm)}({_fmt(eps_prev_norm)})")
        else:
            parts.append(f"|ε|={_fmt(eps_norm)}")
    if x_rms is not None:
        parts.append(f"x_rms={_fmt(x_rms)}")
    if flags:
        parts.append(f"[{flags}]")

    # Compute a coarse risk tag (LOW/MED/HIGH) from available fields
    score = 0
    # Noise fraction risk
    try:
        sn = float(sigma_next.item()) if hasattr(sigma_next, "item") else (float(sigma_next) if sigma_next is not None else None)
        su = float(sigma_up.item()) if hasattr(sigma_up, "item") else (float(sigma_up) if sigma_up is not None else None)
        if sn is not None and su is not None and sn > 0:
            ratio = su / sn
            if ratio > 0.8:
                score += 2
            elif ratio > 0.5:
                score += 1
            # ODE-like mixing with large noise
            try:
                ar = float(alpha_ratio.item()) if hasattr(alpha_ratio, "item") else (float(alpha_ratio) if alpha_ratio is not None else None)
                if ar is not None and ar >= 0.95 and ratio > 0.5:
                    score += 1
            except Exception:
                pass
    except Exception:
        pass
    # Step size and geometry risk
    try:
        if h is not None:
            hv = abs(float(h.item()) if hasattr(h, "item") else float(h))
            if hv < 1e-6:
                score += 2
            elif hv < 1e-3:
                score += 1
    except Exception:
        pass
    try:
        if c2 is not None:
            c2v = abs(float(c2.item()) if hasattr(c2, "item") else float(c2))
            if c2v < 0.1 or c2v > 10.0:
                score += 2
            elif c2v < 0.3 or c2v > 3.0:
                score += 1
    except Exception:
        pass
    try:
        if b1 is not None or b2 is not None:
            b1v = abs(float(b1.item()) if hasattr(b1, "item") else float(b1) if b1 is not None else 0.0)
            b2v = abs(float(b2.item()) if hasattr(b2, "item") else float(b2) if b2 is not None else 0.0)
            s = b1v + b2v
            if s > 10.0:
                score += 2
            elif s > 5.0:
                score += 1
    except Exception:
        pass
    try:
        if eps_norm is not None and eps_prev_norm is not None:
            en = float(eps_norm)
            ep = float(eps_prev_norm)
            if ep > 0:
                rr = en / ep
                if rr > 5.0:
                    score += 2
                elif rr > 2.0:
                    score += 1
    except Exception:
        pass

    risk = "LOW" if score <= 1 else ("MED" if score <= 3 else "HIGH")
    parts.append(f"[RISK={risk}]")

    print(" ".join(parts))
