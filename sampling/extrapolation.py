import torch
import threading


# ---------------------------------------------------------------------------
# Thread-local sigma context (set by engine, read by extrapolation functions)
# ---------------------------------------------------------------------------
_sigma_ctx = threading.local()


def set_sigma_target(target):
    """Set the sigma target for the current step (called by engine)."""
    _sigma_ctx.target = target


def _get_sigma_target():
    """Get the current sigma target, or None if not set."""
    return getattr(_sigma_ctx, 'target', None)


# ---------------------------------------------------------------------------
# Thread-local denoised context (set by engine, read by extrapolation funcs)
# ---------------------------------------------------------------------------
_denoised_ctx = threading.local()


def set_current_x(x):
    """Set the current noisy latent for denoised-mode conversion (called by engine)."""
    _denoised_ctx.current_x = x


def _get_current_x():
    """Get the current noisy latent, or None if not set."""
    return getattr(_denoised_ctx, 'current_x', None)


# ---------------------------------------------------------------------------
# SigmaAwareHistory — list subclass that also tracks sigmas + denoised
# ---------------------------------------------------------------------------
class SigmaAwareHistory(list):
    """Epsilon history that also tracks sigma and denoised per entry.

    Backward-compatible with plain list — all existing sampler code that
    treats epsilon_history as a list works unchanged.
    """

    def __init__(self):
        super().__init__()
        self.sigmas = []
        self.denoised = []
        self._pending_sigma = None
        self._pending_x = None

    def set_pending_sigma(self, sigma):
        """Set the sigma that will be recorded with the next appended epsilon."""
        self._pending_sigma = sigma

    def set_pending_x(self, x):
        """Set the noisy latent so denoised can be computed on append."""
        self._pending_x = x

    def append(self, epsilon):
        super().append(epsilon)
        # Sigma tracking
        if self._pending_sigma is not None:
            self.sigmas.append(self._pending_sigma)
            self._pending_sigma = None
        else:
            self.sigmas.append(None)
        # Denoised tracking: denoised = epsilon + x
        if self._pending_x is not None:
            self.denoised.append(epsilon + self._pending_x)
            self._pending_x = None
        else:
            self.denoised.append(None)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _sigma_aware_ok(epsilon_history, n):
    """Return (target, [s0..s_{n-1}]) if sigma-aware is active and usable, else None."""
    target = _get_sigma_target()
    if target is None:
        return None
    if not hasattr(epsilon_history, 'sigmas'):
        return None
    sigs = epsilon_history.sigmas
    if len(sigs) < n:
        return None
    trailing = [sigs[-(n - i)] for i in range(n)]  # oldest .. newest
    if any(s is None for s in trailing):
        return None
    return target, trailing


def _denoised_mode_ok(epsilon_history, n):
    """Return current_x if denoised extrapolation is active and usable, else None."""
    current_x = _get_current_x()
    if current_x is None:
        return None
    if not hasattr(epsilon_history, 'denoised'):
        return None
    den = epsilon_history.denoised
    if len(den) < n:
        return None
    if any(den[-(n - i)] is None for i in range(n)):
        return None
    return current_x


def _get_values(epsilon_history, n, current_x):
    """Get the N trailing values to extrapolate (denoised if current_x, else epsilon)."""
    if current_x is not None:
        return [epsilon_history.denoised[-(n - i)] for i in range(n)]
    return [epsilon_history[-(n - i)] for i in range(n)]


def _maybe_convert(result, current_x):
    """Convert predicted denoised back to epsilon if in denoised mode."""
    if current_x is not None:
        return result - current_x
    return result


# ---------------------------------------------------------------------------
# Extrapolation functions
# ---------------------------------------------------------------------------

def extrapolate_epsilon_linear(epsilon_history):
    """Linear (2-point) epsilon extrapolation using last two REAL epsilons.

    Args:
        epsilon_history: list[Tensor] of REAL epsilons, oldest..newest
    Returns:
        Tensor or None
    """
    if len(epsilon_history) < 2:
        return None

    current_x = _denoised_mode_ok(epsilon_history, 2)
    v0, v1 = _get_values(epsilon_history, 2, current_x)

    # Sigma-aware branch: 2-point Lagrange extrapolation
    sa = _sigma_aware_ok(epsilon_history, 2)
    if sa is not None:
        target, (s0, s1) = sa
        denom = s1 - s0
        if abs(denom) > 1e-12:
            L0 = (target - s1) / (s0 - s1)
            L1 = (target - s0) / (s1 - s0)
            return _maybe_convert(L0 * v0 + L1 * v1, current_x)

    # Uniform-spacing fallback
    return _maybe_convert(v1 + (v1 - v0), current_x)


def extrapolate_epsilon_richardson(epsilon_history):
    """Richardson (3-point) epsilon extrapolation using last three REAL epsilons.

    Args:
        epsilon_history: list[Tensor] of REAL epsilons, oldest..newest
    Returns:
        Tensor or None
    """
    if len(epsilon_history) < 3:
        return extrapolate_epsilon_linear(epsilon_history)

    current_x = _denoised_mode_ok(epsilon_history, 3)
    v0, v1, v2 = _get_values(epsilon_history, 3, current_x)

    # Sigma-aware branch: 3-point Lagrange extrapolation
    sa = _sigma_aware_ok(epsilon_history, 3)
    if sa is not None:
        target, (s0, s1, s2) = sa
        d01 = s0 - s1
        d02 = s0 - s2
        d10 = s1 - s0
        d12 = s1 - s2
        d20 = s2 - s0
        d21 = s2 - s1
        if abs(d01 * d02) > 1e-12 and abs(d10 * d12) > 1e-12 and abs(d20 * d21) > 1e-12:
            L0 = (target - s1) * (target - s2) / (d01 * d02)
            L1 = (target - s0) * (target - s2) / (d10 * d12)
            L2 = (target - s0) * (target - s1) / (d20 * d21)
            return _maybe_convert(L0 * v0 + L1 * v1 + L2 * v2, current_x)

    # Uniform-spacing fallback
    return _maybe_convert(3 * v2 - 3 * v1 + v0, current_x)


def extrapolate_epsilon_h4(epsilon_history):
    """4-point (cubic) epsilon extrapolation using last four REAL epsilons.

    Assumes uniform step spacing in the prediction index. Uses Lagrange
    coefficients for points at t = [-3, -2, -1, 0] to predict at t = 1:
        eps_hat_{n+1} = -1*eps_{n-3} + 4*eps_{n-2} - 6*eps_{n-1} + 4*eps_{n}

    Falls back to 3-point when history is insufficient.
    """
    if len(epsilon_history) < 4:
        return extrapolate_epsilon_richardson(epsilon_history)

    current_x = _denoised_mode_ok(epsilon_history, 4)
    vals = _get_values(epsilon_history, 4, current_x)

    # Sigma-aware branch: 4-point Lagrange extrapolation
    sa = _sigma_aware_ok(epsilon_history, 4)
    if sa is not None:
        target, (s0, s1, s2, s3) = sa
        nodes = [s0, s1, s2, s3]
        # Check all denominators first
        ok = True
        for i in range(4):
            prod = 1.0
            for j in range(4):
                if i != j:
                    prod *= (nodes[i] - nodes[j])
            if abs(prod) < 1e-12:
                ok = False
                break
        if ok:
            result = torch.zeros_like(vals[0])
            for i in range(4):
                basis = 1.0
                for j in range(4):
                    if i != j:
                        basis *= (target - nodes[j]) / (nodes[i] - nodes[j])
                result = result + basis * vals[i]
            return _maybe_convert(result, current_x)

    # Uniform-spacing fallback
    return _maybe_convert((-1.0) * vals[0] + 4.0 * vals[1] - 6.0 * vals[2] + 4.0 * vals[3], current_x)
