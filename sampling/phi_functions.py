"""Phi functions for exponential integrators (res4lyf parity)."""
import torch
import math


def _phi_res4lyf(j, neg_h):
    """res4lyf-style remainder method for phi_j(neg_h).

    For j in {1,2}, computes:
      phi_j(z) = (exp(z) - sum_{k=0..j-1} z^k/k!) / z^j
    """
    # neg_h may be tensor
    remainder = torch.zeros_like(neg_h)
    for k in range(j):
        remainder = remainder + (neg_h**k) / math.factorial(k)
    return (neg_h.exp() - remainder) / (neg_h**j)


def phi_function(order, step_size):
    """Compute phi function φ_1 or φ_2 using res4lyf's _phi formulation.

    Args:
        order: 1 or 2
        step_size: commonly "-h" where h = -log(sigma_next/sigma_current)
    """
    if order not in (1, 2):
        raise NotImplementedError(f"phi_function order {order} not implemented")
    # Directly use res4lyf's phi with neg_h = step_size
    return _phi_res4lyf(order, step_size)
