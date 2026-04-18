"""
Spectral Diversity Preservation (SDP) — apply at task boundaries
to maintain spectral health of weight matrices.

Interpolates each layer's singular values toward their mean:
    σ'_i = σ̄^γ · σ_i^(1-γ)

This prevents spectral collapse (all singular values → 0) and
spectral blow-up (condition number → ∞) during continual learning.

Extracted from lop-experiments/cifar/cifar100-secondorder-sdp.py:87-117
and lop-experiments/mnist/mnist-secondorder-sdp.py:118-144.
"""

import torch
import torch.nn as nn


@torch.no_grad()
def apply_sdp(net, gamma, skip_output=True):
    """
    Spectral Diversity Preservation (SDP) at task boundary.

    Applies the geometric interpolation:
        σ'_i = σ̄^γ · σ_i^(1-γ)
    to all Conv2d and Linear layers (except the output head).

    Conv2d weights are reshaped to 2D for SVD.

    Args:
        net: nn.Module — the network to apply SDP to.
        gamma: float in [0, 1] — interpolation strength.
               0 = no change, 1 = all singular values become the mean.
        skip_output: if True, skip the last Linear/Conv2d layer
                     (output head typically has structural rank bottleneck).

    Returns:
        cond_numbers: list of condition numbers (σ_max/σ_min) before SDP,
                      one per modified layer.
    """
    if gamma <= 0.0:
        return []

    cond_numbers = []
    modules = [(name, m) for name, m in net.named_modules()
               if isinstance(m, (nn.Linear, nn.Conv2d))]

    for i, (name, module) in enumerate(modules):
        if skip_output and i == len(modules) - 1:
            continue  # skip W_out — output head

        W = module.weight.data
        orig_shape = W.shape
        W2d = W.reshape(orig_shape[0], -1)

        try:
            U, S, Vh = torch.linalg.svd(W2d, full_matrices=False)
        except Exception:
            continue

        if S.numel() == 0 or S[0] < 1e-12:
            continue

        cond_numbers.append((S[0] / S[-1].clamp(min=1e-12)).item())

        # Clamp singular values to avoid log(0)
        S_safe = torch.clamp(S, min=1e-4)
        s_mean = S_safe.mean()

        # Geometric interpolation: σ' = σ̄^γ · σ^(1-γ)
        S_new = (s_mean ** gamma) * (S_safe ** (1.0 - gamma))

        W_new = (U @ torch.diag(S_new) @ Vh).reshape(orig_shape)
        module.weight.data.copy_(W_new)

    return cond_numbers
