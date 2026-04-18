"""
LoP Optimizers — Second-order optimizers for Loss of Plasticity research.

Provides:
    - Adahessian: Adaptive Hessian-based optimizer (Yao et al., AAAI 2021)
    - SophiaH: Hutchinson Hessian + element-wise clipping (Liu et al., ICLR 2024)
    - KFACOptimizer: K-FAC Natural Gradient Descent (Martens & Grosse, ICML 2015)
    - SASSHA: SAM + Hutchinson Hessian trace (Shin et al., ICML 2025)

NOTE: Shampoo and ASAM are excluded by design.
"""

import torch
from torch.optim import SGD, Adam, AdamW

from lop.optimizers.adahessian import Adahessian
from lop.optimizers.sophiaH import SophiaH
from lop.optimizers.kfac_ngd import KFACOptimizer
from lop.optimizers.sassha import SASSHA


__all__ = [
    "Adahessian",
    "SophiaH",
    "KFACOptimizer",
    "SASSHA",
    "get_optimizer",
]

# Registry of supported optimizers
_OPTIMIZER_REGISTRY = {
    'adahessian': Adahessian,
    'sophia': SophiaH,
    'sophiah': SophiaH,
    'kfac': KFACOptimizer,
    'sassha': SASSHA,
    'sgd': SGD,
    'adam': Adam,
    'adamw': AdamW,
}


def get_optimizer(name, model_or_params, **kwargs):
    """
    Factory function to create an optimizer by name.

    Args:
        name: one of 'adahessian', 'sophia', 'kfac', 'sassha',
              'sgd', 'adam', 'adamw' (case-insensitive).
        model_or_params: either an nn.Module (for K-FAC which needs hooks)
                         or an iterable of parameters.
        **kwargs: optimizer-specific keyword arguments (lr, betas, etc.).

    Returns:
        Optimizer instance.
    """
    key = name.lower().strip()
    if key not in _OPTIMIZER_REGISTRY:
        raise ValueError(
            f"Unknown optimizer '{name}'. "
            f"Available: {sorted(_OPTIMIZER_REGISTRY.keys())}"
        )

    opt_cls = _OPTIMIZER_REGISTRY[key]

    # K-FAC needs the full model, not just parameters
    if key == 'kfac':
        if not isinstance(model_or_params, torch.nn.Module):
            raise TypeError(
                "KFACOptimizer requires an nn.Module, not raw parameters."
            )
        return opt_cls(model_or_params, **kwargs)

    # All other optimizers accept parameters
    if isinstance(model_or_params, torch.nn.Module):
        params = model_or_params.parameters()
    else:
        params = model_or_params

    return opt_cls(params, **kwargs)
