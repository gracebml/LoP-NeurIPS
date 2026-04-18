"""
Exponential Moving Average (EMA) wrapper for model parameters.

Maintains shadow copies of all parameters and provides methods to
swap between EMA and current parameters for evaluation.

Extracted from lop-experiments/cifar/cifar100-secondorder-sdp.py:876-902.
"""

import torch


class EMAWrapper:
    """
    Exponential Moving Average (EMA) for model weights.

    Usage:
        ema = EMAWrapper(model, decay=0.999)
        # During training:
        ema.update(model)
        # For evaluation:
        ema.apply(model)       # swap to EMA weights
        evaluate(model)
        ema.restore(model)     # swap back to training weights
        # At task boundary:
        ema.reset(model)       # re-initialize shadow from current weights
    """

    def __init__(self, model, decay=0.999):
        """
        Args:
            model: nn.Module whose parameters to track.
            decay: EMA coefficient. Higher = smoother, slower adaptation.
        """
        self.decay = decay
        self._shadow = {n: p.data.clone()
                        for n, p in model.named_parameters()}
        self._backup = {}

    @torch.no_grad()
    def update(self, model):
        """Update shadow parameters: shadow = decay * shadow + (1-decay) * param."""
        for n, p in model.named_parameters():
            self._shadow[n].mul_(self.decay).add_(p.data, alpha=1 - self.decay)

    @torch.no_grad()
    def apply(self, model):
        """Swap model parameters with EMA shadow (backup originals)."""
        self._backup = {n: p.data.clone()
                        for n, p in model.named_parameters()}
        for n, p in model.named_parameters():
            p.data.copy_(self._shadow[n])

    @torch.no_grad()
    def restore(self, model):
        """Restore original model parameters from backup."""
        for n, p in model.named_parameters():
            if n in self._backup:
                p.data.copy_(self._backup[n])
        self._backup.clear()

    @torch.no_grad()
    def reset(self, model):
        """Re-initialize shadow from current model parameters."""
        self._shadow = {n: p.data.clone()
                        for n, p in model.named_parameters()}
