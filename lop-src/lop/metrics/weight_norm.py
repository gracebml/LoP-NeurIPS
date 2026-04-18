"""
Weight norm metrics for Loss of Plasticity.

Logic copied from lop/incremental_cifar/post_run_analysis.py (canonical source).
"""

import torch


@torch.no_grad()
def compute_average_weight_magnitude(net):
    """
    Computes average magnitude of the weights in the network.

    Canonical source: post_run_analysis.py:92-102

    Args:
        net: a torch.nn.Module.

    Returns:
        float: average absolute weight magnitude.
    """
    num_weights = 0
    sum_weight_magnitude = 0.0

    for p in net.parameters():
        num_weights += p.numel()
        sum_weight_magnitude += torch.sum(torch.abs(p)).item()

    return sum_weight_magnitude / num_weights if num_weights > 0 else 0.0


@torch.no_grad()
def compute_layer_weight_magnitudes(net):
    """
    Computes per-layer average absolute weight magnitude.

    Args:
        net: a torch.nn.Module.

    Returns:
        dict: {layer_name: avg_abs_weight} for all layers with parameters.
    """
    result = {}
    for name, param in net.named_parameters():
        if param.requires_grad:
            result[name] = param.data.abs().mean().item()
    return result
