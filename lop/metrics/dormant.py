"""
Dormant neuron metrics for Loss of Plasticity.

Logic copied from lop/incremental_cifar/post_run_analysis.py (canonical source).
"""

import torch
import numpy as np
from torch.utils.data import DataLoader


@torch.no_grad()
def compute_dormant_units_proportion(net, data_loader: DataLoader,
                                     dormant_unit_threshold: float = 0.01):
    """
    Computes the proportion of dormant units in a network (CNN/ResNet).
    A unit is dormant if its mean activation across samples falls below the threshold.

    Also returns the last-layer activations for further analysis (e.g., SVD for rank).

    Canonical source: post_run_analysis.py:106-131

    Args:
        net: network that accepts forward(image, features_list) where features_list
             is populated with per-layer activations.
        data_loader: DataLoader providing batches with 'image' key.
        dormant_unit_threshold: units with activation proportion below this are dormant.

    Returns:
        proportion_dormant: float, fraction of dormant units across all layers.
        last_layer_activations: numpy array of the last hidden layer's activations.
    """
    device = next(net.parameters()).device
    features_per_layer = []
    last_layer_activations = None

    for i, sample in enumerate(data_loader):
        if isinstance(sample, dict):
            image = sample["image"].to(device)
        elif isinstance(sample, (list, tuple)):
            image = sample[0].to(device)
        else:
            image = sample.to(device)

        temp_features = []
        net.forward(image, temp_features)

        features_per_layer = temp_features
        last_layer_activations = temp_features[-1].cpu()
        break  # Only use first batch (typically 1000 samples)

    dead_neurons = torch.zeros(len(features_per_layer), dtype=torch.float32)
    for layer_idx in range(len(features_per_layer) - 1):
        # For conv layers: average over batch, height, width → per-channel
        dead_neurons[layer_idx] = (
            (features_per_layer[layer_idx] != 0).float().mean(dim=(0, 2, 3))
            < dormant_unit_threshold
        ).sum()
    # Last layer (typically FC after pooling): average over batch only
    dead_neurons[-1] = (
        (features_per_layer[-1] != 0).float().mean(dim=0)
        < dormant_unit_threshold
    ).sum()

    number_of_features = torch.sum(
        torch.tensor([layer_feats.shape[1] for layer_feats in features_per_layer])
    ).item()

    return dead_neurons.sum().item() / number_of_features, last_layer_activations.numpy()


@torch.no_grad()
def compute_dormant_ratio_ffnn(net, probe_x: torch.Tensor,
                               threshold: float = 0.01):
    """
    Computes the proportion of dormant units in a feed-forward network (FFNN).

    Same core logic as compute_dormant_units_proportion but adapted for
    networks that use net.predict(x) -> (output, [hidden_activations]).

    Args:
        net: FFNN with predict(x) method returning (output, list_of_hidden_acts).
        probe_x: input tensor for probing activations.
        threshold: units with alive-score below this are dormant.

    Returns:
        agg_frac: float, overall fraction of dormant units.
        per_layer_frac: list of floats, fraction per layer.
        last_act: numpy array of last hidden layer activations.
    """
    net.eval()
    _, hidden_acts = net.predict(probe_x)
    per_layer_frac = []
    total_d, total_n = 0, 0
    last_act = None
    for i, act in enumerate(hidden_acts):
        alive_score = (act != 0).float().mean(dim=0)
        n_units = act.shape[1]
        dormant = (alive_score < threshold).sum().item()
        per_layer_frac.append(dormant / n_units if n_units > 0 else 0.0)
        total_d += dormant
        total_n += n_units
        if i == len(hidden_acts) - 1:
            last_act = act.cpu().numpy()
    agg_frac = total_d / total_n if total_n > 0 else 0.0
    return agg_frac, per_layer_frac, last_act
