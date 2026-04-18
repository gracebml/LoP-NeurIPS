"""
LoP Metrics Dashboard — compute all plasticity metrics in one call.

Provides a unified interface for computing all LoP metrics at task
boundaries, used by all experiments for consistent logging.

Metric functions are imported from their canonical modules:
    - Rank metrics: lop.metrics.rank
    - Dormant neurons: lop.metrics.dormant
    - Weight magnitude: lop.metrics.weight_norm
    - NTK churn/eigenspectrum: lop.metrics.ntk_churn
"""

import torch
from torch.utils.data import DataLoader, TensorDataset
from lop.metrics.rank import (
    compute_effective_rank,
    compute_stable_rank,
    compute_approximate_rank,
)
from lop.metrics.dormant import (
    compute_dormant_ratio_ffnn,
    compute_dormant_units_proportion,
)
from lop.metrics.weight_norm import compute_average_weight_magnitude
from lop.metrics.ntk_churn import compute_ntk_churn, compute_empirical_ntk_eigenspectrum


@torch.no_grad()
def compute_task_metrics(model, ref_batch, features, prev_ref_outputs=None,
                         ntk_subset_size=32, loss_type='ce',
                         network_type='ffnn'):
    """
    Compute all LoP metrics for a model at a task boundary.

    Args:
        model: nn.Module — the network.
        ref_batch: Tensor (B, ...) — reference input batch.
        features: list of Tensors — per-layer activations from a forward pass.
        prev_ref_outputs: Tensor or None — previous model outputs for NTK churn.
        ntk_subset_size: int — number of samples for NTK eigenspectrum.
        loss_type: str — 'ce' or 'mse' for NTK churn computation.
        network_type: str — 'ffnn' or 'cnn'. Controls which dormant function to call.

    Returns:
        dict with keys:
            - eff_rank: list[float] — effective rank per layer
            - stable_rank: list[int] — stable rank per layer
            - approx_rank: list[int] — approximate rank per layer
            - dormant_proportion: float — overall dormant fraction
            - dormant_per_layer: list — per-layer dormant info
            - avg_weight_mag: float
            - ntk_churn: float (0.0 if prev_ref_outputs is None)
            - ntk_eff_rank: float
            - ntk_cond: float
    """
    results = {}

    # ── Per-layer representation metrics ──
    eff_ranks = []
    stable_ranks = []
    approx_ranks = []

    for feat in features:
        if feat.ndim == 4:
            feat_2d = feat.mean(dim=(2, 3))  # (B, C)
        elif feat.ndim == 2:
            feat_2d = feat
        else:
            continue

        sv = torch.linalg.svdvals(feat_2d.float()).cpu().numpy()
        sv = sv[sv > 1e-10]

        if len(sv) > 0:
            eff_ranks.append(round(float(compute_effective_rank(sv)), 2))
            stable_ranks.append(int(compute_stable_rank(sv)))
            approx_ranks.append(int(compute_approximate_rank(
                torch.tensor(sv, dtype=torch.float32))))
        else:
            eff_ranks.append(0.0)
            stable_ranks.append(0)
            approx_ranks.append(0)

    results['eff_rank'] = eff_ranks
    results['stable_rank'] = stable_ranks
    results['approx_rank'] = approx_ranks

    # ── Dormant neurons (calling functions from lop.metrics.dormant) ──
    if network_type == 'ffnn':
        # compute_dormant_ratio_ffnn: calls net.predict(probe_x)
        agg_frac, per_layer_frac, _ = compute_dormant_ratio_ffnn(model, ref_batch)
        results['dormant_proportion'] = round(agg_frac, 4)
        results['dormant_per_layer'] = [round(f, 4) for f in per_layer_frac]
    else:
        # compute_dormant_units_proportion: needs a DataLoader, calls net.forward(x, features)
        dummy_labels = torch.zeros(ref_batch.shape[0], device=ref_batch.device)
        wrapper_loader = DataLoader(
            TensorDataset(ref_batch, dummy_labels),
            batch_size=ref_batch.shape[0], shuffle=False)
        proportion, _ = compute_dormant_units_proportion(model, wrapper_loader)
        results['dormant_proportion'] = round(proportion, 4)
        results['dormant_per_layer'] = []  # CNN returns aggregate only

    # ── Global metrics ──
    results['avg_weight_mag'] = round(compute_average_weight_magnitude(model), 6)

    # NTK churn
    if prev_ref_outputs is not None:
        results['ntk_churn'] = round(float(compute_ntk_churn(
            model, ref_batch, prev_ref_outputs, loss_type=loss_type)), 6)
    else:
        results['ntk_churn'] = 0.0

    # NTK eigenspectrum (needs gradients — temporarily exit no_grad)
    with torch.enable_grad():
        eigenvals, cond, eff_r = compute_empirical_ntk_eigenspectrum(
            model, ref_batch[:ntk_subset_size])
    results['ntk_eff_rank'] = round(float(eff_r), 2)
    results['ntk_cond'] = round(min(float(cond), 1e10), 2)

    return results


def print_task_summary(task_idx, before, after, loss, train_acc, test_acc):
    """
    Print a formatted task boundary summary with before/after metrics.

    Args:
        task_idx: int — task index.
        before: dict — metrics from compute_task_metrics before training.
        after: dict — metrics from compute_task_metrics after training.
        loss: float — training loss for this task.
        train_acc: float — training accuracy for this task.
        test_acc: float — test accuracy for this task.
    """
    print(f"\n{'='*70}")
    print(f"  Task {task_idx} Summary")
    print(f"{'='*70}")
    print(f"  loss: {loss:.4f}  |  train_acc: {train_acc:.4f}  |  test_acc: {test_acc:.4f}")
    print(f"  avg_weight_mag:      {before['avg_weight_mag']:.6f} -> {after['avg_weight_mag']:.6f}")
    print(f"  ntk_churn:           {after['ntk_churn']:.6f}")
    print(f"  ntk_eff_rank:        {before['ntk_eff_rank']:.2f} -> {after['ntk_eff_rank']:.2f}")
    print(f"  ntk_cond:            {before['ntk_cond']:.2f} -> {after['ntk_cond']:.2f}")

    # Print only last hidden layer's rank metrics (scalar, not list)
    def _last(lst):
        return lst[-1] if lst else 0
    b_eff = _last(before['eff_rank']); a_eff = _last(after['eff_rank'])
    b_stb = _last(before['stable_rank']); a_stb = _last(after['stable_rank'])
    b_app = _last(before['approx_rank']); a_app = _last(after['approx_rank'])

    print(f"  eff_rank:            {b_eff:.2f} -> {a_eff:.2f}")
    print(f"  stable_rank:         {b_stb} -> {a_stb}")
    print(f"  approx_rank:         {b_app} -> {a_app}")
    print(f"  dormant_proportion:  {before['dormant_proportion']:.4f} -> {after['dormant_proportion']:.4f}")
    print(f"{'='*70}\n")

