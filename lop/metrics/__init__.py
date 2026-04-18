"""
Loss of Plasticity Metrics.

All metric logic is the canonical implementation copied from
lop/incremental_cifar/post_run_analysis.py.
"""

from lop.metrics.dormant import (
    compute_dormant_units_proportion,
    compute_dormant_ratio_ffnn,
)
from lop.metrics.rank import (
    compute_effective_rank,
    compute_stable_rank,
    compute_approximate_rank,
    compute_abs_approximate_rank,
    compute_matrix_rank_summaries,
    compute_stable_rank_from_activations,
)
from lop.metrics.weight_norm import (
    compute_average_weight_magnitude,
    compute_layer_weight_magnitudes,
)
from lop.metrics.ntk_churn import (
    compute_ntk_churn,
    compute_empirical_ntk_eigenspectrum,
)
from lop.metrics.dashboard import (
    compute_task_metrics,
    print_task_summary,
)

__all__ = [
    # Dormant neurons
    "compute_dormant_units_proportion",
    "compute_dormant_ratio_ffnn",
    # Rank metrics
    "compute_effective_rank",
    "compute_stable_rank",
    "compute_approximate_rank",
    "compute_abs_approximate_rank",
    "compute_matrix_rank_summaries",
    "compute_stable_rank_from_activations",
    # Weight norm
    "compute_average_weight_magnitude",
    "compute_layer_weight_magnitudes",
    # NTK churn
    "compute_ntk_churn",
    "compute_empirical_ntk_eigenspectrum",
    # Dashboard
    "compute_task_metrics",
    "print_task_summary",
]
