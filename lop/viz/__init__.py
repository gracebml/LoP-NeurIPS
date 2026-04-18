"""
LoP Visualization — Centralized plotting for Loss of Plasticity research.
"""

from lop.viz.style import apply_lop_style, LOP_COLORS
from lop.viz.plot_online import generate_online_performance_plot
from lop.viz.plot_sensitivity import generate_parameter_sensitivity_plot
from lop.viz.plot_metrics import (
    plot_metric_dashboard,
    plot_metric_comparison,
    plot_ntk_eigenspectrum,
)

__all__ = [
    "apply_lop_style",
    "LOP_COLORS",
    "generate_online_performance_plot",
    "generate_parameter_sensitivity_plot",
    "plot_metric_dashboard",
    "plot_metric_comparison",
    "plot_ntk_eigenspectrum",
]
