"""
Multi-panel LoP metric dashboards and NTK eigenspectrum visualization.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from lop.viz.style import apply_lop_style, LOP_COLORS


def plot_metric_dashboard(results_dict, metrics=None, save_path=None,
                          title='LoP Metrics Dashboard'):
    """
    Create a multi-panel figure showing all LoP metrics over training.

    Args:
        results_dict: dict with structure:
            {
                'method_name': {
                    'metric_name': array of shape (num_checkpoints,),
                    ...
                },
                ...
            }
        metrics: list of metric names to plot. If None, plot all available.
        save_path: path to save figure.
        title: figure title.
    """
    apply_lop_style()

    if metrics is None:
        # Collect all metric names across all methods
        metrics = set()
        for method_data in results_dict.values():
            metrics.update(method_data.keys())
        metrics = sorted(metrics)

    n_metrics = len(metrics)
    if n_metrics == 0:
        return

    cols = min(3, n_metrics)
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(5 * cols, 4 * rows))
    if n_metrics == 1:
        axes = np.array([axes])
    axes = axes.flatten()

    color_list = list(LOP_COLORS.values())

    for i, metric in enumerate(metrics):
        ax = axes[i]
        for j, (method, data) in enumerate(results_dict.items()):
            if metric in data:
                values = np.array(data[metric])
                color = color_list[j % len(color_list)]
                ax.plot(values, label=method, color=color, linewidth=1.5)
        ax.set_title(metric.replace('_', ' ').title(), fontsize=11)
        ax.set_xlabel('Checkpoint')
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)

    # Hide unused subplots
    for i in range(n_metrics, len(axes)):
        axes[i].set_visible(False)

    fig.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()

    if save_path is None:
        save_path = 'metric_dashboard.png'
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'metric_dashboard.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_metric_comparison(results_per_method, metric_name,
                           save_path=None, ylabel=None):
    """
    Compare methods on a single metric with mean ± std error.

    Args:
        results_per_method: dict {method_name: array (num_runs, num_checkpoints)}.
        metric_name: name of the metric to plot.
        save_path: path to save figure.
        ylabel: y-axis label. If None, uses metric_name.
    """
    apply_lop_style()

    fig, ax = plt.subplots(figsize=(8, 5))
    color_list = list(LOP_COLORS.values())

    for i, (method, data) in enumerate(results_per_method.items()):
        data = np.array(data)
        if data.ndim == 1:
            ax.plot(data, label=method,
                    color=color_list[i % len(color_list)], linewidth=1.5)
        else:
            mean = data.mean(axis=0)
            std_err = data.std(axis=0) / np.sqrt(data.shape[0])
            x = np.arange(len(mean))
            color = color_list[i % len(color_list)]
            ax.plot(x, mean, '-', label=method, color=color, linewidth=1.5)
            ax.fill_between(x, mean - std_err, mean + std_err,
                            color=color, alpha=0.2)

    ax.set_xlabel('Checkpoint')
    ax.set_ylabel(ylabel or metric_name.replace('_', ' ').title())
    ax.set_title(f'{metric_name.replace("_", " ").title()} Comparison')
    ax.legend()
    plt.tight_layout()

    if save_path is None:
        save_path = f'{metric_name}_comparison.png'
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, f'{metric_name}_comparison.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()


def plot_ntk_eigenspectrum(eigenvalues_dict, save_path=None,
                           title='Empirical NTK Eigenspectrum'):
    """
    Visualize the empirical NTK eigenspectrum for multiple checkpoints
    or methods.

    Tracks rank collapse — when eigenvalues concentrate or decay rapidly,
    the network is losing plasticity.

    Args:
        eigenvalues_dict: dict with structure:
            {
                'label': eigenvalues_array (1D numpy),
                ...
            }
            where each label could be a checkpoint time or method name.
        save_path: path to save figure.
        title: figure title.
    """
    apply_lop_style()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    color_list = list(LOP_COLORS.values())

    for i, (label, eigenvals) in enumerate(eigenvalues_dict.items()):
        eigenvals = np.array(eigenvals)
        eigenvals = eigenvals[eigenvals > 0]  # filter zero/negative
        color = color_list[i % len(color_list)]

        # Left: log-scale eigenvalue decay
        ax1.semilogy(eigenvals, label=label, color=color,
                     linewidth=1.5, alpha=0.8)

        # Right: normalized cumulative energy
        cumsum = np.cumsum(eigenvals) / np.sum(eigenvals)
        ax2.plot(cumsum, label=label, color=color,
                 linewidth=1.5, alpha=0.8)

    ax1.set_xlabel('Index')
    ax1.set_ylabel('Eigenvalue (log scale)')
    ax1.set_title('Eigenvalue Decay')
    ax1.legend(fontsize=8)
    ax1.grid(alpha=0.3)

    ax2.set_xlabel('Index')
    ax2.set_ylabel('Cumulative Energy')
    ax2.set_title('Spectral Energy Distribution')
    ax2.axhline(y=0.99, color='gray', linestyle='--', alpha=0.5,
                label='99% energy')
    ax2.legend(fontsize=8)
    ax2.grid(alpha=0.3)

    fig.suptitle(title, fontsize=14)
    plt.tight_layout()

    if save_path is None:
        save_path = 'ntk_eigenspectrum.png'
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'ntk_eigenspectrum.png')

    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
