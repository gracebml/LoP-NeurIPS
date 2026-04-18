"""
Parameter sensitivity plotting for Loss of Plasticity experiments.

Refactored from lop/utils/plot_param_sensetivity.py.
"""

import os
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from lop.viz.style import apply_lop_style


def generate_parameter_sensitivity_plot(
        final_performances=None,
        param_axis_1=None,
        colors=None,
        yticks=None,
        xticks=None,
        labels=None,
        xlabel='',
        ylabel='',
        save_path=None,
):
    """
    Plot parameter sensitivity for various hyper-parameter settings.
    Plots the mean and std-error for each configuration.

    Args:
        final_performances: list of arrays, each of shape (num_param_values, num_runs).
        param_axis_1: parameter values for x-axis.
        colors: list of RGBA tuples.
        yticks: y-axis tick positions.
        xticks: x-axis tick positions.
        labels: legend labels.
        xlabel: x-axis label.
        ylabel: y-axis label.
        save_path: path to save figure. If None, saves to 'sens_plot.png'.
    """
    if param_axis_1 is None:
        param_axis_1 = []
    if yticks is None:
        yticks = []
    if xticks is None:
        xticks = []

    apply_lop_style()

    if colors is None:
        colors = [(0, 1, 0, 1), (0, 0, 1, 1), (0.5, 0.5, 0, 1), (1, 0, 0, 1)]

    fig, ax = plt.subplots()

    for idx in range(len(final_performances)):
        means = np.mean(final_performances[idx], axis=1)
        num_runs = final_performances[idx].shape[1]
        stds = np.std(final_performances[idx], axis=1) / sqrt(num_runs)

        x = np.array(param_axis_1[:len(final_performances[idx])])
        color = colors[idx]
        label = labels[idx] if labels is not None else ''
        plt.plot(x, means, '-', color=color, label=label)
        plt.fill_between(x, means - stds, means + stds,
                         alpha=0.2, color=color)

    plt.xscale('log')

    if len(yticks) > 0:
        ax.set_yticks(yticks)
        ax.set_yticklabels(yticks, fontsize=12)
        ax.set_ylim(yticks[0], yticks[-1])
    if len(xticks) > 0:
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=12)

    plt.legend(fontsize=10)
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.tight_layout()

    if save_path is None:
        save_path = 'sens_plot.png'
    elif os.path.isdir(save_path):
        save_path = os.path.join(save_path, 'sens_plot.png')

    plt.savefig(save_path, dpi=500)
    plt.close()
