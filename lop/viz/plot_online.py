"""
Online performance plotting for Loss of Plasticity experiments.

Refactored from lop/utils/plot_online_performance.py.
"""

import os
import numpy as np
from math import sqrt
import matplotlib.pyplot as plt
from lop.viz.style import apply_lop_style


def generate_online_performance_plot(
        performances=None,
        colors=None,
        xticks=None,
        xticks_labels=None,
        yticks=None,
        yticks_labels=None,
        m=20000,
        xlabel='',
        ylabel='',
        labels=None,
        caption=None,
        fontsize=24,
        log_scale_x=False,
        log_scale_y=False,
        save_path=None,
        svg=False,
):
    """
    Plot online performance of algorithms across hyper-parameter settings.
    Plots the mean and std-error for each configuration.

    Args:
        performances: array of shape (num_configs, num_runs, num_steps).
        colors: list of RGBA tuples for each config.
        xticks: x-axis tick positions.
        xticks_labels: x-axis tick labels.
        yticks: y-axis tick positions.
        yticks_labels: y-axis tick labels.
        m: step multiplier for x-axis.
        xlabel: x-axis label.
        ylabel: y-axis label.
        labels: legend labels for each config.
        caption: plot title.
        fontsize: font size for tick labels.
        log_scale_x: use log scale on x-axis.
        log_scale_y: use log scale on y-axis.
        save_path: path to save the figure (directory or full path).
                   If None, saves to 'comparison.png' in cwd.
        svg: if True, save as SVG instead of PNG.
    """
    if xticks is None:
        xticks = []
    if yticks is None:
        yticks = []

    apply_lop_style()
    shape = np.shape(performances)

    if colors is None:
        colors = [(1, 0, 0, 1), (0.5, 0.5, 0, 1), (0, 1, 0, 1)]

    fig, ax = plt.subplots()

    for idx in range(shape[0]):
        x = np.array([i for i in range(shape[-1])])
        mean = np.mean(performances[idx], axis=0)
        num_samples = shape[1]
        std_err = np.std(performances[idx], axis=0) / sqrt(num_samples)
        label = labels[idx] if labels is not None else ''
        color = colors[idx]
        plt.plot(x * m, mean, '-', label=label, color=color)
        plt.fill_between(x * m, mean - std_err, mean + std_err,
                         color=color, alpha=0.2)

    if xticks_labels is None:
        xticks_labels = xticks
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks_labels, fontsize=fontsize)

    if len(yticks) > 0:
        ax.set_yticks(yticks)
        ax.set_ylim(yticks[0], yticks[-1])
    if yticks_labels is not None:
        ax.set_yticklabels(yticks_labels, fontsize=fontsize)
    elif len(yticks) > 0:
        ax.set_yticklabels(yticks, fontsize=fontsize)
        ax.set_ylim(yticks[0], yticks[-1])

    if log_scale_y:
        ax.set_yscale('log')
    if log_scale_x:
        ax.set_xscale('log')

    if labels is not None:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if caption is not None:
        ax.set_title(caption)

    # Determine save path
    if save_path is None:
        ext = 'svg' if svg else 'png'
        save_path = f'comparison.{ext}'
    elif os.path.isdir(save_path):
        ext = 'svg' if svg else 'png'
        save_path = os.path.join(save_path, f'comparison.{ext}')

    plt.savefig(save_path, bbox_inches='tight', dpi=500)
    plt.close()
