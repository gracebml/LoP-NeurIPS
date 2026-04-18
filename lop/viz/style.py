"""
Shared matplotlib style configuration for LoP research plots.

Call apply_lop_style() once at the start of your script/notebook
to apply the standard LoP visual style.
"""

import matplotlib
import matplotlib.pyplot as plt


# Standard color palette for method comparisons
LOP_COLORS = {
    'bp':         '#1f77b4',   # blue
    'cbp':        '#ff7f0e',   # orange
    'adam':        '#2ca02c',   # green
    'sgd':        '#d62728',   # red
    'adahessian':  '#9467bd',  # purple
    'sophia':      '#8c564b',  # brown
    'kfac':        '#e377c2',  # pink
    'sassha':      '#7f7f7f',  # gray
    'sdp':         '#bcbd22',  # olive
    'no_sdp':      '#17becf',  # teal
}

# Paired colors for SDP/no-SDP comparisons
LOP_PAIRED_COLORS = [
    ('#1f77b4', '#aec7e8'),   # blue pair
    ('#ff7f0e', '#ffbb78'),   # orange pair
    ('#2ca02c', '#98df8a'),   # green pair
    ('#d62728', '#ff9896'),   # red pair
    ('#9467bd', '#c5b0d5'),   # purple pair
]


def apply_lop_style():
    """Apply the standard LoP matplotlib style."""
    matplotlib.rcParams['font.family'] = 'serif'
    matplotlib.rcParams['font.size'] = 11
    matplotlib.rcParams['axes.spines.top'] = False
    matplotlib.rcParams['axes.spines.right'] = False
    matplotlib.rcParams['axes.grid'] = True
    matplotlib.rcParams['grid.alpha'] = 0.3
    matplotlib.rcParams['figure.dpi'] = 150
    matplotlib.rcParams['savefig.dpi'] = 500
    matplotlib.rcParams['savefig.bbox'] = 'tight'
    matplotlib.rcParams['legend.framealpha'] = 0.8
    matplotlib.rcParams['legend.fontsize'] = 9
