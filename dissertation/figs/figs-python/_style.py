"""Shared style helpers for thesis/paper plots.

Usage:
    from _style import apply_style, C
    apply_style()
"""

import matplotlib.pyplot as plt

# Colorblind-friendly palette used by every figure script.
C = {
    "best": "#1f77b4",  # blue
    "stier": "#e377c2",  # pink
    "shaped": "#1f77b4",  # blue
    "sparse": "#d62728",  # red
    "wr": "#2ca02c",  # green
    "len": "#9467bd",  # purple
    "ret": "#ff7f0e",  # orange
    "bc": "#1f77b4",
    "scratch": "#d62728",
    "bc_only": "#2ca02c",
}


def apply_style():
    """Apply thesis-quality matplotlib rcParams."""
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.family": "serif",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "legend.fontsize": 8.5,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "figure.facecolor": "white",
            "axes.facecolor": "#fafafa",
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "lines.linewidth": 1.8,
            "legend.framealpha": 0.9,
            "legend.edgecolor": "#cccccc",
        }
    )
