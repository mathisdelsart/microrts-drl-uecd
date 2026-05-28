"""Shared style and helpers for thesis/paper plots.

Usage:
    from _style import apply_style, FIGURES_DIR, C, BOT_COLORS
    apply_style()
    fig.savefig(FIGURES_DIR / "my_plot.pdf", bbox_inches="tight")
"""

import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent  # MasterThesis/
FIGURES_DIR = ROOT / "figures"
RUNS_DIR = ROOT / "outputs" / "runs"

FIGURES_DIR.mkdir(exist_ok=True)

# ── Colorblind-friendly palette ──
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

BOT_COLORS = {
    "CoacAI": "#1f77b4",
    "Mayari": "#d62728",
    "POLightRush": "#2ca02c",
    "POWorkerRush": "#ff7f0e",
    "RandomBiasedAI": "#9467bd",
    "Self-play": "#7f7f7f",
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


# ── Data helpers ────────────────────────────────────────────────────────────
def M(steps):
    """Convert steps (scalar or array) to millions."""
    return np.asarray(steps, dtype=float) / 1e6


def smooth(y, window=20):
    """Centered rolling mean, robust to short series."""
    if len(y) < window:
        return y
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


def load_eval(run_name):
    """Load a run's eval_results.csv as a DataFrame."""
    return pd.read_csv(RUNS_DIR / run_name / "eval_results.csv")


def overall_wr(df):
    """Collapse a per-bot eval DataFrame to overall WR per global_step."""
    g = df.groupby("global_step").agg(wins=("wins", "sum"), games=("games", "sum"))
    g["win_rate"] = g["wins"] / g["games"]
    return g.reset_index()


_TRAIN_LOG_PATTERN = re.compile(
    r"step=([\d,]+)\s+\d+ eps:.*"
    r"ret=\s*([\d.]+)\s+len=\s*(\d+)\s+"
    r"WR\[(.+?)\]"
)


def parse_train_log(run_name, sample_every=500):
    """Parse train.log to extract per-episode step, return, length, per-bot WR.

    Returns a DataFrame with columns: step, ret, len, wr_<Bot>... .
    """
    path = RUNS_DIR / run_name / "train.log"
    steps, rets, lens = [], [], []
    bot_wrs = {}
    count = 0
    with open(path) as f:
        for line in f:
            m = _TRAIN_LOG_PATTERN.search(line)
            if not m:
                continue
            count += 1
            if count % sample_every != 0:
                continue
            steps.append(int(m.group(1).replace(",", "")))
            rets.append(float(m.group(2)))
            lens.append(int(m.group(3)))
            for tok in m.group(4).split():
                k, v = tok.split("=")
                bot_wrs.setdefault(k, []).append(float(v.replace("%", "")) / 100.0)
    rows = []
    for i in range(len(steps)):
        row = {"step": steps[i], "ret": rets[i], "len": lens[i]}
        for k, v_list in bot_wrs.items():
            if i < len(v_list):
                row[f"wr_{k}"] = v_list[i]
        rows.append(row)
    return pd.DataFrame(rows)


# ── Diagnostic plot (shared, called by 3 thin wrappers) ─────────────────────
def plot_training_diagnostics(run_name):
    """3-panel diagnostic (return / length / per-bot WR) for one run.

    Reads train.log only (no TB). Output: figures/diagnostics_<run_name>.pdf
    """
    import matplotlib.ticker as mticker

    log = parse_train_log(run_name, sample_every=300)
    if log.empty:
        print(f"  No train log data for {run_name}")
        return

    wr_cols = [c for c in log.columns if c.startswith("wr_")]
    short_name = {
        "wr_Coac": "CoacAI",
        "wr_Maya": "Mayari",
        "wr_POLi": "POLightRush",
        "wr_POWo": "POWorkerRush",
        "wr_Rand": "RandomBiasedAI",
        "wr_Self": "Self-play",
    }

    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    fig.subplots_adjust(hspace=0.18)

    # Episodic return
    ax = axes[0]
    ax.plot(M(log["step"]), smooth(log["ret"], 30), color=C["ret"], linewidth=1.2, alpha=0.8)
    ax.set_ylabel("Episodic return")
    ax.set_ylim(0, None)
    max_step_M = M(log["step"].max())
    xlim_max = int(np.ceil(max_step_M / 50) * 50)
    ax.set_xlim(0, xlim_max)

    # Episode length
    ax = axes[1]
    ax.plot(M(log["step"]), smooth(log["len"], 30), color=C["len"], linewidth=1.2, alpha=0.8)
    ax.set_ylabel("Episode length (frames)")
    ax.set_ylim(0, None)

    # Per-bot training WR
    ax = axes[2]
    for col in wr_cols:
        name = short_name.get(col, col)
        color = BOT_COLORS.get(name, "gray")
        ax.plot(
            M(log["step"]), smooth(log[col], 30), color=color, linewidth=1.2, alpha=0.8, label=name
        )
    ax.set_xlabel("Training steps (M)")
    ax.set_ylabel("Training win-rate (%)")
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))
    ax.set_ylim(0, 1.08)
    ax.legend(loc="lower right", ncol=2, fontsize=8)

    fig.tight_layout()
    out = FIGURES_DIR / f"diagnostics_{run_name}.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"Saved: {out}")
    plt.close(fig)
