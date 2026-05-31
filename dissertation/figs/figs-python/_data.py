"""Shared data + path helpers for thesis/paper plots.

Resolves run names against the shipped `data/` tier (with `outputs/runs/`
as fallback) and parses `eval_results.csv` and `train.log`.

Usage:
    from _data import FIGURES_DIR, find_run_dir, load_eval, parse_train_log
"""

import re
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent.parent.parent  # repo root (microrts-drl-uecd/)
FIGURES_DIR = Path(__file__).resolve().parent.parent / "figs-pdf"  # dissertation/figs/figs-pdf
RUNS_DIR = ROOT / "outputs" / "runs"  # legacy outputs/ root, kept for user-trained runs

# Candidate roots where a "run" can live, in priority order. The shipped
# data/ tier comes first so figure scripts work out of the box on a fresh
# clone; outputs/runs/ is the fallback used while the user is actively
# training.
_RUN_ROOTS = [
    ROOT / "data" / "agents",
    ROOT / "data" / "ablation" / "arch" / "agent",
    ROOT / "data" / "ablation" / "feat" / "agent",
    RUNS_DIR,
]

FIGURES_DIR.mkdir(exist_ok=True)


def find_run_dir(run_name):
    """Resolve a run name (or relative path) to its on-disk directory.

    Tries the shipped data/ tier first, then falls back to outputs/runs/.
    Returns the first existing candidate; if none match, returns the
    outputs/runs/<run_name> path so the caller's FileNotFoundError is
    informative.
    """
    for root in _RUN_ROOTS:
        candidate = root / run_name
        if candidate.exists():
            return candidate
    return RUNS_DIR / run_name


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
    return pd.read_csv(find_run_dir(run_name) / "eval_results.csv")


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
    path = find_run_dir(run_name) / "train.log"
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
