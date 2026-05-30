"""
Training run analysis: per-run PDF plots + quantitative summaries.

Generates individual PDF figures and a text summary for each training run,
suitable for direct inclusion in a LaTeX thesis report.

Usage:
    # Single run (any outputs/runs/<run>/ produced by `python -m microrts_agent train ...`)
    python -m microrts_agent analysis metrics outputs/runs/<run>

    # Multiple runs (each analysed independently)
    python -m microrts_agent analysis metrics outputs/runs/UECD-SingleMap-* outputs/runs/UECD-MultiMap-*

    # All runs under outputs/runs/
    python -m microrts_agent analysis metrics --all

    # Custom smoothing (EMA alpha, higher = smoother)
    python -m microrts_agent analysis metrics outputs/runs/<run> --smoothing 0.97
"""

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from datetime import datetime
from glob import glob

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from microrts_agent.paths import RUNS_DIR

# ---------------------------------------------------------------------------
# Style
# ---------------------------------------------------------------------------


def setup_style():
    sns.set_style("whitegrid")
    plt.rcParams.update(
        {
            "figure.dpi": 150,
            "savefig.dpi": 300,
            "font.size": 11,
            "pdf.fonttype": 42,
            "figure.max_open_warning": 0,
        }
    )


def get_colors(n):
    return sns.color_palette("husl", max(n, 1))


def clean_axes(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------


def _find_event_files(run_dir):
    """Find TensorBoard event files and return their total size in bytes."""
    total = 0
    for f in os.listdir(run_dir):
        if f.startswith("events.out.tfevents"):
            total += os.path.getsize(os.path.join(run_dir, f))
    return total


def load_tensorboard(run_dir, max_points=0):
    """Load all scalar tags from TensorBoard events.

    Args:
        run_dir: Path to the run directory.
        max_points: Max data points per tag (0 = load all). Setting this lower
                    (e.g. 5000) makes loading much faster on large event files
                    by telling EventAccumulator to subsample.

    Returns dict: tag -> (steps_array, values_array)
    """
    from tensorboard.backend.event_processing.event_accumulator import (
        EventAccumulator,  # type: ignore
    )

    event_size = _find_event_files(run_dir)
    if event_size > 0:
        size_mb = event_size / (1024 * 1024)
        print(
            f"  Event file(s): {size_mb:.1f} MB"
            + (" (large file, this may take a few minutes...)" if size_mb > 100 else "")
        )

    t0 = time.time()
    ea = EventAccumulator(str(run_dir), size_guidance={"scalars": max_points})
    ea.Reload()
    elapsed = time.time() - t0
    print(f"  Loaded in {elapsed:.1f}s")

    tags = ea.Tags().get("scalars", [])
    data = {}
    for tag in tags:
        events = ea.Scalars(tag)
        steps = np.array([e.step for e in events])
        values = np.array([e.value for e in events])
        data[tag] = (steps, values)
    return data


def load_config(run_dir):
    path = os.path.join(run_dir, "config.json")
    if not os.path.exists(path):
        return {}
    with open(path) as f:
        return json.load(f)


def load_eval_csv(run_dir):
    """Load eval_results.csv into structured data.

    Returns dict: bot_name -> list of {step, wr, wr_p0, wr_p1, games, avg_return, avg_length}
    Returns None if no eval data.
    """
    path = os.path.join(run_dir, "eval_results.csv")
    if not os.path.exists(path):
        return None

    data = defaultdict(list)
    with open(path) as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            return None
        has_position = "win_rate_p0" in reader.fieldnames
        for row in reader:
            if not row.get("global_step"):
                continue
            bot = row["bot"]
            entry = {
                "step": int(row["global_step"]),
                "wr": float(row["win_rate"]),
                "games": int(row["games"]),
                "avg_return": float(row.get("avg_return", 0)),
                "avg_length": float(row.get("avg_length", 0)),
            }
            if has_position and row.get("win_rate_p0"):
                entry["wr_p0"] = float(row["win_rate_p0"])
                entry["wr_p1"] = float(row["win_rate_p1"])
                entry["games_p0"] = int(row.get("games_p0", 0))
                entry["games_p1"] = int(row.get("games_p1", 0))
            data[bot].append(entry)

    return dict(data) if data else None


# ---------------------------------------------------------------------------
# Smoothing
# ---------------------------------------------------------------------------


def ema(values, alpha=0.95):
    """Exponential moving average."""
    if len(values) == 0:
        return values
    smoothed = np.zeros_like(values)
    smoothed[0] = values[0]
    for i in range(1, len(values)):
        smoothed[i] = alpha * smoothed[i - 1] + (1 - alpha) * values[i]
    return smoothed


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def format_steps(steps):
    """Format step count for display: 1200000 -> '1.2M'."""
    if steps >= 1_000_000:
        return f"{steps / 1_000_000:.1f}M"
    elif steps >= 1_000:
        return f"{steps / 1_000:.0f}K"
    return str(steps)


def save_fig(fig, path):
    fig.savefig(path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)
    print(f"  -> {os.path.basename(path)}")


def has_tags(tb_data, prefixes):
    """Check if any tag starts with any of the given prefixes."""
    return any(tag.startswith(p) for tag in tb_data for p in prefixes)


# ---------------------------------------------------------------------------
# Plot functions
# ---------------------------------------------------------------------------


def plot_training_curves(tb_data, out_dir, alpha):
    """01: Win rate, episodic return, episode length, SPS, total episodes vs steps."""
    metrics = [
        ("charts/win_rate", "Win Rate", "Win Rate"),
        ("charts/episodic_return", "Episodic Return", "Return"),
        ("charts/episodic_length", "Episode Length", "Steps"),
        ("charts/sps", "Steps Per Second", "SPS"),
        ("charts/total_episodes", "Total Episodes", "Episodes"),
    ]
    available = [(tag, title, ylabel) for tag, title, ylabel in metrics if tag in tb_data]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)), squeeze=False)
    for i, (tag, title, ylabel) in enumerate(available):
        ax = axes[i, 0]
        steps, vals = tb_data[tag]
        ax.plot(steps, vals, alpha=0.15, color="C0", linewidth=0.5)
        ax.plot(steps, ema(vals, alpha), color="C0", linewidth=1.5)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_ylabel(ylabel)
        if tag == "charts/win_rate":
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        clean_axes(ax)
    axes[-1, 0].set_xlabel("Training Steps")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "01_training_curves.pdf"))


def plot_loss_curves(tb_data, out_dir, alpha):
    """02: Policy loss, value loss, entropy, KL divergence, cost loss, aux losses."""
    metrics = [
        ("losses/policy_loss", "Policy Loss"),
        ("losses/value_loss", "Value Loss"),
        ("losses/entropy", "Entropy"),
        ("losses/approx_kl", "Approx KL"),
        ("losses/value_loss_cost", "Value Loss (Cost Head)"),
    ]
    # Dynamically discover aux loss tags (aux_spatial, aux_unit_count, aux_contrastive, etc.)
    aux_tags = sorted([t for t in tb_data if t.startswith("losses/aux_")])
    for tag in aux_tags:
        label = tag.replace("losses/", "").replace("_", " ").title()
        metrics.append((tag, label))

    available = [(tag, title) for tag, title in metrics if tag in tb_data]
    if not available:
        return

    ncols = 2
    nrows = (len(available) + 1) // 2
    fig, axes = plt.subplots(nrows, ncols, figsize=(12, 3.5 * nrows), squeeze=False)
    for i, (tag, title) in enumerate(available):
        ax = axes[i // ncols, i % ncols]
        steps, vals = tb_data[tag]
        ax.plot(steps, vals, alpha=0.15, color="C1", linewidth=0.5)
        ax.plot(steps, ema(vals, alpha), color="C1", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        clean_axes(ax)
    # Hide empty subplot if odd count
    for j in range(len(available), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    axes[-1, 0].set_xlabel("Training Steps")
    if ncols > 1 and len(available) > 1:
        axes[-1, min(1, ncols - 1)].set_xlabel("Training Steps")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "02_loss_curves.pdf"))

    # Also plot value_loss_sparse if present
    if "losses/value_loss_sparse" in tb_data:
        fig2, ax2 = plt.subplots(figsize=(12, 4))
        steps, vals = tb_data["losses/value_loss_sparse"]
        ax2.plot(steps, vals, alpha=0.15, color="C3", linewidth=0.5)
        ax2.plot(steps, ema(vals, alpha), color="C3", linewidth=1.5)
        ax2.set_title("Sparse Value Loss (WinLoss head)", fontsize=13, fontweight="bold")
        ax2.set_xlabel("Training Steps")
        clean_axes(ax2)
        fig2.tight_layout()
        save_fig(fig2, os.path.join(out_dir, "02b_value_loss_sparse.pdf"))


def plot_position_bias(eval_data, out_dir, n_last=5):
    """03: P0 vs P1 win rate per bot (grouped bar chart, aggregated over last N evals)."""
    if not eval_data:
        return

    # Check if position data is available
    bots = sorted(eval_data.keys())
    has_pos = any("wr_p0" in e for entries in eval_data.values() for e in entries)
    if not has_pos:
        return

    # Aggregate over last N eval rounds (weighted by games for accuracy)
    bot_data = {}
    for bot in bots:
        entries = [e for e in eval_data[bot] if "wr_p0" in e]
        if not entries:
            continue
        last_n = entries[-n_last:]
        total_games_p0 = sum(e.get("games_p0", 0) for e in last_n)
        total_games_p1 = sum(e.get("games_p1", 0) for e in last_n)
        # Weighted average by number of games per position
        if total_games_p0 > 0 and total_games_p1 > 0:
            wr_p0 = sum(e["wr_p0"] * e.get("games_p0", 1) for e in last_n) / total_games_p0
            wr_p1 = sum(e["wr_p1"] * e.get("games_p1", 1) for e in last_n) / total_games_p1
        else:
            # Fallback: simple average
            wr_p0 = np.mean([e["wr_p0"] for e in last_n])
            wr_p1 = np.mean([e["wr_p1"] for e in last_n])
        bot_data[bot] = {"wr_p0": wr_p0, "wr_p1": wr_p1, "n_evals": len(last_n)}

    if not bot_data:
        return

    bot_names = list(bot_data.keys())
    p0_wrs = [bot_data[b]["wr_p0"] for b in bot_names]
    p1_wrs = [bot_data[b]["wr_p1"] for b in bot_names]
    deltas = [p0 - p1 for p0, p1 in zip(p0_wrs, p1_wrs, strict=False)]
    n_used = bot_data[bot_names[0]]["n_evals"]

    x = np.arange(len(bot_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(
        x - width / 2,
        p0_wrs,
        width,
        label="P0 (first player)",
        color="C0",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )
    bars2 = ax.bar(
        x + width / 2,
        p1_wrs,
        width,
        label="P1 (second player)",
        color="C3",
        alpha=0.8,
        edgecolor="black",
        linewidth=0.5,
    )

    # Annotate deltas
    for i, (b1, b2, d) in enumerate(zip(bars1, bars2, deltas, strict=False)):
        y_max = max(b1.get_height(), b2.get_height())
        sign = "+" if d >= 0 else ""
        ax.text(
            i,
            y_max + 0.03,
            f"{sign}{d:.2f}",
            ha="center",
            va="bottom",
            fontsize=9,
            fontweight="bold",
            color="gray",
        )

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_xticks(x)
    ax.set_xticklabels(bot_names, rotation=30, ha="right")
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title(f"Position Bias: P0 vs P1 (last {n_used} evals)", fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    clean_axes(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "03_eval_position_bias.pdf"))


def plot_reward_decomposition(tb_data, out_dir, alpha):
    """04: Individual reward function contributions (one subplot each)."""
    prefix = "charts/episodic_return/"
    # Only keep raw (non-discounted) reward tags for clarity
    reward_tags = sorted([t for t in tb_data if t.startswith(prefix) and "discounted" not in t])
    if not reward_tags:
        return

    colors = get_colors(len(reward_tags))
    n = len(reward_tags)
    ncols = min(n, 3)
    nrows = (n + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for i, tag in enumerate(reward_tags):
        ax = axes[i // ncols, i % ncols]
        name = tag[len(prefix) :]
        short_name = name.replace("RewardFunction", "").replace("Produce", "Prod.")
        steps, vals = tb_data[tag]
        ax.plot(steps, vals, alpha=0.12, color=colors[i], linewidth=0.5)
        ax.plot(steps, ema(vals, alpha), color=colors[i], linewidth=1.5)
        ax.set_title(short_name, fontsize=12, fontweight="bold")
        ax.set_ylabel("Return", fontsize=9)
        clean_axes(ax)

    # Hide empty subplots
    for j in range(n, nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)

    # Add x-label to bottom row
    for c in range(ncols):
        if (nrows - 1) * ncols + c < n:
            axes[nrows - 1, c].set_xlabel("Training Steps", fontsize=10)

    fig.suptitle("Reward Decomposition", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "04_reward_decomposition.pdf"))


def plot_schedules(tb_data, out_dir, alpha):
    """05: Learning rate, entropy coefficient, reward weight, advantage weight, VF coef schedules."""
    schedule_tags = [
        ("charts/learning_rate", "Learning Rate"),
        ("schedule/ent_coef", "Entropy Coefficient"),
        ("schedule/reward_weight_winloss", "WinLoss Reward Weight"),
        ("schedule/reward_weight_shaped_sum", "Shaped Reward Weights (sum)"),
        ("schedule/adv_w_shaped", "Advantage Weight: Shaped"),
        ("schedule/adv_w_sparse", "Advantage Weight: Sparse"),
        ("schedule/adv_w_cost", "Advantage Weight: Cost"),
        ("schedule/eff_vf_coef", "Effective VF Coef: Shaped"),
        ("schedule/eff_vf_sparse", "Effective VF Coef: Sparse"),
        ("schedule/eff_vf_cost", "Effective VF Coef: Cost"),
    ]
    available = [(tag, title) for tag, title in schedule_tags if tag in tb_data]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3 * len(available)), squeeze=False)
    for i, (tag, title) in enumerate(available):
        ax = axes[i, 0]
        steps, vals = tb_data[tag]
        ax.plot(steps, vals, color="C2", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        clean_axes(ax)
    axes[-1, 0].set_xlabel("Training Steps")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "05_schedules.pdf"))


def plot_selfplay_pfsp(tb_data, out_dir, alpha):
    """06: Self-play win rate and PFSP stats."""
    sp_tags = [
        ("charts/selfplay_win_rate", "Self-Play Win Rate"),
        ("pfsp/current_opponent_wr", "PFSP Opponent Win Rate"),
        ("pfsp/pool_size", "PFSP Pool Size"),
    ]
    available = [(tag, title) for tag, title in sp_tags if tag in tb_data]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)), squeeze=False)
    for i, (tag, title) in enumerate(available):
        ax = axes[i, 0]
        steps, vals = tb_data[tag]
        if "win_rate" in tag.lower() or "wr" in tag.lower():
            ax.plot(steps, vals, alpha=0.15, color="C4", linewidth=0.5)
            ax.plot(steps, ema(vals, alpha), color="C4", linewidth=1.5)
            ax.set_ylim(-0.05, 1.05)
            ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        else:
            ax.plot(steps, vals, color="C4", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        clean_axes(ax)
    axes[-1, 0].set_xlabel("Training Steps")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "06_selfplay_pfsp.pdf"))


def plot_adaptive_opponents(tb_data, out_dir):
    """07: Adaptive opponent env allocation over time (stacked area)."""
    prefix = "adaptive/"
    suffix = "_envs"
    adaptive_tags = sorted([t for t in tb_data if t.startswith(prefix) and t.endswith(suffix)])
    if not adaptive_tags:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = get_colors(len(adaptive_tags))
    tb_data[adaptive_tags[0]][0]

    # Build common step grid and interpolate all series onto it
    all_step_sets = [np.asarray(tb_data[tag][0]) for tag in adaptive_tags]
    common_steps = np.unique(np.concatenate(all_step_sets))

    labels = []
    stacks = []
    for tag in adaptive_tags:
        name = tag[len(prefix) : -len(suffix)]
        labels.append(name)
        steps, vals = tb_data[tag]
        stacks.append(np.interp(common_steps, np.asarray(steps), np.asarray(vals)))

    ax.stackplot(common_steps, *stacks, labels=labels, colors=colors, alpha=0.75)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Number of Environments", fontsize=12)
    ax.set_title("Adaptive Opponent Allocation", fontsize=14, fontweight="bold")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)
    clean_axes(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "07_adaptive_opponents.pdf"))


def plot_ppo_diagnostics(tb_data, out_dir, alpha):
    """08: Clip fraction, explained variance, gradient norm."""
    diag_tags = [
        ("losses/clip_fraction", "Clip Fraction"),
        ("losses/explained_variance", "Explained Variance"),
        ("losses/grad_norm", "Gradient Norm"),
    ]
    available = [(tag, title) for tag, title in diag_tags if tag in tb_data]
    if not available:
        return

    fig, axes = plt.subplots(len(available), 1, figsize=(12, 3.5 * len(available)), squeeze=False)
    for i, (tag, title) in enumerate(available):
        ax = axes[i, 0]
        steps, vals = tb_data[tag]
        ax.plot(steps, vals, alpha=0.15, color="C5", linewidth=0.5)
        ax.plot(steps, ema(vals, alpha), color="C5", linewidth=1.5)
        ax.set_title(title, fontsize=13, fontweight="bold")
        if "clip_fraction" in tag:
            ax.set_ylim(-0.05, 1.05)
        if "explained_variance" in tag:
            ax.set_ylim(-0.5, 1.05)
            ax.axhline(0.0, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        clean_axes(ax)
    axes[-1, 0].set_xlabel("Training Steps")
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "08_ppo_diagnostics.pdf"))


def _build_bot_color_map(tb_data, eval_data):
    """Build a consistent color map across train and eval bot names."""
    prefix = "charts/train_wr/"
    train_names = [t[len(prefix) :] for t in tb_data if t.startswith(prefix)]
    eval_names = sorted(eval_data.keys()) if eval_data else []
    all_names = sorted(set(train_names) | set(eval_names))
    return dict(zip(all_names, get_colors(len(all_names)), strict=False))


def plot_per_opponent_training_wr(tb_data, out_dir, alpha, color_map=None):
    """09: Per-opponent training win rate (from charts/train_wr/*)."""
    prefix = "charts/train_wr/"
    wr_tags = sorted([t for t in tb_data if t.startswith(prefix)])
    if not wr_tags:
        return

    if color_map is None:
        names = [t[len(prefix) :] for t in wr_tags]
        color_map = dict(zip(names, get_colors(len(names)), strict=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    for tag in wr_tags:
        name = tag[len(prefix) :]
        steps, vals = tb_data[tag]
        c = color_map.get(name, "gray")
        ax.plot(steps, ema(vals, alpha), color=c, linewidth=1.5, label=name, alpha=0.85)

    ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Training Win Rate per Opponent", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    clean_axes(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "09_training_wr_per_opponent.pdf"))


def plot_per_opponent_stats(tb_data, out_dir, alpha, color_map=None):
    """09b: Per-opponent training return and game length (from charts/train_ret/* and charts/train_len/*)."""
    ret_prefix = "charts/train_ret/"
    len_prefix = "charts/train_len/"
    ret_tags = sorted([t for t in tb_data if t.startswith(ret_prefix)])
    len_tags = sorted([t for t in tb_data if t.startswith(len_prefix)])
    if not ret_tags and not len_tags:
        return

    # Use shared color map or build one from available opponent names
    if color_map is None:
        names = sorted(
            set([t[len(ret_prefix) :] for t in ret_tags] + [t[len(len_prefix) :] for t in len_tags])
        )
        color_map = dict(zip(names, get_colors(len(names)), strict=False))

    nplots = (1 if ret_tags else 0) + (1 if len_tags else 0)
    fig, axes = plt.subplots(nplots, 1, figsize=(12, 5 * nplots), squeeze=False)
    plot_idx = 0

    # Top subplot: return per opponent
    if ret_tags:
        ax = axes[plot_idx, 0]
        for tag in ret_tags:
            name = tag[len(ret_prefix) :]
            steps, vals = tb_data[tag]
            c = color_map.get(name, "gray")
            ax.plot(steps, ema(vals, alpha), color=c, linewidth=1.5, label=name, alpha=0.85)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Episodic Return", fontsize=12)
        ax.set_title("Training Return per Opponent", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        clean_axes(ax)
        plot_idx += 1

    # Bottom subplot: length per opponent
    if len_tags:
        ax = axes[plot_idx, 0]
        for tag in len_tags:
            name = tag[len(len_prefix) :]
            steps, vals = tb_data[tag]
            c = color_map.get(name, "gray")
            ax.plot(steps, ema(vals, alpha), color=c, linewidth=1.5, label=name, alpha=0.85)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Episode Length", fontsize=12)
        ax.set_title("Training Episode Length per Opponent", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        clean_axes(ax)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "09b_training_stats_per_opponent.pdf"))


def plot_eval_wr_per_opponent(eval_data, out_dir, color_map=None):
    """10: Per-bot evaluation win rate over training."""
    if not eval_data:
        return

    bots = sorted(eval_data.keys())
    if color_map is None:
        color_map = dict(zip(bots, get_colors(len(bots)), strict=False))

    fig, ax = plt.subplots(figsize=(12, 5))
    for bot in bots:
        entries = eval_data[bot]
        steps = [e["step"] for e in entries]
        wrs = [e["wr"] for e in entries]
        final_wr = wrs[-1] if wrs else 0
        c = color_map.get(bot, "gray")
        ax.plot(steps, wrs, color=c, linewidth=1.5, label=f"{bot} ({final_wr:.0%})", alpha=0.85)

    # Overall WR
    if len(bots) > 1:
        step_to_wr = defaultdict(list)
        for entries in eval_data.values():
            for e in entries:
                step_to_wr[e["step"]].append(e["wr"])
        overall_steps = sorted(step_to_wr.keys())
        overall_wrs = [np.mean(step_to_wr[s]) for s in overall_steps]
        ax.plot(
            overall_steps,
            overall_wrs,
            color="black",
            linewidth=2.5,
            linestyle="--",
            label=f"Overall ({overall_wrs[-1]:.0%})",
            alpha=0.9,
        )

    ax.axhline(0.5, color="gray", linestyle=":", alpha=0.4, linewidth=0.8)
    ax.set_ylim(-0.05, 1.05)
    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Win Rate", fontsize=12)
    ax.set_title("Evaluation Win Rate per Bot", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
    clean_axes(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "10_eval_wr_per_opponent.pdf"))


def plot_combined_wr_per_opponent(tb_data, eval_data, out_dir, alpha, color_map=None):
    """11: Per-opponent subplots, train (solid) + eval (dashed) on each."""
    prefix = "charts/train_wr/"
    train_bots = {t[len(prefix) :]: t for t in tb_data if t.startswith(prefix)}
    eval_bots = set(eval_data.keys()) if eval_data else set()
    all_bots = sorted(set(train_bots.keys()) | eval_bots)
    if not all_bots:
        return

    if color_map is None:
        color_map = dict(zip(all_bots, get_colors(len(all_bots)), strict=False))

    ncols = min(len(all_bots), 3)
    nrows = (len(all_bots) + ncols - 1) // ncols
    fig, axes = plt.subplots(nrows, ncols, figsize=(5 * ncols, 3.5 * nrows), squeeze=False)

    for i, bot in enumerate(all_bots):
        ax = axes[i // ncols, i % ncols]
        c = color_map.get(bot, "gray")

        # Training WR
        if bot in train_bots:
            steps, vals = tb_data[train_bots[bot]]
            ax.plot(steps, vals, alpha=0.1, color=c, linewidth=0.5)
            ax.plot(steps, ema(vals, alpha), color=c, linewidth=1.5, label="Train")

        # Eval WR
        if eval_data and bot in eval_data:
            entries = eval_data[bot]
            e_steps = [e["step"] for e in entries]
            e_wrs = [e["wr"] for e in entries]
            ax.plot(
                e_steps,
                e_wrs,
                color=c,
                linewidth=1.5,
                linestyle="--",
                marker="o",
                markersize=3,
                label="Eval",
            )

        ax.axhline(0.5, color="gray", linestyle=":", alpha=0.3, linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_title(bot, fontsize=12, fontweight="bold")
        ax.legend(fontsize=8, loc="lower right")
        clean_axes(ax)

    # Hide empty subplots
    for j in range(len(all_bots), nrows * ncols):
        axes[j // ncols, j % ncols].set_visible(False)
    for c in range(ncols):
        if (nrows - 1) * ncols + c < len(all_bots):
            axes[nrows - 1, c].set_xlabel("Training Steps", fontsize=10)

    fig.suptitle("Win Rate per Opponent (Train + Eval)", fontsize=14, fontweight="bold", y=1.01)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "11_combined_wr_per_opponent.pdf"))


def plot_action_distribution(tb_data, out_dir, alpha):
    """12: Action type distribution and produce unit type over training."""
    # Action type percentages
    action_prefix = "actions/pct_"
    action_tags = sorted([t for t in tb_data if t.startswith(action_prefix)])

    # Produce type percentages
    produce_prefix = "actions/produce_"
    produce_tags = sorted([t for t in tb_data if t.startswith(produce_prefix)])

    if not action_tags:
        return

    nplots = 1 + (1 if produce_tags else 0)
    fig, axes = plt.subplots(nplots, 1, figsize=(12, 4.5 * nplots), squeeze=False)

    # Action type stacked area
    ax = axes[0, 0]
    labels = [t[len(action_prefix) :].capitalize() for t in action_tags]
    colors = get_colors(len(action_tags))
    stacks = []
    ref_steps = None
    for tag in action_tags:
        steps, vals = tb_data[tag]
        stacks.append(ema(vals, alpha))
        ref_steps = steps
    ax.stackplot(ref_steps, *stacks, labels=labels, colors=colors, alpha=0.8)
    ax.set_ylabel("Fraction", fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_title("Action Type Distribution", fontsize=13, fontweight="bold")
    ax.legend(loc="upper right", fontsize=9, framealpha=0.9)
    clean_axes(ax)

    # Produce unit type stacked area
    if produce_tags:
        ax2 = axes[1, 0]
        labels2 = [t[len(produce_prefix) :].capitalize() for t in produce_tags]
        colors2 = get_colors(len(produce_tags))
        stacks2 = []
        produce_steps = None
        for tag in produce_tags:
            produce_steps, vals = tb_data[tag]
            stacks2.append(ema(vals, alpha))
        ax2.stackplot(produce_steps, *stacks2, labels=labels2, colors=colors2, alpha=0.8)
        ax2.set_ylabel("Fraction", fontsize=11)
        ax2.set_ylim(0, 1.05)
        ax2.set_title("Produce Unit Type Distribution", fontsize=13, fontweight="bold")
        ax2.legend(loc="upper right", fontsize=9, framealpha=0.9)
        clean_axes(ax2)

    axes[-1, 0].set_xlabel("Training Steps", fontsize=11)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "12_action_distribution.pdf"))


def plot_plr_stats(tb_data, out_dir, alpha):
    """13: PLR per-map win rates and sampling probabilities (if plr/ tags exist)."""
    wr_prefix = "plr/wr_"
    prob_prefix = "plr/prob_"
    wr_tags = sorted([t for t in tb_data if t.startswith(wr_prefix)])
    prob_tags = sorted([t for t in tb_data if t.startswith(prob_prefix)])
    if not wr_tags and not prob_tags:
        return

    nplots = (1 if wr_tags else 0) + (1 if prob_tags else 0)
    fig, axes = plt.subplots(nplots, 1, figsize=(12, 5 * nplots), squeeze=False)
    plot_idx = 0

    # Top subplot: per-map win rate
    if wr_tags:
        ax = axes[plot_idx, 0]
        colors = get_colors(len(wr_tags))
        for i, tag in enumerate(wr_tags):
            name = tag[len(wr_prefix) :]
            steps, vals = tb_data[tag]
            ax.plot(steps, ema(vals, alpha), color=colors[i], linewidth=1.5, label=name, alpha=0.85)
        ax.axhline(0.5, color="gray", linestyle="--", alpha=0.5, linewidth=0.8)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Win Rate", fontsize=12)
        ax.set_title("PLR Per-Map Win Rate", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="lower right", framealpha=0.9)
        clean_axes(ax)
        plot_idx += 1

    # Bottom subplot: per-map sampling probability
    if prob_tags:
        ax = axes[plot_idx, 0]
        colors = get_colors(len(prob_tags))
        for i, tag in enumerate(prob_tags):
            name = tag[len(prob_prefix) :]
            steps, vals = tb_data[tag]
            ax.plot(steps, vals, color=colors[i], linewidth=1.5, label=name, alpha=0.85)
        ax.set_ylim(-0.05, 1.05)
        ax.set_xlabel("Training Steps", fontsize=12)
        ax.set_ylabel("Sampling Probability", fontsize=12)
        ax.set_title("PLR Per-Map Sampling Probability", fontsize=14, fontweight="bold")
        ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
        clean_axes(ax)

    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "13_plr_stats.pdf"))


def plot_prioritized_weights(tb_data, out_dir, alpha):
    """14: Per-opponent importance weights from prioritized sampling (opponents/weight_*)."""
    prefix = "opponents/weight_"
    weight_tags = sorted([t for t in tb_data if t.startswith(prefix)])
    if not weight_tags:
        return

    fig, ax = plt.subplots(figsize=(12, 5))
    colors = get_colors(len(weight_tags))
    for i, tag in enumerate(weight_tags):
        name = tag[len(prefix) :]
        steps, vals = tb_data[tag]
        ax.plot(steps, ema(vals, alpha), color=colors[i], linewidth=1.5, label=name, alpha=0.85)

    ax.set_xlabel("Training Steps", fontsize=12)
    ax.set_ylabel("Importance Weight", fontsize=12)
    ax.set_title("Prioritized Sampling Weights per Opponent", fontsize=14, fontweight="bold")
    ax.legend(fontsize=9, loc="upper right", framealpha=0.9)
    clean_axes(ax)
    fig.tight_layout()
    save_fig(fig, os.path.join(out_dir, "14_prioritized_weights.pdf"))


# ---------------------------------------------------------------------------
# Quantitative analysis
# ---------------------------------------------------------------------------


def compute_convergence(eval_data, thresholds=(0.5, 0.7, 0.9)):
    """Steps to reach each WR threshold per bot."""
    results = {}
    for bot, entries in eval_data.items():
        bot_results = {}
        for thr in thresholds:
            reached = None
            for e in entries:
                if e["wr"] >= thr:
                    reached = e["step"]
                    break
            bot_results[thr] = reached
        results[bot] = bot_results
    return results


def compute_stability(eval_data, last_fraction=0.2):
    """Win rate statistics over the last N% of evaluations, per bot."""
    results = {}
    for bot, entries in eval_data.items():
        if not entries:
            continue
        n = max(1, int(len(entries) * last_fraction))
        last_wrs = [e["wr"] for e in entries[-n:]]
        results[bot] = {
            "mean": np.mean(last_wrs),
            "std": np.std(last_wrs),
            "min": np.min(last_wrs),
            "max": np.max(last_wrs),
        }
    return results


def write_summary(config, tb_data, eval_data, out_dir):
    """Write summary.txt with config + quantitative analysis."""
    lines = []
    exp_name = config.get("exp_name", os.path.basename(out_dir.removesuffix("/analysis")))

    lines.append(f"{'=' * 60}")
    lines.append(f"  Run Analysis: {exp_name}")
    lines.append(f"  Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append(f"{'=' * 60}")
    lines.append("")

    # Config summary
    lines.append("--- Configuration ---")
    arch = config.get("architecture", "?")
    ch = config.get("arch_channels")
    lines.append(f"Architecture:     {arch}" + (f" ({ch}ch)" if ch else ""))
    lines.append(f"Total steps:      {config.get('total_timesteps', '?'):,}")
    bot_envs = config.get("num_bot_envs", "?")
    sp_envs = config.get("num_selfplay_envs", 0)
    lines.append(f"Num envs:         {bot_envs} bot + {sp_envs} self-play")
    lines.append(f"Seed:             {config.get('seed', '?')}")

    # Active features
    features = []
    if config.get("filtered_masks"):
        features.append("filtered_masks")
    if config.get("reserved_obs"):
        features.append("reserved_obs")
    if config.get("adaptive_opponents"):
        features.append("adaptive")
    if config.get("pfsp"):
        features.append(f"pfsp({config.get('pfsp_mode', '?')})")
    if config.get("dual_value_heads"):
        features.append("dual_heads")
    if config.get("triple_value_heads"):
        features.append("triple_heads")
    if config.get("extended_obs"):
        features.append("extended_obs")
    frame_stack = config.get("frame_stack", 0)
    if frame_stack and frame_stack > 0:
        features.append(f"frame_stack({config.get('frame_stack')})")
    if config.get("multi_map"):
        features.append("multi_map")
    if config.get("aux_spatial"):
        features.append("aux_spatial")
    if config.get("aux_unit_count"):
        features.append("aux_unit_count")
    if config.get("aux_contrastive"):
        features.append("aux_contrastive")
    if config.get("plr"):
        features.append("PLR")
    if config.get("prioritized_sampling"):
        features.append("prioritized_sampling")
    if config.get("augment_symmetry"):
        features.append("symmetry_aug")
    if config.get("alternate_players"):
        features.append("P0/P1_alternate")
    if config.get("reward_schedule", "none") != "none":
        features.append(
            f"reward_sched({config.get('reward_schedule_start', '?')}-{config.get('reward_schedule_end', '?')})"
        )
    if config.get("ent_coef_end") is not None:
        features.append(f"ent_anneal({config.get('ent_coef')}->{config.get('ent_coef_end')})")
    lines.append(f"Features:         {', '.join(features) if features else 'none'}")
    lines.append(f"Reward weights:   {config.get('reward_weight', '?')}")
    lines.append(f"Learning rate:    {config.get('learning_rate', '?')}")
    lines.append(f"Gamma:            {config.get('gamma', '?')}")
    lines.append("")

    # Training summary from TB
    if "charts/win_rate" in tb_data:
        steps, wrs = tb_data["charts/win_rate"]
        if len(wrs) > 0:
            lines.append("--- Training Summary ---")
            lines.append(f"Final train WR:   {wrs[-1]:.3f}")
            best_idx = np.argmax(wrs)
            lines.append(
                f"Best train WR:    {wrs[best_idx]:.3f} (at step {format_steps(steps[best_idx])})"
            )
            lines.append(f"Total steps run:  {format_steps(steps[-1])}")
            lines.append("")

    if "charts/sps" in tb_data:
        _, sps = tb_data["charts/sps"]
        if len(sps) > 10:
            lines.append(f"Avg SPS:          {np.mean(sps[10:]):.0f} steps/sec")
            lines.append("")

    # Eval results
    if eval_data:
        lines.append("--- Final Eval Win Rates (last eval round) ---")
        bots = sorted(eval_data.keys())
        has_pos = any("wr_p0" in eval_data[b][-1] for b in bots if eval_data[b])

        if has_pos:
            lines.append(
                f"{'Bot':<20s} {'WR':>6s} {'P0':>6s} {'P1':>6s} {'Delta':>7s} {'Games':>6s}"
            )
        else:
            lines.append(f"{'Bot':<20s} {'WR':>6s} {'Games':>6s}")

        overall_wrs = []
        for bot in bots:
            last = eval_data[bot][-1]
            wr = last["wr"]
            overall_wrs.append(wr)
            if has_pos and "wr_p0" in last:
                delta = last["wr_p0"] - last["wr_p1"]
                sign = "+" if delta >= 0 else ""
                lines.append(
                    f"{bot:<20s} {wr:>6.2f} {last['wr_p0']:>6.2f} {last['wr_p1']:>6.2f} {sign}{delta:>6.2f} {last['games']:>6d}"
                )
            else:
                lines.append(f"{bot:<20s} {wr:>6.2f} {last['games']:>6d}")

        lines.append("\u2500" * 50)
        lines.append(f"{'Overall':<20s} {np.mean(overall_wrs):>6.2f}")
        lines.append("")

        # Convergence
        convergence = compute_convergence(eval_data)
        lines.append("--- Convergence (steps to reach WR threshold) ---")
        lines.append(f"{'Bot':<20s} {'50% WR':>10s} {'70% WR':>10s} {'90% WR':>10s}")
        for bot in bots:
            row = convergence[bot]
            vals = []
            for thr in (0.5, 0.7, 0.9):
                s = row[thr]
                vals.append(format_steps(s) if s is not None else "\u2014")
            lines.append(f"{bot:<20s} {vals[0]:>10s} {vals[1]:>10s} {vals[2]:>10s}")
        lines.append("")

        # Stability
        stability = compute_stability(eval_data)
        lines.append("--- Stability (last 20% of evaluations) ---")
        for bot in bots:
            if bot in stability:
                s = stability[bot]
                lines.append(
                    f"{bot:<20s} mean={s['mean']:.3f}  std={s['std']:.3f}  "
                    f"min={s['min']:.2f}  max={s['max']:.2f}"
                )
        lines.append("")

        # Position bias (aggregated over last 5 evals)
        if has_pos:
            lines.append("--- Position Bias (last 5 evals) ---")
            deltas = []
            for bot in bots:
                entries = [e for e in eval_data[bot] if "wr_p0" in e]
                if entries:
                    last_n = entries[-5:]
                    p0_mean = np.mean([e["wr_p0"] for e in last_n])
                    p1_mean = np.mean([e["wr_p1"] for e in last_n])
                    d = p0_mean - p1_mean
                    deltas.append(d)
                    sign = "+" if d >= 0 else ""
                    lines.append(
                        f"  {bot:<18s} P0={p0_mean:.2f}  P1={p1_mean:.2f}  delta={sign}{d:.2f}"
                    )
            if deltas:
                avg_delta = np.mean(deltas)
                sign = "+" if avg_delta >= 0 else ""
                bias = (
                    "P0 advantage"
                    if avg_delta > 0.02
                    else "P1 advantage"
                    if avg_delta < -0.02
                    else "balanced"
                )
                lines.append(f"Average P0-P1 delta: {sign}{avg_delta:.3f} ({bias})")
            lines.append("")

    summary_path = os.path.join(out_dir, "summary.txt")
    with open(summary_path, "w") as f:
        f.write("\n".join(lines))
    print("  -> summary.txt")


# ---------------------------------------------------------------------------
# Main orchestration
# ---------------------------------------------------------------------------


def analyze_single_run(run_dir, smoothing=0.95, max_points=0):
    """Generate all analysis outputs for a single run."""
    run_dir = str(run_dir)
    exp_name = os.path.basename(run_dir)
    out_dir = os.path.join(run_dir, "analysis")
    os.makedirs(out_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"  Analyzing: {exp_name}")
    print(f"{'=' * 50}")

    # Load data
    config = load_config(run_dir)
    print("  Loading TensorBoard events...")
    tb_data = load_tensorboard(run_dir, max_points=max_points)
    if not tb_data:
        print("  WARNING: No TensorBoard data found, skipping plots.")
    else:
        print(f"  Found {len(tb_data)} scalar tags")

    eval_data = load_eval_csv(run_dir)
    if eval_data:
        total_rows = sum(len(v) for v in eval_data.values())
        print(f"  Found eval data: {len(eval_data)} bots, {total_rows} rows")
    else:
        print("  No eval_results.csv found (eval plots will be skipped)")

    # Generate plots
    alpha = smoothing
    if tb_data:
        plot_training_curves(tb_data, out_dir, alpha)
        plot_loss_curves(tb_data, out_dir, alpha)
        plot_reward_decomposition(tb_data, out_dir, alpha)
        plot_schedules(tb_data, out_dir, alpha)
        plot_selfplay_pfsp(tb_data, out_dir, alpha)
        plot_adaptive_opponents(tb_data, out_dir)
        plot_plr_stats(tb_data, out_dir, alpha)
        plot_ppo_diagnostics(tb_data, out_dir, alpha)

        # WR per opponent: shared color map for consistency across plots
        bot_colors = _build_bot_color_map(tb_data, eval_data)
        plot_per_opponent_training_wr(tb_data, out_dir, alpha, color_map=bot_colors)
        plot_per_opponent_stats(tb_data, out_dir, alpha, color_map=bot_colors)
        plot_prioritized_weights(tb_data, out_dir, alpha)
        if eval_data:
            plot_eval_wr_per_opponent(eval_data, out_dir, color_map=bot_colors)
            plot_combined_wr_per_opponent(tb_data, eval_data, out_dir, alpha, color_map=bot_colors)

        plot_action_distribution(tb_data, out_dir, alpha)

    if eval_data:
        plot_position_bias(eval_data, out_dir)

    # Quantitative analysis
    write_summary(config, tb_data, eval_data, out_dir)

    print(f"\n  Done! Results in: {out_dir}/")


def find_all_runs():
    """Find all valid run directories (those with at least a config.json or TB events)."""
    runs_dir = str(RUNS_DIR)
    if not os.path.exists(runs_dir):
        return []
    runs = []
    for name in sorted(os.listdir(runs_dir)):
        d = os.path.join(runs_dir, name)
        if os.path.isdir(d):
            has_config = os.path.exists(os.path.join(d, "config.json"))
            has_events = any(f.startswith("events.out.tfevents") for f in os.listdir(d))
            if has_config or has_events:
                runs.append(d)
    return runs


def main():
    parser = argparse.ArgumentParser(
        description="Analyze training runs: generate per-run PDF plots and summaries.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("runs", nargs="*", help="Run directory path(s)")
    parser.add_argument("--all", action="store_true", help="Analyze all runs in outputs/runs/")
    parser.add_argument(
        "--smoothing",
        type=float,
        default=0.95,
        help="EMA smoothing alpha (0=none, 0.99=very smooth, default: 0.95)",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=0,
        help="Max data points per TB tag (0=all, 5000=fast). "
        "Lower values load much faster on large runs.",
    )
    args = parser.parse_args()

    if args.all:
        run_dirs = find_all_runs()
        if not run_dirs:
            print("No runs found in outputs/runs/")
            sys.exit(1)
        print(f"Found {len(run_dirs)} runs to analyze")
    elif args.runs:
        run_dirs = []
        for pattern in args.runs:
            matches = glob(pattern)
            run_dirs.extend(d for d in matches if os.path.isdir(d))
        if not run_dirs:
            print(f"No valid run directories found for: {args.runs}")
            sys.exit(1)
    else:
        parser.print_help()
        sys.exit(1)

    setup_style()

    for run_dir in sorted(set(run_dirs)):
        try:
            analyze_single_run(run_dir, smoothing=args.smoothing, max_points=args.max_points)
        except Exception as e:
            print(f"\n  ERROR analyzing {run_dir}: {e}")
            import traceback

            traceback.print_exc()

    print(f"\n{'=' * 50}")
    print(f"  All done! Analyzed {len(run_dirs)} run(s)")
    print(f"{'=' * 50}")


if __name__ == "__main__":
    main()
