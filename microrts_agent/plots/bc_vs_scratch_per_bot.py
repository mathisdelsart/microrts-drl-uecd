"""BC pre-training vs from-scratch: per-opponent win-rate convergence.

One panel per opponent (5 bots) showing BC+VF→PPO vs from-scratch PPO curves,
with the BC-only (no RL) baseline as a horizontal dashed reference.

Output: figures/bc_vs_scratch_per_bot.pdf
"""

import re

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd
from _style import FIGURES_DIR, RUNS_DIR, C, apply_style

apply_style()

# ── Eval data (hardcoded from train.log per-bot lines, 10M intervals) ──
eval_steps = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])

SHORT2LONG = {
    "Coac": "CoacAI",
    "Maya": "Mayari",
    "POLi": "POLightRush",
    "POWo": "POWorkerRush",
    "Rand": "RandomBiasedAI",
}

PATTERN = re.compile(r"step=\s*([\d,]+)\s+\d+ eps:.*WR\[(.+?)\]")


def parse_per_bot_wr(run_path, sample_every=200):
    """Return dict {bot: (steps_M, wr)} from train.log."""
    acc = {v: ([], []) for v in SHORT2LONG.values()}
    count = 0
    with open(run_path / "train.log") as f:
        for line in f:
            m = PATTERN.search(line)
            if not m or "warming" in m.group(2):
                continue
            count += 1
            if count % sample_every != 0:
                continue
            step_M = int(m.group(1).replace(",", "")) / 1e6
            for tok in m.group(2).split():
                k, v = tok.split("=")
                bot = SHORT2LONG.get(k)
                if bot is None:
                    continue
                acc[bot][0].append(step_M)
                acc[bot][1].append(float(v.replace("%", "")) / 100.0)
    return {b: (np.array(s), np.array(w)) for b, (s, w) in acc.items()}


def smooth(y, window=30):
    return pd.Series(y).rolling(window, min_periods=1, center=True).mean().values


bc_run = RUNS_DIR / "bc" / "bc_v3_finetune_100M_s1"
fs_run = RUNS_DIR / "arch_ablation" / "arch_ablation_unet_entity_cbam_deep_s1"
bc_tr = parse_per_bot_wr(bc_run)
fs_tr = parse_per_bot_wr(fs_run)

bc = {
    "RandomBiasedAI": [100, 100, 100, 100, 100, 100, 100, 100, 100, 100],
    "POWorkerRush": [40, 40, 70, 70, 70, 60, 90, 100, 70, 100],
    "POLightRush": [60, 80, 100, 90, 100, 100, 100, 100, 100, 100],
    "CoacAI": [50, 60, 90, 90, 70, 70, 70, 80, 90, 80],
    "Mayari": [20, 20, 80, 90, 80, 100, 100, 100, 80, 100],
}
fs = {
    "RandomBiasedAI": [0, 30, 100, 100, 100, 100, 100, 100, 100, 100],
    "POWorkerRush": [0, 0, 30, 70, 40, 70, 60, 100, 80, 100],
    "POLightRush": [0, 0, 10, 90, 80, 100, 100, 100, 100, 100],
    "CoacAI": [0, 0, 0, 30, 30, 40, 90, 80, 60, 70],
    "Mayari": [0, 0, 0, 50, 80, 70, 70, 80, 100, 100],
}
bc_pure = {"RandomBiasedAI": 100, "POWorkerRush": 70, "POLightRush": 90, "CoacAI": 80, "Mayari": 50}

ORDER = ["RandomBiasedAI", "POWorkerRush", "POLightRush", "CoacAI", "Mayari"]


# ── Plot: 2×3 grid, last panel used for legend ──
fig, axes = plt.subplots(2, 3, figsize=(11.5, 6.5), sharex=True, sharey=True)
axes = axes.flatten()

for idx, bot in enumerate(ORDER):
    ax = axes[idx]
    y_bc = np.array(bc[bot]) / 100.0
    y_fs = np.array(fs[bot]) / 100.0
    y_pure = bc_pure[bot] / 100.0

    # Faint in-training WR (background)
    s_tr, w_tr = bc_tr[bot]
    if len(s_tr):
        ax.plot(s_tr, smooth(w_tr, 30), color=C["bc"], alpha=0.32, linewidth=1.0)
    s_tr, w_tr = fs_tr[bot]
    if len(s_tr):
        ax.plot(s_tr, smooth(w_tr, 30), color=C["scratch"], alpha=0.32, linewidth=1.0)

    # Eval curves (bold)
    ax.plot(
        eval_steps,
        y_bc,
        "o-",
        color=C["bc"],
        markersize=4.5,
        linewidth=1.8,
        label="BC+VF $\\rightarrow$ PPO — eval",
    )
    ax.plot(
        eval_steps,
        y_fs,
        "s-",
        color=C["scratch"],
        markersize=4.5,
        linewidth=1.8,
        label="From scratch — eval",
    )
    ax.axhline(
        y_pure, color=C["bc_only"], linestyle="--", linewidth=1.2, alpha=0.8, label="BC+VF only"
    )

    # Proxy entries for training curves
    if idx == 0:
        ax.plot(
            [],
            [],
            color=C["bc"],
            alpha=0.45,
            linewidth=1.5,
            label="BC+VF $\\rightarrow$ PPO — training",
        )
        ax.plot(
            [], [], color=C["scratch"], alpha=0.45, linewidth=1.5, label="From scratch — training"
        )

    ax.set_title(bot[2:] if bot.startswith("PO") else bot, fontweight="bold")
    ax.set_xlim(0, 100)
    ax.set_ylim(-0.03, 1.05)
    ax.set_xticks([0, 25, 50, 75, 100])
    ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))

    if idx >= 3:
        ax.set_xlabel("Training steps (M)")
    if idx % 3 == 0:
        ax.set_ylabel("Win-rate (%)")

# ── Legend panel ──
legend_ax = axes[5]
legend_ax.axis("off")
handles, labels = axes[0].get_legend_handles_labels()
legend_ax.legend(
    handles,
    labels,
    loc="center",
    fontsize=10,
    frameon=True,
    title="Training regime",
    title_fontsize=11,
)

fig.tight_layout()
out = FIGURES_DIR / "bc_vs_scratch_per_bot.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
