"""Phased 300M combined diagnostic: WR + episode length + per-component returns.

Three stacked panels sharing the x-axis:
  1. Overall win-rate (eval + faint training)
  2. Average episode length, with rush-emergence annotation
  3. Per-component episodic returns (unweighted, symlog)

All panels show the reward-transition zone (90M-210M) as a red shaded span.

Output: figures/phased_300M_combined.pdf
"""

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from _style import (
    FIGURES_DIR,
    C,
    M,
    apply_style,
    load_eval,
    overall_wr,
    parse_train_log,
    smooth,
)

apply_style()

RUN = "single_map/PhasedRL-300M"

# ── Data ──
df_eval = load_eval(RUN)
wr = overall_wr(df_eval)
avg_len = df_eval.groupby("global_step")["avg_length"].mean().reset_index()

train_log = parse_train_log(RUN, sample_every=300)
wr_cols = [c for c in train_log.columns if c.startswith("wr_") and c != "wr_Self"]
if wr_cols:
    train_log["overall_wr"] = train_log[wr_cols].mean(axis=1)

# Reward transition zone (linear anneal between 30% and 70% of total)
total = 300e6
rs_start, rs_end = 0.3 * total, 0.7 * total  # 90M -> 210M

# Per-component tb caches — look in figures/ first, fall back to outputs/comparisons/
COMP_TAGS = {
    "WinDrawLoss": "charts__episodic_return__WinDrawLossRewardFunction",
    "ResourceGather": "charts__episodic_return__ResourceGatherRewardFunction",
    "ProduceWorker": "charts__episodic_return__ProduceWorkerRewardFunction",
    "ProduceBase": "charts__episodic_return__ProduceBaseRewardFunction",
    "ProduceBarracks": "charts__episodic_return__ProduceBarracksRewardFunction",
    "Attack": "charts__episodic_return__AttackRewardFunction",
    "ProduceLight": "charts__episodic_return__ProduceLightUnitRewardFunction",
    "ProduceHeavy": "charts__episodic_return__ProduceHeavyUnitRewardFunction",
    "ProduceRanged": "charts__episodic_return__ProduceRangedUnitRewardFunction",
}
COMP_COLORS = [
    "#d62728",
    "#2ca02c",
    "#1f77b4",
    "#9467bd",
    "#17becf",
    "#ff7f0e",
    "#8c564b",
    "#bcbd22",
    "#e377c2",
]

CACHE_DIRS = [
    FIGURES_DIR,
    FIGURES_DIR.parent / "outputs" / "comparisons",
]


def find_cache(safe_tag: str) -> Optional[Path]:
    for d in CACHE_DIRS:
        p = d / f"tb_cache_{safe_tag}.csv"
        if p.exists():
            return p
    return None


# ── Figure ──
fig, axes = plt.subplots(
    3, 1, figsize=(10, 9.5), sharex=True, gridspec_kw={"height_ratios": [1, 1, 1]}
)
fig.subplots_adjust(hspace=0.12)

# Panel 1: Overall WR
ax = axes[0]
if "overall_wr" in train_log.columns:
    ax.plot(
        M(train_log["step"]),
        smooth(train_log["overall_wr"], 30),
        color=C["wr"],
        alpha=0.3,
        linewidth=1.0,
    )
    ax.plot([], [], color=C["wr"], alpha=0.4, linewidth=1.5, label="Training WR")
ax.plot(
    M(wr["global_step"]),
    wr["win_rate"],
    "o-",
    color=C["wr"],
    markersize=3.5,
    linewidth=1.5,
    label="Eval WR",
    zorder=5,
)
ax.set_ylabel("Overall win-rate (%)")
ax.set_ylim(0, 1.08)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))
ax.axvspan(
    rs_start / 1e6, rs_end / 1e6, alpha=0.10, color=C["sparse"], label="Reward transition zone"
)
ax.legend(loc="lower right", fontsize=8)

# Panel 2: Average episode length
ax = axes[1]
ax.plot(
    M(avg_len["global_step"]),
    avg_len["avg_length"],
    "o-",
    color=C["len"],
    markersize=3.5,
    linewidth=1.5,
    label="Eval average length",
)
if "len" in train_log.columns:
    ax.plot(
        M(train_log["step"]), smooth(train_log["len"], 30), color=C["len"], alpha=0.3, linewidth=1.0
    )
    ax.plot([], [], color=C["len"], alpha=0.4, linewidth=1.5, label="Training average length")
ax.set_ylabel("Average episode length (frames)")
ax.set_ylim(0, None)
ax.axvspan(rs_start / 1e6, rs_end / 1e6, alpha=0.10, color=C["sparse"])

drop_row = avg_len[avg_len["global_step"] >= 200e6].iloc[0]
ax.annotate(
    "Rush strategy\nemerges",
    xy=(M(drop_row["global_step"]), drop_row["avg_length"]),
    xytext=(M(drop_row["global_step"]) + 25, drop_row["avg_length"] + 250),
    arrowprops={"arrowstyle": "->", "color": "black", "lw": 1.0},
    fontsize=9,
    ha="center",
    bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#ccc", "alpha": 0.8},
)
ax.legend(loc="upper right", fontsize=8)

# Panel 3: Per-component episodic returns (symlog)
ax = axes[2]
has_data = False
for (name, safe_tag), color in zip(COMP_TAGS.items(), COMP_COLORS):
    path = find_cache(safe_tag)
    if path is None:
        continue
    df = pd.read_csv(path)
    ax.plot(
        M(df["step"]), smooth(df["value"], 40), color=color, linewidth=1.3, alpha=0.85, label=name
    )
    has_data = True

if not has_data:
    print("  No TB cache for reward components — components panel empty")

ax.set_xlabel("Training steps (M)")
ax.set_ylabel("Unweighted episodic return")
ax.set_yscale("symlog", linthresh=1)
ax.set_xlim(0, 300)
ax.axvspan(rs_start / 1e6, rs_end / 1e6, alpha=0.10, color=C["sparse"])
if has_data:
    ax.legend(loc="lower left", ncol=3, fontsize=7.5)

fig.tight_layout()
out = FIGURES_DIR / "phased_300M_combined.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
