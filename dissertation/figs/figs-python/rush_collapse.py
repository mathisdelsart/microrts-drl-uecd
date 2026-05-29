"""Phased 300M compact: WR + episode length (twin axes) stacked above reward components.

Two stacked panels (weight-schedule panel removed — red zone already marks the anneal):
  1. Overall win-rate (left axis, green) + avg episode length (right axis, purple),
     with the reward-transition zone shaded in red.
  2. Unweighted episodic return per reward component (symlog), same x-axis.

Output: figures/rush_collapse.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
from _style import (
    FIGURES_DIR,
    ROOT,
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
TB_CACHE_DIR = ROOT / "outputs" / "comparisons"

df_eval = load_eval(RUN)
wr = overall_wr(df_eval)
avg_len = df_eval.groupby("global_step")["avg_length"].mean().reset_index()

train_log = parse_train_log(RUN, sample_every=300)
wr_cols = [c for c in train_log.columns if c.startswith("wr_") and c != "wr_Self"]
if wr_cols:
    train_log["overall_wr"] = train_log[wr_cols].mean(axis=1)

# Reward transition zone: linear anneal between 30% and 70% of total
total = 300e6
rs_start, rs_end = 0.3 * total, 0.7 * total

phase_M = [100, 150, 250, 280]

fig, axes = plt.subplots(
    2,
    1,
    figsize=(11, 7.0),
    sharex=True,
    gridspec_kw={"height_ratios": [1.0, 1.0]},
    constrained_layout=True,
)

# ── Panel 1: WR (left) + Episode length (right) on twin axes ──
ax_wr = axes[0]
ax_len = ax_wr.twinx()

# Win-rate (left axis, green) — bold training line, eval as markers only
wr_train_line = None
if "overall_wr" in train_log.columns:
    (wr_train_line,) = ax_wr.plot(
        M(train_log["step"]),
        smooth(train_log["overall_wr"], 30),
        color=C["wr"],
        alpha=0.65,
        linewidth=2.0,
        label="Training WR",
        zorder=4,
    )
(wr_line,) = ax_wr.plot(
    M(wr["global_step"]),
    wr["win_rate"],
    "o",
    color=C["wr"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    label="Evaluation WR",
    zorder=6,
)
ax_wr.set_ylabel("Overall Win-Rate (%)", color="#1a6b1a", fontsize=14, fontweight="bold")
ax_wr.tick_params(axis="y", labelcolor=C["wr"], labelsize=13)
ax_wr.tick_params(axis="x", labelsize=13)
ax_wr.spines["left"].set_color(C["wr"])
ax_wr.set_ylim(0, 1.08)
ax_wr.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))

# Episode length (right axis, purple) — explicitly re-enable the right spine
# because apply_style() hides it globally.
len_train_line = None
if "len" in train_log.columns:
    (len_train_line,) = ax_len.plot(
        M(train_log["step"]),
        smooth(train_log["len"], 30),
        color=C["len"],
        alpha=0.65,
        linewidth=2.0,
        label="Training Length",
        zorder=4,
    )
(len_line,) = ax_len.plot(
    M(avg_len["global_step"]),
    avg_len["avg_length"],
    "D",
    color=C["len"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    label="Evaluation Length",
    zorder=6,
)
ax_len.set_ylabel(
    "Average Episode Length (frames)", color="#5d3c87", fontsize=12, fontweight="bold"
)
ax_len.tick_params(axis="y", labelcolor=C["len"], labelsize=11)
ax_len.spines["right"].set_visible(True)
ax_len.spines["right"].set_color(C["len"])
ax_len.set_ylim(0, 1800)
ax_len.yaxis.set_major_locator(mticker.FixedLocator([0, 300, 600, 900, 1200, 1500, 1800]))

# Phase markers + reward transition zone (drawn on the WR axis)
for ps in phase_M:
    ax_wr.axvline(ps, color="#888", ls="--", alpha=0.4, lw=0.8)
ax_wr.axvspan(
    rs_start / 1e6, rs_end / 1e6, alpha=0.10, color=C["sparse"], label="Linear Reward Anneal"
)
ax_wr.axvspan(200, 210, alpha=0.30, color="#b22222", zorder=2, label="Reward Crash")
ax_wr.set_xlim(0, 300)

# Combined legend grouped by colour: col 1 = WR (Eval / Train),
# col 2 = Length (Eval / Train), col 3 = anneal zone + crash.
# matplotlib fills column-major, so list entries column by column.
anneal_patch = plt.Rectangle((0, 0), 1, 1, color=C["sparse"], alpha=0.18)
crash_patch = plt.Rectangle((0, 0), 1, 1, color="#b22222", alpha=0.45)
handles = [wr_line, wr_train_line, len_line, len_train_line, anneal_patch, crash_patch]
labels = [
    "Evaluation WR",
    "Training WR",
    "Evaluation Length",
    "Training Length",
    "Linear Reward Anneal",
    "Reward Crash",
]
ax_wr.legend(
    handles,
    labels,
    loc="lower left",
    fontsize=11,
    ncol=3,
    columnspacing=1.0,
    handletextpad=0.5,
    prop={"weight": "bold", "size": 11},
)

# ── Panel 2: Reward component returns (symlog) ──
ax_rc = axes[1]

SINGLE_COMPS = {
    "WinDrawLoss": "charts__episodic_return__WinDrawLossRewardFunction",
    "ProduceWorker": "charts__episodic_return__ProduceWorkerRewardFunction",
    "ProduceBarracks": "charts__episodic_return__ProduceBarracksRewardFunction",
}
MILITARY_TAGS = [
    "charts__episodic_return__ProduceLightUnitRewardFunction",
    "charts__episodic_return__ProduceHeavyUnitRewardFunction",
    "charts__episodic_return__ProduceRangedUnitRewardFunction",
]
COMP_COLORS = {
    "WinDrawLoss": "#d62728",
    "ProduceWorker": "#1f77b4",
    "ProduceBarracks": "#17becf",
    "Military Production": "#ff7f0e",
}

for name, safe_tag in SINGLE_COMPS.items():
    path = TB_CACHE_DIR / f"tb_cache_{safe_tag}.csv"
    if not path.exists():
        continue
    df = pd.read_csv(path)
    ax_rc.plot(
        M(df["step"]),
        smooth(df["value"], 40),
        color=COMP_COLORS[name],
        linewidth=1.5,
        alpha=0.9,
        label=name,
    )

# Military production = sum of Light + Heavy + Ranged
mil_dfs = [
    pd.read_csv(TB_CACHE_DIR / f"tb_cache_{t}.csv")
    for t in MILITARY_TAGS
    if (TB_CACHE_DIR / f"tb_cache_{t}.csv").exists()
]
if mil_dfs:
    n = min(len(d) for d in mil_dfs)
    mil_steps = mil_dfs[0]["step"].values[:n]
    mil_sum = sum(d["value"].values[:n] for d in mil_dfs)
    ax_rc.plot(
        M(mil_steps),
        smooth(mil_sum, 40),
        color=COMP_COLORS["Military Production"],
        linewidth=1.5,
        alpha=0.9,
        label="Military Production",
    )

for ps in phase_M:
    ax_rc.axvline(ps, color="#888", ls="--", alpha=0.4, lw=0.8)
ax_rc.axvspan(rs_start / 1e6, rs_end / 1e6, alpha=0.10, color=C["sparse"])
ax_rc.axvspan(200, 210, alpha=0.30, color="#b22222", zorder=2)

ax_rc.set_xlabel("Training Steps (M)", fontsize=14, fontweight="bold")
ax_rc.set_ylabel("Unweighted Episodic Return", fontsize=13, fontweight="bold")
ax_rc.set_yscale("symlog", linthresh=1)
ax_rc.set_ylim(bottom=-1.3)
ax_rc.yaxis.set_major_locator(mticker.FixedLocator([-1, 0, 1, 10, 100]))
ax_rc.yaxis.set_major_formatter(mticker.FixedFormatter(["-1", "0", "1", "10", "100"]))
ax_rc.tick_params(axis="both", labelsize=15)
ax_rc.legend(
    loc="lower left",
    ncol=4,
    fontsize=13,
    columnspacing=1.0,
    handletextpad=0.5,
    prop={"weight": "bold", "size": 13},
)

out = FIGURES_DIR / "rush_collapse.pdf"
fig.savefig(out, bbox_inches="tight", pad_inches=0.15)
print(f"Saved: {out}")
