"""Phased 300M: reward scheduling overview + in-training WR + episode length.

Three stacked panels:
  1. Reward weight schedule (shaped vs win/loss) over the 3 phases
  2. Overall win-rate (eval + faint training), with phase markers
  3. Average episode length, with annotation at rush-strategy emergence

Output: figures/phased_300M_reward_annealing.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
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

RUN = "phased_single_map_300M_s1"

df_eval = load_eval(RUN)
wr = overall_wr(df_eval)
avg_len = df_eval.groupby("global_step")["avg_length"].mean().reset_index()

train_log = parse_train_log(RUN, sample_every=300)
wr_cols = [c for c in train_log.columns if c.startswith("wr_") and c != "wr_Self"]
if wr_cols:
    train_log["overall_wr"] = train_log[wr_cols].mean(axis=1)

# ── Reward schedule (linear between 30% and 70% of total) ──
total = 300e6
rs_start, rs_end = 0.3 * total, 0.7 * total  # 90M–210M
x_s = np.linspace(0, 300, 500)
shaped_w = np.clip(1.0 - (x_s * 1e6 - rs_start) / (rs_end - rs_start), 0, 1)
winloss_w = 1.0 - shaped_w

phase_M = [100, 150, 250, 280]

fig, axes = plt.subplots(
    3, 1, figsize=(10, 8.5), sharex=True, gridspec_kw={"height_ratios": [0.8, 1, 1]}
)
fig.subplots_adjust(hspace=0.15)

# ── Panel 1: Reward schedule ──
ax = axes[0]
ax.fill_between(x_s, shaped_w, alpha=0.25, color=C["shaped"])
ax.plot(x_s, shaped_w, color=C["shaped"], linewidth=1.8, label="Shaped reward weight")
ax.fill_between(x_s, winloss_w, alpha=0.20, color=C["sparse"])
ax.plot(x_s, winloss_w, color=C["sparse"], linewidth=1.8, label="Win/Loss reward weight")
ax.set_ylabel("Weight")
ax.set_ylim(0, 1.15)
for ps in phase_M:
    ax.axvline(ps, color="#888", ls="--", alpha=0.4, lw=0.8)
ax.text(45, 1.05, "Phase 1", ha="center", fontsize=9, fontweight="bold", color="#555")
ax.text(150, 1.05, "Phase 2", ha="center", fontsize=9, fontweight="bold", color="#555")
ax.text(255, 1.05, "Phase 3", ha="center", fontsize=9, fontweight="bold", color="#555")
ax.legend(loc="lower left", fontsize=8)

# ── Panel 2: Overall win-rate (eval + training) ──
ax = axes[1]
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
for ps in phase_M:
    ax.axvline(ps, color="#888", ls="--", alpha=0.4, lw=0.8)
ax.axvspan(
    rs_start / 1e6, rs_end / 1e6, alpha=0.06, color=C["sparse"], label="Reward transition zone"
)
ax.legend(loc="lower right", fontsize=8)

# ── Panel 3: Average episode length ──
ax = axes[2]
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
ax.set_xlabel("Training steps (M)")
ax.set_ylabel("Average episode length (frames)")
ax.set_ylim(0, None)
ax.set_xlim(0, 300)
for ps in phase_M:
    ax.axvline(ps, color="#888", ls="--", alpha=0.4, lw=0.8)
ax.axvspan(rs_start / 1e6, rs_end / 1e6, alpha=0.06, color=C["sparse"])

# Annotate rush emergence (first eval point after 200M)
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

fig.tight_layout()
out = FIGURES_DIR / "phased_300M_reward_annealing.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
