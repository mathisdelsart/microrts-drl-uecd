"""Best-config vs Top-features (100M): overall win-rate, eval + training.

Compares two 100M single-map runs:
  - UECD-SingleMap-AllFeats  (all features; on-disk run name: AllFeatsRL-100M)
  - UECD-SingleMap-TopFeats  (top features only; on-disk run name: TopFeatsRL-100M)

Output: figures/best_vs_stier_100M.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from _data import FIGURES_DIR, M, load_eval, overall_wr, parse_train_log, smooth
from _style import C, apply_style

apply_style()
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{lmodern}"})

best_eval = overall_wr(load_eval("UECD-SingleMap-AllFeats"))
stier_eval = overall_wr(load_eval("UECD-SingleMap-TopFeats"))

best_train = parse_train_log("UECD-SingleMap-AllFeats", sample_every=200)
stier_train = parse_train_log("UECD-SingleMap-TopFeats", sample_every=200)

wr_cols = [c for c in best_train.columns if c.startswith("wr_")]
best_train["overall_wr"] = best_train[wr_cols].mean(axis=1)
stier_train["overall_wr"] = stier_train[
    [c for c in stier_train.columns if c.startswith("wr_")]
].mean(axis=1)

fig, ax = plt.subplots(figsize=(9, 3.5))

# Smoothed training WR on common interpolation grid (for fill_between)
best_train_x = np.asarray(M(best_train["step"]))
best_train_y = np.asarray(smooth(best_train["overall_wr"], 30))
stier_train_x = np.asarray(M(stier_train["step"]))
stier_train_y = np.asarray(smooth(stier_train["overall_wr"], 30))

x_lo = max(best_train_x.min(), stier_train_x.min())
x_hi = min(best_train_x.max(), stier_train_x.max())
x_grid = np.linspace(x_lo, x_hi, 1000)
best_y = np.interp(x_grid, best_train_x, best_train_y)
stier_y = np.interp(x_grid, stier_train_x, stier_train_y)

# Advantage shading (signed by leader)
top_ahead = ax.fill_between(
    x_grid,
    best_y,
    stier_y,
    where=(stier_y >= best_y),
    color=C["stier"],
    alpha=0.18,
    interpolate=True,
    linewidth=0,
    zorder=1,
)
all_ahead = ax.fill_between(
    x_grid,
    best_y,
    stier_y,
    where=(stier_y < best_y),
    color=C["best"],
    alpha=0.18,
    interpolate=True,
    linewidth=0,
    zorder=1,
)

# Training WR (bold lines — primary signal)
(all_train,) = ax.plot(
    best_train_x, best_train_y, color=C["best"], linewidth=2.0, alpha=0.65, zorder=3
)
(top_train,) = ax.plot(
    stier_train_x, stier_train_y, color=C["stier"], linewidth=2.0, alpha=0.65, zorder=3
)

# Eval WR (markers only — discrete sampled measurements)
(all_eval,) = ax.plot(
    M(best_eval["global_step"]),
    best_eval["win_rate"],
    "o",
    color=C["best"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=5,
)
(top_eval,) = ax.plot(
    M(stier_eval["global_step"]),
    stier_eval["win_rate"],
    "D",
    color=C["stier"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=5,
)

ax.set_xlim(0, 105)
ax.set_ylim(0, 1.05)
ax.set_xlabel(r"\textbf{Training steps (M)}", fontsize=18)
ax.set_ylabel(r"\textbf{Overall win rate (\%)}", fontsize=18)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))
ax.tick_params(axis="both", labelsize=15)

# 2-column legend (column-major fill): each column groups one variant.
#   col 1 = TopFeats (zone + curves), col 2 = AllFeats (zone + curves).
handles = [top_ahead, top_train, top_eval, all_ahead, all_train, all_eval]
labels = [
    r"\textbf{\texttt{UECD-TopFeats} ahead}",
    r"\textbf{\texttt{UECD-TopFeats} -- Training WR}",
    r"\textbf{\texttt{UECD-TopFeats} -- Evaluation WR}",
    r"\textbf{\texttt{UECD-AllFeats} ahead}",
    r"\textbf{\texttt{UECD-AllFeats} -- Training WR}",
    r"\textbf{\texttt{UECD-AllFeats} -- Evaluation WR}",
]
ax.legend(
    handles, labels, loc="lower right", fontsize=13, ncol=2, columnspacing=1.0, handletextpad=0.5
)

fig.tight_layout()
out = FIGURES_DIR / "best_vs_stier_100M.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
