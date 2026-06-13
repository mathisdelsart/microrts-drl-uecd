#!/usr/bin/env python3
"""Generate the full feature-ablation diverging bar chart (Finding 1 slide).

Data: dissertation chapter 10, Table tab:feat-ablation-summary (mean +/- std,
delta vs the UECD baseline 58.0% +/- 7.5). All 21 features are shown with std
error bars; the top 4 "reliable gains" are highlighted in uclBlue, other positive
features in green, the two below-baseline features in red. Output:
defense/figures/feat_ablation_full.png. Re-run after any data change.
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# (label, delta_pp, std)
DATA = [
    ("Extended Obs (73ch)", 26.3, 1.7),
    ("Filtered Masks + Reserved Obs", 22.6, 5.6),
    ("Matchup Competitiveness Weighting (MCW)", 20.4, 0.4),
    ("Opponent Modeling", 19.8, 4.0),
    ("Triple Value Heads", 14.7, 4.0),
    ("PAE (keep 95%)", 13.3, 8.0),
    ("Hierarchical Mask", 13.1, 11.5),
    ("Adaptive Opponents", 11.7, 1.4),
    ("PopArt", 10.6, 8.7),
    ("Aux Unit Count", 9.6, 12.6),
    ("Bots + Self-Play + PFSP", 8.9, 9.1),
    ("GELU", 8.3, 8.2),
    ("Frame Stack (4)", 7.3, 4.5),
    ("Autoregressive + Hier. Mask", 7.2, 7.6),
    ("Autoregressive", 5.8, 7.6),
    ("Aux Spatial", 5.3, 5.2),
    ("SPP Critic", 5.2, 8.3),
    ("Aux Contrastive", 3.7, 6.6),
    ("Build-Time Rewards", 3.2, 11.2),
    ("Augment Symmetry", -0.5, 4.0),
    ("HL-Gauss", -18.5, 17.0),
]
BG, GREEN, RED, BLUE, GREY = "#FAFAFA", "#33A968", "#D0473A", "#1F3A5F", "#5A5A5A"
TOP = 4
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "defense", "figures", "feat_ablation_full.png")

labels = [d[0] for d in DATA]
deltas = [d[1] for d in DATA]
stds = [d[2] for d in DATA]
n = len(DATA)
ys = list(range(n))[::-1]
nfirst = next(i for i, (_, d, _) in enumerate(DATA) if d < 0)
colors = [RED if d < 0 else (BLUE if i < TOP else GREEN) for i, (_, d, _) in enumerate(DATA)]

fig, ax = plt.subplots(figsize=(11.6, 6.7), dpi=200)
fig.patch.set_facecolor(BG)
ax.set_facecolor(BG)
ax.barh(
    ys,
    deltas,
    color=colors,
    height=0.66,
    zorder=3,
    xerr=stds,
    error_kw={
        "ecolor": GREY,
        "elinewidth": 1.0,
        "capsize": 2.5,
        "capthick": 1.0,
        "zorder": 4,
        "alpha": 0.85,
    },
)
ax.set_yticks(ys)
ax.set_yticklabels(labels, fontsize=10.5)
for y, d, s in zip(ys, deltas, stds, strict=True):
    off = s + 0.6
    ax.text(
        d + (off if d >= 0 else -off),
        y,
        f"{d:+.1f}",
        va="center",
        ha="left" if d >= 0 else "right",
        fontsize=9.5,
        color="#333333",
        zorder=5,
    )
ax.axvline(0, color="#888888", lw=1.1, zorder=2)
ax.set_xlabel(r"$\Delta$ win rate vs UECD baseline = 58.0% $\pm$ 7.5  (pp)", fontsize=11)
ax.set_xlim(-40, 36)
ax.tick_params(axis="x", labelsize=9.5)
for sp in ("top", "right", "left"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_color("#bbbbbb")
ax.grid(axis="x", color="#dddddd", lw=0.6, zorder=0)
ax.margins(y=0.01)
ax.axhline(ys[TOP - 1] - 0.5, color=BLUE, lw=0.9, ls=(0, (4, 3)), alpha=0.5, zorder=2)
ax.text(
    -39,
    ys[1],
    "Reliable gains:\ntop 4 (blue),\nlow variance",
    fontsize=12.5,
    color=BLUE,
    ha="left",
    va="center",
    style="italic",
    linespacing=1.3,
    fontweight="bold",
)
yb = ys[nfirst] + 0.5
ax.axhline(yb, color=GREY, lw=1.0, ls=(0, (3, 3)), alpha=0.7, zorder=2)
ax.text(
    35.5,
    yb - 0.28,
    "UECD baseline (58.0% $\\pm$ 7.5)",
    fontsize=9.5,
    color=GREY,
    ha="right",
    va="top",
    style="italic",
)
plt.tight_layout()
plt.savefig(OUT, facecolor=BG, bbox_inches="tight")
print("saved", OUT)
