"""UECD-Best generalization probes: layout shift vs scale shift.
Dumbbell chart (win-rate), marker size = episode length. Data from results.tex
Table tab:gen-probe. Output: ShortPaperCoG/figures/generalization_probes.pdf"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "sans-serif",
        "axes.linewidth": 1.0,
        "font.size": 13,
    }
)

RED = "#c0392b"  # layout shift  : TwoBasesBarracks16x16
BLUE = "#2171b5"  # scale  shift  : basesWorkers32x32A

# name, layout_WR, layout_len, scale_WR, scale_len  (sorted by scale WR desc)
DATA = [
    ("RandomBiasedAI", 100, 517, 100, 1003),
    ("WorkerRush", 94, 491, 100, 1117),
    ("CoacAI", 85, 542, 96, 1586),
    ("Mayari", 65, 640, 95, 1455),
    ("RAISocketAI", 22, 468, 95, 1502),
    ("LightRush", 0, 500, 60, 1997),
    ("ObiBotKenobi", 4, 437, 46, 2090),
    ("TMA", 4, 548, 11, 2160),
]


def msize(length):
    return length * 0.22  # marker area (pt^2)


fig, ax = plt.subplots(figsize=(8.2, 5.1))
n = len(DATA)
y = list(range(n))[::-1]  # first row on top

for yi, (_name, lwr, llen, swr, slen) in zip(y, DATA):
    ax.plot([lwr, swr], [yi, yi], color="#999999", lw=2.0, zorder=1)
    ax.scatter(lwr, yi, s=msize(llen), color=RED, edgecolor="black", linewidth=0.8, zorder=3)
    ax.scatter(swr, yi, s=msize(slen), color=BLUE, edgecolor="black", linewidth=0.8, zorder=3)

ax.set_yticks(y)
ax.set_yticklabels([d[0] for d in DATA], fontsize=13, fontweight="bold")
ax.set_ylim(-0.6, n - 0.4)
ax.set_xlim(-4, 104)
ax.set_xticks([0, 20, 40, 60, 80, 100])
ax.tick_params(axis="x", labelsize=12)
ax.set_xlabel("Win rate (%)", fontsize=15, fontweight="bold")
ax.set_axisbelow(True)
ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb")
for s in ("top", "right"):
    ax.spines[s].set_visible(False)

# ── legend 1: distribution shift (colour) ────────────────────────────
shift_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=RED,
        markeredgecolor="black",
        markersize=10,
        label="Layout: $\\it{TwoBasesBarracks}$16x16",
    ),
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor=BLUE,
        markeredgecolor="black",
        markersize=10,
        label="Scale: $\\it{basesWorkers}$32x32A",
    ),
]
leg1 = ax.legend(
    handles=shift_handles,
    title="Distribution shift",
    loc="upper left",
    bbox_to_anchor=(0.16, 1.0),
    fontsize=12,
    title_fontsize=12,
    frameon=True,
    framealpha=0.95,
    handletextpad=0.4,
    borderpad=0.6,
)
leg1.get_title().set_fontweight("bold")
ax.add_artist(leg1)

# ── legend 2: episode length (marker size) ───────────────────────────
size_handles = [
    Line2D(
        [0],
        [0],
        marker="o",
        color="none",
        markerfacecolor="#777777",
        markeredgecolor="black",
        markersize=(msize(L) ** 0.5),
        label=f"{L}f",
    )
    for L in (500, 1200, 2000)
]
leg2 = ax.legend(
    handles=size_handles,
    title="Episode length",
    loc="lower right",
    ncol=3,
    fontsize=11,
    title_fontsize=12,
    frameon=True,
    framealpha=0.95,
    handletextpad=0.3,
    columnspacing=1.0,
    borderpad=0.6,
)
leg2.get_title().set_fontweight("bold")

fig.tight_layout()
out = os.path.join(
    os.path.dirname(__file__), "..", "..", "ShortPaperCoG", "figures", "generalization_probes.pdf"
)
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.abspath(out))
