"""Rush fragility: 150M (balanced) vs 300M (rush) checkpoints.
Slopegraph of win rate and episode length; held-out opponents collapse,
in-training opponents stay flat. Output: figs-pdf/rush_fragility.pdf."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.0,
        "font.size": 14,
    }
)

HELD = "#c0392b"  # held-out   : red (collapse)
INTR = "#1e8449"  # in-training: green (stable)
MS, MEW = 4.5, 1.3

# name, group-colour, marker, wr150(avg P0/P1), wr300, len150, len300
DATA = [
    ("TMA", HELD, "o", 56.5, 0.0, 1375, 946),
    ("ObiBotKenobi", HELD, "s", 70.4, 1.5, 1248, 765),
    ("CoacAI", INTR, "o", 98.0, 98.0, 987, 306),
    ("Mayari", INTR, "s", 99.8, 100.0, 864, 295),
]

fig, (axw, axl) = plt.subplots(1, 2, figsize=(11, 4.3))


def declutter(items, gap, lo, hi):
    """items: (y, text, colour, marker_y). Spread only labels that would
    overlap, keeping each cluster centred on its own markers (so close pairs
    stay aligned with their points). Clamp each cluster inside [lo, hi]."""
    items = sorted(items, key=lambda t: t[0])
    ys = [it[0] for it in items]
    n = len(ys)
    for i in range(1, n):
        if ys[i] - ys[i - 1] < gap:
            ys[i] = ys[i - 1] + gap
    i = 0
    while i < n:
        j = i
        while j + 1 < n and ys[j + 1] - ys[j] < gap + 1e-6:
            j += 1
        cur = (ys[i] + ys[j]) / 2
        mk = sum(items[k][3] for k in range(i, j + 1)) / (j - i + 1)
        for k in range(i, j + 1):
            ys[k] += mk - cur
        if ys[i] < lo:
            d = lo - ys[i]
            for k in range(i, j + 1):
                ys[k] += d
        if ys[j] > hi:
            d = hi - ys[j]
            for k in range(i, j + 1):
                ys[k] += d
        i = j + 1
    return [(ys[k], items[k][1], items[k][2], items[k][3]) for k in range(n)]


def draw(ax, i0, i1, ymax, fmt, gap, lo, hi):
    left, right = [], []
    for _name, col, mk, *vals in DATA:
        y0, y1 = vals[i0], vals[i1]
        ax.plot([0, 1], [y0, y1], color=col, lw=2.4, alpha=0.9, zorder=2)
        ax.plot(
            [0, 1], [y0, y1], marker=mk, color=col, ms=MS, mfc="white", mew=MEW, ls="none", zorder=3
        )
        left.append((y0, fmt(y0), col, y0))
        right.append((y1, fmt(y1), col, y1))
    for ly, txt, col, _my in declutter(left, gap, lo, hi):
        ax.text(-0.07, ly, txt, ha="right", va="center", color=col, fontsize=10)
    for ly, txt, col, _my in declutter(right, gap, lo, hi):
        ax.text(1.07, ly, txt, ha="left", va="center", color=col, fontsize=10)
    ax.set_xlim(-0.5, 1.5)
    ax.set_ylim(0, ymax)
    ax.set_xticks([0, 1])
    ax.set_xticklabels(["Balanced Policy", "Rush Policy"], fontsize=13)
    ax.grid(axis="y", color="0.9", lw=0.7, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.tick_params(axis="y", labelsize=12)


# vals indices: 0=wr150, 1=wr300, 2=len150, 3=len300
draw(axw, 0, 1, 110, lambda v: f"{v:.0f}%", gap=7.0, lo=3, hi=107)
axw.set_ylabel("Win rate (%)", fontsize=12.5)
axw.set_yticks([0, 25, 50, 75, 100])

draw(axl, 2, 3, 1550, lambda v: f"{v:,.0f}", gap=62, lo=35, hi=1515)
axl.set_ylabel("Avg. episode length (frames)", fontsize=12.5)
axl.set_yticks([0, 300, 600, 900, 1200, 1500])

# ── inline, boxed legend with group headers ──────────────────────────
none = Line2D([], [], color="none")
h = {
    n: Line2D([], [], color=c, marker=m, ms=MS, mfc="white", mew=MEW, lw=2.4)
    for n, c, m, *_ in DATA
}
handles = [none, h["TMA"], h["ObiBotKenobi"], none, h["CoacAI"], h["Mayari"]]
labels = ["Held-out:", "TMA", "ObiBotKenobi", "In-training:", "CoacAI", "Mayari"]
leg = fig.legend(
    handles,
    labels,
    loc="lower center",
    ncol=6,
    frameon=True,
    fancybox=True,
    fontsize=12.5,
    handletextpad=0.4,
    columnspacing=1.3,
    borderpad=0.7,
    bbox_to_anchor=(0.5, -0.01),
)
leg.get_frame().set_edgecolor("0.65")
leg.get_frame().set_linewidth(0.9)
leg.get_frame().set_facecolor("white")
txts = leg.get_texts()
txts[0].set_color(HELD)
txts[0].set_fontweight("bold")
txts[3].set_color(INTR)
txts[3].set_fontweight("bold")

fig.tight_layout(rect=[0, 0.08, 1, 1])
out = os.path.join(os.path.dirname(__file__), "..", "figs-pdf", "rush_fragility.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.abspath(out))
