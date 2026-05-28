"""Learning-rate and entropy-coefficient schedules for the 300M reward-scheduling run.
Dual-axis, proportional line plot. Output: figs-pdf/phased_schedule.pdf."""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ── style: Computer-Modern-like to match the LaTeX report ────────────
plt.rcParams.update(
    {
        "font.family": "serif",
        "mathtext.fontset": "cm",
        "axes.labelweight": "bold",
        "axes.linewidth": 1.1,
        "font.size": 15,
        "xtick.major.size": 5,
        "ytick.major.size": 5,
    }
)

LRC = (40 / 255, 102 / 255, 173 / 255)  # learning rate : blue
ENTC = (146 / 255, 70 / 255, 168 / 255)  # entropy coef. : purple

# ── schedule definitions (x in millions of steps) ────────────────────
lr_x = [0, 250, 280, 300]
lr_y = [2.5e-4, 2.5e-4, 5e-5, 5e-5]
lr_mk_x, lr_mk_y = [250, 280], [2.5e-4, 5e-5]

ent_x = [0, 100, 150, 250, 280, 300]
ent_y = [0.01, 0.01, 0.005, 0.005, 0.001, 0.001]
ent_mk_x = [100, 150, 250, 280]
ent_mk_y = [0.01, 0.005, 0.005, 0.001]

LR_TOP, ENT_TOP = 2.75e-4, 0.0125  # axis tops (different ranges)

fig, ax = plt.subplots(figsize=(11.5, 4.6))
ax2 = ax.twinx()
trx = ax.get_xaxis_transform()  # x in data coords, y in axes fraction

# ── reference gridlines: every plateau value of BOTH axes sits on one ─
grid_lr = [5e-5, 2.5e-4] + [v * LR_TOP / ENT_TOP for v in (0.001, 0.005, 0.01)]
ax.set_axisbelow(True)
for gy in grid_lr:
    ax.axhline(gy, color="0.88", lw=0.8, zorder=0)

# ── transition bands + phase boundaries ──────────────────────────────
for a, b in [(100, 150), (250, 280)]:
    ax.axvspan(a, b, color="0.55", alpha=0.10, lw=0, zorder=1)
for xb in [100, 150, 250, 280]:
    ax.axvline(xb, color="0.6", alpha=0.55, ls=(0, (3, 3)), lw=0.9, zorder=1.5)

# ── curves ───────────────────────────────────────────────────────────
lk = {"lw": 3.0, "solid_joinstyle": "round", "solid_capstyle": "round", "zorder": 3}
mk = {"marker": "o", "ms": 8, "mfc": "white", "mew": 2.4, "ls": "none", "zorder": 4}
ax.plot(lr_x, lr_y, color=LRC, **lk)
ax.plot(lr_mk_x, lr_mk_y, mec=LRC, **mk)
ax2.plot(ent_x, ent_y, color=ENTC, **lk)
ax2.plot(ent_mk_x, ent_mk_y, mec=ENTC, **mk)

# ── left axis : learning rate (blue) ─────────────────────────────────
ax.set_ylim(0, LR_TOP)
ax.set_yticks([5e-5, 2.5e-4])
ax.set_yticklabels(["5×10⁻⁵", "2.5×10⁻⁴"], fontsize=17, color=LRC)
ax.set_ylabel("learning rate", color=LRC, fontsize=18)
ax.tick_params(axis="y", colors=LRC)
ax.spines["left"].set_color(LRC)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# ── right axis : entropy coefficient (purple) ────────────────────────
ax2.set_ylim(0, ENT_TOP)
ax2.set_yticks([0.001, 0.005, 0.01])
ax2.set_yticklabels(["0.001", "0.005", "0.01"], fontsize=17, color=ENTC)
ax2.set_ylabel("entropy coefficient", color=ENTC, rotation=90, labelpad=14, fontsize=18)
ax2.tick_params(axis="y", colors=ENTC)
ax2.spines["right"].set_color(ENTC)
ax2.spines["left"].set_color(LRC)
ax2.spines["top"].set_visible(False)

# ── x axis ───────────────────────────────────────────────────────────
ax.set_xlim(0, 305)
ax.set_xticks([0, 100, 150, 250, 280, 300])
ax.tick_params(axis="x", labelsize=16)
ax.set_xlabel("training steps (millions)", fontsize=18)

# ── phase labels (bold, no brace) ────────────────────────────────────
for cx, lab in [(50, "phase 1"), (200, "phase 2"), (290, "phase 3")]:
    ax.text(
        cx,
        1.045,
        lab,
        transform=trx,
        ha="center",
        va="bottom",
        fontsize=16,
        fontweight="bold",
        color="0.30",
    )

fig.tight_layout()
out = os.path.join(os.path.dirname(__file__), "..", "figs-pdf", "phased_schedule.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.abspath(out))
