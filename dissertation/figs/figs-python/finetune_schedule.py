"""Two-phase fine-tuning schedule (150M -> 350M) to the final UECD-Best agent.
Triple-axis, proportional line plot. Output: figs-pdf/finetune_schedule.pdf."""

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
SHC = (216 / 255, 124 / 255, 38 / 255)  # shaped weight : orange

bx = [150, 240, 350]  # breakpoints (millions of steps)
lr_y = [5e-5, 2e-5, 9.5e-6]
ent_y = [0.005, 0.002, 0.0012]
shp_y = [1.0, 0.25, 0.10]

fig, ax = plt.subplots(figsize=(11.5, 4.8))
ax2 = ax.twinx()  # entropy
ax3 = ax.twinx()  # shaped weight
ax3.spines["right"].set_position(("outward", 78))
trx = ax.get_xaxis_transform()

# ── phase boundary + light edge guides ───────────────────────────────
for gx in (150, 240, 350):
    ax.axvline(gx, color="0.82", lw=0.8, zorder=0)
ax.axvline(240, color="0.55", ls=(0, (4, 3)), lw=1.0, zorder=1)

# ── curves ───────────────────────────────────────────────────────────
lk = {"lw": 3.0, "solid_joinstyle": "round", "solid_capstyle": "round", "zorder": 3}
mk = {"marker": "o", "ms": 8, "mfc": "white", "mew": 2.4, "ls": "none", "zorder": 4}
ax.plot(bx, lr_y, color=LRC, **lk)
ax.plot(bx, lr_y, mec=LRC, **mk)
ax2.plot(bx, ent_y, color=ENTC, **lk)
ax2.plot(bx, ent_y, mec=ENTC, **mk)
ax3.plot(bx, shp_y, color=SHC, **lk)
ax3.plot(bx, shp_y, mec=SHC, **mk)

# ── left axis : learning rate (blue) ─────────────────────────────────
ax.set_ylim(0, 6.5e-5)
ax.set_yticks(lr_y)
ax.set_yticklabels(["5×10⁻⁵", "2×10⁻⁵", "9.5×10⁻⁶"], fontsize=16, color=LRC)
ax.set_ylabel("learning rate", color=LRC, fontsize=18)
ax.tick_params(axis="y", colors=LRC)
ax.spines["left"].set_color(LRC)
ax.spines["right"].set_visible(False)
ax.spines["top"].set_visible(False)

# ── right axis : entropy coefficient (purple) ────────────────────────
ax2.set_ylim(0, 0.0054)
ax2.set_yticks(ent_y)
ax2.set_yticklabels(["0.005", "0.002", "0.0012"], fontsize=16, color=ENTC)
ax2.set_ylabel("entropy coefficient", color=ENTC, rotation=90, labelpad=12, fontsize=18)
ax2.tick_params(axis="y", colors=ENTC)
ax2.spines["right"].set_color(ENTC)
ax2.spines["top"].set_visible(False)

# ── outer-right axis : shaped weight (orange) ────────────────────────
ax3.set_ylim(0, 1.45)
ax3.set_yticks(shp_y)
ax3.set_yticklabels(["1.0", "0.25", "0.10"], fontsize=16, color=SHC)
ax3.set_ylabel("shaped weight", color=SHC, rotation=90, labelpad=14, fontsize=18)
ax3.tick_params(axis="y", colors=SHC)
ax3.spines["right"].set_color(SHC)
ax3.spines["top"].set_visible(False)

# ── x axis ───────────────────────────────────────────────────────────
ax.set_xlim(150, 350)
ax.set_xticks([150, 240, 350])
ax.tick_params(axis="x", labelsize=16)
ax.set_xlabel("training steps (millions)", fontsize=18)

# ── phase labels (bold) ──────────────────────────────────────────────
for cx, lab in [(195, "Phase 1"), (295, "Phase 2")]:
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
out = os.path.join(os.path.dirname(__file__), "..", "figs-pdf", "finetune_schedule.pdf")
fig.savefig(out, bbox_inches="tight")
print("wrote", os.path.abspath(out))
