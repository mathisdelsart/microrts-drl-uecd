"""BC pre-training vs from-scratch: overall win-rate convergence.

Data source: 10-game in-training evaluations against 5 opponents (10M intervals).
  - BC+VF → PPO: UECD-BC-PPO/train.log (data/agents/)
  - From scratch: unet_entity_cbam_deep_s1/train.log (data/ablation/arch/agent/)

Output: figures/bc_vs_scratch_overall.pdf
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
from _data import FIGURES_DIR, parse_train_log, smooth
from _style import C, apply_style

apply_style()
plt.rcParams.update({"text.usetex": True, "text.latex.preamble": r"\usepackage{lmodern}"})

# ── Eval data (hardcoded from train.log OVERALL lines, every 10M) ──
eval_steps = np.array([10, 20, 30, 40, 50, 60, 70, 80, 90, 100])
bc_v3_ppo = np.array([54, 60, 88, 88, 84, 86, 92, 96, 88, 96]) / 100.0
from_scratch = np.array([0, 6, 28, 68, 66, 76, 84, 92, 88, 94]) / 100.0
bc_only_wr = 0.78  # BC+VF only, no RL


def _mean_in_training_wr(run_name, sample_every=200):
    """Collapse per-episode per-bot WR from train.log to a single overall mean WR per sample."""
    log = parse_train_log(run_name, sample_every=sample_every)
    if log.empty:
        return np.array([]), np.array([])
    wr_cols = [c for c in log.columns if c.startswith("wr_")]
    steps_m = log["step"].to_numpy() / 1e6
    mean_wr = log[wr_cols].mean(axis=1).to_numpy()
    return steps_m, mean_wr


bc_tr_s, bc_tr_wr = _mean_in_training_wr("UECD-BC-PPO")
fs_tr_s, fs_tr_wr = _mean_in_training_wr("unet_entity_cbam_deep_s1")


# ── Plot ──
fig, ax = plt.subplots(figsize=(9, 4.6))

# Training WR (bold lines — primary signal)
bc_train = fs_train = None
if len(bc_tr_s):
    (bc_train,) = ax.plot(
        bc_tr_s, smooth(bc_tr_wr, 30), color=C["bc"], linewidth=2.0, alpha=0.65, zorder=3
    )
if len(fs_tr_s):
    (fs_train,) = ax.plot(
        fs_tr_s, smooth(fs_tr_wr, 30), color=C["scratch"], linewidth=2.0, alpha=0.65, zorder=3
    )

# Eval WR (markers only — discrete sampled measurements)
(bc_eval,) = ax.plot(
    eval_steps,
    bc_v3_ppo,
    "o",
    color=C["bc"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=5,
)
(fs_eval,) = ax.plot(
    eval_steps,
    from_scratch,
    "s",
    color=C["scratch"],
    markersize=8,
    markeredgecolor="white",
    markeredgewidth=0.8,
    zorder=5,
)

bc_only = ax.axhline(
    bc_only_wr, color=C["bc_only"], linestyle="--", linewidth=1.5, alpha=0.8, zorder=2
)

ax.set_xlim(0, 100)
ax.set_ylim(-0.03, 1.05)
ax.set_xlabel(r"\textbf{Training steps (M)}", fontsize=18)
ax.set_ylabel(r"\textbf{Overall win rate (\%)}", fontsize=18)
ax.yaxis.set_major_formatter(mticker.PercentFormatter(1.0, symbol=""))
ax.tick_params(axis="both", labelsize=15)

# 2-column legend (column-major): col 1 = BC+VF, col 2 = from scratch.
handles = [bc_train, bc_eval, bc_only, fs_train, fs_eval]
labels = [
    r"\textbf{BC+VF $\rightarrow$ PPO -- Training WR}",
    r"\textbf{BC+VF $\rightarrow$ PPO -- Evaluation WR}",
    r"\textbf{BC+VF only (no RL)}",
    r"\textbf{From scratch -- Training WR}",
    r"\textbf{From scratch -- Evaluation WR}",
]
ax.legend(
    handles, labels, loc="lower right", fontsize=9, ncol=2, columnspacing=1.0, handletextpad=0.5
)

fig.tight_layout()
out = FIGURES_DIR / "bc_vs_scratch_overall.pdf"
fig.savefig(out, bbox_inches="tight")
print(f"Saved: {out}")
