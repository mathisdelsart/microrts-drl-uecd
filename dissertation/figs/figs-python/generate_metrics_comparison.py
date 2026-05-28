"""
Generate the comparative figure for game-theoretic metrics
(Nash, Alpha-Rank, Copeland, Regret) applied to the same win-rate matrix.

Dependencies: numpy, matplotlib, scipy
Usage: python generate_metrics_comparison.py
Output: metrics_comparison.pdf (vector, for LaTeX inclusion)
        metrics_comparison.png (raster, for previews)
"""

import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.ticker import FormatStrFormatter, MaxNLocator
from scipy.optimize import linprog

# ============================================================
#  Toy 4-agent win-rate matrix
# ============================================================
# Designed to produce distinct rankings across the four metrics:
#   - Non-transitive 3-cycle among A, B, C
#   - D beats only C
W = np.array(
    [
        [0.50, 0.65, 0.35, 0.70],  # A
        [0.35, 0.50, 0.70, 0.60],  # B
        [0.65, 0.30, 0.50, 0.40],  # C
        [0.30, 0.40, 0.60, 0.50],  # D
    ]
)
n = 4
agents = ["A", "B", "C", "D"]

# Consistent categorical colors per agent (used across all bar panels)
agent_colors = ["#2563EB", "#F59E0B", "#0D9488", "#DC2626"]  # blue, amber, teal, red

# Off-diagonal heatmap palette: muted diverging (red below 0.5, neutral at 0.5, green above)
HEATMAP_CMAP = LinearSegmentedColormap.from_list(
    "muted_div",
    ["#B91C1C", "#F87171", "#FEF3C7", "#86EFAC", "#15803D"],
    N=256,
)
DIAG_FILL = "#E5E7EB"  # light grey for self-play diagonal
DIAG_LINE = "#6B7280"  # darker grey for the horizontal bar


# ============================================================
#  Metric computations
# ============================================================


def compute_copeland(W):
    """Signed Copeland score: wins - losses, with c_ij in {-1, 0, +1}."""
    n = W.shape[0]
    C = np.zeros(n)
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if W[i, j] > 0.5:
                C[i] += 1
            elif W[i, j] < 0.5:
                C[i] -= 1
    return C


def compute_nash(W):
    """Nash averaging via linear programming on the zero-sum meta-game.

    Returns (weights, scores):
      weights : equilibrium mixed strategy p* over the agent pool
      scores  : expected payoff of each agent against the Nash mix, s_i = (A p*)_i
                (this is the ranking metric reported in the tournament).
    """
    n = W.shape[0]
    A = W - 0.5
    c = np.zeros(n + 1)
    c[-1] = -1
    A_ub = np.column_stack([-A.T, np.ones(n)])
    b_ub = np.zeros(n)
    A_eq = np.zeros((1, n + 1))
    A_eq[0, :n] = 1
    b_eq = np.array([1.0])
    bounds = [(0, None)] * n + [(None, None)]
    res = linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    weights = np.maximum(res.x[:n], 0)
    weights /= weights.sum()
    scores = A @ weights
    # Snap solver/floating-point noise to exactly zero (avoids "-0.00" labels)
    scores = np.where(np.abs(scores) < 1e-9, 0.0, scores)
    return weights, scores


def compute_alpharank(W, m=50, alpha=1.0):
    """Alpha-Rank stationary distribution under Fermi selection."""
    n = W.shape[0]

    def fixation_prob(i, j):
        if i == j:
            return 0.0
        s = 0.0
        for k in range(1, m):
            inner = 0.0
            for ell in range(1, k + 1):
                f_j = ((ell - 1) * W[j, j] + (m - ell) * W[j, i]) / (m - 1)
                f_i = (ell * W[i, j] + (m - ell - 1) * W[i, i]) / (m - 1)
                inner -= alpha * (f_j - f_i)
            s += np.exp(inner)
        return 1.0 / (1.0 + s)

    T = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                T[i, j] = fixation_prob(i, j) / (n - 1)
        T[i, i] = 1.0 - T[i, :].sum()

    eigvals, eigvecs = np.linalg.eig(T.T)
    idx = np.argmin(np.abs(eigvals - 1.0))
    pi = np.real(eigvecs[:, idx])
    if (pi < 0).any():
        pi = -pi
    return pi / pi.sum()


def compute_avg_regret(W):
    """Average regret across opponents: max_k W[k,j] - W[i,j], averaged over j != i."""
    n = W.shape[0]
    R = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j:
                R[i, j] = W[:, j].max() - W[i, j]
    return R.sum(axis=1) / (n - 1)


# ============================================================
#  Compute all metrics
# ============================================================

copeland = compute_copeland(W)
nash_weights, nash_score = compute_nash(W)
alpharank = compute_alpharank(W)
regret = compute_avg_regret(W)

print("Win-rate matrix W:")
print(W)
print()
print(f"  Nash p*:     {np.round(nash_weights, 3)}")
print(f"  Nash score:  {np.round(nash_score, 3)}")
print(f"  Alpha-Rank:  {np.round(alpharank, 3)}")
print(f"  Copeland:    {np.round(copeland, 3)}")
print(f"  Avg regret:  {np.round(regret, 3)}")


# ============================================================
#  Plotting
# ============================================================

# Paper-style global rcParams
plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Computer Modern Roman", "DejaVu Serif"],
        "mathtext.fontset": "cm",
        "font.size": 10,
        "axes.titlesize": 10.5,
        "axes.labelsize": 10,
        "xtick.labelsize": 9.5,
        "ytick.labelsize": 9,
        "axes.linewidth": 0.7,
        "axes.edgecolor": "#374151",
        "xtick.color": "#374151",
        "ytick.color": "#374151",
        "axes.labelcolor": "#111827",
        "axes.titlecolor": "#111827",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

fig = plt.figure(figsize=(11.5, 3.0))
gs = gridspec.GridSpec(
    1,
    5,
    width_ratios=[1.45, 1, 1, 1, 1],
    wspace=0.62,
    left=0.05,
    right=0.985,
    top=0.92,
    bottom=0.18,
)


# ─────────────────────────────────────────────────────────────
#  Panel 1: win-rate heatmap with grey self-play diagonal
# ─────────────────────────────────────────────────────────────
ax_W = fig.add_subplot(gs[0, 0])

# Show the heatmap WITHOUT the diagonal (mask it for clarity)
W_off = np.ma.masked_where(np.eye(n, dtype=bool), W)
HEATMAP_CMAP.set_bad(color=DIAG_FILL)  # diagonal cells appear in light grey

im = ax_W.imshow(W_off, cmap=HEATMAP_CMAP, vmin=0.0, vmax=1.0, aspect="equal")

# Tick & axis styling
ax_W.set_xticks(range(n))
ax_W.set_yticks(range(n))
ax_W.set_xticklabels(agents, fontweight="bold")
ax_W.set_yticklabels(agents, fontweight="bold")
ax_W.set_xlabel("Opponent $j$", labelpad=4)
ax_W.set_ylabel("Agent $i$", labelpad=4)
ax_W.set_title(r"Win-rate matrix $W_{ij}$", pad=8, fontweight="bold")
for spine in ax_W.spines.values():
    spine.set_visible(False)
ax_W.tick_params(length=0)

# Light separating grid between cells (subtle white lines)
ax_W.set_xticks(np.arange(-0.5, n, 1), minor=True)
ax_W.set_yticks(np.arange(-0.5, n, 1), minor=True)
ax_W.grid(which="minor", color="white", linewidth=1.4)
ax_W.tick_params(which="minor", length=0)

# Off-diagonal cell value annotations
for i in range(n):
    for j in range(n):
        if i == j:
            continue
        val = W[i, j]
        # Choose text colour by luminance of the cell colour
        rgb = HEATMAP_CMAP(val)[:3]
        lum = 0.299 * rgb[0] + 0.587 * rgb[1] + 0.114 * rgb[2]
        txt = "white" if lum < 0.55 else "#111827"
        ax_W.text(
            j,
            i,
            f"{val:.2f}",
            ha="center",
            va="center",
            color=txt,
            fontsize=9.5,
            fontweight="semibold",
        )

# Diagonal: horizontal grey bar instead of "0.50"
for i in range(n):
    ax_W.plot(
        [i - 0.28, i + 0.28],
        [i, i],
        color=DIAG_LINE,
        linewidth=2.6,
        solid_capstyle="round",
        zorder=4,
    )

# Discreet colorbar
cbar = fig.colorbar(im, ax=ax_W, fraction=0.044, pad=0.04)
cbar.outline.set_linewidth(0.6)
cbar.outline.set_edgecolor("#374151")
cbar.set_label("Win rate", fontsize=9)
cbar.ax.tick_params(labelsize=8, length=2.5, color="#374151")
cbar.set_ticks([0.0, 0.25, 0.5, 0.75, 1.0])


# ─────────────────────────────────────────────────────────────
#  Helper for the four metric panels
# ─────────────────────────────────────────────────────────────
def make_bar(
    ax,
    scores,
    title,
    *,
    fmt="{:.2f}",
    higher_is_better=True,
    y_zero_line=False,
    value_pad_frac=0.06,
    mark_winner=True,
):
    bars = ax.bar(
        agents, scores, color=agent_colors, edgecolor="#1F2937", linewidth=0.6, width=0.66, zorder=3
    )

    # Highlight the winner (top bar) with a thicker outline.
    # Skipped when mark_winner=False (e.g. a tie, where no single agent leads).
    best_idx = -1
    if mark_winner:
        best_idx = np.argmax(scores) if higher_is_better else np.argmin(scores)
        bars[best_idx].set_linewidth(1.8)
        bars[best_idx].set_edgecolor("#111827")

    # Per-bar value labels (with smart placement & font weight on winner)
    rng = max(abs(max(scores)), abs(min(scores)), 1e-3)
    pad = value_pad_frac * rng
    for k, (bar, s) in enumerate(zip(bars, scores)):
        h = bar.get_height()
        va = "bottom" if h >= 0 else "top"
        off = pad if h >= 0 else -pad
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            h + off,
            fmt.format(s),
            ha="center",
            va=va,
            fontsize=9,
            fontweight="bold" if k == best_idx else "normal",
            color="#111827" if k == best_idx else "#374151",
        )

    # Mark winner on the x-axis tick
    xticks = list(ax.get_xticklabels())
    for k, lbl in enumerate(xticks):
        if k == best_idx:
            lbl.set_color("#111827")
            lbl.set_fontweight("bold")

    ax.set_title(title, pad=7, fontweight="bold")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#9CA3AF")
    ax.spines["bottom"].set_color("#9CA3AF")
    ax.grid(
        axis="y", linestyle=(0, (1.2, 2.0)), linewidth=0.6, color="#9CA3AF", alpha=0.55, zorder=0
    )
    ax.set_axisbelow(True)
    ax.tick_params(length=2.5, color="#374151")

    if y_zero_line:
        ax.axhline(0, color="#374151", linewidth=0.7, zorder=2)


# Panels 2-5: the four metrics
ax_nash = fig.add_subplot(gs[0, 1])
make_bar(
    ax_nash, nash_score, r"Nash score $s_i$", fmt="{:+.2f}", y_zero_line=True, mark_winner=False
)
_nash_mag = max(abs(nash_score.min()), abs(nash_score.max()), 1e-3)
ax_nash.set_ylim(-_nash_mag * 1.6, _nash_mag * 0.7)
# Keep y-tick labels narrow (2 decimals, few ticks) so they don't collide
# with the heatmap colorbar label to the left.
ax_nash.yaxis.set_major_locator(MaxNLocator(nbins=4))
ax_nash.yaxis.set_major_formatter(FormatStrFormatter("%.2f"))

ax_ar = fig.add_subplot(gs[0, 2])
make_bar(ax_ar, alpharank, r"$\alpha$-Rank $\pi_i$")
ax_ar.set_ylim(0, max(alpharank) * 1.32 + 0.05)

ax_cop = fig.add_subplot(gs[0, 3])
make_bar(ax_cop, copeland, r"Copeland $C_i$", fmt="{:+.0f}", y_zero_line=True)
ax_cop.set_ylim(min(copeland) - 0.7, max(copeland) + 0.9)

ax_reg = fig.add_subplot(gs[0, 4])
make_bar(ax_reg, regret, r"Average regret $r_i$", higher_is_better=False)
ax_reg.set_ylim(0, max(regret) * 1.32 + 0.02)


# ─────────────────────────────────────────────────────────────
#  Save
# ─────────────────────────────────────────────────────────────
fig.savefig("metrics_comparison.pdf", bbox_inches="tight")
print("\nSaved: metrics_comparison.pdf (for LaTeX)")
