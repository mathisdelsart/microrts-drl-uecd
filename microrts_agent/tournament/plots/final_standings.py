"""Final tournament standings — horizontal bar chart of total points.

Agents ranked by total points (Win=1, Draw=0.5, Loss=0). Bar length = total score.
The bar gradient mirrors the head-to-head matrix (red→white→blue, 0→50%→100% WR
of the agent's max-possible points). The champion bar carries a gold outline.
Useful to see the overall tournament hierarchy at a glance.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from .common import calculate_rankings, clean_name


def plot_final_standings(data, console, output_path: Path):
    from matplotlib.colors import Normalize

    rankings = calculate_rankings(data.games, data.ais)
    ai_names = [clean_name(r[0]) for r in rankings]
    scores = [r[1] for r in rankings]

    n = len(ai_names)
    fig, ax = plt.subplots(figsize=(12, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y_pos = np.arange(n)

    # Diverging RdBu gradient mapped to total points, on the same scale as the
    # head-to-head matrix: 0 points → 0% WR (deep red), max possible points
    # (= games each agent plays) → 100% WR (deep blue), midpoint 50% → white.
    max_score = max(scores)
    games_per_agent = 2 * len(data.games) / n if n else 1
    norm = Normalize(vmin=0, vmax=games_per_agent)
    cmap = plt.cm.RdBu
    bar_colors = [cmap(norm(s)) for s in scores]

    bars = ax.barh(
        y_pos,
        scores,
        color=bar_colors,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.8,
        zorder=3,
    )

    # Champion: gold edge border (slightly thicker for visibility)
    bars[0].set_edgecolor("#FFC400")
    bars[0].set_linewidth(3.2)

    # Score labels (right of bar)
    score_strs = [f"{int(s)}" if s == int(s) else f"{s:.1f}" for s in scores]

    for i, bar in enumerate(bars):
        bw = bar.get_width()
        # Score label outside on the right of the bar
        ax.text(
            bw + max_score * 0.012,
            i,
            score_strs[i],
            va="center",
            fontsize=14,
            fontweight="bold",
            color="black",
            family="sans-serif",
            zorder=5,
        )

    ax.set_yticks(y_pos)
    ax.set_yticklabels(ai_names, fontsize=14, fontweight="bold", family="monospace")
    ax.tick_params(axis="x", labelsize=16)
    ax.set_xlabel("Total Points", fontsize=17, fontweight="bold", family="sans-serif")

    if len(data.maps) == 1:
        map_name = Path(data.maps[0]).stem
        maps_descriptor = f"{map_name} Map"
    else:
        maps_descriptor = f"{len(data.maps)} Maps"

    ax.text(
        0.5,
        1.018,
        f"{len(data.games)} Games  \u2022  {maps_descriptor}"
        "  \u2022  Win = +1   |   Draw = +0.5   |   Loss = +0",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=16,
        color="#555555",
        family="sans-serif",
        fontstyle="italic",
        fontweight="bold",
    )

    # 50%-score reference line. Drawn behind the bars (zorder=0) so opaque
    # bars cover it where they overlap.
    games_per_agent = 2 * len(data.games) / n if n else 0  # each game involves two agents
    ref_x = games_per_agent / 2
    if ref_x > 0:
        ax.axvline(ref_x, ls="--", color="#888888", alpha=0.8, linewidth=1.5, zorder=0)
        # Label at the foot of the line, where bars stop covering it.
        ax.text(
            ref_x + max_score * 0.005,
            0.015,
            "50% WR",
            transform=ax.get_xaxis_transform(),
            ha="left",
            va="bottom",
            fontsize=14,
            color="#666666",
            fontstyle="italic",
        )

    ax.grid(False)
    ax.invert_yaxis()
    # Squeeze top padding so the title sits right above the first bar and
    # the dashed grid lines below stop at the top of that bar.
    ax.set_ylim(n - 0.5, -0.4)

    # X ticks across the [0, games_per_agent] range — derive the step so we
    # land on round numbers and end up with ~6-7 ticks regardless of tournament
    # size (10 bots × 20 games = 180; multi-map has different totals).
    import matplotlib.ticker as mticker

    max_x = int(np.ceil(games_per_agent))
    if max_x <= 0:
        max_x = 1
    # Pick a step that gives ~6 ticks (rounded to the nearest "nice" number).
    raw_step = max_x / 6
    step = (
        max(1, int(round(raw_step / 10) * 10)) if raw_step >= 10 else max(1, int(round(raw_step)))
    )
    tick_positions = list(range(0, max_x + 1, step))
    ax.xaxis.set_major_locator(mticker.FixedLocator(tick_positions))

    # Vertical dashed reference lines (skip endpoints) — drawn behind bars.
    for x in tick_positions:
        if x in (0, max_x):
            continue
        ax.vlines(
            x, -0.4, n - 0.5, linestyles="--", colors="#bbbbbb", linewidth=0.8, alpha=0.6, zorder=0
        )

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)
    ax.spines["left"].set_linewidth(ax.spines["bottom"].get_linewidth())
    ax.spines["left"].set_color(ax.spines["bottom"].get_edgecolor())

    # Small right-side padding (~4%) so the score labels next to the top bars
    # don't bump against the axis frame.
    ax.set_xlim(0, max_x * 1.04 if max_x > 0 else 1)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Final Standings → {output_path.name}")
