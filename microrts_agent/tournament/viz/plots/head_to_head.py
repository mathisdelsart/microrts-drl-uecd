"""Head-to-head pairwise win-rate heatmap.

RdBu matrix where cell (i, j) = win rate of row agent i against column agent j.
Red (0%) means agent i always loses, blue (100%) means agent i always wins.
Agents are ranked by tournament standing; diagonal is masked (self-play).
Reveals dominant matchups and exploitable weaknesses between specific pairs.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from .common import calculate_rankings, clean_name


def plot_head_to_head_matrix(data, console, output_path: Path):
    n_ais = len(data.ais)

    win_matrix = np.zeros((n_ais, n_ais))
    game_count = np.zeros((n_ais, n_ais))

    for game in data.games:
        i = data.ais.index(game.ai1_name)
        j = data.ais.index(game.ai2_name)

        game_count[i, j] += 1
        game_count[j, i] += 1

        if game.winner == 0:
            win_matrix[i, j] += 1
        elif game.winner == 1:
            win_matrix[j, i] += 1
        else:
            win_matrix[i, j] += 0.5
            win_matrix[j, i] += 0.5

    winrate = np.full((n_ais, n_ais), np.nan)
    for i in range(n_ais):
        for j in range(n_ais):
            if game_count[i, j] > 0:
                winrate[i, j] = 100 * win_matrix[i, j] / game_count[i, j]

    winrate = np.ma.array(winrate, mask=np.eye(n_ais, dtype=bool))

    rankings = calculate_rankings(data.games, data.ais)
    sorted_ais = [r[0] for r in rankings]

    ai_to_idx = {ai: i for i, ai in enumerate(data.ais)}
    order = [ai_to_idx[ai] for ai in sorted_ais]

    winrate = winrate[np.ix_(order, order)]
    names = [clean_name(ai) for ai in sorted_ais]

    _, ax = plt.subplots(figsize=(10, 8.5))

    # Diagonal cells (self-play) rendered as a uniform light gray block
    cmap = mpl.colormaps.get_cmap("RdBu").copy()
    cmap.set_bad(color="#d8d8d8")
    im = ax.imshow(
        winrate,
        cmap=cmap,
        vmin=0,
        vmax=100,
        interpolation="nearest",
        aspect="equal",
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Win rate (%)",
        fontsize=16,
        fontweight="bold",
        rotation=90,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=17)

    ax.set_xticks(np.arange(n_ais))
    ax.set_yticks(np.arange(n_ais))
    ax.set_xticklabels(
        names, rotation=45, ha="right", fontsize=15, fontweight="bold", family="monospace"
    )
    ax.set_yticklabels(names, fontsize=15, fontweight="bold", family="monospace")
    ax.grid(False)

    for i in range(n_ais):
        for j in range(n_ais):
            if i == j:
                ax.text(
                    j,
                    i,
                    "—",
                    ha="center",
                    va="center",
                    fontsize=18,
                    fontweight="bold",
                    color="#777777",
                )
                continue
            if np.isnan(winrate[i, j]):
                continue
            wr = winrate[i, j]
            # Skip 0 and 100: deep-red and deep-blue cells are unambiguous on their own
            if wr == 0 or wr == 100:
                continue
            ax.text(
                j,
                i,
                f"{wr:.0f}",
                ha="center",
                va="center",
                fontsize=12,
                fontweight="bold",
                color="white",
                path_effects=[
                    pe.Stroke(linewidth=2.8, foreground="black"),
                    pe.Normal(),
                ],
            )

    # Highlight the showcase UECD agents (not the ablation variants) with a gold border
    gold = "#FFC400"
    _ablation_variants = ("UECD-Rushed", "UECD-TopFeats", "UECD-AllFeats")
    for idx, name in enumerate(names):
        if not name.startswith("UECD") or name in _ablation_variants:
            continue
        # Row
        ax.add_patch(
            mpatches.Rectangle(
                (-0.5, idx - 0.5),
                n_ais,
                1,
                fill=False,
                edgecolor=gold,
                linewidth=3.0,
                zorder=10,
            )
        )
        # Column
        ax.add_patch(
            mpatches.Rectangle(
                (idx - 0.5, -0.5),
                1,
                n_ais,
                fill=False,
                edgecolor=gold,
                linewidth=3.0,
                zorder=10,
            )
        )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Head-to-Head Matrix → {output_path.name}")
