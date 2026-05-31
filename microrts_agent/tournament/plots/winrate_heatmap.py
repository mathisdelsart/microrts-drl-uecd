"""Per-map win rate heatmap (rows = AIs ranked, columns = maps + Overall).

RdBu matrix where cell (i, j) = overall win rate of agent i on map j, on the
same 0-100 colour scale as the head-to-head matrix (red = struggles, blue =
strong, white = 50%). A trailing Overall column ties each agent's per-map
profile back to its tournament standing. Agents are ranked by standing and any
UECD-* agent is highlighted with a gold border, matching final_standings and
the head-to-head matrix.
"""

from pathlib import Path

import matplotlib as mpl
import matplotlib.patches as mpatches
import matplotlib.patheffects as pe
import matplotlib.pyplot as plt
import numpy as np

from .common import calculate_rankings, clean_name


def plot_per_map_winrates(data, console, output_path: Path):
    if len(data.maps) <= 1:
        if console:
            console.print("  [dim]⊘ Per-Map Win Rates skipped (single map)[/dim]")
        return

    rankings = calculate_rankings(data.games, data.ais)
    sorted_ais = [r[0] for r in rankings]

    n_ais = len(sorted_ais)
    n_maps = len(data.maps)

    wins = np.zeros((n_ais, n_maps))
    games_count = np.zeros((n_ais, n_maps))

    ai_to_rank = {ai: i for i, ai in enumerate(sorted_ais)}

    for game in data.games:
        map_idx = game.map_id
        ai1_rank = ai_to_rank[game.ai1_name]
        ai2_rank = ai_to_rank[game.ai2_name]

        games_count[ai1_rank, map_idx] += 1
        games_count[ai2_rank, map_idx] += 1

        if game.winner == 0:
            wins[ai1_rank, map_idx] += 1
        elif game.winner == 1:
            wins[ai2_rank, map_idx] += 1
        else:
            wins[ai1_rank, map_idx] += 0.5
            wins[ai2_rank, map_idx] += 0.5

    winrate = np.where(games_count > 0, 100 * wins / games_count, np.nan)

    # Overall win rate per agent (true ratio over all maps, not a row mean),
    # appended as a trailing column set apart from the per-map block.
    total_games = games_count.sum(axis=1)
    overall = np.where(total_games > 0, 100 * wins.sum(axis=1) / total_games, np.nan)
    matrix = np.column_stack([winrate, overall])
    n_cols = n_maps + 1
    overall_col = n_maps

    clean_ais = [clean_name(ai) for ai in sorted_ais]
    clean_maps = [Path(m).stem for m in data.maps]
    col_labels = clean_maps + ["Overall"]

    fig, ax = plt.subplots(figsize=(max(10, n_cols * 1.5), max(6, n_ais * 0.5)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    cmap = mpl.colormaps.get_cmap("RdBu").copy()
    cmap.set_bad(color="#d8d8d8")
    im = ax.imshow(
        matrix,
        cmap=cmap,
        vmin=0,
        vmax=100,
        interpolation="nearest",
        aspect="auto",
    )

    cbar = plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(
        "Win rate (%)",
        fontsize=16,
        fontweight="bold",
        rotation=90,
        labelpad=10,
    )
    cbar.ax.tick_params(labelsize=14)

    ax.set_xticks(np.arange(n_cols))
    ax.set_yticks(np.arange(n_ais))
    ax.set_xticklabels(
        col_labels, rotation=45, ha="right", fontsize=14, fontweight="bold", family="monospace"
    )
    ax.set_yticklabels(clean_ais, fontsize=14, fontweight="bold", family="monospace")
    # Mark the Overall column header so it reads as a summary, not a map.
    ax.get_xticklabels()[overall_col].set_color("#222222")
    ax.get_xticklabels()[overall_col].set_fontstyle("italic")

    # Thin white separators between cells for a crisper grid.
    ax.set_xticks(np.arange(-0.5, n_cols), minor=True)
    ax.set_yticks(np.arange(-0.5, n_ais), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.0)
    ax.tick_params(which="minor", length=0)
    ax.grid(which="major", visible=False)

    for i in range(n_ais):
        for j in range(n_cols):
            if np.isnan(matrix[i, j]):
                continue
            is_overall = j == overall_col
            ax.text(
                j,
                i,
                f"{matrix[i, j]:.0f}",
                ha="center",
                va="center",
                fontsize=13 if is_overall else 12,
                fontweight="bold",
                color="white",
                path_effects=[
                    pe.Stroke(linewidth=2.2 if is_overall else 1.8, foreground="black"),
                    pe.Normal(),
                ],
            )

    # Bold separator between the per-map block and the Overall column.
    ax.axvline(overall_col - 0.5, color="#222222", linewidth=2.5, zorder=6)

    # Gold border on UECD-* rows (matches final_standings champion + h2h).
    gold = "#FFC400"
    for i, name in enumerate(clean_ais):
        if name.startswith("UECD"):
            ax.add_patch(
                mpatches.Rectangle(
                    (-0.5, i - 0.5),
                    n_cols,
                    1,
                    fill=False,
                    edgecolor=gold,
                    linewidth=3.0,
                    zorder=10,
                )
            )

    # Italic subtitle in place of a title (matches final_standings).
    ax.text(
        0.5,
        1.015,
        f"{len(data.games)} Games  •  {n_maps} Maps  •  per-map win rate (%)",
        transform=ax.transAxes,
        ha="center",
        va="bottom",
        fontsize=15,
        color="#555555",
        family="sans-serif",
        fontstyle="italic",
        fontweight="bold",
    )

    for spine in ax.spines.values():
        spine.set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Per-Map Win Rates → {output_path.name}")
