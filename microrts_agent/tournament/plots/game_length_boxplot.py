"""Game length distribution by AI: boxplot ordered by final rankings.

One boxplot per agent showing the distribution of game durations (in frames).
Short games suggest decisive play or early rushes; long games suggest defensive styles.
Outliers reveal occasional timeouts or stalemates. Agents ordered by tournament rank.
Comparing medians across agents helps identify aggressive vs passive strategies.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np

from .common import calculate_rankings, clean_name


def plot_game_length_distribution(data, console, output_path: Path):
    ai_game_lengths = defaultdict(list)
    for game in data.games:
        ai_game_lengths[game.ai1_name].append(game.time)
        ai_game_lengths[game.ai2_name].append(game.time)

    rankings = calculate_rankings(data.games, data.ais)
    sorted_ais = [r[0] for r in rankings]

    clean_names = [clean_name(ai) for ai in sorted_ais]
    box_data = [ai_game_lengths[ai] for ai in sorted_ais]

    # Sort by median game length descending
    medians = [float(np.median(d)) for d in box_data]
    order = np.argsort(medians)[::-1]
    clean_names = [clean_names[i] for i in order]
    box_data = [box_data[i] for i in order]

    fig, ax = plt.subplots(figsize=(14, 6.5))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    bp = ax.boxplot(
        box_data,
        tick_labels=clean_names,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.6,
        medianprops={"color": "#1f4e79", "linewidth": 2.2},
        meanprops={"color": "#1f4e79", "linewidth": 1.5, "linestyle": "--"},
        flierprops={
            "marker": "o",
            "markersize": 3,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.4,
        },
    )

    STEEL_BLUE = "#4682B4"
    for patch in bp["boxes"]:
        patch.set_facecolor(STEEL_BLUE)
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.3)

    for whisker in bp["whiskers"]:
        whisker.set(linewidth=1.3, linestyle="--", alpha=0.6)
    for cap in bp["caps"]:
        cap.set(linewidth=1.3)

    ax.set_ylabel("Game Length", fontsize=16, fontweight="bold")
    ax.set_xticklabels(clean_names, rotation=45, ha="right", fontsize=11)

    ax.grid(False)
    ax.yaxis.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax.set_axisbelow(True)

    for spine in ["top", "right"]:
        ax.spines[spine].set_visible(False)

    max_length = max(max(lengths) for lengths in box_data)
    ax.set_ylim(-50, max_length * 1.05)

    # Legend
    median_line = mlines.Line2D([], [], color="#1f4e79", linewidth=2.2, label="Median")
    mean_line = mlines.Line2D([], [], color="#1f4e79", linewidth=1.5, linestyle="--", label="Mean")
    ax.legend(
        handles=[median_line, mean_line],
        loc="upper right",
        fontsize=10,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Game Length Distribution → {output_path.name}")
