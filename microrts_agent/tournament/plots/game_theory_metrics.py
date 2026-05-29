"""
Game-theoretic tournament visualizations.

Generates 4 PDFs (Nash weights, Alpha-Rank, Copeland, Robustness)
+ JSON metrics export + Rich console summary.
Runs globally and per-map when multiple maps exist.
"""

from collections import defaultdict
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from rich import box
from rich.table import Table

from .common import clean_name

# Thesis agent: highlighted with a gold "champion" edge across all plots.
HERO = "UECD-Best"
GOLD = "#FFC400"


# ──────────────────────────────────────────────────────────────────────────
#  Public entry point
# ──────────────────────────────────────────────────────────────────────────


def generate_game_theory(data, console, output_dir: Path, per_map_dir: Path = None):
    """
    Compute game-theoretic metrics and generate one PDF per metric
    + console summary.  Runs globally and per-map.

    Args:
        per_map_dir: If provided, per-map plots are saved into existing
                     subdirectories of this path (per_map/<MapName>/).
                     If None, per-map analysis is skipped.
    """
    # ---- Global analysis ----
    _run_analysis(data.games, data.ais, console, output_dir)

    # ---- Per-map analysis (reuse existing per_map/ folders) ----
    if per_map_dir is not None and len(data.maps) > 1:
        games_by_map = defaultdict(list)
        for game in data.games:
            games_by_map[game.map_name].append(game)

        if console:
            console.print("\n[bold cyan]Per-Map Game-Theoretic Analysis[/bold cyan]")

        for map_name in sorted(games_by_map.keys()):
            map_games = games_by_map[map_name]
            short = Path(map_name).stem
            map_dir = per_map_dir / short

            if console:
                console.print(f"\n  [bold]{short}[/bold]")

            _run_analysis(map_games, data.ais, console, map_dir)


# ──────────────────────────────────────────────────────────────────────────
#  Core pipeline
# ──────────────────────────────────────────────────────────────────────────


def _run_analysis(games, ai_names, console, output_dir: Path):
    from ..analysis.game_theory import (
        alpha_rank,
        build_winrate_matrix,
        condorcet_winner,
        copeland_scores,
        nash_averaging,
        regret_metrics,
    )

    winrate = build_winrate_matrix(games, ai_names)

    nash_w, nash_s = nash_averaging(winrate)
    ar_scores = alpha_rank(winrate)
    copeland = copeland_scores(winrate)
    avg_reg, worst_reg, worst_match = regret_metrics(winrate)
    cw = condorcet_winner(winrate, ai_names)
    cw_clean = clean_name(cw) if cw else None

    names = [clean_name(ai) for ai in ai_names]

    _plot_nash_scores(output_dir / "nash_scores.pdf", names, nash_s, console)
    _plot_alpha_sweep(output_dir / "alpha_rank_sweep.pdf", names, winrate, console)
    _plot_copeland(output_dir / "copeland_scores.pdf", names, copeland, cw_clean, console)
    _plot_robustness(
        output_dir / "robustness_score.pdf", names, worst_reg, avg_reg, worst_match, console
    )

    if console:
        _print_summary(
            console,
            names,
            nash_w,
            nash_s,
            ar_scores,
            copeland,
            avg_reg,
            worst_reg,
            worst_match,
            cw_clean,
        )


# ──────────────────────────────────────────────────────────────────────────
#  Individual plots
# ──────────────────────────────────────────────────────────────────────────


def _plot_nash_scores(output_path, names, nash_s, console):
    from matplotlib.colors import TwoSlopeNorm

    order = np.argsort(nash_s)[::-1]
    sel_names = [names[i] for i in order]
    sel_scores = nash_s[order]
    n = len(sel_names)

    # Divergent color: red (negative) → white (zero) → blue (positive)
    abs_max = max(abs(sel_scores.min()), abs(sel_scores.max()), 0.01)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = plt.cm.RdBu
    sel_colors = [cmap(norm(s)) for s in sel_scores]

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.46)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(n)
    bars = ax.barh(y, sel_scores, color=sel_colors, alpha=1.0, edgecolor="black", linewidth=0.8)
    for k, bar in enumerate(bars):
        if sel_names[k] == HERO:
            bar.set_edgecolor(GOLD)
            bar.set_linewidth(3.2)

    off = abs_max * 0.03
    for k in range(n):
        val = sel_scores[k]
        x_pos, ha = (val + off, "left") if val >= 0 else (val - off, "right")
        ax.text(
            x_pos,
            k,
            f"{val:+.3f}",
            va="center",
            ha=ha,
            fontsize=13,
            fontweight="bold",
            family="sans-serif",
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sel_names, fontsize=14, fontweight="bold", family="monospace")
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel("Nash Score", fontsize=17, fontweight="bold", family="sans-serif")
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)

    margin = abs_max * 0.38
    ax.set_xlim(min(0, sel_scores.min()) - margin, max(0, sel_scores.max()) + margin)
    ax.invert_yaxis()
    ax.set_axisbelow(True)
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb", zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Nash Scores → {output_path.name}")


def _plot_alpha_sweep(output_path, names, winrate, console):
    from ..analysis.game_theory import alpha_rank

    alphas = np.logspace(-3, 3, 80)
    n = len(names)
    scores = np.zeros((len(alphas), n))
    for i, a in enumerate(alphas):
        scores[i] = alpha_rank(winrate, alpha=a)

    # Sort agents by score at default alpha (0.02)
    ref_idx = np.argmin(np.abs(alphas - 0.02))
    final_order = np.argsort(-scores[ref_idx])

    fig, ax = plt.subplots(figsize=(14, 7))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    top_n = 5
    cmap_lines = plt.cm.tab10

    import matplotlib.patheffects as pe

    for rank, i in enumerate(final_order):
        if rank < top_n:
            kw = {"color": cmap_lines(rank), "linewidth": 3.2, "alpha": 0.95}
            if names[i] == HERO:
                # Thin gold "champion" halo around the thesis-agent curve
                kw["linewidth"] = 3.0
                kw["zorder"] = 10
                kw["path_effects"] = [
                    pe.Stroke(linewidth=5.8, foreground=GOLD, alpha=0.95),
                    pe.Normal(),
                ]
            ax.plot(alphas, scores[:, i], label=names[i], **kw)
        else:
            ax.plot(alphas, scores[:, i], color="#CCCCCC", linewidth=0.8, alpha=0.5)

    ax.set_xscale("log")
    ax.set_xlabel(r"Selection Intensity ($\alpha$)", fontsize=17, fontweight="bold")
    ax.set_ylabel("Stationary Distribution", fontsize=17, fontweight="bold")
    ax.tick_params(axis="both", labelsize=15)
    leg = ax.legend(loc="upper left", fontsize=15, framealpha=0.95)
    leg.set_title("Top 5 agents", prop={"size": 14, "weight": "bold"})
    ax.grid(True, linestyle="--", alpha=0.25)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Alpha-Rank Sweep → {output_path.name}")


def _plot_copeland(output_path, names, copeland, cw_clean, console):
    n = len(names)
    order = np.argsort(copeland)[::-1]
    sel_names = [names[i] for i in order]
    sel_scores = copeland[order]

    # Divergent color gradient: red (negative) → white (zero) → blue (positive)
    from matplotlib.colors import TwoSlopeNorm

    score_min = sel_scores.min()
    score_max = sel_scores.max()
    abs_max = max(abs(score_min), abs(score_max), 1)
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0, vmax=abs_max)
    cmap = plt.cm.RdBu  # red → white → blue
    sel_colors = [cmap(norm(s)) for s in sel_scores]

    fig, ax = plt.subplots(figsize=(12, max(6, n * 0.46)))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    y = np.arange(n)
    bars = ax.barh(
        y,
        sel_scores,
        color=sel_colors,
        alpha=1.0,
        edgecolor="black",
        linewidth=0.8,
    )

    max_abs = max(abs(sel_scores.max()), abs(sel_scores.min()), 1)
    for k, bar in enumerate(bars):
        name = sel_names[k]
        score = int(sel_scores[k])
        label = f"{score:+d}"
        # Gold champion edge on the thesis agent
        if name == HERO:
            bar.set_edgecolor(GOLD)
            bar.set_linewidth(3.2)
        # Position: positive scores to the right, negative to the left
        if score >= 0:
            x_pos = bar.get_width() + max_abs * 0.03
            ha = "left"
        else:
            x_pos = bar.get_width() - max_abs * 0.03
            ha = "right"
        ax.text(
            x_pos, k, label, va="center", ha=ha, fontsize=14, fontweight="bold", family="sans-serif"
        )

    ax.set_yticks(y)
    ax.set_yticklabels(sel_names, fontsize=14, fontweight="bold", family="monospace")
    ax.tick_params(axis="x", labelsize=15)
    ax.set_xlabel(
        "Copeland Score (#Beaten \u2212 #Lost)",
        fontsize=17,
        fontweight="bold",
        family="sans-serif",
    )
    # Solid thin black line at x=0
    ax.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax.invert_yaxis()

    ax.set_axisbelow(True)
    ax.grid(False)
    ax.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb", zorder=0)
    for s in ["top", "right"]:
        ax.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Copeland Score → {output_path.name}")


def _plot_robustness(output_path, names, worst_reg, avg_reg, worst_match, console):
    from matplotlib.colors import Normalize

    n = len(names)
    worst_rob = 1.0 - worst_reg
    mean_rob = 1.0 - avg_reg

    fig, (ax_w, ax_m) = plt.subplots(
        1,
        2,
        figsize=(22, max(6, n * 0.46)),
        sharey=False,
    )
    fig.patch.set_facecolor("white")

    # ---- (a) Worst-case, sorted by worst-case ----
    order_w = np.argsort(worst_rob)[::-1]
    names_w = [names[i] for i in order_w]
    vals_w = worst_rob[order_w]

    norm_w = Normalize(vmin=vals_w.min() - (vals_w.max() - vals_w.min()) * 0.15, vmax=vals_w.max())
    colors_w = [plt.cm.Blues(norm_w(s)) for s in vals_w]

    ax_w.set_facecolor("white")
    y = np.arange(n)

    bars_w = ax_w.barh(y, vals_w, color=colors_w, alpha=1.0, edgecolor="black", linewidth=0.8)
    for k, bar in enumerate(bars_w):
        if names_w[k] == HERO:
            bar.set_edgecolor(GOLD)
            bar.set_linewidth(3.2)

    # Score + nemesis (highest-regret opponent), attached to each bar's end
    ax_w.text(
        0.09,
        -0.9,
        "↯ worst opponent",
        va="center",
        fontsize=12,
        fontweight="bold",
        color="#555555",
        family="sans-serif",
        clip_on=False,
    )
    for k in range(n):
        ax_w.text(
            vals_w[k] + 0.012,
            k,
            f"{vals_w[k]:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            family="sans-serif",
        )
        opp = names[worst_match[order_w[k]]]
        ax_w.text(
            vals_w[k] + 0.08,
            k,
            opp,
            va="center",
            fontsize=12,
            fontweight="bold",
            color="#333333",
            family="monospace",
            clip_on=False,
        )

    ax_w.set_yticks(y)
    ax_w.set_yticklabels(names_w, fontsize=14, fontweight="bold", family="monospace")
    ax_w.tick_params(axis="x", labelsize=14)
    ax_w.set_xlabel(
        "Worst-Case Robustness Score", fontsize=17, fontweight="bold", family="sans-serif"
    )
    ax_w.axvline(0, color="black", linestyle="-", linewidth=1.0)
    # Tight x-range: max bar + room for the score & opponent label
    ax_w.set_xlim(0, vals_w.max() + 0.32)
    ax_w.invert_yaxis()
    ax_w.set_axisbelow(True)
    ax_w.grid(False)
    ax_w.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb", zorder=0)
    for s in ["top", "right"]:
        ax_w.spines[s].set_visible(False)

    # ---- (b) Mean-case, sorted by mean-case ----
    order_m = np.argsort(mean_rob)[::-1]
    names_m = [names[i] for i in order_m]
    vals_m = mean_rob[order_m]

    norm_m = Normalize(vmin=vals_m.min() - (vals_m.max() - vals_m.min()) * 0.15, vmax=vals_m.max())
    colors_m = [plt.cm.Blues(norm_m(s)) for s in vals_m]

    ax_m.set_facecolor("white")

    bars_m = ax_m.barh(y, vals_m, color=colors_m, alpha=1.0, edgecolor="black", linewidth=0.8)
    for k, bar in enumerate(bars_m):
        if names_m[k] == HERO:
            bar.set_edgecolor(GOLD)
            bar.set_linewidth(3.2)

    for k in range(n):
        ax_m.text(
            vals_m[k] + 0.012,
            k,
            f"{vals_m[k]:.2f}",
            va="center",
            fontsize=12,
            fontweight="bold",
            color="black",
            family="sans-serif",
        )

    ax_m.set_yticks(y)
    ax_m.set_yticklabels(names_m, fontsize=14, fontweight="bold", family="monospace")
    ax_m.tick_params(axis="x", labelsize=14)
    ax_m.set_xlabel(
        "Mean-Case Robustness Score", fontsize=17, fontweight="bold", family="sans-serif"
    )
    ax_m.axvline(0, color="black", linestyle="-", linewidth=1.0)
    ax_m.set_xlim(0, vals_m.max() + 0.08)
    ax_m.invert_yaxis()
    ax_m.set_axisbelow(True)
    ax_m.grid(False)
    ax_m.xaxis.grid(True, linestyle="--", alpha=0.35, color="#bbbbbb", zorder=0)
    for s in ["top", "right"]:
        ax_m.spines[s].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close()

    if console:
        console.print(f"  [green]✓[/green] Robustness → {output_path.name}")


# ──────────────────────────────────────────────────────────────────────────
#  Console summary
# ──────────────────────────────────────────────────────────────────────────


def _print_summary(
    console, names, nash_w, nash_s, ar_scores, copeland, avg_reg, worst_reg, worst_match, cw_clean
):
    order = np.argsort(nash_s)[::-1]

    table = Table(
        title="Game-Theoretic Rankings",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold cyan",
        border_style="cyan",
    )
    table.add_column("Rank", justify="center", style="dim", width=4)
    table.add_column("Agent", style="bright_white", min_width=15)
    table.add_column("Nash Score", justify="center", style="green")
    table.add_column("Nash Wt", justify="center", style="yellow")
    table.add_column("α-Rank", justify="center", style="magenta")
    table.add_column("Copeland", justify="center", style="blue")
    table.add_column("Avg Regret", justify="center", style="red")
    table.add_column("Max Regret", justify="center", style="red")
    table.add_column("Worst vs", justify="center", style="dim")

    for rank, i in enumerate(order, 1):
        table.add_row(
            str(rank),
            names[i],
            f"{nash_s[i]:+.3f}",
            f"{nash_w[i]:.3f}",
            f"{ar_scores[i]:.3f}",
            f"{int(copeland[i]):+d}",
            f"{avg_reg[i]:.3f}",
            f"{worst_reg[i]:.3f}",
            names[worst_match[i]],
        )

    console.print()
    console.print(table)

    if cw_clean:
        console.print(
            f"\n  [bold green]Condorcet Winner: {cw_clean}[/bold green]"
            " (beats all opponents pairwise)"
        )
    else:
        console.print(
            "\n  [bold yellow]No Condorcet Winner[/bold yellow] (intransitive cycle detected)"
        )
