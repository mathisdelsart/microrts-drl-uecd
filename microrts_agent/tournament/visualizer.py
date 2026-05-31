"""
Tournament Visualizer: Orchestrator

Loads tournament data and delegates to individual plot modules in plots/.

Usage:
    from microrts_agent.tournament import TournamentData, TournamentVisualizer

    data = TournamentData(results_dir / "tournament_parsed.json")
    viz = TournamentVisualizer(data, console)
    viz.generate_all(results_dir / "visualizations")

Generates PDFs:
    Final Standings, Head-to-Head Matrix, Game Length Distribution,
    Per-Map Win Rates, Game-Theoretic Metrics (Nash, Alpha-Rank, Copeland, Robustness)
"""

import json
from pathlib import Path

from .plots import (
    generate_game_theory,
    plot_final_standings,
    plot_game_length_distribution,
    plot_head_to_head_matrix,
    plot_per_map_winrates,
)
from .ranking.data import GameData


class TournamentData:
    """Container for tournament data loaded from JSON."""

    def __init__(self, json_path: Path):
        self.tournament_dir = json_path.parent

        with open(json_path) as f:
            data = json.load(f)

        self.tournament_type = data["tournament"]["type"]
        self.ais = data["tournament"]["ais"]
        self.maps = data["tournament"]["maps"]
        self.iterations = data["tournament"]["iterations"]
        self.config = data["configuration"]

        self.games = []
        for game in data["games"]:
            self.games.append(
                GameData(
                    iteration=game["iteration"],
                    map_id=game["map"]["id"],
                    map_name=game["map"]["name"],
                    ai1_id=game["players"]["ai1"]["id"],
                    ai1_name=game["players"]["ai1"]["name"],
                    ai2_id=game["players"]["ai2"]["id"],
                    ai2_name=game["players"]["ai2"]["name"],
                    time=game["result"]["time"],
                    winner=game["result"]["winner"],
                    crashed=game["result"]["crashed"],
                    timedout=game["result"]["timedout"],
                    ai1_time_ns=game["result"].get("ai1_time_ns", 0),
                    ai2_time_ns=game["result"].get("ai2_time_ns", 0),
                )
            )


def _clean_map_name(map_path: str) -> str:
    """Extract a clean map name from a full path like 'maps/open_competition/basesWorkers16x16A.xml'."""
    return Path(map_path).stem


def _filter_by_map(data: TournamentData, map_name: str) -> TournamentData:
    """Create a lightweight copy of TournamentData filtered to a single map."""
    filtered = TournamentData.__new__(TournamentData)
    filtered.tournament_dir = data.tournament_dir
    filtered.tournament_type = data.tournament_type
    filtered.ais = data.ais
    filtered.maps = [map_name]
    filtered.iterations = data.iterations
    filtered.config = data.config
    filtered.games = [g for g in data.games if g.map_name == map_name]
    return filtered


class TournamentVisualizer:
    """Thin orchestrator: delegates to plot modules in plots/.

    Output layout (under ``output_dir`` = ``<results_dir>/visualizations/``):

    Single-map tournament::

        basic-metrics/{final_standings,h2h_matrix,games_length}.pdf
        game-theoretic-metrics/{nash_scores,alpha_rank_sweep,
                                copeland_scores,robustness_score}.pdf

    Multi-map tournament::

        global/basic-metrics/{final_standings,h2h_matrix,games_length,
                              winrates_per_map}.pdf
        global/game-theoretic-metrics/{...4 PDFs}.pdf
        individual/<map>/basic-metrics/{...3 PDFs}.pdf
        individual/<map>/game-theoretic-metrics/{...4 PDFs}.pdf
    """

    def __init__(self, tournament_data: TournamentData, console=None):
        self.data = tournament_data
        self.console = console

    def generate_all(self, output_dir: Path):
        d, con = self.data, self.console
        multi_map = len(d.maps) > 1

        # ── Global plots ─────────────────────────────────────────────────
        # For single-map tournaments there is no meaningful "per-map" view,
        # so the global subdirs sit directly under output_dir.
        global_root = output_dir / "global" if multi_map else output_dir
        basic_global = global_root / "basic-metrics"
        gt_global = global_root / "game-theoretic-metrics"
        basic_global.mkdir(parents=True, exist_ok=True)
        gt_global.mkdir(parents=True, exist_ok=True)

        if con:
            con.print("\n[bold cyan]Core Tournament Analysis[/bold cyan]")

        plot_final_standings(d, con, basic_global / "final_standings.pdf")
        plot_head_to_head_matrix(d, con, basic_global / "h2h_matrix.pdf")
        plot_game_length_distribution(d, con, basic_global / "games_length.pdf")
        if multi_map:
            plot_per_map_winrates(d, con, basic_global / "winrates_per_map.pdf")

        # ── Per-map plots (only when multiple maps were played) ──────────
        individual_dir = None
        if multi_map:
            if con:
                con.print(f"\n[bold cyan]Per-Map Analysis ({len(d.maps)} maps)[/bold cyan]")

            individual_dir = output_dir / "individual"
            individual_dir.mkdir(parents=True, exist_ok=True)

            for map_path in d.maps:
                clean = _clean_map_name(map_path)
                map_basic = individual_dir / clean / "basic-metrics"
                map_basic.mkdir(parents=True, exist_ok=True)
                # game-theoretic subdir is created lazily by generate_game_theory

                filtered = _filter_by_map(d, map_path)
                plot_final_standings(filtered, None, map_basic / "final_standings.pdf")
                plot_head_to_head_matrix(filtered, None, map_basic / "h2h_matrix.pdf")
                plot_game_length_distribution(filtered, None, map_basic / "games_length.pdf")

            if con:
                con.print(
                    f"  [green]✓[/green] Per-map plots → individual/ ({len(d.maps)} maps x 3 plots)"
                )

        # ── Game theory (global into gt_global; per-map into individual/) ─
        generate_game_theory(d, con, gt_global, individual_dir=individual_dir)
