"""
Tournament Visualizer — Orchestrator

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
    """Thin orchestrator — delegates to plot modules in plots/."""

    def __init__(self, tournament_data: TournamentData, console=None):
        self.data = tournament_data
        self.console = console

    def generate_all(self, output_dir: Path):
        d, con = self.data, self.console

        # --- Aggregate plots (all maps) ---
        if con:
            con.print("\n[bold cyan]Core Tournament Analysis[/bold cyan]")

        plot_final_standings(d, con, output_dir / "final_standings.pdf")
        plot_head_to_head_matrix(d, con, output_dir / "h2h_matrix.pdf")
        plot_game_length_distribution(d, con, output_dir / "games_length.pdf")
        plot_per_map_winrates(d, con, output_dir / "winrates_per_map.pdf")

        # --- Per-map plots (create folders, reused by game theory below) ---
        per_map_dir = None
        if len(d.maps) > 1:
            if con:
                con.print(f"\n[bold cyan]Per-Map Analysis ({len(d.maps)} maps)[/bold cyan]")

            per_map_dir = output_dir / "per_map"
            per_map_dir.mkdir(exist_ok=True, parents=True)

            for map_path in d.maps:
                clean = _clean_map_name(map_path)
                map_dir = per_map_dir / clean
                map_dir.mkdir(exist_ok=True, parents=True)

                filtered = _filter_by_map(d, map_path)

                plot_final_standings(filtered, None, map_dir / "final_standings.pdf")
                plot_head_to_head_matrix(filtered, None, map_dir / "h2h_matrix.pdf")
                plot_game_length_distribution(filtered, None, map_dir / "games_length.pdf")

            if con:
                con.print(
                    f"  [green]✓[/green] Per-map plots → per_map/ ({len(d.maps)} maps x 3 plots)"
                )

        # --- Game theory (global + per-map into existing per_map/ folders) ---
        generate_game_theory(d, con, output_dir, per_map_dir=per_map_dir)
