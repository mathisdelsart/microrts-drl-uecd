"""
Tournament CSV Parser for MicroRTS

Parses the CSV files produced by the Java MicroRTS tournament runner.
Handles two CSV layouts:
    - Unified:    single block containing all maps (maps annotated with maxGameLength=N)
    - Multi-block: one block per map, each with its own header + summary (legacy)

Both formats are merged into a single ParsedTournamentConfig + flat list of GameResults.

Also detects and extracts trace ZIP files (game replays) when present.

Usage:
    from microrts_agent.tournament import TournamentParser

    parser = TournamentParser("results/tournament.csv").parse()
    config = parser.config      # ParsedTournamentConfig
    games  = parser.games       # List[GameResult]
    parser.export_to_json()     # -> tournament_parsed.json
"""

import json
import zipfile
from pathlib import Path
from typing import Any, Optional

from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.progress import BarColumn, Progress, SpinnerColumn, TextColumn
from rich.table import Table

from .ranking.data import GameResult, ParsedTournamentConfig


class TournamentParser:
    """
    Tournament CSV Parser
    Reads and parses MicroRTS tournament CSV files
    """

    def __init__(self, csv_path: str):
        self.csv_path = Path(csv_path)
        self.tournament_dir = self.csv_path.parent
        self.config: Optional[ParsedTournamentConfig] = None
        self.games: list[GameResult] = []
        self.traces_available: bool = False
        self.trace_files: list[str] = []

    def parse(self):
        """
        Parse tournament data from CSV file.

        Supports both formats:
        - Unified: single block with all maps (new, maps annotated with maxGameLength=N)
        - Multi-block: one block per map, each with own header + summary (legacy)
        Both are merged into a single ParsedTournamentConfig + list of GameResults.
        """
        with open(self.csv_path) as f:
            lines = f.readlines()

        # Collect data across all blocks (unified or per-map)
        tournament_type = "tournaments.RoundRobinTournament"
        all_ais = []
        all_maps = []
        all_max_game_lengths = []
        config_data = {}
        block_map_offset = 0  # offset for map IDs in current block

        CONFIG_KEYS = {
            "iterations",
            "maxGameLength",
            "timeBudget",
            "iterationsBudget",
            "pregameAnalysisBudget",
            "preAnalysis",
            "fullObservability",
            "timeoutCheck",
            "runGC",
            "saveTraces",
            "saveGameLogs",
            "selfMatches",
        }

        i = 0
        while i < len(lines):
            line = lines[i].strip()

            # Block start
            if line.startswith("tournaments."):
                tournament_type = line
                i += 1
                continue

            # AIs section
            if line == "AIs":
                block_ais = []
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("maps"):
                    ai = lines[i].strip()
                    if ai:
                        block_ais.append(ai)
                    i += 1
                if not all_ais:
                    all_ais = block_ais
                continue

            # Maps section (may contain 1 map per block or all maps)
            if line == "maps":
                block_map_offset = len(all_maps)
                i += 1
                while i < len(lines) and lines[i].strip() and not lines[i].startswith("iterations"):
                    m = lines[i].strip()
                    if m:
                        parts = m.split("\t")
                        map_path = parts[0]
                        mgl = 3000
                        for part in parts[1:]:
                            if part.startswith("maxGameLength="):
                                mgl = int(part.split("=")[1])
                        all_maps.append(map_path)
                        all_max_game_lengths.append(mgl)
                    i += 1
                continue

            # Config parameters
            if line.split("\t")[0] in CONFIG_KEYS:
                parts = line.split("\t")
                key = parts[0]
                if len(parts) > 2:
                    config_data[key] = parts[1:]
                else:
                    config_data[key] = parts[1] if len(parts) > 1 else ""
                i += 1
                continue

            # Game result rows (after "iteration\tmap\t..." header)
            if line.startswith("iteration\t"):
                i += 1
                while i < len(lines) and not lines[i].startswith("Wins:"):
                    parts = lines[i].strip().split("\t")
                    if len(parts) >= 8 and parts[0].isdigit():
                        # map column is local to the block; add offset for global ID
                        # Columns 8-9 (ai1_time_ns, ai2_time_ns) are optional
                        # for backward compat with old CSVs
                        self.games.append(
                            GameResult(
                                iteration=int(parts[0]),
                                map_id=block_map_offset + int(parts[1]),
                                ai1_id=int(parts[2]),
                                ai2_id=int(parts[3]),
                                time=int(parts[4]),
                                winner=int(parts[5]),
                                crashed=int(parts[6]),
                                timedout=int(parts[7]),
                                ai1_time_ns=int(parts[8]) if len(parts) > 8 else 0,
                                ai2_time_ns=int(parts[9]) if len(parts) > 9 else 0,
                            )
                        )
                    i += 1
                continue

            # Skip everything else (Wins:, Ties:, summary matrices, blanks)
            i += 1

        # Build config
        pregame_budget = config_data.get("pregameAnalysisBudget", ["1000", "1000"])
        if isinstance(pregame_budget, str):
            pregame_budget = [pregame_budget, pregame_budget]

        if not all_max_game_lengths:
            global_mgl = int(config_data.get("maxGameLength", 3000))
            all_max_game_lengths = [global_mgl] * len(all_maps)

        self.config = ParsedTournamentConfig(
            tournament_type=tournament_type,
            ais=all_ais,
            maps=all_maps,
            iterations=int(config_data.get("iterations", 10)),
            max_game_lengths=all_max_game_lengths,
            time_budget=int(config_data.get("timeBudget", 100)),
            pregame_analysis_budget_ai1=int(pregame_budget[0]),
            pregame_analysis_budget_ai2=int(pregame_budget[1]),
            pre_analysis=config_data.get("preAnalysis", "true").lower() == "true",
            full_observability=config_data.get("fullObservability", "true").lower() == "true",
            timeout_check=config_data.get("timeoutCheck", "true").lower() == "true",
            run_gc=config_data.get("runGC", "false").lower() == "true",
            self_matches=config_data.get("selfMatches", "false").lower() == "true",
        )

        # Check for trace files
        self._check_traces()

        # Extract traces if available
        if self.traces_available:
            self._extract_all_traces()

        return self

    def _check_traces(self):
        """Check if trace files are available in traces/ directory"""
        traces_dir = self.tournament_dir / "traces"
        if traces_dir.exists() and traces_dir.is_dir():
            self.traces_available = True
            # Find all .zip trace files
            self.trace_files = sorted([f.name for f in traces_dir.glob("*.zip")])
        else:
            self.traces_available = False
            self.trace_files = []

    def _extract_all_traces(self):
        """Extract all trace ZIP files to traces_extracted/ directory"""
        traces_dir = self.tournament_dir / "traces"
        extracted_dir = self.tournament_dir / "traces_extracted"

        # Create traces_extracted directory
        extracted_dir.mkdir(exist_ok=True, parents=True)

        console = Console()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"[cyan]Extracting {len(self.trace_files)} trace files...",
                total=len(self.trace_files),
            )

            extracted_count = 0
            skipped_count = 0

            for trace_file in self.trace_files:
                trace_path = traces_dir / trace_file
                trace_name = trace_path.stem
                trace_output_dir = extracted_dir / trace_name

                # Skip if already extracted
                if (trace_output_dir / "game.xml").exists():
                    skipped_count += 1
                    progress.advance(task)
                    continue

                # Extract trace
                try:
                    trace_output_dir.mkdir(exist_ok=True, parents=True)
                    with zipfile.ZipFile(trace_path, "r") as zip_ref:
                        zip_ref.extractall(trace_output_dir)
                    extracted_count += 1
                except Exception as e:
                    console.print(f"[red]✗[/red] Error extracting {trace_file}: {e}")

                progress.advance(task)

        # Report results
        if extracted_count > 0:
            console.print(
                f"[green]✓[/green] Extracted {extracted_count} new trace files to [cyan]traces_extracted/[/cyan]"
            )
        if skipped_count > 0:
            console.print(f"[dim]  Skipped {skipped_count} already extracted traces[/dim]")

    def get_trace_path(
        self, ai1_id: int, ai2_id: int, map_id: int, iteration: int
    ) -> Optional[Path]:
        """
        Get the path to a specific game trace file
        Format: {AI1_ID}-vs-{AI2_ID}-{MapID}-{Iteration}.zip
        """
        if not self.traces_available:
            return None

        trace_name = f"{ai1_id}-vs-{ai2_id}-{map_id}-{iteration}.zip"
        trace_path = self.tournament_dir / "traces" / trace_name

        if trace_path.exists():
            return trace_path
        return None

    def get_ai_traces(self, ai_idx: int) -> list[Path]:
        """Get all trace files for games involving a specific AI"""
        if not self.traces_available:
            return []

        traces = []
        for game in self.games:
            if game.ai1_id == ai_idx or game.ai2_id == ai_idx:
                trace_path = self.get_trace_path(
                    game.ai1_id, game.ai2_id, game.map_id, game.iteration
                )
                if trace_path:
                    traces.append(trace_path)

        return traces

    def extract_trace(self, trace_path: Path, output_dir: Optional[Path] = None) -> Optional[Path]:
        """
        Extract a trace ZIP file to get the game.xml file

        Args:
            trace_path: Path to the .zip trace file
            output_dir: Directory to extract to (default: traces_extracted/)

        Returns:
            Path to the extracted game.xml file, or None if extraction fails
        """
        if not trace_path.exists():
            return None

        if output_dir is None:
            output_dir = self.tournament_dir / "traces_extracted"

        output_dir.mkdir(exist_ok=True)

        # Create subdirectory for this trace
        trace_name = trace_path.stem  # e.g., "0-vs-0-0-0"
        trace_output_dir = output_dir / trace_name
        trace_output_dir.mkdir(exist_ok=True)

        try:
            with zipfile.ZipFile(trace_path, "r") as zip_ref:
                zip_ref.extractall(trace_output_dir)

            xml_path = trace_output_dir / "game.xml"
            if xml_path.exists():
                return xml_path
        except Exception as e:
            print(f"Error extracting {trace_path}: {e}")

        return None

    def export_to_json(self, output_path: Optional[str] = None) -> Path:
        """
        Export parsed tournament data to JSON format
        Returns the path to the created JSON file
        """
        if output_path is None:
            output_path = self.csv_path.parent / f"{self.csv_path.stem}_parsed.json"
        else:
            output_path = Path(output_path)

        # Build JSON structure
        data = {
            "tournament": {
                "type": self.config.tournament_type,
                "ais": self.config.ais,
                "maps": self.config.maps,
                "iterations": self.config.iterations,
            },
            "configuration": {
                "maxGameLengths": self.config.max_game_lengths,
                "timeBudget": self.config.time_budget,
                "pregameAnalysisBudget": {
                    "ai1": self.config.pregame_analysis_budget_ai1,
                    "ai2": self.config.pregame_analysis_budget_ai2,
                },
                "preAnalysis": self.config.pre_analysis,
                "fullObservability": self.config.full_observability,
                "timeoutCheck": self.config.timeout_check,
                "runGC": self.config.run_gc,
            },
            "games": [
                {
                    "iteration": game.iteration,
                    "map": {
                        "id": game.map_id,
                        "name": self.config.maps[game.map_id]
                        if game.map_id < len(self.config.maps)
                        else f"Map{game.map_id}",
                    },
                    "players": {
                        "ai1": {
                            "id": game.ai1_id,
                            "name": self.config.ais[game.ai1_id]
                            if game.ai1_id < len(self.config.ais)
                            else f"AI{game.ai1_id}",
                        },
                        "ai2": {
                            "id": game.ai2_id,
                            "name": self.config.ais[game.ai2_id]
                            if game.ai2_id < len(self.config.ais)
                            else f"AI{game.ai2_id}",
                        },
                    },
                    "result": {
                        "time": game.time,
                        "winner": game.winner,
                        "crashed": game.crashed,
                        "timedout": game.timedout,
                        "ai1_time_ns": game.ai1_time_ns,
                        "ai2_time_ns": game.ai2_time_ns,
                    },
                }
                for game in self.games
            ],
            "traces": {
                "available": self.traces_available,
                "count": len(self.trace_files),
                "directory": str(self.tournament_dir / "traces") if self.traces_available else None,
            },
        }

        # Write JSON file with indentation
        with open(output_path, "w") as f:
            json.dump(data, f, indent=2)

        return output_path

    def get_statistics(self) -> dict[str, Any]:
        """Calculate tournament statistics"""
        if not self.games:
            return {}

        stats = {
            "total_games": len(self.games),
            "wins": [0] * len(self.config.ais),
            "ties": 0,
            "crashes": 0,
            "timeouts": 0,
            "avg_game_time": sum(g.time for g in self.games) / len(self.games),
            "max_game_time": max(g.time for g in self.games),
            "min_game_time": min(g.time for g in self.games),
        }

        for game in self.games:
            if game.winner == -1:
                stats["ties"] += 1
            elif game.winner == 0:
                stats["wins"][game.ai1_id] += 1
            elif game.winner == 1:
                stats["wins"][game.ai2_id] += 1

            if game.crashed > 0:
                stats["crashes"] += 1
            if game.timedout > 0:
                stats["timeouts"] += 1

        return stats


def print_summary(parser: "TournamentParser", console: Console = None):
    """Pretty-print tournament summary with Rich."""
    if console is None:
        console = Console()

    stats = parser.get_statistics()

    console.print()
    console.rule("[bold cyan] MicroRTS Tournament Parser[/bold cyan]", style="cyan")
    console.print()

    # Tournament Info Panel
    info_text = f"[bold]Type:[/bold] {parser.config.tournament_type}\n"
    info_text += f"[bold]Iterations:[/bold] {parser.config.iterations}\n"
    info_text += f"[bold]Total Games:[/bold] {stats['total_games']}"

    console.print(
        Panel(
            info_text,
            title="[bold yellow]Tournament Info[/bold yellow]",
            border_style="yellow",
            box=box.ROUNDED,
        )
    )
    console.print()

    # AIs Table
    ai_table = Table(
        title="AI Players",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold magenta",
        border_style="magenta",
    )
    ai_table.add_column("ID", justify="center", style="cyan", width=6)
    ai_table.add_column("Name", style="bright_white")
    ai_table.add_column("Wins", justify="center", style="green")
    ai_table.add_column("Win Rate", justify="center", style="yellow")

    for idx, ai in enumerate(parser.config.ais):
        wins = stats["wins"][idx]
        games_played = sum(1 for g in parser.games if g.ai1_id == idx or g.ai2_id == idx)
        win_rate = (wins / games_played * 100) if games_played > 0 else 0
        ai_table.add_row(str(idx), ai, str(wins), f"{win_rate:.1f}%")

    console.print(ai_table)
    console.print()

    # Maps Table
    map_table = Table(
        title="Maps",
        box=box.ROUNDED,
        show_header=True,
        header_style="bold green",
        border_style="green",
    )
    map_table.add_column("ID", justify="center", style="cyan", width=6)
    map_table.add_column("Name", style="bright_white", overflow="fold")
    map_table.add_column("Games", justify="center", style="yellow")

    for idx, map_name in enumerate(parser.config.maps):
        games_on_map = sum(1 for g in parser.games if g.map_id == idx)
        map_table.add_row(str(idx), map_name, str(games_on_map))

    console.print(map_table)
    console.print()

    # Configuration Panel
    config_text = f"[bold]Max Game Length:[/bold] {parser.config.max_game_lengths} frames\n"
    config_text += f"[bold]Time Budget:[/bold] {parser.config.time_budget}ms\n"
    config_text += (
        f"[bold]Pregame Analysis:[/bold] AI1: {parser.config.pregame_analysis_budget_ai1}ms, "
    )
    config_text += f"AI2: {parser.config.pregame_analysis_budget_ai2}ms\n"
    config_text += f"[bold]Pre-Analysis:[/bold] {parser.config.pre_analysis}\n"
    config_text += f"[bold]Full Observability:[/bold] {parser.config.full_observability}\n"
    config_text += f"[bold]Timeout Check:[/bold] {parser.config.timeout_check}\n"
    config_text += f"[bold]Run GC:[/bold] {parser.config.run_gc}"

    console.print(
        Panel(
            config_text,
            title="[bold blue] Configuration[/bold blue]",
            border_style="blue",
            box=box.ROUNDED,
        )
    )
    console.print()

    # Statistics Panel
    stats_text = f"[bold]Average Game Time:[/bold] {stats['avg_game_time']:.0f} frames\n"
    stats_text += (
        f"[bold]Min / Max Time:[/bold] {stats['min_game_time']} / {stats['max_game_time']} frames\n"
    )
    stats_text += f"[bold]Ties:[/bold] {stats['ties']}\n"
    stats_text += f"[bold]Crashes:[/bold] {stats['crashes']}\n"
    stats_text += f"[bold]Timeouts:[/bold] {stats['timeouts']}"

    console.print(
        Panel(
            stats_text,
            title="[bold red] Game Statistics[/bold red]",
            border_style="red",
            box=box.ROUNDED,
        )
    )
    console.print()

    # Traces Info
    if parser.traces_available:
        traces_text = "[bold green]Available[/bold green]\n"
        traces_text += f"[bold]Files:[/bold] {len(parser.trace_files)}\n"
        traces_text += f"[bold]Directory:[/bold] {parser.tournament_dir / 'traces'}"
        console.print(
            Panel(
                traces_text,
                title="[bold cyan] Trace Files[/bold cyan]",
                border_style="cyan",
                box=box.ROUNDED,
            )
        )
    else:
        console.print(
            Panel(
                "[bold red]No trace files found[/bold red]",
                title="[bold cyan] Trace Files[/bold cyan]",
                border_style="red",
                box=box.ROUNDED,
            )
        )
    console.print()

    console.rule("[bold cyan]Parser Complete[/bold cyan]", style="cyan")
    console.print()
