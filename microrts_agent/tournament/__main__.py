"""
MicroRTS Tournament: Run, Parse & Visualize.

    microrts-agent tournament run <name> [--chunk 0 --total-chunks 10]
    microrts-agent tournament parse <name>
    microrts-agent tournament viz <name>
    microrts-agent tournament analyze <name>   # parse + viz

<name> resolves to tournament_configs/<name>.json (run)
or outputs/tournaments/<name>/ (parse/viz/analyze).
"""

import argparse
import os
import sys

from microrts_agent.paths import TOURNAMENT_CONFIGS_DIR
from microrts_agent.registries import ai as microrts_ai
from microrts_agent.tournament import (
    TournamentConfig,
    TournamentData,
    TournamentParser,
    TournamentVisualizer,
    ensure_tournament_csv,
    needs_uts_imass,
    resolve_results_dir,
    run_tournament,
    start_uts_imass_server,
    stop_uts_imass_server,
)

# ── CLI ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(
        description="MicroRTS Tournament: Run, Parse & Visualize",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command")

    # ---- run ----
    p_run = sub.add_parser(
        "run",
        help="Run a tournament from a JSON config",
        description="Run a full round-robin tournament. Iterations are vectorized.",
    )
    p_run.add_argument(
        "name", help="Config name (e.g. single_map, multi_map) or path to JSON config"
    )
    p_run.add_argument(
        "--chunk", type=int, default=-1, help="Chunk index for SLURM parallelization (0-based)"
    )
    p_run.add_argument("--total-chunks", type=int, default=1, help="Total number of chunks")

    # ---- parse ----
    p_parse = sub.add_parser(
        "parse",
        help="Parse CSV -> JSON (auto-merges chunks if needed)",
        description="Parse tournament results. If only chunk_*.csv exist, "
        "merges them into tournament.csv first.",
    )
    p_parse.add_argument(
        "results", help="Config name (e.g. single_map, multi_map) or full path to results dir"
    )

    # ---- viz ----
    p_viz = sub.add_parser(
        "viz",
        help="Generate PDF visualizations from parsed JSON",
        description="Generate all tournament visualizations from tournament_parsed.json.",
    )
    p_viz.add_argument("results", help="Config name or full path to results dir")

    # ---- analyze ----
    p_analyze = sub.add_parser(
        "analyze",
        help="Parse + Visualize (combined)",
        description="Run parse then viz in sequence. Auto-merges chunks if needed.",
    )
    p_analyze.add_argument("results", help="Config name or full path to results dir")

    args = parser.parse_args()

    if args.command == "run":
        _run_tournament(args)
    elif args.command == "parse":
        results_dir = resolve_results_dir(args.results)
        csv_path = ensure_tournament_csv(results_dir)
        _run_parser(csv_path)
    elif args.command == "viz":
        results_dir = resolve_results_dir(args.results)
        _run_visualizer(results_dir)
    elif args.command == "analyze":
        results_dir = resolve_results_dir(args.results)
        csv_path = ensure_tournament_csv(results_dir)
        _run_parser(csv_path)
        _run_visualizer(results_dir)
    else:
        parser.print_help()
        sys.exit(1)


def _run_parser(csv_path):
    """Parse a tournament CSV -> produces tournament_parsed.json."""
    parser = TournamentParser(str(csv_path))
    parser.parse()
    json_path = parser.export_to_json()
    print(f"  Parsed -> {json_path.name}")


def _run_visualizer(results_dir):
    """Generate PDF visualizations from parsed JSON."""
    from pathlib import Path

    from rich.console import Console

    json_path = Path(results_dir) / "tournament_parsed.json"
    if not json_path.exists():
        print(f"ERROR: {json_path} not found. Run 'parse' first.")
        sys.exit(1)

    console = Console()
    data = TournamentData(json_path)
    output_dir = Path(results_dir) / "visualizations"
    output_dir.mkdir(exist_ok=True, parents=True)

    visualizer = TournamentVisualizer(data, console)
    visualizer.generate_all(output_dir)
    print(f"  Visualizations -> {output_dir}/")


def _run_tournament(args):
    """Execute the 'run' subcommand: run a tournament."""
    # Resolve config path
    config_path = args.name
    if not os.path.exists(config_path):
        alt = TOURNAMENT_CONFIGS_DIR / config_path
        if not alt.suffix:
            alt = alt.with_suffix(".json")
        if alt.exists():
            config_path = str(alt)
        else:
            print(f"ERROR: Config not found: {config_path}")
            sys.exit(1)

    config = TournamentConfig.load(config_path)

    # Validate bot names
    for idx, name in enumerate(config.ai_names):
        if not config.ai_is_agent[idx] and name not in microrts_ai.AI_MAPPING:
            print(
                f"ERROR: Unknown bot '{name}'.\n"
                f"  Available: {sorted(microrts_ai.AI_MAPPING.keys())}"
            )
            sys.exit(1)

    # Auto-start external servers if needed
    uts_proc = None
    if needs_uts_imass(config):
        uts_proc = start_uts_imass_server()

    try:
        run_tournament(config, args.chunk, args.total_chunks)
    finally:
        stop_uts_imass_server(uts_proc)

    # Force exit (JVM daemon threads prevent clean Python shutdown)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
