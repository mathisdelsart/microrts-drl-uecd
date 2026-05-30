"""
Tournament CSV I/O: write results, merge chunks, resolve results directories.
"""

import sys
from pathlib import Path

from microrts_agent.paths import TOURNAMENTS_DIR


def write_csv_header(f, config):
    """Write CSV header in the exact format TournamentParser expects."""
    f.write("tournaments.RoundRobinTournament\n")
    f.write("AIs\n")
    for name in config.ai_names:
        f.write(f"\t{name}\n")
    f.write("maps\n")
    for i, map_path in enumerate(config.maps):
        mgl = config.max_game_lengths[i]
        f.write(f"\t{map_path}\tmaxGameLength={mgl}\n")
    f.write(f"iterations\t{config.iterations}\n")
    f.write(f"timeBudget\t{config.time_budget}\n")
    f.write(f"iterationsBudget\t{config.iterations_budget}\n")
    pab = config.pre_analysis_budget
    f.write(f"pregameAnalysisBudget\t{pab}\t{pab}\n")
    f.write(f"preAnalysis\t{str(pab > 0).lower()}\n")
    f.write(f"fullObservability\t{str(config.full_observability).lower()}\n")
    f.write(f"timeoutCheck\t{str(config.timeout_check).lower()}\n")
    f.write(f"runGC\t{str(config.run_gc).lower()}\n")
    f.write("iteration\tmap\tai1\tai2\ttime\twinner\tcrashed\ttimedout\tai1_time_ns\tai2_time_ns\n")


def write_game_result(f, result: dict):
    """Write one game result line (tab-separated)."""
    f.write(
        f"{result['iteration']}\t{result['map_idx']}\t{result['ai1_idx']}\t"
        f"{result['ai2_idx']}\t{result['time']}\t{result['winner']}\t"
        f"{result['crashed']}\t{result['timedout']}\t"
        f"{result.get('ai1_time_ns', 0)}\t{result.get('ai2_time_ns', 0)}\n"
    )
    f.flush()


def resolve_results_dir(name_or_path: str) -> Path:
    """Resolve a config name or directory path to a results directory."""
    p = Path(name_or_path)
    if p.is_dir():
        return p.resolve()
    candidate = TOURNAMENTS_DIR / name_or_path
    if candidate.is_dir():
        return candidate.resolve()
    print(f"ERROR: Results directory not found: {name_or_path}")
    print(f"  Tried: {p.absolute()}")
    print(f"  Tried: {candidate}")
    sys.exit(1)


def ensure_tournament_csv(results_dir: Path) -> Path:
    """Return path to tournament.csv, merging chunk files if needed."""
    csv_path = results_dir / "tournament.csv"
    if csv_path.exists():
        print(f"  Found {csv_path.name}")
        return csv_path

    # Auto-merge chunks (now under results_dir/chunks/)
    chunks_dir = results_dir / "chunks"
    chunks = sorted(chunks_dir.glob("chunk_*.csv")) if chunks_dir.is_dir() else []
    if not chunks:
        print(f"ERROR: No tournament.csv or chunks/chunk_*.csv in {results_dir}")
        sys.exit(1)

    print(f"  No tournament.csv -- merging {len(chunks)} chunks...")

    header_lines = []
    with open(chunks[0]) as f:
        for line in f:
            header_lines.append(line)
            if line.startswith("iteration\t"):
                break

    data_lines = []
    for chunk_file in chunks:
        count = 0
        with open(chunk_file) as f:
            for line in f:
                stripped = line.strip()
                if stripped and stripped[0].isdigit() and stripped.count("\t") >= 7:
                    data_lines.append(line if line.endswith("\n") else line + "\n")
                    count += 1
        print(f"    {chunk_file.name}: {count} games")

    with open(csv_path, "w") as f:
        for line in header_lines:
            f.write(line)
        for line in data_lines:
            f.write(line)

    print(f"  Merged {len(data_lines)} games -> {csv_path.name}")
    return csv_path
