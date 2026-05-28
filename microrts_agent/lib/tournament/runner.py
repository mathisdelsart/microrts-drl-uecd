"""
Tournament runner: orchestrates matchup execution and CSV output.
"""

import os
import time
from pathlib import Path

import torch

from lib.paths import TOURNAMENT_RESULTS_DIR

from .config import TournamentConfig, generate_matchups, group_matchups, select_chunk_groups
from .csv_io import write_csv_header, write_game_result
from .env_pool import EnvPool
from .game_loops import play_matchup_group


def fmt_time(seconds):
    """Format elapsed seconds as 'm:ss' or 'h:mm:ss'."""
    m, s = divmod(int(seconds), 60)
    h, m = divmod(m, 60)
    return f"{h}:{m:02d}:{s:02d}" if h > 0 else f"{m}:{s:02d}"


def run_tournament(config: TournamentConfig, chunk: int = -1, total_chunks: int = 1) -> str:
    """Run the tournament and return the path to the output CSV.

    Iterations of the same matchup are vectorized: played in parallel
    using MicroRTS vec envs with batched neural network inference.
    """
    device = torch.device("cpu")

    # Generate matchups -> group -> chunk
    all_matchups = generate_matchups(config)
    all_groups = group_matchups(all_matchups)

    if chunk >= 0:
        groups = select_chunk_groups(all_groups, config, chunk, total_chunks)
    else:
        groups = all_groups

    total_games = sum(len(g.iterations) for g in groups)
    if total_games == 0:
        print("No matchups to play.")
        return ""

    # Output directory
    output_dir = str(TOURNAMENT_RESULTS_DIR / config.config_name)
    os.makedirs(output_dir, exist_ok=True)

    # CSV path
    if chunk >= 0:
        csv_path = os.path.join(output_dir, f"chunk_{chunk}.csv")
    else:
        csv_path = os.path.join(output_dir, "tournament.csv")

    # Header
    print(f"\n{'=' * 65}")
    print("  MicroRTS Python Tournament")
    print(f"{'=' * 65}")
    print(f"  Config:      {config.config_name}")
    print(f"  AIs:         {', '.join(config.ai_names)}")
    print(f"  Maps:        {len(config.maps)}")
    print(f"  Iterations:  {config.iterations}")
    print(f"  Total games: {total_games}{f' (chunk {chunk}/{total_chunks})' if chunk >= 0 else ''}")
    print(f"  Matchup groups: {len(groups)} (vectorized x{config.iterations} iterations)")
    print(f"  Output:      {csv_path}")
    if config.save_traces:
        print(f"  Traces:      {output_dir}/traces/")
    if config.pre_analysis_budget > 0:
        print(f"  Pre-analysis: {config.pre_analysis_budget / 1000:.0f}s per bot/map")
    print(f"{'=' * 65}\n")

    game_log_path = os.path.join(output_dir, "game_logs.txt") if config.save_game_logs else None
    env_pool = EnvPool(config, device, game_log_path=game_log_path)
    t_start = time.time()

    # Win/loss counters
    wins = [0] * len(config.ais)
    losses = [0] * len(config.ais)
    draws = 0

    game_num = 0
    with open(csv_path, "w") as f:
        write_csv_header(f, config)

        for group in groups:
            t_group = time.time()
            batch_results = play_matchup_group(env_pool, group, config, output_dir)
            elapsed = time.time() - t_group

            for result in batch_results:
                write_game_result(f, result)
                game_num += 1

                w = result["winner"]
                if w == 0:
                    wins[group.ai1_idx] += 1
                    losses[group.ai2_idx] += 1
                elif w == 1:
                    wins[group.ai2_idx] += 1
                    losses[group.ai1_idx] += 1
                else:
                    draws += 1

            # Progress
            N = len(batch_results)
            pct = game_num * 100 / total_games
            ai1 = config.ai_names[group.ai1_idx]
            ai2 = config.ai_names[group.ai2_idx]
            map_name = Path(config.maps[group.map_idx]).stem
            batch_wins = sum(1 for r in batch_results if r["winner"] == 0)
            batch_losses = sum(1 for r in batch_results if r["winner"] == 1)
            batch_draws = N - batch_wins - batch_losses
            avg_steps = sum(r["time"] for r in batch_results) // N
            batch_tag = (
                f"{batch_wins}W/{batch_losses}L/{batch_draws}D"
                if N > 1
                else {0: "P0 WIN", 1: "P1 WIN", -1: "DRAW"}.get(batch_results[0]["winner"], "???")
            )
            print(
                f"  {game_num:>{len(str(total_games))}}/{total_games} "
                f"({pct:5.1f}%) | {ai1} vs {ai2} on {map_name} "
                f"[x{N}] | {batch_tag} (~{avg_steps} steps, "
                f"{fmt_time(elapsed)})"
            )

    assert game_num == total_games, f"Result count mismatch: expected {total_games}, got {game_num}"

    total_elapsed = time.time() - t_start
    env_pool.close_all()

    # Summary
    print(f"\n{'=' * 65}")
    print("  RESULTS")
    print(f"{'=' * 65}")
    print(f"  Games played: {total_games}")
    print(f"  Draws:        {draws}")
    print(f"  Time:         {fmt_time(total_elapsed)}")
    print(f"  CSV:          {csv_path}")
    print()
    for idx, name in enumerate(config.ai_names):
        total = wins[idx] + losses[idx]
        wr = wins[idx] / total * 100 if total > 0 else 0
        print(f"    {name:<25s}  {wins[idx]}W / {losses[idx]}L  ({wr:.0f}% WR)")
    print(f"{'=' * 65}\n")

    return csv_path
