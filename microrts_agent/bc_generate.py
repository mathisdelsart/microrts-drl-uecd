"""
Generate behavior cloning data from bot-vs-bot games.

Records (observation, action_grid) tuples from the perspective of --bot.
Supports multiple opponents in a single run for diverse training data.

Usage:
    python bc_generate.py --bot RAISocketAI --opponents RAISocketAI CoacAI Mayari POWorkerRush POLightRush \
                          --games-per-opponent 1000 500 500 500 500
    python bc_generate.py --bot RAISocketAI  # defaults: RAISocketAI vs itself, 1000 games
"""

import argparse
import contextlib
import os
import shutil
import sys
import time

import numpy as np
from jpype.types import JInt

from lib.env_factory import JVM_ARGS, make_bot_env
from lib.envs.base_vec_env import suppress_java_output
from lib.mappings.ai import AI_MAPPING
from lib.paths import OUTPUTS_DIR


def parse_args():
    p = argparse.ArgumentParser(description="Generate BC replay data from bot games")
    p.add_argument(
        "--bot", type=str, default="RAISocketAI", help="Bot to imitate (default: RAISocketAI)"
    )
    p.add_argument(
        "--opponents",
        type=str,
        nargs="+",
        default=None,
        help="Opponent bot(s) (default: same as --bot)",
    )
    p.add_argument(
        "--games-per-opponent",
        type=int,
        nargs="+",
        default=None,
        help="Number of games per opponent (default: 1000 each)",
    )
    p.add_argument(
        "--map", type=str, default="maps/open_competition/basesWorkers16x16A.xml", help="Map path"
    )
    p.add_argument("--max-steps", type=int, default=4000, help="Max steps per game (default: 4000)")
    p.add_argument(
        "--output", type=str, default=None, help="Output .npz path (auto-generated if omitted)"
    )
    return p.parse_args()


def action_vector_to_grid(action_vectors, height, width, action_dims):
    """Convert Java int[numUnits][8] to grid (H*W, 7) matching RL agent format."""
    grid = np.zeros((height * width, len(action_dims)), dtype=np.int32)
    for av in action_vectors:
        cell_idx = int(av[0])
        if 0 <= cell_idx < height * width:
            grid[cell_idx] = av[1:8]
    return grid


DEFAULT_REWARD_WEIGHT = np.array(
    [10.0, 1.0, 1.0, 0.2, 0.2, 1.0, 4.0, 4.0, 4.0, 0.0], dtype=np.float32
)


def record_games(
    bot_factory,
    opp_factory,
    opp_name,
    num_games,
    map_path,
    max_steps,
    height,
    width,
    action_dims,
    num_planes,
    num_planes_prefix_sum,
    total_ch,
):
    """Record games between bot and opponent, return chunk file paths."""

    env = make_bot_env(
        ai1s=[bot_factory],
        ai2s=[opp_factory],
        map_paths=[map_path],
        partial_obs=False,
        max_steps=max_steps,
        jvm_args=JVM_ARGS,
        full_rewards=True,
    )
    client = env.vec_client.botClients[0]
    client.reset(JInt(0))

    def encode_standard(raw_obs):
        n_cells = height * width
        raw_full = np.array(raw_obs)  # (13, H, W) or (13, H*W)
        # Standard encoding uses only first 6 features (same as RL env)
        raw = raw_full[: len(num_planes)].reshape(len(num_planes), n_cells)
        raw = np.clip(raw, 0, np.array(num_planes).reshape(-1, 1) - 1)
        result = np.zeros((n_cells, total_ch), dtype=np.float32)
        for f_idx in range(len(num_planes)):
            offset = num_planes_prefix_sum[f_idx]
            result[np.arange(n_cells), offset + raw[f_idx]] = 1.0
        return result.reshape(height, width, total_ch)

    obs_list = []
    act_list = []
    rew_list = []
    chunk_files = []
    chunk_idx = 0
    flush_every = 50  # save to disk every N games to avoid OOM
    games_completed = 0
    total_steps = 0
    t0 = time.time()

    games_per_side_p0 = (num_games + 1) // 2
    games_per_side_p1 = num_games - games_per_side_p0

    for player, target_games in [(0, games_per_side_p0), (1, games_per_side_p1)]:
        if target_games == 0:
            continue

        if player == 0:
            current_client = client
        else:
            env_swapped = make_bot_env(
                ai1s=[bot_factory],
                ai2s=[opp_factory],
                map_paths=[map_path],
                partial_obs=False,
                max_steps=max_steps,
                jvm_args=JVM_ARGS,
                full_rewards=True,
            )
            current_client = env_swapped.vec_client.botClients[0]
            current_client.reset(JInt(1))

        games_this_side = 0

        while games_this_side < target_games:
            current_client.reset(JInt(player))

            game_steps = 0
            while True:
                raw_obs = current_client.getObservation(player)
                response = current_client.gameStep(JInt(player))
                game_steps += 1
                total_steps += 1

                action_vecs = current_client.getLastAction(0)
                action_vecs_np = (
                    np.array(action_vecs)
                    if len(action_vecs) > 0
                    else np.zeros((0, 8), dtype=np.int32)
                )

                # Weighted scalar reward (same as RL training: w^T @ r_t)
                raw_reward = np.array(response.reward, dtype=np.float32)  # (10,)
                reward = float(raw_reward @ DEFAULT_REWARD_WEIGHT)

                obs = encode_standard(raw_obs)
                action_grid = action_vector_to_grid(action_vecs_np, height, width, action_dims)

                obs_list.append(obs)
                act_list.append(action_grid)
                rew_list.append(reward)

                done = bool(response.done[0])
                if done or game_steps >= max_steps:
                    games_this_side += 1
                    games_completed += 1
                    elapsed = time.time() - t0
                    avg_spg = total_steps / games_completed
                    remaining = (
                        (num_games - games_completed) * avg_spg * (elapsed / total_steps)
                        if total_steps > 0
                        else 0
                    )
                    print(
                        f"\r    [{opp_name}] {games_completed}/{num_games} games | "
                        f"{total_steps} steps | "
                        f"{elapsed:.0f}s elapsed | "
                        f"~{remaining:.0f}s remaining",
                        end="",
                        flush=True,
                    )

                    # Flush to disk periodically to avoid OOM
                    if (
                        games_completed % flush_every == 0 or games_completed == num_games
                    ) and obs_list:
                        chunk_path = f"/tmp/bc_chunk_{opp_name}_{chunk_idx}.npz"
                        np.savez_compressed(
                            chunk_path,
                            obs=np.stack(obs_list),
                            actions=np.stack(act_list),
                            rewards=np.array(rew_list, dtype=np.float32),
                        )
                        chunk_files.append(chunk_path)
                        chunk_idx += 1
                        obs_list.clear()
                        act_list.clear()
                        rew_list.clear()
                    break

    # Flush any remaining data
    if obs_list:
        chunk_path = f"/tmp/bc_chunk_{opp_name}_{chunk_idx}.npz"
        np.savez_compressed(
            chunk_path,
            obs=np.stack(obs_list),
            actions=np.stack(act_list),
            rewards=np.array(rew_list, dtype=np.float32),
        )
        chunk_files.append(chunk_path)
        obs_list.clear()
        act_list.clear()
        rew_list.clear()

    with contextlib.suppress(Exception):
        env.close()

    return chunk_files


def main():
    args = parse_args()

    # Defaults
    if args.opponents is None:
        args.opponents = [args.bot]
    if args.games_per_opponent is None:
        args.games_per_opponent = [1000] * len(args.opponents)

    # Validate
    if len(args.games_per_opponent) == 1 and len(args.opponents) > 1:
        args.games_per_opponent = args.games_per_opponent * len(args.opponents)
    if len(args.games_per_opponent) != len(args.opponents):
        print(f"ERROR: --games-per-opponent must have 1 or {len(args.opponents)} values")
        sys.exit(1)

    for opp in args.opponents:
        if opp not in AI_MAPPING:
            print(f"Unknown opponent: {opp}. Valid: {sorted(AI_MAPPING.keys())}")
            sys.exit(1)
    if args.bot not in AI_MAPPING:
        print(f"Unknown bot: {args.bot}. Valid: {sorted(AI_MAPPING.keys())}")
        sys.exit(1)

    total_games = sum(args.games_per_opponent)
    map_name = os.path.splitext(os.path.basename(args.map))[0]

    # Output path
    if args.output is None:
        bc_dir = os.path.join(OUTPUTS_DIR, "bc_data")
        os.makedirs(bc_dir, exist_ok=True)
        opp_tag = (
            "_".join(args.opponents) if len(args.opponents) <= 3 else f"{len(args.opponents)}opps"
        )
        args.output = os.path.join(bc_dir, f"{args.bot}_vs_{opp_tag}_{map_name}_{total_games}g.npz")

    print("=" * 60)
    print("  BC Data Generation")
    print("=" * 60)
    print(f"  Bot:          {args.bot}")
    print(f"  Opponents:    {list(zip(args.opponents, args.games_per_opponent))}")
    print(f"  Total games:  {total_games}")
    print(f"  Map:          {args.map}")
    print(f"  Max steps:    {args.max_steps}")
    print(f"  Output:       {args.output}")
    print("=" * 60)

    # First, create a temp env just to get map dimensions and UTT
    bot_factory = AI_MAPPING[args.bot]
    temp_env = make_bot_env(
        ai1s=[bot_factory],
        ai2s=[bot_factory],
        map_paths=[args.map],
        partial_obs=False,
        max_steps=args.max_steps,
        jvm_args=JVM_ARGS,
    )
    suppress_java_output()
    temp_client = temp_env.vec_client.botClients[0]
    temp_client.reset(JInt(0))

    height = temp_client.gs.getPhysicalGameState().getHeight()
    width = temp_client.gs.getPhysicalGameState().getWidth()
    action_dims = [6, 4, 4, 4, 4, 7, 49]

    import json

    utt = json.loads(str(temp_client.sendUTT()))
    num_unit_types = len(utt["unitTypes"]) + 1
    num_planes = [5, 5, 3, num_unit_types, 6, 2]
    num_planes_prefix_sum = [0]
    for n in num_planes:
        num_planes_prefix_sum.append(num_planes_prefix_sum[-1] + n)
    total_ch = num_planes_prefix_sum[-1]

    print(f"  Map size:     {width}x{height}")
    print(f"  Obs channels: {total_ch}")
    print()

    with contextlib.suppress(Exception):
        temp_env.close()

    # Record games for each opponent — save per-opponent to avoid OOM
    output_dir = os.path.dirname(args.output)
    os.makedirs(output_dir, exist_ok=True)

    saved_files = []
    t0 = time.time()

    for opp_name, num_games in zip(args.opponents, args.games_per_opponent):
        print(f"  Recording {num_games} games vs {opp_name}...")
        opp_factory = AI_MAPPING[opp_name]

        chunk_files = record_games(
            bot_factory=bot_factory,
            opp_factory=opp_factory,
            opp_name=opp_name,
            num_games=num_games,
            map_path=args.map,
            max_steps=args.max_steps,
            height=height,
            width=width,
            action_dims=action_dims,
            num_planes=num_planes,
            num_planes_prefix_sum=num_planes_prefix_sum,
            total_ch=total_ch,
        )

        # Move chunks from /tmp to output dir with proper names
        for cf in chunk_files:
            dest = os.path.join(output_dir, os.path.basename(cf))
            shutil.move(cf, dest)
            saved_files.append(dest)

        # Count steps from chunk files without loading into RAM
        total_opp_steps = 0
        for sf in chunk_files:
            dest = os.path.join(output_dir, os.path.basename(sf))
            d = np.load(dest)
            total_opp_steps += len(d["obs"])
        print(f"\n    -> {total_opp_steps} steps in {len(chunk_files)} chunk(s)")

    elapsed = time.time() - t0
    print(f"\nDone! {len(saved_files)} chunk files saved to {output_dir}")
    print(f"  Total time: {elapsed:.0f}s")
    print("  Files:")
    for f in saved_files:
        print(f"    {f}")
    print("\nTo train BC+VF:")
    print(f"  python bc_train.py --data {' '.join(saved_files)} ...")


if __name__ == "__main__":
    main()
