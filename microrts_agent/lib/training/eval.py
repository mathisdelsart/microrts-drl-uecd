"""Periodic evaluation: play N games vs each bot, log WR to console/CSV/TB."""

import os

import numpy as np
import torch
from stable_baselines3.common.vec_env import VecMonitor  # type: ignore[import-not-found]

from lib.env_factory import JVM_ARGS, make_agent_env
from lib.mappings.ai import AI_MAPPING
from lib.wrapper_factory import apply_env_wrappers

# ── Env creation ──────────────────────────────────────────────────────────


def create_eval_envs(
    eval_bots,
    map_path,
    partial_obs,
    player,
    reward_weight,
    *,
    gamma=0.99,
    frame_stack=0,
    filtered_masks=False,
    reserved_obs=False,
    extended_obs=False,
    padded=False,
    max_height=0,
    max_width=0,
    max_steps=3000,
):
    """Create one parallel env with one sub-env per bot. Reused across evals
    (JPype JVM cannot restart). alternate_players=False so we control P0/P1."""
    envs = make_agent_env(
        num_bot_envs=len(eval_bots),
        num_selfplay_envs=0,
        ai2s=[AI_MAPPING[b] for b in eval_bots],
        map_paths=[map_path],
        player=player,
        partial_obs=partial_obs,
        max_steps=max_steps,
        reward_weight=reward_weight,
        jvm_args=JVM_ARGS,
        padded=padded,
        max_height=max_height,
        max_width=max_width,
        alternate_players=False,
        filtered_masks=filtered_masks,
        extended_obs=extended_obs,
    )
    envs = apply_env_wrappers(envs, gamma=gamma, frame_stack=frame_stack, reserved_obs=reserved_obs)
    envs = VecMonitor(envs)
    return envs, eval_bots


# ── Evaluation logic ──────────────────────────────────────────────────────


def _run_eval_one_position(agent, eval_envs, eval_bot_names, games_per_bot, device):
    """Play N games per bot on one position (P0 or P1). Returns per-bot W/L/D + stats."""
    agent.eval()

    bot_stats = {b: {"w": 0, "l": 0, "d": 0, "ret": [], "len": []} for b in eval_bot_names}
    bot_games = dict.fromkeys(eval_bot_names, 0)

    o = torch.as_tensor(eval_envs.reset()).to(device)

    # Keep stepping until every bot has played enough games
    while any(bot_games[b] < games_per_bot[b] for b in eval_bot_names):
        with torch.no_grad():
            masks = torch.tensor(eval_envs.get_action_mask()).to(device)
            action = agent.forward(o, invalid_action_masks=masks)["action"]

        o, _, _, infos = eval_envs.step(action.cpu().numpy().reshape(eval_envs.num_envs, -1))
        o = torch.as_tensor(o).to(device)

        # Record finished episodes
        for env_idx, info in enumerate(infos):
            if "episode" in info:
                bot = eval_bot_names[env_idx]
                if bot_games[bot] >= games_per_bot[bot]:
                    continue  # already have enough games for this bot
                bot_games[bot] += 1
                wl = info["microrts_stats"]["WinDrawLossRewardFunction"]
                if wl > 0:
                    bot_stats[bot]["w"] += 1
                elif wl < 0:
                    bot_stats[bot]["l"] += 1
                else:
                    bot_stats[bot]["d"] += 1
                bot_stats[bot]["ret"].append(info["episode"]["r"])
                bot_stats[bot]["len"].append(info["episode"]["l"])

    return bot_stats


def run_evaluation(agent, eval_envs_p0, eval_envs_p1, eval_bot_names, games_per_bot, device):
    """Play games as P0 and P1, merge results. Returns per-bot stats dict.

    games_per_bot: {bot: total_games} — split evenly between P0 and P1.
    """
    games_p0 = {b: g // 2 for b, g in games_per_bot.items()}
    games_p1 = {b: g - games_p0[b] for b, g in games_per_bot.items()}

    stats_p0 = _run_eval_one_position(agent, eval_envs_p0, eval_bot_names, games_p0, device)
    stats_p1 = _run_eval_one_position(agent, eval_envs_p1, eval_bot_names, games_p1, device)

    agent.train()  # back to training mode

    results = {}
    for bot in eval_bot_names:
        s0, s1 = stats_p0[bot], stats_p1[bot]
        g0, g1 = games_p0[bot], games_p1[bot]
        total_g = g0 + g1
        total_w = s0["w"] + s1["w"]
        total_l = s0["l"] + s1["l"]
        total_d = s0["d"] + s1["d"]
        all_ret = s0["ret"] + s1["ret"]
        all_len = s0["len"] + s1["len"]
        results[bot] = {
            "wins": total_w,
            "losses": total_l,
            "draws": total_d,
            "games": total_g,
            "win_rate": total_w / total_g if total_g > 0 else 0,
            "avg_return": float(np.mean(all_ret)) if all_ret else 0,
            "avg_length": float(np.mean(all_len)) if all_len else 0,
            "wins_p0": s0["w"],
            "games_p0": g0,
            "win_rate_p0": s0["w"] / g0 if g0 > 0 else 0,
            "wins_p1": s1["w"],
            "games_p1": g1,
            "win_rate_p1": s1["w"] / g1 if g1 > 0 else 0,
        }
    return results


# ── Logging ───────────────────────────────────────────────────────────────


def log_eval_results(eval_results, writer, global_step, eval_csv_path):
    """Print table + write CSV + log TB for single-map eval. Returns overall WR."""
    total_w, total_l, total_d, total_g = 0, 0, 0, 0
    all_returns, all_lengths = [], []

    for bot_name, stats in eval_results.items():
        w, losses, d, g = stats["wins"], stats["losses"], stats["draws"], stats["games"]
        total_w += w
        total_l += losses
        total_d += d
        total_g += g
        all_returns.append(stats["avg_return"])
        all_lengths.append(stats["avg_length"])

        # Console table row
        print(
            f"  {bot_name:<18s} {g:>3d} {w:>3d} {losses:>3d} {d:>3d} "
            f"{stats['win_rate'] * 100:>6.1f}% "
            f"{stats['win_rate_p0'] * 100:>5.0f}% "
            f"{stats['win_rate_p1'] * 100:>5.0f}% "
            f"{stats['avg_return']:>9.2f} {stats['avg_length']:>8.0f}"
        )

        # TensorBoard
        writer.add_scalar(f"eval/{bot_name}/win_rate", stats["win_rate"], global_step)
        writer.add_scalar(f"eval/{bot_name}/win_rate_p0", stats["win_rate_p0"], global_step)
        writer.add_scalar(f"eval/{bot_name}/win_rate_p1", stats["win_rate_p1"], global_step)
        writer.add_scalar(f"eval/{bot_name}/avg_return", stats["avg_return"], global_step)
        writer.add_scalar(f"eval/{bot_name}/avg_length", stats["avg_length"], global_step)

        # CSV
        with open(eval_csv_path, "a") as f:
            f.write(
                f"{global_step},{bot_name},{w},{losses},{d},{g},"
                f"{stats['win_rate']:.4f},"
                f"{stats['wins_p0']},{stats['games_p0']},{stats['win_rate_p0']:.4f},"
                f"{stats['wins_p1']},{stats['games_p1']},{stats['win_rate_p1']:.4f},"
                f"{stats['avg_return']:.4f},{stats['avg_length']:.1f}\n"
            )

    # Overall summary
    overall_wr = total_w / total_g if total_g > 0 else 0
    print(f"  {'-' * 70}")
    if all_returns:
        print(
            f"  {'OVERALL':<18s} {total_g:>3d} {total_w:>3d} {total_l:>3d} {total_d:>3d} "
            f"{overall_wr * 100:>6.1f}% {' ':>6s} {' ':>6s} "
            f"{np.mean(all_returns):>9.2f} {np.mean(all_lengths):>8.0f}"
        )
    else:
        print(
            f"  {'OVERALL':<18s} {total_g:>3d} {total_w:>3d} {total_l:>3d} {total_d:>3d} "
            f"{overall_wr * 100:>6.1f}%"
        )

    writer.add_scalar("eval/overall_win_rate", overall_wr, global_step)
    return overall_wr


# ── Multi-map evaluation ─────────────────────────────────────────────────


def _unwrap_padded_env(env):
    """Find PaddedMicroRTSRLVecEnv in wrapper chain."""
    from lib.envs.padded_rl_vec_env import PaddedMicroRTSRLVecEnv

    cur = env
    while cur is not None:
        if isinstance(cur, PaddedMicroRTSRLVecEnv):
            return cur
        cur = getattr(cur, "venv", None)
    raise RuntimeError("PaddedMicroRTSRLVecEnv not found in wrapper chain")


def run_multimap_evaluation(
    agent, eval_envs_p0, eval_envs_p1, eval_bot_names, map_pool, games_per_bot_per_map, device
):
    """Evaluate on each map, return (per_map_results, aggregate_results).

    Switches the padded eval envs to each map, plays games, then aggregates.
    """
    per_map_results = {}
    games_per_bot = dict.fromkeys(eval_bot_names, games_per_bot_per_map)

    for map_path in map_pool:
        _unwrap_padded_env(eval_envs_p0).switch_map(map_path)
        _unwrap_padded_env(eval_envs_p1).switch_map(map_path)
        per_map_results[map_path] = run_evaluation(
            agent, eval_envs_p0, eval_envs_p1, eval_bot_names, games_per_bot, device
        )

    # Sum stats across all maps
    aggregate = {}
    for bot in eval_bot_names:
        total = {
            "wins": 0,
            "losses": 0,
            "draws": 0,
            "games": 0,
            "wins_p0": 0,
            "games_p0": 0,
            "wins_p1": 0,
            "games_p1": 0,
            "ret": [],
            "len": [],
        }
        for mp in map_pool:
            s = per_map_results[mp][bot]
            total["wins"] += s["wins"]
            total["losses"] += s["losses"]
            total["draws"] += s["draws"]
            total["games"] += s["games"]
            total["wins_p0"] += s["wins_p0"]
            total["games_p0"] += s["games_p0"]
            total["wins_p1"] += s["wins_p1"]
            total["games_p1"] += s["games_p1"]
            total["ret"].append(s["avg_return"])
            total["len"].append(s["avg_length"])
        g = total["games"]
        aggregate[bot] = {
            "wins": total["wins"],
            "losses": total["losses"],
            "draws": total["draws"],
            "games": g,
            "win_rate": total["wins"] / g if g > 0 else 0,
            "avg_return": float(np.mean(total["ret"])) if total["ret"] else 0,
            "avg_length": float(np.mean(total["len"])) if total["len"] else 0,
            "wins_p0": total["wins_p0"],
            "games_p0": total["games_p0"],
            "win_rate_p0": total["wins_p0"] / total["games_p0"] if total["games_p0"] > 0 else 0,
            "wins_p1": total["wins_p1"],
            "games_p1": total["games_p1"],
            "win_rate_p1": total["wins_p1"] / total["games_p1"] if total["games_p1"] > 0 else 0,
        }

    return per_map_results, aggregate


def log_multimap_eval_results(
    per_map_results, aggregate_results, map_pool, writer, global_step, eval_csv_path
):
    """Print per-map tables + aggregate, write CSV + TB."""

    # Per-map tables
    for map_path in map_pool:
        short = os.path.splitext(os.path.basename(map_path))[0]
        results = per_map_results[map_path]
        map_w, map_g = 0, 0

        print(f"\n  -- {short} --")
        for bot_name, stats in results.items():
            w, losses, d, g = stats["wins"], stats["losses"], stats["draws"], stats["games"]
            map_w += w
            map_g += g
            print(
                f"  {bot_name:<18s} {g:>3d} {w:>3d} {losses:>3d} {d:>3d} "
                f"{stats['win_rate'] * 100:>6.1f}%"
            )
            writer.add_scalar(
                f"eval_map/{short}/{bot_name}/win_rate", stats["win_rate"], global_step
            )
            with open(eval_csv_path, "a") as f:
                f.write(
                    f"{global_step},{short},{bot_name},{w},{losses},{d},{g},"
                    f"{stats['win_rate']:.4f},"
                    f"{stats['wins_p0']},{stats['games_p0']},{stats['win_rate_p0']:.4f},"
                    f"{stats['wins_p1']},{stats['games_p1']},{stats['win_rate_p1']:.4f},"
                    f"{stats['avg_return']:.4f},{stats['avg_length']:.1f}\n"
                )

        map_wr = map_w / map_g if map_g > 0 else 0
        writer.add_scalar(f"eval_map/{short}/overall_win_rate", map_wr, global_step)

    # Aggregate across all maps
    print("\n  -- AGGREGATE (all maps) --")
    total_w, total_g = 0, 0
    for bot_name, stats in aggregate_results.items():
        w, g = stats["wins"], stats["games"]
        total_w += w
        total_g += g
        print(
            f"  {bot_name:<18s} {g:>3d} {w:>3d} "
            f"{stats['losses']:>3d} {stats['draws']:>3d} "
            f"{stats['win_rate'] * 100:>6.1f}%"
        )
        writer.add_scalar(f"eval/{bot_name}/win_rate", stats["win_rate"], global_step)

    total_l = sum(s["losses"] for s in aggregate_results.values())
    total_d = sum(s["draws"] for s in aggregate_results.values())
    overall_wr = total_w / total_g if total_g > 0 else 0
    print(f"  {'-' * 50}")
    print(
        f"  {'OVERALL':<18s} {total_g:>3d} {total_w:>3d} "
        f"{total_l:>3d} {total_d:>3d} "
        f"{overall_wr * 100:>6.1f}%"
    )
    writer.add_scalar("eval/overall_win_rate", overall_wr, global_step)
    return overall_wr
