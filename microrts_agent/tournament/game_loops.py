"""
Tournament game loop implementations (vectorized across iterations).

Each function plays all iterations of a matchup in parallel using
MicroRTS vectorized environments with batched neural network inference.
"""

import time

import torch

from .config import MatchupGroup, TournamentConfig
from .env_pool import EnvPool, save_client_trace


def play_bot_vs_bot_batch(
    env_pool: EnvPool, group: MatchupGroup, config: TournamentConfig, output_dir: str
) -> list[dict]:
    """Play all iterations of a bot-vs-bot matchup in parallel.

    Each JNIBotClient is stepped independently (manual loop), allowing
    per-client exception handling and per-client trace saving.
    """
    from jpype.types import JInt

    N = len(group.iterations)
    map_path = config.maps[group.map_idx]
    max_steps = config.max_game_lengths[group.map_idx]
    ai1_name = config.ai_names[group.ai1_idx]
    ai2_name = config.ai_names[group.ai2_idx]

    env = env_pool.get_bot_env(map_path, max_steps)

    # Setup first N clients
    # (Time budget is applied post-reset by configure_cloned_ai, since reset()
    # internally clones the AI; setting timeBudgetMs on the wrapper here would
    # be discarded.)
    for i in range(N):
        client = env.vec_client.botClients[i]
        client.ai1 = env_pool.get_bot_ai(ai1_name, map_path, env.real_utt)
        client.ai2 = env_pool.get_bot_ai(ai2_name, map_path, env.real_utt)
        if config.save_traces:
            client.startTrace()
        client.reset(JInt(0))
        client.ai1 = env_pool.configure_cloned_ai(client.ai1)
        client.ai2 = env_pool.configure_cloned_ai(client.ai2)
        env_pool.run_preanalysis(client.ai1, ai1_name, map_path, client.gs)
        env_pool.run_preanalysis(client.ai2, ai2_name, map_path, client.gs)

    # Game loop
    active = [True] * N
    steps = [0] * N
    crashed = [-1] * N
    timedout = [-1] * N
    last_responses = [None] * N

    while any(active):
        if config.run_gc:
            env_pool.run_java_gc()

        for i in range(N):
            if not active[i]:
                continue
            client = env.vec_client.botClients[i]
            try:
                response = client.gameStep(JInt(0))
                last_responses[i] = response
            except Exception as e:
                print(f"  WARNING: bot_vs_bot game {i} crashed: {e}")
                crashed[i] = 0
                active[i] = False
                if config.save_traces:
                    save_client_trace(
                        client,
                        group.ai1_idx,
                        group.ai2_idx,
                        group.map_idx,
                        group.iterations[i],
                        output_dir,
                    )
                continue

            steps[i] += 1

            if config.timeout_check:
                if int(client.ai1TimeoutCount) > 0:
                    timedout[i] = 0
                    active[i] = False
                    if config.save_traces:
                        save_client_trace(
                            client,
                            group.ai1_idx,
                            group.ai2_idx,
                            group.map_idx,
                            group.iterations[i],
                            output_dir,
                        )
                    continue
                if int(client.ai2TimeoutCount) > 0:
                    timedout[i] = 1
                    active[i] = False
                    if config.save_traces:
                        save_client_trace(
                            client,
                            group.ai1_idx,
                            group.ai2_idx,
                            group.map_idx,
                            group.iterations[i],
                            output_dir,
                        )
                    continue

            d = [bool(x) for x in response.done]
            if d[0] or steps[i] >= max_steps:
                active[i] = False
                if config.save_traces:
                    save_client_trace(
                        client,
                        group.ai1_idx,
                        group.ai2_idx,
                        group.map_idx,
                        group.iterations[i],
                        output_dir,
                    )

    # Build results
    results = []
    for i in range(N):
        client = env.vec_client.botClients[i]
        if crashed[i] != -1:
            winner = 1 - crashed[i]
        elif timedout[i] != -1:
            winner = 1 - timedout[i]
        else:
            wl = float(last_responses[i].reward[0])
            if wl > 0:
                winner = 0
            elif wl < 0:
                winner = 1
            else:
                winner = -1

        game_time = int(client.gs.getTime())
        results.append(
            {
                "iteration": group.iterations[i],
                "map_idx": group.map_idx,
                "ai1_idx": group.ai1_idx,
                "ai2_idx": group.ai2_idx,
                "time": game_time,
                "winner": winner,
                "crashed": crashed[i],
                "timedout": timedout[i],
                "ai1_timeouts": int(client.ai1TimeoutCount),
                "ai2_timeouts": int(client.ai2TimeoutCount),
                "ai1_time_ns": int(client.ai1TotalTimeNs),
                "ai2_time_ns": int(client.ai2TotalTimeNs),
            }
        )
    return results


def play_agent_vs_bot_batch(
    env_pool: EnvPool, group: MatchupGroup, config: TournamentConfig, player: int, output_dir: str
) -> list[dict]:
    """Play all iterations of an agent-vs-bot matchup in parallel.

    Uses batched neural network inference (predict_batch) for all N games.
    """
    N = len(group.iterations)
    map_path = config.maps[group.map_idx]
    max_steps = config.max_game_lengths[group.map_idx]

    if player == 0:
        agent_idx = group.ai1_idx
        bot_idx = group.ai2_idx
    else:
        agent_idx = group.ai2_idx
        bot_idx = group.ai1_idx

    run_dir = config.ai_run_dirs[agent_idx]
    bot_name = config.ai_names[bot_idx]

    agent, agent_cfg = env_pool.get_agent(run_dir)
    device = env_pool.device
    env = env_pool.get_agent_env(map_path, max_steps, player, agent_cfg=agent_cfg)
    num_slots = env.num_envs

    # Unwrap to access vec_client through wrapper chain
    base_env = env_pool._get_base_env(env)

    # Setup first N clients with real bot
    # (Time budget is applied post-reset by configure_cloned_ai below; setting
    # it here on the pre-clone wrapper would be discarded.)
    for i in range(N):
        client = base_env.vec_client.clients[i]
        client.ai2 = env_pool.get_bot_ai(bot_name, map_path, base_env.real_utt)
        if config.save_traces:
            client.startTrace()

    # Reset all envs (batched)
    raw_obs = env.reset()
    obs = torch.as_tensor(raw_obs).to(device)

    # Configure first N bots (after reset which clones)
    for i in range(N):
        client = base_env.vec_client.clients[i]
        client.ai2 = env_pool.configure_cloned_ai(client.ai2)
        env_pool.run_preanalysis(client.ai2, bot_name, map_path, client.gs)

    active = [True] * N
    steps = [0] * N
    result_data = [None] * N
    agent_time_ns = [0] * N
    bot_time_ns = [0] * N

    while any(active):
        if config.run_gc:
            env_pool.run_java_gc()

        try:
            with torch.no_grad():
                masks = torch.as_tensor(env.get_action_mask()).to(device)
                t0 = time.perf_counter_ns()
                actions = agent.predict_batch(obs, masks)
                inference_ns = time.perf_counter_ns() - t0
            # Divide the batched-inference cost by the initial env count N (not the
            # live count) so each game's per-step share stays constant over the match.
            # Dividing by n_active would charge the final survivor the full batch cost,
            # even though GPU work doesn't shrink as games end.
            per_game_ns = inference_ns // N
            for i in range(N):
                if active[i]:
                    agent_time_ns[i] += per_game_ns
            raw_obs, _, done_arr, infos = env.step(actions.cpu().numpy().reshape(num_slots, -1))
            obs = torch.as_tensor(raw_obs).to(device)
        except Exception as e:
            print(f"  WARNING: agent_vs_bot batch crashed: {e}")
            for i in range(N):
                if active[i]:
                    result_data[i] = {
                        "winner": 1 - player,  # opponent (bot) wins
                        "crashed": player,  # agent side crashed
                        "timedout": -1,
                        "time": steps[i],
                        "bot_timeouts": 0,
                    }
                    active[i] = False
                    if config.save_traces:
                        save_client_trace(
                            base_env.vec_client.clients[i],
                            group.ai1_idx,
                            group.ai2_idx,
                            group.map_idx,
                            group.iterations[i],
                            output_dir,
                        )
            break

        for i in range(N):
            if not active[i]:
                continue
            steps[i] += 1
            client = base_env.vec_client.clients[i]

            if config.timeout_check and int(client.ai2TimeoutCount) > 0:
                bot_to = 1 if player == 0 else 0
                result_data[i] = {
                    "winner": 1 - bot_to,
                    "crashed": -1,
                    "timedout": bot_to,
                    "time": steps[i],
                    "bot_timeouts": int(client.ai2TimeoutCount),
                }
                active[i] = False
                if config.save_traces:
                    save_client_trace(
                        client,
                        group.ai1_idx,
                        group.ai2_idx,
                        group.map_idx,
                        group.iterations[i],
                        output_dir,
                    )
                    client.collectTrace = False
                continue

            if done_arr[i]:
                wl = infos[i]["raw_rewards"][0]
                if player == 1:
                    wl = -wl
                if wl > 0:
                    winner = 0
                elif wl < 0:
                    winner = 1
                else:
                    winner = -1
                result_data[i] = {
                    "winner": winner,
                    "crashed": -1,
                    "timedout": -1,
                    "time": steps[i],
                    "bot_timeouts": 0,
                }
                active[i] = False
                if config.save_traces:
                    save_client_trace(
                        client,
                        group.ai1_idx,
                        group.ai2_idx,
                        group.map_idx,
                        group.iterations[i],
                        output_dir,
                    )
                    client.collectTrace = False
            else:
                bot_time_ns[i] = int(client.ai2TotalTimeNs)

    # Build result dicts
    results = []
    for i in range(N):
        d = result_data[i]
        bot_to = d.get("bot_timeouts", 0)
        if player == 0:
            ai1_to, ai2_to = 0, bot_to
            ai1_ns = agent_time_ns[i]
            ai2_ns = bot_time_ns[i]
        else:
            ai1_to, ai2_to = bot_to, 0
            ai1_ns = bot_time_ns[i]
            ai2_ns = agent_time_ns[i]
        results.append(
            {
                "iteration": group.iterations[i],
                "map_idx": group.map_idx,
                "ai1_idx": group.ai1_idx,
                "ai2_idx": group.ai2_idx,
                "time": d["time"],
                "winner": d["winner"],
                "crashed": d["crashed"],
                "timedout": d["timedout"],
                "ai1_timeouts": ai1_to,
                "ai2_timeouts": ai2_to,
                "ai1_time_ns": ai1_ns,
                "ai2_time_ns": ai2_ns,
            }
        )
    return results


def _make_obs_adapter(config, num_envs):
    """Create a per-model obs adapter (shared implementation in microrts_agent/obs_adapter.py)."""
    from microrts_agent.obs_adapter import ObsAdapter

    return ObsAdapter(config, num_envs)


def _get_reserved_planes_raw_tournament(base_env):
    """Compute reserved-position planes (delegates to microrts_agent/obs_adapter.py)."""
    from microrts_agent.obs_adapter import get_reserved_planes

    return get_reserved_planes(base_env)


def _process_obs_agent_vs_agent(
    base_env, raw_obs, p0_adapter, p1_adapter, device, env_extended_obs
):
    """Delegates to shared implementation in microrts_agent/obs_adapter.py."""
    from microrts_agent.obs_adapter import process_obs_rl_vs_rl

    return process_obs_rl_vs_rl(
        base_env,
        raw_obs,
        p0_adapter,
        p1_adapter,
        device,
        base_env=base_env,
        env_extended_obs=env_extended_obs,
    )


def play_agent_vs_agent_batch(
    env_pool: EnvPool, group: MatchupGroup, config: TournamentConfig, output_dir: str
) -> list[dict]:
    """Play all iterations of an agent-vs-agent matchup in parallel.

    Uses a self-play env where each game occupies 2 slots (even=P0, odd=P1).
    Supports mismatched observation configs (extended_obs, frame_stack,
    reserved_obs) between the two agents via per-model adapters.
    """
    N = len(group.iterations)
    map_path = config.maps[group.map_idx]
    max_steps = config.max_game_lengths[group.map_idx]

    run_dir1 = config.ai_run_dirs[group.ai1_idx]
    run_dir2 = config.ai_run_dirs[group.ai2_idx]

    agent1, cfg1 = env_pool.get_agent(run_dir1)
    agent2, cfg2 = env_pool.get_agent(run_dir2)
    po1 = cfg1.get("partial_obs", False)
    po2 = cfg2.get("partial_obs", False)
    if po1 != po2:
        n1 = config.ai_names[group.ai1_idx]
        n2 = config.ai_names[group.ai2_idx]
        print(
            f"  WARNING: partial_obs mismatch -- {n1} ({po1}) vs {n2} ({po2}). "
            f"Using partial_obs={po1} (from {n1})."
        )
    device = env_pool.device

    # Detect obs config mismatches
    ext1 = cfg1.get("extended_obs", False)
    ext2 = cfg2.get("extended_obs", False)
    fs1 = cfg1.get("frame_stack", 0)
    fs2 = cfg2.get("frame_stack", 0)
    ro1 = cfg1.get("reserved_obs", False)
    ro2 = cfg2.get("reserved_obs", False)

    needs_adapters = ext1 != ext2 or fs1 != fs2 or fs1 > 1 or fs2 > 1 or ro1 or ro2

    env = env_pool.get_selfplay_env(map_path, max_steps, agent_cfg=cfg1, agent_cfg2=cfg2)
    base_env = env_pool._get_base_env(env)
    env_extended_obs = base_env.extended_obs
    num_slots = env.num_envs

    # Create per-model adapters
    p0_adapter = _make_obs_adapter(cfg1, N)
    p1_adapter = _make_obs_adapter(cfg2, N)

    # Reset all envs
    raw_obs = env.reset()
    if needs_adapters:
        obs_p0, obs_p1 = _process_obs_agent_vs_agent(
            base_env, raw_obs, p0_adapter, p1_adapter, device, env_extended_obs
        )
    else:
        obs_p0 = torch.as_tensor(raw_obs[0::2]).to(device)
        obs_p1 = torch.as_tensor(raw_obs[1::2]).to(device)

    active_games = [True] * N
    steps = [0] * N
    result_data = [None] * N
    agent1_time_ns = [0] * N
    agent2_time_ns = [0] * N

    while any(active_games):
        if config.run_gc:
            env_pool.run_java_gc()

        try:
            with torch.no_grad():
                masks = torch.as_tensor(env.get_action_mask()).to(device)

                masks_p0 = masks[0::2]
                masks_p1 = masks[1::2]

                t0 = time.perf_counter_ns()
                actions_p0 = agent1.predict_batch(obs_p0, masks_p0)
                ns1 = time.perf_counter_ns() - t0

                t0 = time.perf_counter_ns()
                actions_p1 = agent2.predict_batch(obs_p1, masks_p1)
                ns2 = time.perf_counter_ns() - t0

                all_actions = torch.zeros(
                    num_slots,
                    *actions_p0.shape[1:],
                    dtype=actions_p0.dtype,
                    device=actions_p0.device,
                )
                all_actions[0::2] = actions_p0
                all_actions[1::2] = actions_p1

            # See comment in play_agent_vs_bot_batch: divide by N (initial env count)
            # to keep each game's per-step inference share stable over the match.
            per1 = ns1 // N
            per2 = ns2 // N
            for i in range(N):
                if active_games[i]:
                    agent1_time_ns[i] += per1
                    agent2_time_ns[i] += per2

            raw_obs, _, done_arr, infos = env.step(all_actions.cpu().numpy().reshape(num_slots, -1))
            if needs_adapters:
                obs_p0, obs_p1 = _process_obs_agent_vs_agent(
                    base_env, raw_obs, p0_adapter, p1_adapter, device, env_extended_obs
                )
            else:
                obs_p0 = torch.as_tensor(raw_obs[0::2]).to(device)
                obs_p1 = torch.as_tensor(raw_obs[1::2]).to(device)
        except Exception as e:
            print(f"  WARNING: agent_vs_agent batch crashed: {e}")
            for i in range(N):
                if active_games[i]:
                    result_data[i] = {
                        "winner": -1,
                        "crashed": -1,
                        "timedout": -1,
                        "time": steps[i],
                    }
                    active_games[i] = False
            break

        for i in range(N):
            if not active_games[i]:
                continue
            steps[i] += 1
            p0_slot = 2 * i

            if done_arr[p0_slot]:
                wl = infos[p0_slot]["raw_rewards"][0]
                if wl > 0:
                    winner = 0
                elif wl < 0:
                    winner = 1
                else:
                    winner = -1
                result_data[i] = {
                    "winner": winner,
                    "crashed": -1,
                    "timedout": -1,
                    "time": steps[i],
                }
                active_games[i] = False

    # Build result dicts
    results = []
    for i in range(N):
        d = result_data[i]
        results.append(
            {
                "iteration": group.iterations[i],
                "map_idx": group.map_idx,
                "ai1_idx": group.ai1_idx,
                "ai2_idx": group.ai2_idx,
                "time": d["time"],
                "winner": d["winner"],
                "crashed": d["crashed"],
                "timedout": d["timedout"],
                "ai1_timeouts": 0,
                "ai2_timeouts": 0,
                "ai1_time_ns": agent1_time_ns[i],
                "ai2_time_ns": agent2_time_ns[i],
            }
        )
    return results


def play_matchup_group(
    env_pool: EnvPool, group: MatchupGroup, config: TournamentConfig, output_dir: str
) -> list[dict]:
    """Dispatch to the appropriate batch game loop based on matchup type."""
    if group.matchup_type == "bot_vs_bot":
        return play_bot_vs_bot_batch(env_pool, group, config, output_dir)
    elif group.matchup_type == "agent_vs_bot":
        return play_agent_vs_bot_batch(env_pool, group, config, player=0, output_dir=output_dir)
    elif group.matchup_type == "bot_vs_agent":
        return play_agent_vs_bot_batch(env_pool, group, config, player=1, output_dir=output_dir)
    else:  # agent_vs_agent
        return play_agent_vs_agent_batch(env_pool, group, config, output_dir=output_dir)
