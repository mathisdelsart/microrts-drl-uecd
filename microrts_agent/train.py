"""
PPO training for MicroRTS.

    python train.py --total-timesteps 1000000
    python train.py --architecture impala --ent-coef 0.05
    tensorboard --logdir outputs/runs/
"""

import os
import sys
import time
from collections import deque

import numpy as np
import torch
import torch.optim as optim
from gymnasium.spaces import MultiDiscrete  # type: ignore[import-not-found]
from stable_baselines3.common.vec_env import VecMonitor  # type: ignore[import-not-found]

from lib.arch_factory import create_agent
from lib.env_factory import JVM_ARGS, make_agent_env
from lib.envs.base_vec_env import suppress_java_output
from lib.mappings.ai import AI_MAPPING
from lib.paths import RUNS_DIR
from lib.training.checkpoint import (
    Tee,
    resume_checkpoint,
    save_checkpoint,
    setup_device_and_seed,
    setup_tensorboard,
)
from lib.training.cli import parse_args
from lib.training.config import print_config_banner, save_run_config
from lib.training.logging import log_step_episodes, log_update
from lib.training.opponents import build_opponent_config
from lib.training.ppo import ppo_update
from lib.training.scheduling import (
    AdaptiveOpponentScheduler,
    MapPLR,
    OpponentTracker,
    build_importance_weights,
    compute_schedules,
)
from lib.training.selfplay import SelfPlayManager
from lib.training.setup import (
    _unwrap_base_env,
    build_aux_coefs,
    build_aux_targets,
    compute_aux_channels,
    compute_gae_from_rollout,
    maybe_switch_map,
    run_eval_block,
    setup_eval,
)
from lib.wrapper_factory import apply_env_wrappers

# ── Training ─────────────────────────────────────────────────────────────


def main():
    args = parse_args()

    # ── Setup ────────────────────────────────────────────────────────────
    # Run directory, logging, device, seeds

    run_dir = RUNS_DIR / args.exp_name
    os.makedirs(run_dir, exist_ok=True)
    sys.stdout = Tee(os.path.join(run_dir, "train.log"))

    writer = setup_tensorboard(args, run_dir)
    device = setup_device_and_seed(args)

    # Opponent labels for each env: [SP_latest... | SP_pool... | bot_labels...]
    env_opponent_labels, _ = build_opponent_config(args)
    ai_list = [AI_MAPPING[name] for name in env_opponent_labels]
    n_latest = args.num_latest_selfplay_envs
    sp_labels = ["SelfPlayLatest"] * n_latest + ["SelfPlay"] * (args.num_selfplay_envs - n_latest)
    env_opponent_labels = sp_labels + env_opponent_labels

    # Starting map (first from pool if multi-map, else the single --map)
    current_map = args.map_pool[0] if args.multi_map else args.map

    # N parallel MicroRTS games running in a Java JVM
    envs = make_agent_env(
        num_bot_envs=args.num_bot_envs,
        num_selfplay_envs=args.num_selfplay_envs,
        ai2s=ai_list,
        map_paths=[current_map],
        player=0,
        partial_obs=args.partial_obs,
        max_steps=args.max_steps,
        reward_weight=np.array(args.reward_weight),
        jvm_args=JVM_ARGS,
        padded=args.multi_map,
        max_height=args.max_map_size,
        max_width=args.max_map_size,
        alternate_players=args.alternate_players,
        filtered_masks=args.filtered_masks,
        extended_obs=args.extended_obs,
    )
    envs = apply_env_wrappers(
        envs,
        gamma=args.gamma,
        frame_stack=args.frame_stack,
        reserved_obs=args.reserved_obs,
        augment_symmetry=args.augment_symmetry,
    )
    envs = VecMonitor(envs)
    suppress_java_output()
    assert isinstance(envs.action_space, MultiDiscrete)

    # Env-derived shapes (depend on map size + obs config)
    h, w, obs_channels = envs.observation_space.shape
    action_nvec = envs.action_plane_space.nvec.tolist()
    mapsize = h * w

    # Save all args + env shapes to config.json for reproducibility
    config_path = save_run_config(
        args, obs_channels, action_nvec, mapsize, run_dir, map_pool=args.map_pool
    )

    # Win rate tracker per opponent (+ optional importance weights for prioritized sampling)
    opp_tracker = OpponentTracker(
        env_opponent_labels,
        window=args.opp_log_window,
        prioritized=args.prioritized_sampling,
        priority_window=args.priority_window,
    )

    # Prioritized Level Replay: sample maps where we lose more (optional)
    map_plr = None
    if args.plr and args.multi_map:
        map_plr = MapPLR(args.map_pool, window=args.plr_window)

    # Adaptive opponents: auto-promote mastered bots to harder ones (optional)
    adaptive_scheduler = None
    if args.adaptive_opponents:
        adaptive_scheduler = AdaptiveOpponentScheduler(
            env_opponent_labels[args.num_selfplay_envs :],
            threshold=args.adaptive_threshold,
            window=args.adaptive_window,
            map_pool=args.map_pool if args.multi_map else None,
            criteria=args.adaptive_criteria,
            hybrid_min=args.adaptive_hybrid_min,
        )

    # Separate eval envs (P0 + P1) for unbiased periodic evaluation
    eval_envs_p0, eval_envs_p1, eval_bot_names, eval_csv_path = setup_eval(args, run_dir)

    # Neural network + Adam optimizer
    agent = create_agent(args, obs_channels, action_nvec, device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)
    starting_update, global_step = resume_checkpoint(args, agent, optimizer, device)
    total_params = sum(p.nelement() for p in agent.parameters())

    # BC teacher model for KL penalty (AlphaStar-style BC→PPO transition)
    bc_teacher = None
    if args.bc_teacher_model is not None:
        bc_teacher = create_agent(args, obs_channels, action_nvec, device)
        bc_teacher.load_state_dict(
            torch.load(args.bc_teacher_model, map_location=device, weights_only=False)
        )
        bc_teacher.eval()
        for p in bc_teacher.parameters():
            p.requires_grad = False
        print(f"BC teacher loaded from {args.bc_teacher_model} (frozen)")

    # Verify architecture works with this map size (catches stride/pooling mismatches)
    try:
        with torch.no_grad():
            agent.forward(
                torch.zeros((1,) + envs.observation_space.shape, device=device),
                invalid_action_masks=torch.ones((1, mapsize, sum(action_nvec)), device=device),
            )
    except RuntimeError as e:
        print(f"\n  ERROR: {args.architecture} incompatible with {args.map} ({h}x{w}): {e}\n")
        envs.close()
        writer.close()
        sys.exit(1)

    # Print full config banner (after agent creation so we have param count)
    print_config_banner(args, run_dir, title="PPO", total_params=total_params)
    print(f"Device: {device}")
    print(f"Config: {config_path}\n")

    # Self-play: opponent model, checkpoint pool, side alternation
    sp = SelfPlayManager(args, agent, obs_channels, action_nvec, device, run_dir)

    # ── Rollout buffers ──────────────────────────────────────────────────
    # Pre-allocated tensors filled during rollout, then flattened for PPO.
    # T = rollout length (steps per update), N = number of parallel envs.

    T, N = args.num_steps, args.num_envs
    action_space_shape = (mapsize, len(agent.action_nvec))  # (H*W, 7)
    invalid_action_shape = (mapsize, sum(agent.action_nvec))  # (H*W, 78)

    obs = torch.zeros((T, N) + envs.observation_space.shape).to(device)  # (T, N, H, W, C)
    actions = torch.zeros((T, N) + action_space_shape).to(device)  # (T, N, H*W, 7)
    logprobs = torch.zeros((T, N)).to(device)  # (T, N)
    dones = torch.zeros((T, N)).to(device)  # (T, N)
    invalid_action_masks = torch.zeros((T, N) + invalid_action_shape).to(device)  # (T, N, H*W, 78)

    values_shaped = torch.zeros((T, N)).to(device)  # (T, N)
    rewards_shaped = torch.zeros((T, N)).to(device)  # (T, N)

    values_sparse = (
        torch.zeros((T, N)).to(device) if args.dual_value_heads else None
    )  # win/loss value
    rewards_sparse = (
        torch.zeros((T, N)).to(device) if args.dual_value_heads else None
    )  # win/loss reward
    values_cost = (
        torch.zeros((T, N)).to(device) if args.triple_value_heads else None
    )  # military value
    rewards_cost = (
        torch.zeros((T, N)).to(device) if args.triple_value_heads else None
    )  # cost reward (score delta per step)
    prev_cost_scores = (
        np.zeros(N, dtype=np.float32) if args.triple_value_heads else None
    )  # previous normalized score per env (for delta computation)
    sp_agent_mask = torch.ones((T, N), device=device) if sp.enabled else None  # 1=agent, 0=opp

    # Aux task setup (channel indices for spatial + unit_count + opponent_modeling targets)
    aux_coefs = build_aux_coefs(args)
    aux_p0_ch, aux_p1_ch, aux_ut_start, aux_action_start = 0, 0, 0, 0
    if args.aux_tasks:
        aux_p0_ch, aux_p1_ch, aux_ut_start, aux_action_start = compute_aux_channels(
            obs_channels, args.frame_stack, extended_obs=args.extended_obs
        )

    # ── Training loop ────────────────────────────────────────────────────
    # Each iteration: collect rollout -> compute advantages -> PPO update -> log

    start_time = time.time()
    next_obs = torch.as_tensor(envs.reset()).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    win_buffer = deque(maxlen=100)
    total_episodes = 0

    for update in range(starting_update, args.num_updates + 1):
        # Update LR, entropy, advantage weights, reward weights, VF coefs
        sched = compute_schedules(args, update, global_step)
        optimizer.param_groups[0]["lr"] = sched["lr"]
        if args.reward_schedule != "none":
            _unwrap_base_env(envs).reward_weight = sched["reward_weight"]

        # Multi-map: switch to next map (cyclic or PLR-sampled)
        new_map = maybe_switch_map(
            args,
            envs,
            update,
            global_step,
            map_plr,
            adaptive_scheduler,
            env_opponent_labels,
            sched["reward_weight"],
            writer,
        )
        if new_map is not None:
            current_map = new_map
            next_obs = torch.as_tensor(envs.reset()).to(device)
            next_done = torch.zeros(args.num_envs).to(device)

        # ---- Collect rollout (T steps × N envs) ----
        # Each step: observe -> act -> step env -> record transition
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done
            sp.record_step_mask(sp_agent_mask, step)  # track agent vs opponent envs

            with torch.no_grad():
                # Agent predicts action + value from observation
                invalid_action_masks[step] = torch.tensor(envs.get_action_mask()).to(device)
                out = agent.forward(next_obs, invalid_action_masks=invalid_action_masks[step])
                action = out["action"]
                logproba = out["logprob"]
                values_shaped[step] = out["v_shaped"].flatten()
                if args.dual_value_heads:
                    values_sparse[step] = out["v_sparse"].flatten()
                if args.triple_value_heads:
                    values_cost[step] = out["v_cost"].flatten()

                # Self-play: pool opponent uses its own model for its envs
                sp_indices, sp_actions = sp.get_opponent_actions(
                    next_obs, invalid_action_masks[step]
                )
                if sp_indices:
                    action[sp_indices] = sp_actions

            actions[step] = action
            logprobs[step] = logproba

            # All N envs advance one game tick simultaneously
            next_obs, rs, ds, infos = envs.step(action.cpu().numpy().reshape(envs.num_envs, -1))
            next_obs = torch.as_tensor(next_obs).to(device)
            rewards_shaped[step] = torch.as_tensor(rs).to(device)
            next_done = torch.as_tensor(ds, dtype=torch.float32).to(device)

            # Split reward into per-head components (dual/triple heads only)
            if args.dual_value_heads:
                for env_idx, info in enumerate(infos):
                    rr = info["raw_rewards"]
                    rewards_sparse[step, env_idx] = float(rr[0])  # win/loss only
                    if args.triple_value_heads:
                        # RAI-style cost: delta of normalized army score between consecutive steps.
                        # rr[9] is MilitaryScoreRewardFunction = (own-opp)/(own+opp+1) ∈ [-1, 1].
                        # Taking deltas gives a dense signal that measures per-step change in
                        # army balance rather than the absolute score. On episode end we reset the
                        # previous score so the next episode's first-step delta starts from 0.
                        current_score = float(rr[9])
                        rewards_cost[step, env_idx] = current_score - prev_cost_scores[env_idx]
                        prev_cost_scores[env_idx] = 0.0 if ds[env_idx] else current_score

            # ---- Process finished episodes ----
            # Record results in trackers, accumulate stats for logging
            step_eps = []  # (opponent_name, wl_signal, episode_return, episode_length, player_id)
            step_stats_acc = {}  # reward_component_name -> [values] for averaging
            for env_idx, info in enumerate(infos):
                if "episode" not in info:
                    continue
                if env_idx in sp.opp_indices:  # skip SP opponent side
                    continue
                if sp.adaptive and not sp.pool_active and sp.n_latest <= env_idx < sp.n_selfplay:
                    continue  # skip pre-activation pool SP (not latest SP)

                wl = info["microrts_stats"]["WinDrawLossRewardFunction"]
                win = 1 if wl > 0 else 0
                opp = env_opponent_labels[env_idx]

                ep_ret = info["episode"]["r"]
                ep_len = info["episode"]["l"]
                win_buffer.append(win)
                opp_tracker.record(opp, win, current_map, ep_return=ep_ret, ep_length=ep_len)
                if adaptive_scheduler is not None and env_idx >= args.num_selfplay_envs:
                    adaptive_scheduler.record(opp, win, current_map)
                if map_plr is not None:
                    map_plr.record(current_map, win)
                if opp in ("SelfPlay", "SelfPlayLatest"):
                    sp.record_result(win)

                step_eps.append((opp, wl, ep_ret, ep_len, info.get("player", 0)))
                for key, val in info["microrts_stats"].items():
                    step_stats_acc.setdefault(key, []).append(val)

            # Print step summary + log to TensorBoard
            total_episodes = log_step_episodes(
                step_eps,
                step_stats_acc,
                total_episodes,
                global_step,
                win_buffer,
                opp_tracker,
                sp,
                args,
                current_map,
                writer,
            )

            sp.on_episode_end(ds)  # flip SP sides when games end

        # ---- Compute advantages (GAE) ----
        # advantages     (T, N) — how much better was this action vs average?
        # returns        (T, N) — discounted shaped return targets
        # returns_sparse (T, N) — win/loss return targets (or None)
        # returns_cost   (T, N) — military cost targets (or None)
        advantages, returns, returns_sparse, returns_cost = compute_gae_from_rollout(
            agent,
            args,
            values_shaped=values_shaped,
            rewards_shaped=rewards_shaped,
            dones=dones,
            next_obs=next_obs,
            next_done=next_done,
            values_sparse=values_sparse,
            rewards_sparse=rewards_sparse,
            values_cost=values_cost,
            rewards_cost=rewards_cost,
            advantage_weights=sched["adv_weights"],
        )

        # Zero opponent-side advantages (handles both (T,N) and (T,N,H) shapes)
        sp.mask_advantages(advantages, sp_agent_mask)

        # PAE: zero late-rollout advantages for envs where no episode ended
        # Also recompute returns for the kept portion since GAE propagated
        # stale advantages backward through the recursive computation.
        if args.pae_cutoff > 0:
            any_done = dones.bool().any(dim=0)
            mask = ~any_done  # envs that never had an episode end
            advantages[args.pae_cutoff :, mask] = 0.0
            returns[args.pae_cutoff :, mask] = 0.0
            # For multi-head advantages, use the shaped column (index 0) to
            # recompute returns_shaped (the returns we track separately).
            shaped_adv = advantages[..., 0] if advantages.dim() == 3 else advantages
            returns[: args.pae_cutoff, mask] = (
                shaped_adv[: args.pae_cutoff, mask] + values_shaped[: args.pae_cutoff, mask]
            )

        # ---- Flatten (T, N, ...) -> (T*N, ...) for random minibatch sampling ----
        b_obs = obs.reshape((-1,) + envs.observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + action_space_shape)
        # advantages is (T, N) for single-head or (T, N, H) for multi-head
        if advantages.dim() == 3:
            b_advantages = advantages.reshape(-1, advantages.shape[-1])  # (T*N, H)
        else:
            b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values_shaped.reshape(-1)
        b_invalid_action_masks = invalid_action_masks.reshape((-1,) + invalid_action_shape)
        b_values_sparse = values_sparse.reshape(-1) if values_sparse is not None else None
        b_returns_sparse = returns_sparse.reshape(-1) if returns_sparse is not None else None
        b_values_cost = values_cost.reshape(-1) if values_cost is not None else None
        b_returns_cost = returns_cost.reshape(-1) if returns_cost is not None else None

        # Harder opponents get more weight in the gradient (optional)
        b_importance_weights = build_importance_weights(
            opp_tracker, env_opponent_labels, args.num_envs, args.num_steps, device
        )

        # Aux targets: spatial ownership, unit counts, contrastive embeddings, opponent actions
        b_aux_targets = build_aux_targets(
            args,
            agent,
            obs,
            next_obs,
            b_obs,
            device,
            aux_p0_ch=aux_p0_ch,
            aux_p1_ch=aux_p1_ch,
            aux_ut_start=aux_ut_start,
            aux_action_start=aux_action_start,
        )

        # ---- PPO gradient update ----
        # metrics: dict with pg_loss, v_loss, entropy_loss, approx_kl,
        #          clip_fraction, explained_variance, grad_norm,
        #          + v_loss_sparse, v_loss_cost, aux_* (when enabled)
        metrics = ppo_update(
            agent,
            optimizer,
            b_obs,
            b_actions,
            b_logprobs,
            b_advantages,
            b_returns,
            b_values,
            b_invalid_action_masks,
            batch_size=args.batch_size,
            minibatch_size=args.minibatch_size,
            update_epochs=args.update_epochs,
            clip_coef=args.clip_coef,
            vf_coef=sched["vf_coef"],
            current_ent_coef=sched["ent_coef"],
            max_grad_norm=args.max_grad_norm,
            norm_adv=args.norm_adv,
            clip_vloss=args.clip_vloss,
            dual_value_heads=args.dual_value_heads,
            b_values_sparse=b_values_sparse,
            b_returns_sparse=b_returns_sparse,
            vf_coef_sparse=sched["vf_sparse"],
            triple_value_heads=args.triple_value_heads,
            b_values_cost=b_values_cost,
            b_returns_cost=b_returns_cost,
            vf_coef_cost=sched["vf_cost"],
            b_importance_weights=b_importance_weights,
            aux_tasks=args.aux_tasks,
            aux_coefs=aux_coefs,
            b_aux_targets=b_aux_targets,
            hl_gauss=args.hl_gauss,
            hl_gauss_bins=args.hl_gauss_bins,
            popart=args.popart,
            bc_teacher=bc_teacher,
            bc_kl_coef=args.bc_kl_coef * max(0, 1 - global_step / args.bc_kl_anneal_steps)
            if bc_teacher
            else 0.0,
            advantage_weights=sched["adv_weights"]
            if (args.dual_value_heads or args.triple_value_heads)
            else None,
        )

        # ---- Post-update bookkeeping ----

        if (update - 1) % args.save_frequency == 0:
            save_checkpoint(agent, optimizer, global_step, update, run_dir)

        # Save current weights to pool, swap opponent to a past checkpoint
        sp.manage_pool(agent, update, global_step, adaptive_scheduler)

        # Promote mastered bots to harder opponents (hot-swap Java AIs)
        if adaptive_scheduler is not None:
            promotions = adaptive_scheduler.check_promotions()
            if promotions:
                base_env = _unwrap_base_env(envs)
                for env_idx, new_type, old_type, wr in promotions:
                    new_ai = AI_MAPPING[new_type](base_env.real_utt)
                    base_env.vec_client.clients[env_idx].ai2 = new_ai
                    env_opponent_labels[args.num_selfplay_envs + env_idx] = new_type
                    print(
                        f"  [ADAPT] {old_type} mastered (WR={wr:.2f}) "
                        f"-> env {env_idx} promoted to {new_type}"
                    )

        # Log losses, schedules, WRs, action stats to TensorBoard
        log_update(
            writer,
            args,
            global_step,
            update,
            start_time,
            optimizer,
            metrics,
            sched,
            opp_tracker,
            adaptive_scheduler,
            env_opponent_labels,
            sp,
            map_plr,
            actions,
            T,
            N,
        )

        # Periodic eval tournament against fixed bot pool
        if args.eval_interval > 0 and update % args.eval_interval == 0:
            run_eval_block(
                agent,
                args,
                eval_envs_p0,
                eval_envs_p1,
                eval_bot_names,
                writer,
                global_step,
                update,
                eval_csv_path,
                device,
                map_pool=args.map_pool if args.multi_map else None,
            )

    # ── Teardown ─────────────────────────────────────────────────────────

    # Only save final checkpoint if the loop didn't already save on the last update
    if (update - 1) % args.save_frequency != 0:
        save_checkpoint(agent, optimizer, global_step, update, run_dir)

    wr = opp_tracker.get_win_rates()
    wr_str = "  ".join(
        f"{name}={rate:.0%}"
        for name, rate in sorted(wr.items())
        if opp_tracker.is_window_full(name)
    )

    print("\nTraining complete!")
    print(f"  Total steps:    {global_step:,}")
    print(f"  Total episodes: {total_episodes:,}")
    print(f"  Final model:    {run_dir}/agent.pt")
    if wr_str:
        print(f"  Win rates:      {wr_str}")

    envs.close()
    writer.close()


if __name__ == "__main__":
    main()
