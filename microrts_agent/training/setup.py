"""Eval environment setup, auxiliary task helpers, and GAE dispatch.

Called once at training init (setup_eval, build_aux_coefs, compute_aux_channels)
and once per update (build_aux_targets, compute_gae_from_rollout, maybe_switch_map,
run_eval_block).
"""

import os
import time

import numpy as np
import torch

from microrts_agent.training.eval import create_eval_envs

# ═══════════════════════════════════════════════════════════════════════════
# Eval setup — create envs + CSV once at training start
# ═══════════════════════════════════════════════════════════════════════════

_EVAL_CSV_HEADER = (
    "global_step,bot,wins,losses,draws,games,win_rate,"
    "wins_p0,games_p0,win_rate_p0,wins_p1,games_p1,win_rate_p1,"
    "avg_return,avg_length\n"
)

_EVAL_CSV_HEADER_MULTIMAP = (
    "global_step,map,bot,wins,losses,draws,games,win_rate,"
    "wins_p0,games_p0,win_rate_p0,wins_p1,games_p1,win_rate_p1,"
    "avg_return,avg_length\n"
)


def setup_eval(args, run_dir):
    """Create P0/P1 eval environments and initialize the results CSV.

    Two separate env sets (P0 and P1) are needed to measure win rate from
    both starting positions, eliminating first-player advantage bias.
    Envs are created once here because JPype cannot restart the JVM.

    Returns (eval_envs_p0, eval_envs_p1, eval_bot_names, eval_csv_path).
    All are None/empty when eval is disabled (eval_interval <= 0).
    """
    eval_envs_p0 = None
    eval_envs_p1 = None
    eval_bot_names = []
    eval_csv_path = os.path.join(run_dir, "eval_results.csv")

    if args.eval_interval > 0:
        # Multi-map: pad all envs to max_map_size so they can switch maps
        multi_map = args.multi_map
        pad_h = args.max_map_size if multi_map else None
        pad_w = args.max_map_size if multi_map else None

        rw = np.array(args.reward_weight)

        # P0 envs: agent plays as player 0 (top-left spawn)
        eval_envs_p0, eval_bot_names = create_eval_envs(
            args.eval_bots,
            args.map,
            args.partial_obs,
            0,
            rw,
            gamma=args.gamma,
            frame_stack=args.frame_stack,
            filtered_masks=args.filtered_masks,
            reserved_obs=args.reserved_obs,
            extended_obs=args.extended_obs,
            padded=multi_map,
            max_height=pad_h,
            max_width=pad_w,
            max_steps=args.max_steps,
        )
        # P1 envs: agent plays as player 1 (bottom-right spawn)
        eval_envs_p1, _ = create_eval_envs(
            args.eval_bots,
            args.map,
            args.partial_obs,
            1,
            rw,
            gamma=args.gamma,
            frame_stack=args.frame_stack,
            filtered_masks=args.filtered_masks,
            reserved_obs=args.reserved_obs,
            extended_obs=args.extended_obs,
            padded=multi_map,
            max_height=pad_h,
            max_width=pad_w,
            max_steps=args.max_steps,
        )

        # Initialize CSV (skip if resuming and file already exists)
        if not os.path.exists(eval_csv_path):
            header = _EVAL_CSV_HEADER_MULTIMAP if multi_map else _EVAL_CSV_HEADER
            with open(eval_csv_path, "w") as f:
                f.write(header)

    return eval_envs_p0, eval_envs_p1, eval_bot_names, eval_csv_path


# ═══════════════════════════════════════════════════════════════════════════
# Auxiliary tasks — coefficient mapping, channel indexing, target extraction
# ═══════════════════════════════════════════════════════════════════════════


def build_aux_coefs(args):
    """Map active auxiliary tasks to their loss coefficients.

    Returns {'aux_spatial': 0.1, 'aux_contrastive': 0.05, ...} or None.
    These coefficients multiply each aux loss before adding to the total loss.
    """
    coefs = {}
    if args.aux_tasks:
        if "spatial" in args.aux_tasks:
            coefs["aux_spatial"] = args.aux_spatial_coef
        if "unit_count" in args.aux_tasks:
            coefs["aux_unit_count"] = args.aux_unit_count_coef
        if "contrastive" in args.aux_tasks:
            coefs["aux_contrastive"] = args.aux_contrastive_coef
        if "opponent_modeling" in args.aux_tasks:
            coefs["aux_opponent_modeling"] = args.aux_opponent_modeling_coef
    return coefs if coefs else None


def compute_aux_channels(obs_channels, frame_stack, extended_obs):
    """Find the channel indices for ownership, unit-type, and action planes in the observation.

    The observation tensor is (H, W, C) with channels packed as:
      Standard: HP(5) + Resources(5) + Owner(3) + UnitType(nUT+1) + Action(6) + Terrain(2)
      Extended: HP(1) + Resources(2) + Owner(3) + UnitType(nUT+2) + Action(6) + ...

    With frame stacking, we want the most recent frame (last in the stack),
    so we offset by (frame_stack - 1) * channels_per_frame.

    Returns (p0_ch, p1_ch, ut_start, action_start):
      p0_ch:        channel index for "cell owned by P0" (binary)
      p1_ch:        channel index for "cell owned by P1" (binary)
      ut_start:     first unit-type channel (for unit_count aux)
      action_start: first action-type one-hot channel (6 channels, for opponent_modeling aux)
    """
    if frame_stack > 0:
        base_c = obs_channels // frame_stack
        offset = (frame_stack - 1) * base_c
    else:
        offset = 0

    if extended_obs:
        # Extended: HP(1)+Res(2)+Owner(3)+UnitType(nUT+2)+Action(6)+...+Global(14)
        # per_cell = 52 + nUT, total = per_cell + 14.
        # Solve for nUT, then action_start = 6 + (nUT + 2)
        base_c = obs_channels // max(frame_stack, 1)
        n_ut = (base_c - 14) - 52  # 14 global planes, 52 non-UT per-cell channels
        action_start = 6 + n_ut + 2  # UnitType has nUT+2 channels in extended
        return offset + 4, offset + 5, offset + 6, offset + action_start
    else:
        # Standard: HP(5)+Res(5)+Owner(3)+UnitType(nUT+1)+Action(6)+Terrain(2) [+extras]
        # Action block always starts at index 21 (= 5+5+3+8) for nUT=7.
        # Extras (reserved_obs, partial) are appended AFTER Terrain.
        return offset + 11, offset + 12, offset + 13, offset + 21


def build_aux_targets(
    args,
    agent,
    obs,
    next_obs,
    b_obs,
    device,
    *,
    aux_p0_ch,
    aux_p1_ch,
    aux_ut_start,
    aux_action_start,
):
    """Extract ground-truth targets for auxiliary tasks from the rollout.

    Called once per update, before ppo_update. The targets are what the
    auxiliary heads should learn to predict from the encoder's hidden state.

    Args:
        obs:      raw rollout observations, shape (T, N, H, W, C)
        next_obs: observation after the last rollout step, shape (N, H, W, C)
        b_obs:    flattened rollout observations, shape (T*N, H, W, C)
        aux_*_ch: channel indices from compute_aux_channels

    Returns dict of target tensors, or None if no aux tasks active.
    """
    if not args.aux_tasks:
        return None

    targets = {}

    # ── Spatial: "which cells does each player own?" ──
    # Extract binary ownership maps -> target (B, 2, H, W)
    # The aux head predicts this from hidden features (BCE loss)
    if "spatial" in args.aux_tasks:
        p0 = b_obs[:, :, :, aux_p0_ch].float()
        p1 = b_obs[:, :, :, aux_p1_ch].float()
        targets["spatial_target"] = torch.stack([p0, p1], dim=1)

    # ── Unit count: "how many of each unit type does each player have?" ──
    # For each (player, unit_type) pair: sum(ownership × unit_type_channel)
    # -> target (B, 12) = [P0_base, P0_barracks, ..., P0_ranged,
    #                      P1_base, P1_barracks, ..., P1_ranged]
    # The aux head predicts this from hidden features (MSE loss)
    if "unit_count" in args.aux_tasks:
        p0_mask = b_obs[:, :, :, aux_p0_ch].float()  # (B, H, W)
        p1_mask = b_obs[:, :, :, aux_p1_ch].float()
        counts = []
        # 6 playable types: base, barracks, worker, light, heavy, ranged
        # Skip none(0) and resource(1) in the UTT -> offset by 2
        for player_mask in [p0_mask, p1_mask]:
            for ut_idx in range(6):
                ut_ch = aux_ut_start + 2 + ut_idx
                counts.append((player_mask * b_obs[:, :, :, ut_ch].float()).sum(dim=(1, 2)))
        H, W = b_obs.shape[1], b_obs.shape[2]
        targets["unit_count_target"] = torch.stack(counts, dim=1) / (H * W)  # normalize to ~[0, 1]

    # ── Opponent modeling: "what action is each enemy unit executing?" ──
    # Extract action type (argmax of 6 one-hot channels) for enemy cells only.
    # Target: (B, H, W) long tensor with action class 0-5, masked to enemy cells.
    # Enemy mask: (B, H, W) float tensor, 1 where enemy unit present.
    if "opponent_modeling" in args.aux_tasks:
        enemy_mask = b_obs[:, :, :, aux_p1_ch].float()  # (B, H, W)
        action_onehot = b_obs[:, :, :, aux_action_start : aux_action_start + 6]  # (B, H, W, 6)
        action_target = action_onehot.argmax(dim=-1)  # (B, H, W) in {0..5}
        targets["opponent_action_target"] = action_target
        targets["opponent_enemy_mask"] = enemy_mask

    # ── Contrastive (InfoNCE): "consecutive states should embed similarly" ──
    # Encode s_{t+1} as positive pair for s_t. During ppo_update, the aux head
    # encodes s_t (anchor) and tries to match it to this pre-computed embedding.
    # All other transitions in the batch serve as negatives.
    if "contrastive" in args.aux_tasks:
        obs_shape = obs.shape[2:]  # (H, W, C)
        with torch.no_grad():
            # Shift observations by 1 step: obs[1], obs[2], ..., next_obs
            shifted_obs = torch.cat(
                [
                    obs[1:].reshape((-1,) + obs_shape),
                    next_obs.unsqueeze(0)
                    .expand(1, obs.shape[1], *obs_shape)
                    .reshape((-1,) + obs_shape),
                ],
                dim=0,
            )
            # Pre-compute positive embeddings (detached — no gradient through targets)
            targets["contrastive_positive"] = agent.aux_contrastive(
                agent.encoder(shifted_obs)
            ).detach()

    return targets


# ═══════════════════════════════════════════════════════════════════════════
# Periodic evaluation — play games vs eval bots during training
# ═══════════════════════════════════════════════════════════════════════════


def run_eval_block(
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
    map_pool=None,
):
    """Run periodic evaluation: play games vs eval bots, log to TB + CSV.

    Single-map: plays eval_games per bot (half P0, half P1).
    Multi-map:  plays 2 games per bot per map (1 P0 + 1 P1), logs per-map + aggregate.
    """
    from microrts_agent.training.eval import (
        log_eval_results,
        log_multimap_eval_results,
        run_evaluation,
        run_multimap_evaluation,
    )

    eval_start = time.time()

    print(f"\n  {'=' * 72}")
    print(f"  EVAL @ step {global_step:,} (update {update})")
    print(f"  {'=' * 72}")

    if map_pool and len(map_pool) > 1:
        # Multi-map: evaluate on every map in the pool
        print(
            f"  Multi-map eval: {len(map_pool)} maps, "
            f"2 games/bot/map = {2 * len(eval_bot_names) * len(map_pool)} total"
        )

        per_map, aggregate = run_multimap_evaluation(
            agent,
            eval_envs_p0,
            eval_envs_p1,
            eval_bot_names,
            map_pool,
            games_per_bot_per_map=2,
            device=device,
        )
        log_multimap_eval_results(per_map, aggregate, map_pool, writer, global_step, eval_csv_path)
    else:
        # Single-map: play eval_games against each bot
        games_per_bot = dict.fromkeys(eval_bot_names, args.eval_games)
        print(
            f"  {'Bot':<18s} {'G':>3s} {'W':>3s} {'L':>3s} {'D':>3s} "
            f"{'WR%':>7s} {'P0%':>6s} {'P1%':>6s} {'Avg Ret':>9s} {'Avg Len':>8s}"
        )
        print(f"  {'-' * 70}")

        eval_results = run_evaluation(
            agent, eval_envs_p0, eval_envs_p1, eval_bot_names, games_per_bot, device
        )
        log_eval_results(eval_results, writer, global_step, eval_csv_path)

    eval_sec = time.time() - eval_start
    print(f"  Eval took {eval_sec:.0f}s")
    print(f"  {'=' * 72}\n")
    writer.add_scalar("eval/duration_sec", eval_sec, global_step)


# ═══════════════════════════════════════════════════════════════════════════
# GAE from rollout — dispatch to single-head or multi-head GAE
# ═══════════════════════════════════════════════════════════════════════════


def compute_gae_from_rollout(
    agent,
    args,
    *,
    values_shaped,
    rewards_shaped,
    dones,
    next_obs,
    next_done,
    values_sparse,
    rewards_sparse,
    values_cost,
    rewards_cost,
    advantage_weights,
):
    """Compute GAE advantages and returns from a collected rollout.

    Three modes depending on value head configuration:
      - Standard (1 head): single GAE on shaped rewards, advantages (T, N)
      - Dual (2 heads):    per-head GAE, advantages (T, N, 2) — blending is
                           done in ppo_update after per-head normalization
      - Triple (3 heads):  same with cost head added, advantages (T, N, 3)

    The bootstrap step (V(s_{T+1})) is computed here by passing next_obs
    through the critic. Everything is in no_grad — these are targets, not
    part of the optimization graph.

    Returns:
        (advantages, returns_shaped, returns_sparse, returns_cost)
        advantages is (T, N) in single-head mode, (T, N, H) otherwise.
        returns_sparse/returns_cost are None when their heads are inactive.
    """
    from microrts_agent.training.ppo import compute_gae, compute_multi_head_gae

    with torch.no_grad():
        # Bootstrap: get V(s_{T+1}) for each head from the last observation
        last = agent.get_values(next_obs)
        last_v_shaped = last["v_shaped"].reshape(1, -1)  # (1, N)

        if args.triple_value_heads or args.dual_value_heads:
            # Each head config: (rewards, values, bootstrap_v, gamma, gae_lambda)
            heads = [
                (rewards_shaped, values_shaped, last_v_shaped, args.gamma, args.gae_lambda),
                (
                    rewards_sparse,
                    values_sparse,
                    last["v_sparse"].reshape(1, -1),
                    args.gamma_sparse,
                    args.gae_lambda_sparse,
                ),
            ]
            if args.triple_value_heads:
                heads.append(
                    (
                        rewards_cost,
                        values_cost,
                        last["v_cost"].reshape(1, -1),
                        args.gamma_cost,
                        args.gae_lambda_cost,
                    )
                )

            # compute_multi_head_gae now returns per-head advantages (T, N, H);
            # normalization + blending happens later in ppo_update.
            result = compute_multi_head_gae(
                heads, dones, next_done, args.num_steps, advantage_weights
            )
            adv_per_head, ret_shaped, ret_sparse = result[0], result[1], result[2]
            ret_cost = result[3] if args.triple_value_heads else None
            return adv_per_head, ret_shaped, ret_sparse, ret_cost
        else:
            # Single head: standard GAE on shaped rewards
            advantages, returns = compute_gae(
                rewards_shaped,
                values_shaped,
                dones,
                last_v_shaped,
                next_done,
                args.gamma,
                args.gae_lambda,
                args.num_steps,
            )
            return advantages, returns, None, None


# ═══════════════════════════════════════════════════════════════════════════
# Multi-map switching — cycle maps during training
# ═══════════════════════════════════════════════════════════════════════════


def _unwrap_base_env(envs):
    """Traverse the wrapper chain (StatsRecorder -> FrameStack -> ...) to
    reach the raw MicroRTS vec env that has switch_map/vec_client."""
    env = envs
    while hasattr(env, "venv"):
        env = env.venv
    return env


def maybe_switch_map(
    args,
    envs,
    update,
    global_step,
    map_plr,
    adaptive_scheduler,
    env_opponent_labels,
    reward_weight,
    writer,
):
    """Switch to the next map if multi-map is enabled and it's time.

    Map selection:
      - With PLR: sample proportional to learning signal (harder maps sampled more)
      - Without PLR: deterministic round-robin through map_pool

    After switching, two things must be restored because switch_map recreates
    the Java client from scratch:
      1. Adaptive opponents (promoted bots revert to defaults)
      2. Reward weights (if a reward schedule is active)

    Returns the new map path, or None if no switch happened.
    """
    if not args.multi_map or update <= 1:
        return None
    if (update - 1) % args.map_switch_freq != 0:
        return None

    # Select next map
    if map_plr is not None:
        next_map = map_plr.sample()
    else:
        map_idx = ((update - 1) // args.map_switch_freq) % len(args.map_pool)
        next_map = args.map_pool[map_idx]

    base_env = _unwrap_base_env(envs)
    base_env.switch_map(next_map)

    # Restore adaptive opponents (switch_map resets all AIs to defaults)
    if adaptive_scheduler is not None:
        from microrts_agent.registries.ai import AI_MAPPING

        for env_idx in range(args.num_bot_envs):
            opp_type = env_opponent_labels[args.num_selfplay_envs + env_idx]
            new_ai = AI_MAPPING[opp_type](base_env.real_utt)
            base_env.vec_client.clients[env_idx].ai2 = new_ai

    # Restore reward weights (switch_map resets to env defaults)
    if args.reward_schedule != "none":
        base_env.reward_weight = reward_weight

    # Log the switch
    plr_info = ""
    if map_plr is not None:
        probs = map_plr.get_sampling_probs()
        wr = map_plr.get_win_rates()
        plr_info = f" [PLR: WR={wr[next_map]:.2f}, P={probs[next_map]:.2f}]"
    print(
        f"  [Map switch] update={update} -> {next_map} ({base_env.width}x{base_env.height}){plr_info}"
    )
    writer.add_text("map_switch", next_map, global_step)

    return next_map
