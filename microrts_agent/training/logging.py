"""TensorBoard and console logging for training."""

import os
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch


def log_action_stats(actions, writer, global_step, num_steps, num_envs):
    """Log what the agent is doing: % move, attack, produce, and what it produces.

    Reads sub-action[0] = action type, sub-action[5] = produced unit type.
    Only counts active cells (action_type > 0, i.e. not NOOP).
    """
    with torch.no_grad():
        act_np = actions.cpu().numpy()
        action_types = act_np[:, :, :, 0].ravel()  # (T*N*H*W,)
        active_mask = action_types > 0
        if not active_mask.any():
            return

        active_types = action_types[active_mask]
        n_active = len(active_types)

        for idx, name in [
            (1, "move"),
            (2, "harvest"),
            (3, "return"),
            (4, "produce"),
            (5, "attack"),
        ]:
            writer.add_scalar(
                f"actions/pct_{name}", (active_types == idx).sum() / n_active, global_step
            )
        writer.add_scalar(
            "actions/active_cells_avg", active_mask.sum() / (num_steps * num_envs), global_step
        )

        # What unit types are being produced? (action_type 4 = produce)
        produce_mask = action_types == 4
        if produce_mask.any():
            produce_types = act_np[:, :, :, 5].ravel()[produce_mask]
            n_produce = len(produce_types)
            for idx, name in [
                (1, "base"),
                (2, "barracks"),
                (3, "worker"),
                (4, "light"),
                (5, "heavy"),
                (6, "ranged"),
            ]:
                writer.add_scalar(
                    f"actions/produce_{name}", (produce_types == idx).sum() / n_produce, global_step
                )


def log_update(
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
):
    """Log everything for one training update to TensorBoard.

    Called once per update from train.py. Groups scalars by category:
    charts/ = progress, losses/ = PPO metrics, schedule/ = hyperparams,
    actions/ = agent behavior, opponents/ = per-bot stats.
    """
    sps = int(global_step / (time.time() - start_time))
    lr = optimizer.param_groups[0]["lr"]

    # ---- Progress ----
    writer.add_scalar("charts/learning_rate", lr, global_step)
    writer.add_scalar("charts/update", update, global_step)
    writer.add_scalar("charts/sps", sps, global_step)

    # ---- PPO losses ----
    writer.add_scalar("losses/policy_loss", metrics["pg_loss"], global_step)
    writer.add_scalar("losses/value_loss", metrics["v_loss"], global_step)
    writer.add_scalar("losses/entropy", metrics["entropy_loss"], global_step)
    writer.add_scalar("losses/approx_kl", metrics["approx_kl"], global_step)
    writer.add_scalar("losses/clip_fraction", metrics["clip_fraction"], global_step)
    writer.add_scalar("losses/explained_variance", metrics["explained_variance"], global_step)
    writer.add_scalar("losses/grad_norm", metrics["grad_norm"], global_step)
    if args.dual_value_heads:
        writer.add_scalar("losses/value_loss_sparse", metrics["v_loss_sparse"], global_step)
    if args.triple_value_heads:
        writer.add_scalar("losses/value_loss_cost", metrics["v_loss_cost"], global_step)

    # ---- Value-head predictions (collapse/drift monitoring) ----
    writer.add_scalar("values/shaped_mean", metrics["value_shaped_mean"], global_step)
    writer.add_scalar("values/shaped_std", metrics["value_shaped_std"], global_step)
    if args.dual_value_heads and "value_sparse_mean" in metrics:
        writer.add_scalar("values/sparse_mean", metrics["value_sparse_mean"], global_step)
        writer.add_scalar("values/sparse_std", metrics["value_sparse_std"], global_step)
    if args.triple_value_heads and "value_cost_mean" in metrics:
        writer.add_scalar("values/cost_mean", metrics["value_cost_mean"], global_step)
        writer.add_scalar("values/cost_std", metrics["value_cost_std"], global_step)
    if args.aux_tasks:
        for task in args.aux_tasks:
            writer.add_scalar(f"losses/aux_{task}", metrics[f"aux_{task}"], global_step)

    # ---- Scheduled hyperparams (only logged when annealing is active) ----
    if args.ent_coef_end is not None:
        writer.add_scalar("schedule/ent_coef", sched["ent_coef"], global_step)
    if args.reward_schedule != "none":
        writer.add_scalar("schedule/reward_weight_winloss", sched["reward_weight"][0], global_step)
        writer.add_scalar(
            "schedule/reward_weight_shaped_sum", sched["reward_weight"][1:].sum(), global_step
        )
    if sched["adv_weights"] is not None:
        writer.add_scalar("schedule/adv_w_shaped", sched["adv_weights"][0], global_step)
        writer.add_scalar("schedule/adv_w_sparse", sched["adv_weights"][1], global_step)
        if len(sched["adv_weights"]) > 2:
            writer.add_scalar("schedule/adv_w_cost", sched["adv_weights"][2], global_step)
    if args.dual_value_heads:
        writer.add_scalar("schedule/eff_vf_coef", sched["vf_coef"], global_step)
        writer.add_scalar("schedule/eff_vf_sparse", sched["vf_sparse"], global_step)
    if args.triple_value_heads:
        writer.add_scalar("schedule/eff_vf_cost", sched["vf_cost"], global_step)

    # ---- Agent behavior (action distributions) ----
    log_action_stats(actions, writer, global_step, T, N)

    # ---- Per-opponent stats (sliding window) ----
    wr = opp_tracker.get_win_rates()
    avg_ret = opp_tracker.get_avg_returns()
    avg_len = opp_tracker.get_avg_lengths()
    for name in opp_tracker.names:
        if opp_tracker.is_window_full(name):
            writer.add_scalar(f"charts/train_wr/{name}", wr[name], global_step)
            writer.add_scalar(f"charts/train_ret/{name}", avg_ret[name], global_step)
            writer.add_scalar(f"charts/train_len/{name}", avg_len[name], global_step)
        writer.add_scalar(f"charts/episodes/{name}", opp_tracker.episode_counts[name], global_step)
    if args.prioritized_sampling:
        weights = opp_tracker.get_sampling_weights()
        for name in opp_tracker.names:
            writer.add_scalar(f"opponents/weight_{name}", weights[name], global_step)

    # ---- Adaptive opponent env counts ----
    if adaptive_scheduler is not None:
        type_counts = Counter(env_opponent_labels[args.num_selfplay_envs :])
        for t, count in type_counts.items():
            writer.add_scalar(f"adaptive/{t}_envs", count, global_step)

    # ---- Self-play + PLR map stats ----
    sp.log_stats(writer, global_step)
    if map_plr is not None:
        wr = map_plr.get_win_rates()
        probs = map_plr.get_sampling_probs()
        for m in args.map_pool:
            short = os.path.basename(m).replace(".xml", "")
            writer.add_scalar(f"plr/wr_{short}", wr[m], global_step)
            writer.add_scalar(f"plr/prob_{short}", probs[m], global_step)


def log_step_episodes(
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
):
    """Print + log summary when episodes finish during a rollout step.

    Called from the inner rollout loop in train.py. Prints one line to console
    with per-bot WDL and sliding-window WR, and logs episode metrics to TB.

    Returns updated total_episodes count.
    """
    n_ep = len(step_eps)
    if n_ep == 0:
        return total_episodes

    total_episodes += n_ep
    avg_ret = np.mean([e[2] for e in step_eps])
    avg_len = int(np.mean([e[3] for e in step_eps]))

    # Per-bot WDL this step: "CoacA:2W1L Mayar:1W" with P0/P1 tag
    bot_wdl = {}
    # Collect player IDs per bot to determine P0/P1 tag
    bot_players = {}
    for opp, wl, _, _, player in step_eps:
        if opp not in bot_wdl:
            bot_wdl[opp] = [0, 0, 0]
        if wl > 0:
            bot_wdl[opp][0] += 1
        elif wl < 0:
            bot_wdl[opp][2] += 1
        else:
            bot_wdl[opp][1] += 1
        bot_players.setdefault(opp, set()).add(player)
    bot_parts = []
    for name in sorted(bot_wdl):
        w, d, losses = bot_wdl[name]
        parts = []
        if w:
            parts.append(f"{w}W")
        if d:
            parts.append(f"{d}D")
        if losses:
            parts.append(f"{losses}L")
        # Show player position: P0, P1, or P0+P1 if mixed in same step
        players = bot_players[name]
        if players == {0}:
            ptag = " P0"
        elif players == {1}:
            ptag = " P1"
        else:
            ptag = " P0+P1"
        bot_parts.append(f"{name[:5]}:{''.join(parts)}{ptag}")

    # Sliding-window WR across all opponents
    wr_dict = opp_tracker.get_win_rates()
    opp_wr_parts = []
    for oname in sorted(wr_dict.keys()):
        if opp_tracker.is_window_full(oname):
            opp_wr_parts.append(f"{oname[:4]}={wr_dict[oname]:.0%}")
    wr_str = " ".join(opp_wr_parts) if opp_wr_parts else "warming up..."
    map_tag = f"  ({Path(current_map).stem})" if args.multi_map else ""

    # Console: one line per step with episodes
    print(
        f"  step={global_step:>8,d}  {n_ep} eps: {' '.join(bot_parts)}  "
        f"ret={avg_ret:>7.2f}  len={avg_len:>5d}  "
        f"WR[{wr_str}]{map_tag}"
    )

    # TensorBoard
    writer.add_scalar("charts/episodic_return", avg_ret, global_step)
    writer.add_scalar("charts/episodic_length", avg_len, global_step)
    writer.add_scalar("charts/total_episodes", total_episodes, global_step)
    if len(win_buffer) >= win_buffer.maxlen:
        writer.add_scalar("charts/win_rate", np.mean(win_buffer), global_step)
    sp.log_stats(writer, global_step)
    # Average each reward component across all episodes this step
    for key, vals in step_stats_acc.items():
        writer.add_scalar(f"charts/episodic_return/{key}", np.mean(vals), global_step)

    return total_episodes
