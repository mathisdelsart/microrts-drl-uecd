"""Config save (config.json) and training banner display."""

import json
import os
import subprocess

import torch

# ── Config save ──────────────────────────────────────────────────────────


def save_run_config(args, obs_channels, action_nvec, mapsize, run_dir, **extra):
    """Save all args + env shapes + git hash to config.json for reproducibility."""
    try:
        git_hash = (
            subprocess.check_output(["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL)
            .decode()
            .strip()
        )
    except (FileNotFoundError, subprocess.CalledProcessError):
        git_hash = "unknown"

    config = vars(args).copy()
    # Add env-derived values not in CLI args
    config.update(
        {
            "obs_channels": obs_channels,
            "action_nvec": action_nvec,
            "mapsize": mapsize,
            "git_hash": git_hash,
        }
    )
    config.update(extra)

    config_path = os.path.join(run_dir, "config.json")
    with open(config_path, "w") as f:
        json.dump(config, f, indent=2)
    return config_path


# ── Config banner ────────────────────────────────────────────────────────


def print_config_banner(args, run_dir, title="PPO", total_params=None):
    """Print all training config to console, grouped by category."""
    sep = "=" * 60
    print(f"\n{sep}")
    print(f"  {title} + {args.architecture}  —  MicroRTS")
    print(f"{sep}")

    if args.baseline:
        print("  MODE:         BASELINE (original paper config)")

    # ── Run identity ──
    print(f"  Exp name:     {args.exp_name}")
    print(f"  Seed:         {args.seed}")
    arch_str = args.architecture
    if args.arch_channels:
        arch_str += f" (channels={args.arch_channels})"
    extras = []
    if args.gelu:
        extras.append("GELU")
    if args.spp_critic:
        extras.append("SPP-critic")
    if extras:
        arch_str += f" [{', '.join(extras)}]"
    if total_params is not None:
        arch_str += f" ({total_params:,} params)"
    print(f"  Architecture: {arch_str}")

    # ── Environment ──
    if args.opponent_list:
        print(f"  Opponents:    CUSTOM ({args.opponent_list})")
    elif args.diverse_opponents:
        print(f"  Opponents:    diverse (rush_per_type={args.rush_per_type})")
    else:
        print(f"  Opponents:    {args.opponent}")
    print(f"  Bot envs:     {args.num_bot_envs}")
    print(f"  Opp log win:  {args.opp_log_window}")
    print(
        f"  Self-play:    {args.num_selfplay_envs}"
        + (
            f" ({args.num_latest_selfplay_envs} latest)"
            if args.num_latest_selfplay_envs > 0
            else ""
        )
    )
    print(f"  Map:          {args.map}")
    print(f"  Max steps:    {args.max_steps}")

    if args.multi_map:
        pool = args.map_pool
        plr_str = " + PLR" if args.plr else ""
        print(
            f"  Multi-map:    ON ({len(pool)} maps, max_size={args.max_map_size}, "
            f"switch every {args.map_switch_freq}{plr_str})"
        )
        for mp in pool:
            print(f"                - {mp}")
        if args.plr:
            print(f"  PLR:          window={args.plr_window}")

    print(f"  Partial obs:  {args.partial_obs}")
    if args.extended_obs:
        print("  Extended obs: True (RAI-style 73ch)")

    # ── Observation wrappers ──
    if args.filtered_masks:
        print("  Mask filter:  destination-aware")
    if args.reserved_obs:
        print("  Reserved obs: extra channel for reserved positions")
    if args.frame_stack > 0:
        print(f"  Frame stack:  {args.frame_stack} frames")
    if args.alternate_players:
        print("  P0/P1:        alternate each episode")
    if args.augment_symmetry:
        print("  Augment:      symmetry (random H/V flip)")

    # ── PPO hyperparameters ──
    print(f"  Total steps:  {args.total_timesteps:,}")
    print(f"  Batch size:   {args.batch_size} ({args.num_steps} steps x {args.num_envs} envs)")
    print(f"  Minibatches:  {args.n_minibatch}")
    print(f"  PPO epochs:   {args.update_epochs}")
    print(f"  Updates:      {args.num_updates}")

    lr_str = f"{args.learning_rate}"
    if args.anneal_lr:
        lr_end = args.learning_rate_end if args.learning_rate_end is not None else 0.0
        lr_str += f" -> {lr_end} (annealed)"
    print(f"  LR:           {lr_str}")

    print(f"  Gamma:        {args.gamma}")
    print(f"  GAE lambda:   {args.gae_lambda}")
    print(f"  Clip coef:    {args.clip_coef}")
    print(f"  Clip vloss:   {args.clip_vloss}")
    print(f"  Max grad:     {args.max_grad_norm}")
    print(f"  Norm adv:     {args.norm_adv}")

    if args.ent_coef_end is not None:
        print(f"  Entropy:      {args.ent_coef} -> {args.ent_coef_end}")
    else:
        print(f"  Entropy:      {args.ent_coef}")

    # ── Reward ──
    if args.reward_schedule != "none":
        print(f"  Reward:       {args.reward_weight} -> pure WinLoss ({args.reward_schedule})")
        print(f"  Rew sched:    {args.reward_schedule_start:.0%} - {args.reward_schedule_end:.0%}")
    else:
        print(f"  Reward:       {args.reward_weight}")

    # ── Value heads ──
    if args.triple_value_heads:
        print("  Value heads:  triple (shaped + sparse + cost)")
        print(
            f"    VF coefs:   shaped={args.vf_coef}->{args.vf_coef_end}, "
            f"sparse={args.vf_coef_sparse}->{args.vf_coef_sparse_end}, "
            f"cost={args.vf_coef_cost}->{args.vf_coef_cost_end}"
        )
        print(
            f"    Gammas:     shaped={args.gamma}, sparse={args.gamma_sparse}, cost={args.gamma_cost}"
        )
        print(
            f"    GAE lambda: shaped={args.gae_lambda}, sparse={args.gae_lambda_sparse}, cost={args.gae_lambda_cost}"
        )
    elif args.dual_value_heads:
        print("  Value heads:  dual (shaped + sparse)")
        print(
            f"    VF coefs:   shaped={args.vf_coef}->{args.vf_coef_end}, "
            f"sparse={args.vf_coef_sparse}->{args.vf_coef_sparse_end}"
        )
        print(f"    Gammas:     shaped={args.gamma}, sparse={args.gamma_sparse}")
    else:
        print(f"  VF coef:      {args.vf_coef}")

    if args.advantage_weights is not None:
        aw_str = f"{list(args.advantage_weights)}"
        if args.advantage_weights_end is not None:
            aw_str += f" -> {list(args.advantage_weights_end)}"
        print(f"  Adv weights:  {aw_str}")

    # ── Multi-phase schedule ──
    if args.phase_steps is not None:
        n_phases = len(args.phase_steps) // 2 + 1
        print(f"  Phases:       {n_phases} phases, boundaries={args.phase_steps}")
        if args.phase_lr is not None:
            print(f"    Phase LR:   {args.phase_lr}")
        if args.phase_ent_coef is not None:
            print(f"    Phase ent:  {args.phase_ent_coef}")
        if args.phase_adv_weights is not None:
            print(f"    Phase adv:  {args.phase_adv_weights}")
        if args.phase_vf_coefs is not None:
            print(f"    Phase VF:   {args.phase_vf_coefs}")

    # ── Advanced features ──
    if args.hl_gauss:
        print(f"  HL-Gauss:     {args.hl_gauss_bins} bins")
    if args.autoregressive:
        print(f"  Auto-reg:     embed_dim={args.ar_embed_dim}")
    if args.popart:
        print("  PopArt:       adaptive value normalization")
    if args.pae_keep > 0:
        pae_pct = args.pae_cutoff / args.num_steps * 100
        print(
            f"  PAE:          keep first {args.pae_cutoff}/{args.num_steps} steps ({pae_pct:.0f}%)"
        )

    if args.aux_tasks:
        coef_map = {
            "spatial": args.aux_spatial_coef,
            "unit_count": args.aux_unit_count_coef,
            "contrastive": args.aux_contrastive_coef,
            "opponent_modeling": args.aux_opponent_modeling_coef,
        }
        for task in args.aux_tasks:
            print(f"  Aux task:     {task} (coef={coef_map[task]})")

    # ── Opponent scheduling ──
    if args.adaptive_opponents:
        print(f"  Adaptive:     threshold={args.adaptive_threshold}, window={args.adaptive_window}")
    if args.adaptive_selfplay:
        print(f"  Adapt SP:     pool delayed until terminal WR>={args.adaptive_selfplay_threshold}")
    if args.prioritized_sampling:
        print(f"  Prioritized:  window={args.priority_window}")
    if args.selfplay_pool_size > 0 and args.num_selfplay_envs > 0:
        print(
            f"  SP pool:      size={args.selfplay_pool_size}, save_every={args.selfplay_save_interval}, "
            f"swap_every={args.selfplay_swap_interval}"
        )
    if args.pfsp:
        print(f"  PFSP:         mode={args.pfsp_mode}, p={args.pfsp_p}")

    # ── Infra ──
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"  Device:       {device}")
    print(f"  Checkpoints:  {args.num_models}")
    print(f"  Run dir:      {run_dir}")
    if args.load_model:
        print(f"  Load model:   {args.load_model}")
    if args.resume:
        print(f"  Resume from:  {args.resume}")
    if args.eval_interval > 0:
        print(
            f"  Eval every:   ~{args.eval_interval * args.batch_size / 1e6:.1f}M steps ({args.eval_interval} updates)"
        )
        print(f"  Eval bots:    {', '.join(args.eval_bots)} ({args.eval_games} games each)")

    print(f"{sep}\n")
