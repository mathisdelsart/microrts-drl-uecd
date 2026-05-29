"""
CLI argument definitions, parsing, and validation for training.

Single source of truth for all argparse flags, constants, and
post-parse helpers.
"""

import argparse

from microrts_agent.lib.arch_factory import ARCHITECTURE_REGISTRY
from microrts_agent.lib.mappings.ai import AI_MAPPING
from microrts_agent.lib.mappings.maps import COMPETITION_OPEN_MAPS


def strtobool(x):
    """Replace deprecated distutils.util.strtobool."""
    return str(x).lower() in ("1", "true", "yes")


# ── Constants ─────────────────────────────────────────────────────────────


# ── Shared argparse definitions ───────────────────────────────────────────


def add_common_args(parser):
    """Add CLI arguments shared across training scripts."""

    # Architecture & wrappers
    parser.add_argument(
        "--architecture",
        type=str,
        default="gridnet",
        choices=list(ARCHITECTURE_REGISTRY.keys()),
        help="agent architecture (default: gridnet)",
    )
    parser.add_argument(
        "--arch-channels",
        type=int,
        default=None,
        help="channel width for architectures that support it, e.g. doublecone"
        "(default: architecture-specific)",
    )
    parser.add_argument(
        "--constant-channels",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="use constant channel width (C,C,C) instead of expanding (C,2C,4C) (default: False)",
    )
    # Experiment
    parser.add_argument(
        "--exp-name", type=str, default=None, help="experiment name (default: auto-generated)"
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed")

    # Environment
    parser.add_argument(
        "--opponent",
        type=str,
        default="coacAI",
        choices=list(AI_MAPPING.keys()),
        help="opponent bot when not using --diverse-opponents (default: coacAI)",
    )
    parser.add_argument(
        "--diverse-opponents",
        type=strtobool,
        default=True,
        nargs="?",
        const=True,
        help="use diverse opponents (default: True)",
    )
    parser.add_argument(
        "--opponent-list",
        type=str,
        default=None,
        help='custom opponent distribution, e.g. "CoacAI:3,Mayari:3,POWorkerRush:2".'
        "Overrides --opponent and --diverse-opponents."
        "Sum of counts must equal --num-bot-envs.",
    )
    parser.add_argument(
        "--num-bot-envs", type=int, default=24, help="number of parallel bot envs (default: 24)"
    )
    parser.add_argument(
        "--rush-per-type",
        type=int,
        default=2,
        help="max envs per rush bot type in diverse opponent mix (default: 2)",
    )
    parser.add_argument(
        "--opp-log-window",
        type=int,
        default=50,
        help="sliding window for per-opponent win rate logging (default: 50)",
    )
    parser.add_argument(
        "--partial-obs",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="fog-of-war (default: False)",
    )
    parser.add_argument(
        "--map", type=str, default="maps/16x16/basesWorkers16x16A.xml", help="map path"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="max game steps (default: 3000)"
    )
    parser.add_argument(
        "--reward-weight",
        nargs="+",
        type=float,
        default=[10.0, 1.0, 1.0, 0.2, 0.2, 1.0, 4.0, 4.0, 4.0, 0.0],
        help="reward weights [winloss resource worker base barracks attack light heavy ranged military]",
    )
    parser.add_argument(
        "--reward-unit-build-time",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="scale military unit rewards by build-time (Light=4.0, Ranged=5.25, Heavy=6.0) (default: False)",
    )
    parser.add_argument(
        "--reward-schedule",
        type=str,
        default="none",
        choices=["none", "linear"],
        help="reward weight schedule (none=fixed, linear=interpolate to pure WinLoss)",
    )
    parser.add_argument(
        "--reward-schedule-start",
        type=float,
        default=0.3,
        help="fraction of training where reward transition starts",
    )
    parser.add_argument(
        "--reward-schedule-end",
        type=float,
        default=0.7,
        help="fraction of training where reward transition ends",
    )
    parser.add_argument(
        "--reward-schedule-min",
        type=float,
        default=0.0,
        help="minimum fraction of shaped reward to keep (0.0=pure WinLoss, 0.1=keep 10%% shaped)",
    )
    parser.add_argument(
        "--reward-weight-end",
        type=float,
        nargs="+",
        default=None,
        help="target reward weights for linear schedule (overrides --reward-schedule-min); "
        "must have same length as --reward-weight",
    )

    # PPO hyperparameters
    parser.add_argument(
        "--total-timesteps", type=int, default=300000000, help="total env steps (default: 300M)"
    )
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="initial learning rate (default: 1e-4)"
    )
    parser.add_argument(
        "--learning-rate-end",
        type=float,
        default=None,
        help="final learning rate for linear decay (None=anneal to 0 if --anneal-lr, default: None)",
    )
    parser.add_argument(
        "--num-steps",
        type=int,
        default=512,
        help="rollout length per env per update (default: 512)",
    )
    parser.add_argument("--gamma", type=float, default=0.99, help="discount factor")
    parser.add_argument("--gae-lambda", type=float, default=0.95, help="GAE lambda")
    parser.add_argument("--n-minibatch", type=int, default=3, help="number of minibatches")
    parser.add_argument("--update-epochs", type=int, default=2, help="PPO epochs per update")
    parser.add_argument("--clip-coef", type=float, default=0.1, help="PPO clip coef")
    parser.add_argument("--ent-coef", type=float, default=0.01, help="entropy coef")
    parser.add_argument(
        "--ent-coef-end",
        type=float,
        default=None,
        help="final entropy coef (None=no annealing, uses --ent-coef throughout)",
    )
    parser.add_argument(
        "--vf-coef", type=float, default=0.5, help="value loss coef for shaped head"
    )
    parser.add_argument(
        "--vf-coef-end",
        type=float,
        default=0.0,
        help="value loss coef for shaped head at end (default: 0.0)",
    )
    parser.add_argument(
        "--dual-value-heads",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="dual critics: shaped (gamma) + sparse (gamma-sparse) with blended advantage",
    )
    parser.add_argument(
        "--gamma-sparse",
        type=float,
        default=0.999,
        help="discount factor for sparse (win/loss) value head (default: 0.999)",
    )
    parser.add_argument(
        "--gae-lambda-sparse",
        type=float,
        default=0.99,
        help="GAE lambda for sparse value head — higher = longer horizon (default: 0.99)",
    )
    parser.add_argument(
        "--vf-coef-sparse",
        type=float,
        default=0.1,
        help="value loss coef for sparse head at start (default: 0.1, RAISocketAI)",
    )
    parser.add_argument(
        "--vf-coef-sparse-end",
        type=float,
        default=0.5,
        help="value loss coef for sparse head at end (default: 0.5)",
    )
    parser.add_argument("--max-grad-norm", type=float, default=0.5, help="max grad norm")
    parser.add_argument(
        "--lr-warmup-updates",
        type=int,
        default=0,
        help="number of updates for linear LR warmup from 0 to learning-rate (default: 0 = no warmup)",
    )
    parser.add_argument(
        "--anneal-lr",
        type=strtobool,
        default=True,
        nargs="?",
        const=True,
        help="anneal LR (default: True)",
    )
    parser.add_argument(
        "--norm-adv",
        type=strtobool,
        default=True,
        nargs="?",
        const=True,
        help="normalize advantages",
    )
    parser.add_argument(
        "--clip-vloss", type=strtobool, default=True, nargs="?", const=True, help="clip value loss"
    )

    # Data augmentation
    parser.add_argument(
        "--alternate-players",
        type=strtobool,
        default=True,
        nargs="?",
        const=True,
        help="alternate P0/P1 each episode (default: True). "
        "Java already does owner-relative encoding.",
    )
    parser.add_argument(
        "--filtered-masks",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="destination-aware action mask filtering (default: False)",
    )
    parser.add_argument(
        "--reserved-obs",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="add reserved-positions obs channel (requires --filtered-masks) (default: False)",
    )
    parser.add_argument(
        "--extended-obs",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="RAI-style 73-channel extended obs: mixed continuous/one-hot encoding, "
        "action params, ETA, global resource planes "
        "(incompatible with --partial-obs) (default: False)",
    )
    parser.add_argument(
        "--frame-stack",
        type=int,
        default=0,
        help="stack last N observations along channel axis for temporal info (0=disabled, default: 0)",
    )

    # Activation
    parser.add_argument(
        "--gelu",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="use GELU activations instead of ReLU in architecture (default: False)",
    )

    # Critic architecture
    parser.add_argument(
        "--spp-critic",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="use Spatial Pyramid Pooling in critic for scale-invariant value estimation (default: False)",
    )

    # HL-Gauss (value function via classification)
    parser.add_argument(
        "--hl-gauss",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="use HL-Gauss categorical value heads instead of scalar regression (default: False)",
    )
    parser.add_argument(
        "--hl-gauss-bins",
        type=int,
        default=255,
        help="number of bins for HL-Gauss value distribution (default: 255)",
    )

    # PopArt value normalization
    parser.add_argument(
        "--popart",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="PopArt adaptive value normalization for multi-scale returns "
        "(mutually exclusive with --hl-gauss) (default: False)",
    )

    # Partial Advantage Estimator
    parser.add_argument(
        "--pae-keep",
        type=float,
        default=0.0,
        help="PAE: fraction of rollout to keep for incomplete episodes "
        "(0.0 = disabled, 0.9 = keep first 90%%). Cutoff step = "
        "int(num_steps * pae_keep). Adapts automatically to rollout length.",
    )

    # BC teacher KL penalty (AlphaStar-style)
    parser.add_argument(
        "--bc-teacher-model",
        type=str,
        default=None,
        help="path to BC teacher agent.pt for KL penalty during PPO fine-tuning "
        "(default: None = disabled)",
    )
    parser.add_argument(
        "--bc-kl-coef",
        type=float,
        default=1.0,
        help="initial KL penalty coefficient (default: 1.0)",
    )
    parser.add_argument(
        "--bc-kl-anneal-steps",
        type=int,
        default=50000000,
        help="steps over which to anneal KL coef from bc-kl-coef to 0 (default: 50M)",
    )

    # Auto-regressive sub-actions
    parser.add_argument(
        "--autoregressive",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="auto-regressive sub-action sampling: each sub-action conditioned on previous (default: False)",
    )
    parser.add_argument(
        "--ar-embed-dim",
        type=int,
        default=8,
        help="embedding dimension per sub-action for auto-regressive head (default: 8)",
    )

    # Hierarchical sub-action masking (RAISocketAI-style): zero out log_prob/entropy
    # of sub-actions whose parent action_type doesn't activate them.
    # Default: None → auto-enabled when --autoregressive is True, disabled otherwise.
    parser.add_argument(
        "--hierarchical-mask",
        type=strtobool,
        default=None,
        nargs="?",
        const=True,
        help="zero log_prob/entropy of inactive sub-actions given action_type "
        "(default: auto — enabled when --autoregressive is set)",
    )

    # Auxiliary tasks
    parser.add_argument(
        "--aux-spatial",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="auxiliary task: predict spatial ownership map (default: False)",
    )
    parser.add_argument(
        "--aux-spatial-coef",
        type=float,
        default=0.1,
        help="loss coefficient for spatial ownership aux task (default: 0.1)",
    )
    parser.add_argument(
        "--aux-contrastive",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="auxiliary task: temporal contrastive loss (InfoNCE) on encoder embeddings (default: False)",
    )
    parser.add_argument(
        "--aux-contrastive-coef",
        type=float,
        default=0.05,
        help="loss coefficient for contrastive aux task (default: 0.05)",
    )
    parser.add_argument(
        "--aux-unit-count",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="auxiliary task: predict unit counts per player (6 types × 2 players) (default: False)",
    )
    parser.add_argument(
        "--aux-unit-count-coef",
        type=float,
        default=0.1,
        help="loss coefficient for unit count aux task (default: 0.1)",
    )
    parser.add_argument(
        "--aux-opponent-modeling",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="auxiliary task: predict enemy unit action types per cell (default: False)",
    )
    parser.add_argument(
        "--aux-opponent-modeling-coef",
        type=float,
        default=0.05,
        help="loss coefficient for opponent modeling aux task (default: 0.05)",
    )

    # Triple value heads (shaped + sparse + cost)
    parser.add_argument(
        "--triple-value-heads",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="triple critics: shaped + sparse (win/loss) + cost (army differential) value heads (default: False)",
    )
    parser.add_argument(
        "--gamma-cost",
        type=float,
        default=0.999,
        help="discount factor for cost value head (default: 0.999)",
    )
    parser.add_argument(
        "--gae-lambda-cost",
        type=float,
        default=0.99,
        help="GAE lambda for cost value head (default: 0.99)",
    )
    parser.add_argument(
        "--vf-coef-cost",
        type=float,
        default=0.2,
        help="value loss coef for cost head (default: 0.2, RAISocketAI Phase 1)",
    )
    parser.add_argument(
        "--vf-coef-cost-end",
        type=float,
        default=0.1,
        help="value loss coef for cost head at end (default: 0.1, RAISocketAI Phase 2+)",
    )

    # Advantage weight scheduling
    parser.add_argument(
        "--advantage-weights",
        type=float,
        nargs="+",
        default=None,
        help="advantage weights per value head [shaped, WL] or [shaped, WL, cost] "
        "(must sum to 1.0, default: [0.5, 0.5] for dual, [0.5, 0.3, 0.2] for triple)",
    )
    parser.add_argument(
        "--advantage-weights-end",
        type=float,
        nargs="+",
        default=None,
        help="end advantage weights per value head (must sum to 1.0, "
        "transition uses --reward-schedule-start/end fractions)",
    )

    # Multi-phase schedule (overrides legacy 2-point schedule when set)
    parser.add_argument(
        "--phase-steps",
        type=int,
        nargs="+",
        default=None,
        help="transition boundaries in absolute steps: s1 e1 s2 e2 ... "
        "Each (s_i, e_i) pair defines a linear transition between phases. "
        "Example: 90000000 150000000 180000000 240000000 for 3 phases with 2 transitions",
    )
    parser.add_argument(
        "--phase-adv-weights",
        type=str,
        nargs="+",
        default=None,
        help="advantage weights per phase, comma-separated tuples: "
        '"0.8,0.01,0.19" "0.0,0.5,0.5" "0.0,0.99,0.01"',
    )
    parser.add_argument(
        "--phase-vf-coefs",
        type=str,
        nargs="+",
        default=None,
        help="VF loss coefs per phase [shaped,sparse,cost]: "
        '"0.5,0.1,0.2" "0.0,0.4,0.4" "0.0,0.5,0.1"',
    )
    parser.add_argument(
        "--phase-ent-coef",
        type=float,
        nargs="+",
        default=None,
        help="entropy coef per phase: 0.01 0.01 0.001",
    )
    parser.add_argument(
        "--phase-lr",
        type=float,
        nargs="+",
        default=None,
        help="learning rate per phase: 1e-4 1e-4 5e-5",
    )

    # Saving / loading
    parser.add_argument(
        "--num-models", type=int, default=100, help="number of checkpoints to save during training"
    )
    parser.add_argument(
        "--load-model", type=str, default=None, help="path to pretrained weights (for BC warmstart)"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="path to run dir to resume training (e.g. outputs/runs/my_exp)",
    )

    # Evaluation during training
    parser.add_argument(
        "--with-eval",
        type=strtobool,
        default=True,
        nargs="?",
        const=True,
        help="enable periodic evaluation during training (default: True)",
    )
    parser.add_argument(
        "--eval-bots",
        nargs="+",
        type=str,
        default=[
            "RandomBiasedAI",
            "POWorkerRush",
            "POLightRush",
            "CoacAI",
            "Mayari",
        ],
        help="bots to evaluate against (default: RandomBiasedAI POWorkerRush POLightRush CoacAI Mayari)",
    )
    parser.add_argument(
        "--eval-games",
        type=int,
        default=10,
        help="games per bot per eval round, split P0/P1 (default: 10)",
    )

    # Baseline mode
    parser.add_argument(
        "--baseline",
        action="store_true",
        help="use original paper config: P0 only, original opponent mix, no alternate_players",
    )

    # Self-play envs
    parser.add_argument(
        "--num-selfplay-envs", type=int, default=0, help="number of self-play envs (default: 0)"
    )
    parser.add_argument(
        "--num-latest-selfplay-envs",
        type=int,
        default=0,
        help="of --num-selfplay-envs, how many play against the LATEST model "
        "(rest play against checkpoint pool). Must be even. (default: 0 = all use pool)",
    )

    # Multi-map
    parser.add_argument(
        "--multi-map",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="enable multi-map training with padded observations (default: False)",
    )
    parser.add_argument(
        "--map-pool",
        nargs="+",
        type=str,
        default=COMPETITION_OPEN_MAPS,
        help="list of map paths for multi-map training (default: 8 open competition maps)",
    )
    parser.add_argument(
        "--max-map-size",
        type=int,
        default=64,
        help="max height/width for observation padding (default: 64)",
    )
    parser.add_argument(
        "--map-switch-freq", type=int, default=50, help="switch map every N updates (default: 50)"
    )
    parser.add_argument(
        "--plr",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="prioritized level replay for multi-map: sample maps by learning signal "
        "instead of cycling (requires --multi-map) (default: False)",
    )
    parser.add_argument(
        "--plr-window",
        type=int,
        default=50,
        help="sliding window for per-map win rate tracking in PLR (default: 50)",
    )

    # Prioritized opponent sampling
    parser.add_argument(
        "--prioritized-sampling",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="weight PPO transitions by opponent difficulty (default: False)",
    )
    parser.add_argument(
        "--priority-window",
        type=int,
        default=100,
        help="window size for per-opponent win rate tracking (default: 100)",
    )

    # Adaptive opponent scheduling
    parser.add_argument(
        "--adaptive-opponents",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="promote mastered bot types to harder opponents (default: False)",
    )
    parser.add_argument(
        "--adaptive-threshold",
        type=float,
        default=0.7,
        help="WR threshold to consider a bot type mastered (default: 0.7)",
    )
    parser.add_argument(
        "--adaptive-window",
        type=int,
        default=50,
        help="sliding window size for mastery detection (default: 50)",
    )
    parser.add_argument(
        "--adaptive-criteria",
        type=str,
        default="strict",
        choices=["strict", "average", "hybrid"],
        help="multi-map mastery criterion: strict = all maps >= threshold "
        "(original), average = global mean >= threshold, "
        "hybrid = global mean >= threshold AND min per-map >= "
        "--adaptive-hybrid-min (default: strict)",
    )
    parser.add_argument(
        "--adaptive-hybrid-min",
        type=float,
        default=0.4,
        help="minimum per-map WR required under --adaptive-criteria hybrid (default: 0.4)",
    )
    parser.add_argument(
        "--adaptive-selfplay",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="activate self-play pool only when terminal opponents mastered "
        "(requires --adaptive-opponents and --num-selfplay-envs > 0) (default: False)",
    )
    parser.add_argument(
        "--adaptive-selfplay-threshold",
        type=float,
        default=0.9,
        help="WR threshold for terminal-tier bots to activate self-play pool (default: 0.9)",
    )

    # Historical self-play
    parser.add_argument(
        "--selfplay-pool-size",
        type=int,
        default=0,
        help="historical self-play pool size (0=disabled, uses current model for all self-play)",
    )
    parser.add_argument(
        "--selfplay-save-interval",
        type=int,
        default=750,
        help="save checkpoint to pool every N updates (~9M steps with 24 envs × 512 steps)",
    )
    parser.add_argument(
        "--selfplay-swap-interval",
        type=int,
        default=50,
        help="load new opponent from pool every N updates (~600K steps with 24 envs × 512 steps)",
    )

    # PFSP (Prioritized Fictitious Self-Play)
    parser.add_argument(
        "--pfsp",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="enable PFSP sampling for self-play checkpoint pool (default: False)",
    )
    parser.add_argument(
        "--pfsp-mode",
        type=str,
        default="hard",
        choices=["hard", "var"],
        help="PFSP weighting: hard=focus losses, var=focus 50/50 (default: hard)",
    )
    parser.add_argument(
        "--pfsp-p", type=float, default=1.0, help="exponent for f_hard weighting (default: 1.0)"
    )

    # Data augmentation
    parser.add_argument(
        "--augment-symmetry",
        type=strtobool,
        default=False,
        nargs="?",
        const=True,
        help="random spatial flip augmentation (H/V/both per episode) (default: False)",
    )


# ── Argument parsing ─────────────────────────────────────────────────────


def parse_args():
    """Parse all CLI arguments, validate, and compute derived values."""
    import time

    parser = argparse.ArgumentParser(description="PPO training for MicroRTS")
    add_common_args(parser)
    args = parser.parse_args()

    # --baseline overrides opponent / training config to match original paper
    if args.baseline:
        args.diverse_opponents = True
        args.alternate_players = False
        args.num_bot_envs = 24
        args.num_selfplay_envs = 0

    # Validate latest selfplay split
    n_latest = args.num_latest_selfplay_envs
    if n_latest > 0:
        if n_latest % 2 != 0:
            raise ValueError("--num-latest-selfplay-envs must be even (pairs)")
        if n_latest > args.num_selfplay_envs:
            raise ValueError("--num-latest-selfplay-envs cannot exceed --num-selfplay-envs")
    args.num_pool_selfplay_envs = args.num_selfplay_envs - n_latest

    # Validate eval bots
    for bot in args.eval_bots:
        if bot not in AI_MAPPING:
            raise ValueError(f"Unknown eval bot '{bot}'. Valid: {sorted(AI_MAPPING.keys())}")

    args.num_envs = args.num_selfplay_envs + args.num_bot_envs
    compute_derived_args(args, eval_step_interval=10_000_000)
    build_aux_tasks_list(args)
    validate_advantage_weights(args)
    validate_phase_schedule(args)

    if args.exp_name is None:
        opp = "diverse" if args.diverse_opponents else args.opponent
        args.exp_name = f"ppo_{opp}_s{args.seed}_{int(time.time())}"
    return args


# ── Post-parse helpers ────────────────────────────────────────────────────


def build_aux_tasks_list(args):
    """Build args.aux_tasks list from individual --aux-* flags."""
    # Extended obs is incompatible with partial obs (GameStateWrapper
    # does not produce visibility channels).
    if args.extended_obs and args.partial_obs:
        raise ValueError("--extended-obs and --partial-obs are incompatible")

    # PopArt and HL-Gauss are mutually exclusive
    if args.popart and args.hl_gauss:
        raise ValueError(
            "--popart and --hl-gauss are mutually exclusive (both solve value scale normalization)"
        )

    # Scale military unit rewards by build-time if requested
    # Index: 6=Light(80t→4.0), 7=Heavy(120t→6.0), 8=Ranged(100t→5.25)
    if args.reward_unit_build_time:
        args.reward_weight[7] = 6.0  # Heavy (120 ticks)
        args.reward_weight[8] = 5.25  # Ranged (100 ticks)

    # Triple value heads implies dual value heads
    if args.triple_value_heads:
        args.dual_value_heads = True

    # Hierarchical sub-action masking: default to ON when autoregressive is enabled,
    # OFF otherwise. User can always override explicitly.
    if args.hierarchical_mask is None:
        args.hierarchical_mask = bool(args.autoregressive)

    # Default advantage weights when dual/triple heads used without explicit weights
    if args.dual_value_heads and args.advantage_weights is None:
        if args.triple_value_heads:
            args.advantage_weights = (0.5, 0.3, 0.2)
        else:
            args.advantage_weights = (0.5, 0.5)

    aux_tasks = []
    if args.aux_spatial:
        aux_tasks.append("spatial")
    if args.aux_unit_count:
        aux_tasks.append("unit_count")
    if args.aux_contrastive:
        aux_tasks.append("contrastive")
    if args.aux_opponent_modeling:
        aux_tasks.append("opponent_modeling")
    args.aux_tasks = aux_tasks if aux_tasks else None


def validate_advantage_weights(args):
    """Validate --advantage-weights / --advantage-weights-end consistency."""
    aw = args.advantage_weights
    aw_end = args.advantage_weights_end

    if aw is None and aw_end is not None:
        raise ValueError("--advantage-weights-end requires --advantage-weights")

    if aw is not None:
        # Must have dual or triple value heads
        if not args.dual_value_heads:
            raise ValueError(
                "--advantage-weights requires --dual-value-heads or --triple-value-heads"
            )

        expected_len = 3 if args.triple_value_heads else 2
        if len(aw) != expected_len:
            raise ValueError(
                f"--advantage-weights must have {expected_len} elements "
                f"({'triple' if expected_len == 3 else 'dual'} heads), got {len(aw)}"
            )
        if abs(sum(aw) - 1.0) > 1e-4:
            raise ValueError(f"--advantage-weights must sum to 1.0, got {sum(aw):.4f}")

        # Convert to tuple for immutability
        args.advantage_weights = tuple(aw)

    if aw_end is not None:
        expected_len = 3 if args.triple_value_heads else 2
        if len(aw_end) != expected_len:
            raise ValueError(
                f"--advantage-weights-end must have {expected_len} elements, got {len(aw_end)}"
            )
        if abs(sum(aw_end) - 1.0) > 1e-4:
            raise ValueError(f"--advantage-weights-end must sum to 1.0, got {sum(aw_end):.4f}")
        args.advantage_weights_end = tuple(aw_end)


def validate_phase_schedule(args):
    """Validate and parse multi-phase schedule args.

    Parses comma-separated string args (--phase-adv-weights, --phase-vf-coefs)
    into tuples of floats. Validates consistency with --phase-steps.
    """
    phase_steps = args.phase_steps
    if phase_steps is None:
        return

    n_boundaries = len(phase_steps)
    if n_boundaries % 2 != 0:
        raise ValueError(
            f"--phase-steps must have an even number of values (pairs of start,end), "
            f"got {n_boundaries}"
        )
    n_phases = n_boundaries // 2 + 1

    # Validate monotonicity
    for i in range(n_boundaries - 1):
        if phase_steps[i] >= phase_steps[i + 1]:
            raise ValueError(f"--phase-steps must be strictly increasing, got {phase_steps}")

    # Validate each (start, end) pair
    for i in range(n_boundaries // 2):
        if phase_steps[2 * i] >= phase_steps[2 * i + 1]:
            raise ValueError(
                f"--phase-steps transition {i + 1}: start ({phase_steps[2 * i]}) "
                f"must be < end ({phase_steps[2 * i + 1]})"
            )

    # Parse --phase-adv-weights: "0.8,0.01,0.19" "0.0,0.5,0.5" → [tuple, tuple, ...]
    raw_aw = args.phase_adv_weights
    if raw_aw is not None:
        parsed = [tuple(float(x) for x in s.split(",")) for s in raw_aw]
        if len(parsed) != n_phases:
            raise ValueError(
                f"--phase-adv-weights needs {n_phases} entries (one per phase), got {len(parsed)}"
            )
        for i, p in enumerate(parsed):
            if abs(sum(p) - 1.0) > 1e-4:
                raise ValueError(
                    f"--phase-adv-weights entry {i + 1} must sum to 1.0, got {p} (sum={sum(p):.4f})"
                )
        args.phase_adv_weights = parsed

    # Parse --phase-vf-coefs: "0.5,0.1,0.2" "0.0,0.4,0.4" → [tuple, tuple, ...]
    raw_vf = args.phase_vf_coefs
    if raw_vf is not None:
        parsed = [tuple(float(x) for x in s.split(",")) for s in raw_vf]
        if len(parsed) != n_phases:
            raise ValueError(
                f"--phase-vf-coefs needs {n_phases} entries (one per phase), got {len(parsed)}"
            )
        args.phase_vf_coefs = parsed

    # Validate scalar arrays
    for flag, attr in [("--phase-ent-coef", "phase_ent_coef"), ("--phase-lr", "phase_lr")]:
        val = getattr(args, attr)
        if val is not None and len(val) != n_phases:
            raise ValueError(f"{flag} needs {n_phases} entries (one per phase), got {len(val)}")

    # Validate: --phase-steps with --anneal-lr requires --phase-lr
    if args.anneal_lr and args.phase_lr is None:
        raise ValueError(
            "--phase-steps with --anneal-lr requires --phase-lr to define "
            "per-phase learning rates. Without --phase-lr, the LR schedule "
            "is ambiguous (classic annealing ignores phase boundaries)."
        )

    # Multi-phase overrides legacy 2-point schedule
    if args.advantage_weights_end is not None:
        import warnings

        warnings.warn(
            "--phase-steps overrides --advantage-weights-end. "
            "The legacy 2-point schedule will be ignored.",
            stacklevel=2,
        )
        args.advantage_weights_end = None


def compute_derived_args(args, eval_step_interval=10_000_000):
    """Compute derived args (batch size, update count, etc.) from parsed args.

    Caller must set args.num_envs before calling this.
    """
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.n_minibatch)
    # PAE: derive absolute cutoff from keep ratio
    pae_keep = args.pae_keep
    if pae_keep > 0:
        if not (0.0 < pae_keep < 1.0):
            import warnings

            warnings.warn(
                f"--pae-keep must be in (0, 1), got {pae_keep}. PAE disabled.", stacklevel=2
            )
            pae_keep = 0.0
        args.pae_cutoff = int(args.num_steps * pae_keep)
    else:
        args.pae_cutoff = 0

    if args.pae_cutoff > 0 and args.pae_cutoff >= args.num_steps:
        import warnings

        warnings.warn(
            f"PAE cutoff ({args.pae_cutoff}) >= --num-steps ({args.num_steps}), "
            f"PAE disabled (no truncation to apply)",
            stacklevel=2,
        )
        args.pae_cutoff = 0

    args.num_updates = args.total_timesteps // args.batch_size
    args.save_frequency = max(1, int(args.num_updates // args.num_models))
    args.eval_interval = max(1, eval_step_interval // args.batch_size) if args.with_eval else 0
