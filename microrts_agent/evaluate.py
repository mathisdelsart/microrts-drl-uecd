"""
Batch evaluation of a trained agent vs bots (or agent-vs-agent), always vectorized.

Games run in parallel even with --record (recording switches render_client to each
sub-env before capture).

    python -m microrts_agent evaluate --agent data/agents/UECD-SingleMap-Best --opponent CoacAI
    python -m microrts_agent evaluate --agent data/agents/UECD-SingleMap-Best --opponent CoacAI --record
"""

import argparse
import contextlib
import io
import os
import sys
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import torch

try:
    from gymnasium.wrappers.monitoring.video_recorder import VideoRecorder  # type: ignore
except ImportError:
    # gymnasium >= 1.2 removed monitoring.video_recorder. Only --record needs
    # it; everything else (incl. --help) keeps working without this symbol.
    VideoRecorder = None  # type: ignore

from microrts_agent.architectures.factory import load_agent_from_config
from microrts_agent.envs.base_vec_env import get_base_env as _get_base_env
from microrts_agent.envs.base_vec_env import suppress_java_output
from microrts_agent.envs.factory import JVM_ARGS, make_agent_env, make_bot_env
from microrts_agent.obs_adapter import ObsAdapter as _ObsAdapter

# ── Helpers ───────────────────────────────────────────────────────────────
from microrts_agent.obs_adapter import is_agent_dir
from microrts_agent.obs_adapter import process_obs_rl_vs_rl as _process_obs_rl_vs_rl
from microrts_agent.paths import RECORDINGS_DIR
from microrts_agent.registries.ai import AI_MAPPING
from microrts_agent.registries.maps import map_short
from microrts_agent.tournament.servers import (
    needs_uts_imass,
    run_uts_imass_preanalysis,
    start_uts_imass_server,
    stop_uts_imass_server,
)
from microrts_agent.wrappers.factory import apply_env_wrappers


def _padding_kwargs(config):
    """Multi-map padding kwargs from config."""
    if config.get("multi_map", False):
        ms = config.get("max_map_size", 64)
        return {"padded": True, "max_height": ms, "max_width": ms}
    return {"padded": False, "max_height": 0, "max_width": 0}


def _apply_wrappers(env, config):
    """Apply obs wrappers matching training config (no augmentation)."""
    if not config:
        return env
    return apply_env_wrappers(
        env,
        gamma=config.get("gamma", 0.99),
        frame_stack=config.get("frame_stack", 0),
        reserved_obs=config.get("reserved_obs", False),
    )


def wl_to_result(wl, player=0):
    """WinLoss reward -> absolute result (P0_WIN / P1_WIN / DRAW)."""
    if player == 1:
        wl = -wl
    if wl > 0:
        return "P0_WIN"
    elif wl < 0:
        return "P1_WIN"
    return "DRAW"


def _interpret_result(raw_result, swap):
    """Absolute result -> agent perspective (WIN / LOSS / DRAW)."""
    if raw_result == "DRAW":
        return "DRAW"
    if not swap:
        return "WIN" if raw_result == "P0_WIN" else "LOSS"
    return "WIN" if raw_result == "P1_WIN" else "LOSS"


def _record_result(stats, result, steps):
    """Update PositionStats with a game result."""
    if result == "WIN":
        stats.wins += 1
    elif result == "LOSS":
        stats.losses += 1
    else:
        stats.draws += 1
    stats.games += 1
    stats.game_lengths.append(steps)


def get_actions(model, obs, masks, deterministic=False):
    """Forward pass: stochastic sampling or greedy argmax."""
    with torch.no_grad():
        if deterministic:
            return model.predict_batch(obs, masks)
        return model.forward(obs, invalid_action_masks=masks)["action"]


# ── Modes ─────────────────────────────────────────────────────────────────

MODE_BOT_VS_BOT = "bot_vs_bot"
MODE_RL_VS_BOT = "rl_vs_bot"
MODE_RL_VS_RL = "rl_vs_rl"


# ── Data ──────────────────────────────────────────────────────────────────


@dataclass
class SessionConfig:
    """Parsed + validated session configuration."""

    agent_raw: str
    opponent_raw: str
    nb_games: int
    deterministic: bool
    record: bool
    mode: str
    agent_name: str
    opponent_name: str
    maps: list[str]
    max_steps_per_map: list[int]
    device: torch.device = None
    rl_player: Optional[int] = None  # rl_vs_bot: which side is RL
    bot_name: Optional[str] = None  # rl_vs_bot: the Java bot name
    agent_model: object = None
    opponent_model: object = None
    agent_config: dict = None
    opponent_config: dict = None

    @property
    def positions(self):
        """(swap, num_games, label) for P0 and P1 evaluation."""
        return [(False, self.nb_games, "P0"), (True, self.nb_games, "P1")]


@dataclass
class PositionStats:
    """W/L/D stats for one position (P0 or P1)."""

    label: str
    wins: int = 0
    losses: int = 0
    draws: int = 0
    games: int = 0
    game_lengths: list[int] = field(default_factory=list)

    @property
    def win_rate(self):
        return self.wins / self.games if self.games > 0 else 0

    @property
    def avg_length(self):
        return float(np.mean(self.game_lengths)) if self.game_lengths else 0


# ── CLI ───────────────────────────────────────────────────────────────────


def parse_config() -> SessionConfig:
    """Parse CLI, detect mode (bot_vs_bot / rl_vs_bot / rl_vs_rl), return config."""
    p = argparse.ArgumentParser(description="Evaluate MicroRTS games")
    p.add_argument("--agent", required=True, help="bot name or path to RL run dir")
    p.add_argument("--opponent", required=True, help="bot name or path to RL run dir")
    p.add_argument(
        "--nb_games", type=int, default=5, help="games per map per position (default: 5)"
    )
    p.add_argument(
        "--maps",
        nargs="+",
        default=["maps/open_competition/basesWorkers16x16A.xml"],
        help="map path(s) (default: basesWorkers16x16A)",
    )
    p.add_argument(
        "--max-steps",
        type=int,
        nargs="+",
        default=[3000],
        help="max steps per game (default: [3000])",
    )
    p.add_argument(
        "--deterministic", action="store_true", help="greedy argmax (default: stochastic)"
    )
    p.add_argument("--record", action="store_true", help="save .mp4 videos")
    args = p.parse_args()

    # Detect mode from agent/opponent types
    agent_is_rl = is_agent_dir(args.agent)
    opponent_is_rl = is_agent_dir(args.opponent)

    if not agent_is_rl and args.agent not in AI_MAPPING:
        p.error(f"Unknown: '{args.agent}'. Valid bots: {sorted(AI_MAPPING.keys())}")
    if not opponent_is_rl and args.opponent not in AI_MAPPING:
        p.error(f"Unknown: '{args.opponent}'. Valid bots: {sorted(AI_MAPPING.keys())}")

    if agent_is_rl and opponent_is_rl:
        mode = MODE_RL_VS_RL
    elif agent_is_rl or opponent_is_rl:
        mode = MODE_RL_VS_BOT
    else:
        mode = MODE_BOT_VS_BOT

    agent_name = os.path.basename(args.agent.rstrip("/")) if agent_is_rl else args.agent
    opponent_name = os.path.basename(args.opponent.rstrip("/")) if opponent_is_rl else args.opponent

    # Max-steps: broadcast single value to all maps
    n_maps = len(args.maps)
    if len(args.max_steps) == 1:
        max_steps_per_map = args.max_steps * n_maps
    elif len(args.max_steps) == n_maps:
        max_steps_per_map = args.max_steps
    else:
        p.error(f"--max-steps must be 1 or {n_maps} values, got {len(args.max_steps)}")

    # rl_vs_bot: which CLI arg is RL vs bot?
    rl_player, bot_name = None, None
    if mode == MODE_RL_VS_BOT:
        if agent_is_rl:
            rl_player, bot_name = 0, args.opponent
        else:
            rl_player, bot_name = 1, args.agent

    return SessionConfig(
        agent_raw=args.agent,
        opponent_raw=args.opponent,
        nb_games=args.nb_games,
        deterministic=args.deterministic,
        record=args.record,
        mode=mode,
        agent_name=agent_name,
        opponent_name=opponent_name,
        maps=args.maps,
        max_steps_per_map=max_steps_per_map,
        rl_player=rl_player,
        bot_name=bot_name,
    )


# ── Model Loading ─────────────────────────────────────────────────────────

_ENV_KEYS = [
    "extended_obs",
    "filtered_masks",
    "frame_stack",
    "reserved_obs",
    "partial_obs",
    "multi_map",
    "max_map_size",
]


def _print_rl_vs_rl_config(cfg):
    """Print env/wrapper config comparison for rl_vs_rl, highlighting mismatches.
    Crashes on incompatible configs (partial_obs, filtered_masks).
    """
    ac, oc = cfg.agent_config, cfg.opponent_config

    # Hard incompatibilities — crash early with clear message
    if ac.get("partial_obs", False) != oc.get("partial_obs", False):
        raise ValueError(
            f"partial_obs mismatch: agent={ac.get('partial_obs')}, "
            f"opponent={oc.get('partial_obs')}. "
            f"Cannot mix fog-of-war and full observability in the same game."
        )
    if ac.get("filtered_masks", False) != oc.get("filtered_masks", False):
        print("  NOTE: filtered_masks mismatch — per-model mask override active.")

    print("\n  Env config comparison:")
    print(f"  {'Feature':<18s} {'Agent':>12s} {'Opponent':>12s}")
    print(f"  {'-' * 44}")
    has_mismatch = False
    for key in _ENV_KEYS:
        if (
            key == "max_map_size"
            and not ac.get("multi_map", False)
            and not oc.get("multi_map", False)
        ):
            continue
        av = ac.get(key, "-")
        ov = oc.get(key, "-")
        tag = " *" if av != ov else ""
        if av != ov:
            has_mismatch = True
        print(f"  {key:<18s} {str(av):>12s} {str(ov):>12s}{tag}")
    if has_mismatch:
        print("  (* = mismatch — handled by per-model ObsAdapter)")
    print()


def load_models(cfg: SessionConfig):
    """Load RL model(s) and resolve maps from config if not specified."""
    if cfg.mode == MODE_BOT_VS_BOT:
        return

    if cfg.mode == MODE_RL_VS_RL:
        cfg.agent_model, cfg.agent_config = load_agent_from_config(
            cfg.agent_raw, device=str(cfg.device)
        )
        cfg.opponent_model, cfg.opponent_config = load_agent_from_config(
            cfg.opponent_raw, device=str(cfg.device)
        )
        n0 = sum(p.nelement() for p in cfg.agent_model.parameters())
        n1 = sum(p.nelement() for p in cfg.opponent_model.parameters())
        print(f"Loaded agent: {cfg.agent_name} ({n0:,} params)")
        print(f"Loaded opponent: {cfg.opponent_name} ({n1:,} params)")
        _print_rl_vs_rl_config(cfg)
    else:
        # rl_vs_bot: load whichever side is RL
        run_dir = cfg.agent_raw if cfg.rl_player == 0 else cfg.opponent_raw
        name = cfg.agent_name if cfg.rl_player == 0 else cfg.opponent_name
        model, config = load_agent_from_config(run_dir, device=str(cfg.device))
        if cfg.rl_player == 0:
            cfg.agent_model, cfg.agent_config = model, config
        else:
            cfg.opponent_model, cfg.opponent_config = model, config
        n = sum(p.nelement() for p in model.parameters())
        print(f"Loaded: {name} ({n:,} params)")


# ── Output ────────────────────────────────────────────────────────────────


def _player_type(cfg, side):
    """'rl' or 'bot' for a given side."""
    if cfg.mode == MODE_BOT_VS_BOT:
        return "bot"
    if cfg.mode == MODE_RL_VS_RL:
        return "rl"
    if side == "agent":
        return "rl" if cfg.rl_player == 0 else "bot"
    return "rl" if cfg.rl_player == 1 else "bot"


def print_header(cfg: SessionConfig):
    total_games = len(cfg.maps) * sum(g for _, g, _ in cfg.positions)
    action_mode = "deterministic" if cfg.deterministic else "stochastic"
    print(f"\n{'=' * 60}")
    print("  MicroRTS Evaluation")
    print(f"{'=' * 60}")
    print(f"  Agent:       {cfg.agent_name} ({_player_type(cfg, 'agent')})")
    print(f"  Opponent:    {cfg.opponent_name} ({_player_type(cfg, 'opponent')})")
    print(f"  Mode:        {cfg.mode}")
    print(f"  Device:      {cfg.device}")
    print(f"  Maps:        {len(cfg.maps)}")
    for i, m in enumerate(cfg.maps):
        print(f"               - {m} (max_steps={cfg.max_steps_per_map[i]})")
    print(f"  Games/pos:   {cfg.nb_games}")
    print(f"  Total games: {total_games}")
    print(f"  Actions:     {action_mode}")
    if cfg.record:
        print("  Recording:   ON")
    print(f"{'=' * 60}")


def print_summary(pos_stats: list[PositionStats], cfg: SessionConfig):
    print(f"\n{'=' * 60}")
    print(f"  RESULTS — {cfg.agent_name} vs {cfg.opponent_name}")
    print(f"{'=' * 60}")
    total_w, total_l, total_d, total_g = 0, 0, 0, 0
    for s in pos_stats:
        total_w += s.wins
        total_l += s.losses
        total_d += s.draws
        total_g += s.games
        print(
            f"  As {s.label}:  {s.wins:>3d}W / {s.losses:>3d}L / {s.draws:>3d}D  "
            f"({100 * s.win_rate:>5.1f}%)  avg_len={s.avg_length:>.0f}"
        )
    if len(pos_stats) > 1:
        overall_wr = total_w / total_g if total_g > 0 else 0
        print(f"  {'─' * 56}")
        print(
            f"  Total:  {total_w:>3d}W / {total_l:>3d}L / {total_d:>3d}D  "
            f"({100 * overall_wr:>5.1f}%)"
        )
    print(f"{'=' * 60}")


# ── Constants ─────────────────────────────────────────────────────────────

_MAX_BATCH_STEPS = 100_000  # safety guard against infinite loops


# ── Per-game recording via render_client switching ────────────────────────


def _switch_render_client(env, game_idx, mode):
    """Point render_client to a specific sub-env for VideoRecorder.capture_frame()."""
    base = _get_base_env(env)
    vc = base.vec_client
    if mode == MODE_RL_VS_RL:
        # selfPlayClients[i] = pair i (handles both P0 and P1 sub-envs)
        # game_idx is P0 sub-env index (0, 2, 4...) → pair = game_idx // 2
        base.render_client = vc.selfPlayClients[game_idx // 2]
    elif mode == MODE_RL_VS_BOT:
        base.render_client = vc.clients[game_idx]
    else:
        base.render_client = vc.botClients[game_idx]


def _close_recorder(recorder):
    """Close VideoRecorder (encodes mp4), suppressing MoviePy noise."""
    sys.stdout.flush()  # flush before blocking encode
    with contextlib.redirect_stdout(io.StringIO()), contextlib.redirect_stderr(io.StringIO()):
        recorder.close()


# ── Batch play with optional recording ────────────────────────────────────


def play_batch_bot_vs_bot(cfg, map_path, max_steps, swap, num_games, rec_dir=None, on_done=None):
    """N bot-vs-bot games in parallel, with optional per-game recording."""
    if swap:
        ai1s = [AI_MAPPING[cfg.opponent_raw]] * num_games
        ai2s = [AI_MAPPING[cfg.agent_raw]] * num_games
    else:
        ai1s = [AI_MAPPING[cfg.agent_raw]] * num_games
        ai2s = [AI_MAPPING[cfg.opponent_raw]] * num_games

    env = make_bot_env(
        ai1s=ai1s,
        ai2s=ai2s,
        map_paths=[map_path] * num_games,
        partial_obs=False,
        max_steps=max_steps,
        jvm_args=JVM_ARGS,
    )
    suppress_java_output()
    env.reset()

    # UtsImass: run preGameAnalysis per client so the Python server loads
    # the map's opening strategy (same logic as env_pool.run_preanalysis).
    ai1_name = cfg.opponent_raw if swap else cfg.agent_raw
    ai2_name = cfg.agent_raw if swap else cfg.opponent_raw
    ai1_is_uts = ai1_name.lower().replace("_", "") == "utsimass"
    ai2_is_uts = ai2_name.lower().replace("_", "") == "utsimass"
    if ai1_is_uts or ai2_is_uts:
        for i in range(num_games):
            client = env.vec_client.botClients[i]
            if ai1_is_uts:
                run_uts_imass_preanalysis(client.ai1, map_path, client.gs)
            if ai2_is_uts:
                run_uts_imass_preanalysis(client.ai2, map_path, client.gs)

    # Create per-game recorders
    recorders = [None] * num_games
    if rec_dir:
        if VideoRecorder is None:
            raise RuntimeError(
                "--record needs gymnasium.wrappers.monitoring.video_recorder.VideoRecorder, "
                "removed in gymnasium >= 1.2. Pin gymnasium <= 1.1 or migrate to "
                "gymnasium.wrappers.RecordVideo / gymnasium.utils.save_video."
            )
        mname = map_short(map_path)
        label = "P1" if swap else "P0"
        for i in range(num_games):
            vid_base = os.path.join(rec_dir, f"{mname}_{label}_game{i + 1:02d}")
            _switch_render_client(env, i, MODE_BOT_VS_BOT)
            recorders[i] = VideoRecorder(env, base_path=vid_base)
            recorders[i].frames_per_sec = 150
            recorders[i].capture_frame()

    completed = [False] * num_games
    results = [None] * num_games
    step_counts = [0] * num_games
    global_step = 0

    while not all(completed) and global_step < _MAX_BATCH_STEPS:
        _, _, dones, infos = env.step(np.zeros(num_games, dtype=np.int32))
        global_step += 1
        for i in range(num_games):
            if completed[i]:
                continue
            step_counts[i] += 1

            # Capture frame for active games
            if recorders[i] is not None:
                _switch_render_client(env, i, MODE_BOT_VS_BOT)
                recorders[i].capture_frame()

            if dones[i]:
                completed[i] = True
                wl = infos[i]["raw_rewards"][0]
                results[i] = (wl_to_result(wl, player=0), step_counts[i])
                if recorders[i] is not None:
                    _close_recorder(recorders[i])
                if on_done:
                    on_done(i, results[i])

    # Mark unfinished games as DRAW
    for i in range(num_games):
        if not completed[i]:
            print(
                f"  WARNING: game {i} did not finish within {_MAX_BATCH_STEPS} steps, treating as DRAW"
            )
            results[i] = ("DRAW", step_counts[i])
            if recorders[i] is not None:
                _close_recorder(recorders[i])
            if on_done:
                on_done(i, results[i])

    # Don't close vec_client — JVM cleanup is slow and a new env will be created for next position
    return results


def play_batch_rl_vs_bot(cfg, map_path, max_steps, swap, num_games, rec_dir=None, on_done=None):
    """N rl-vs-bot games in parallel, with optional per-game recording."""
    player = (1 - cfg.rl_player) if swap else cfg.rl_player
    rl_model = cfg.agent_model if cfg.rl_player == 0 else cfg.opponent_model
    rl_config = cfg.agent_config if cfg.rl_player == 0 else cfg.opponent_config

    pad = _padding_kwargs(rl_config)
    env = make_agent_env(
        num_bot_envs=num_games,
        num_selfplay_envs=0,
        ai2s=[AI_MAPPING[cfg.bot_name]] * num_games,
        map_paths=[map_path],
        player=player,
        partial_obs=rl_config.get("partial_obs", False),
        max_steps=max_steps,
        reward_weight=np.array(rl_config["reward_weight"]),
        jvm_args=JVM_ARGS,
        alternate_players=False,
        filtered_masks=rl_config.get("filtered_masks", False),
        extended_obs=rl_config.get("extended_obs", False),
        **pad,
    )
    env = _apply_wrappers(env, rl_config)
    suppress_java_output()

    obs = torch.as_tensor(env.reset()).to(cfg.device)

    # UtsImass: run preGameAnalysis per client so the Python server loads
    # the map's opening strategy (same logic as play_batch_bot_vs_bot).
    if cfg.bot_name.lower().replace("_", "") == "utsimass":
        base_env = _get_base_env(env)
        for i in range(num_games):
            client = base_env.vec_client.clients[i]
            run_uts_imass_preanalysis(client.ai2, map_path, client.gs)

    # Create per-game recorders
    recorders = [None] * num_games
    if rec_dir:
        if VideoRecorder is None:
            raise RuntimeError(
                "--record needs gymnasium.wrappers.monitoring.video_recorder.VideoRecorder, "
                "removed in gymnasium >= 1.2. Pin gymnasium <= 1.1 or migrate to "
                "gymnasium.wrappers.RecordVideo / gymnasium.utils.save_video."
            )
        mname = map_short(map_path)
        label = "P1" if swap else "P0"
        for i in range(num_games):
            vid_base = os.path.join(rec_dir, f"{mname}_{label}_game{i + 1:02d}")
            _switch_render_client(env, i, MODE_RL_VS_BOT)
            recorders[i] = VideoRecorder(env, base_path=vid_base)
            recorders[i].frames_per_sec = 150
            recorders[i].capture_frame()

    completed = [False] * num_games
    results = [None] * num_games
    step_counts = [0] * num_games
    global_step = 0

    while not all(completed) and global_step < _MAX_BATCH_STEPS:
        with torch.no_grad():
            masks = torch.tensor(env.get_action_mask()).to(cfg.device)
            actions = get_actions(rl_model, obs, masks, cfg.deterministic)
        obs_np, _, dones, infos = env.step(actions.cpu().numpy().reshape(env.num_envs, -1))
        obs = torch.as_tensor(obs_np).to(cfg.device)
        global_step += 1

        for i in range(num_games):
            if completed[i]:
                continue
            step_counts[i] += 1
            if recorders[i] is not None:
                _switch_render_client(env, i, MODE_RL_VS_BOT)
                recorders[i].capture_frame()
            if dones[i]:
                completed[i] = True
                wl = infos[i]["raw_rewards"][0]
                results[i] = (wl_to_result(wl, player=player), step_counts[i])
                if recorders[i] is not None:
                    _close_recorder(recorders[i])
                if on_done:
                    on_done(i, results[i])

    # Mark unfinished games as DRAW
    for i in range(num_games):
        if not completed[i]:
            print(
                f"  WARNING: game {i} did not finish within {_MAX_BATCH_STEPS} steps, treating as DRAW"
            )
            results[i] = ("DRAW", step_counts[i])
            if recorders[i] is not None:
                _close_recorder(recorders[i])
            if on_done:
                on_done(i, results[i])

    # Don't close vec_client — JVM cleanup is slow and a new env will be created for next position
    return results


def play_batch_rl_vs_rl(cfg, map_path, max_steps, swap, num_games, rec_dir=None, on_done=None):
    """N rl-vs-rl games in parallel (2N sub-envs), with optional per-game recording."""
    p0_model = cfg.opponent_model if swap else cfg.agent_model
    p1_model = cfg.agent_model if swap else cfg.opponent_model
    p0_config = cfg.opponent_config if swap else cfg.agent_config
    p1_config = cfg.agent_config if swap else cfg.opponent_config

    # Multi-map agents expect padded obs (fixed shape). Single-map agents don't.
    # Mixing the two is incompatible — the obs shape can't satisfy both.
    agent_mm = cfg.agent_config.get("multi_map", False)
    opp_mm = cfg.opponent_config.get("multi_map", False)
    if agent_mm and not opp_mm:
        print(
            "  WARNING: agent is multi-map but opponent is single-map. "
            "Opponent may crash on padded obs if map size differs from training."
        )
    elif opp_mm and not agent_mm:
        print(
            "  WARNING: opponent is multi-map but agent is single-map. "
            "Agent may crash on padded obs if map size differs from training."
        )
    if agent_mm or opp_mm:
        # Only consider max_map_size from agents that actually use multi_map
        sizes = []
        if agent_mm:
            sizes.append(cfg.agent_config.get("max_map_size", 64))
        if opp_mm:
            sizes.append(cfg.opponent_config.get("max_map_size", 64))
        ms = max(sizes)
        pad = {"padded": True, "max_height": ms, "max_width": ms}
    else:
        pad = {"padded": False, "max_height": 0, "max_width": 0}
    env_extended_obs = cfg.agent_config.get("extended_obs", False) or cfg.opponent_config.get(
        "extended_obs", False
    )
    env = make_agent_env(
        num_bot_envs=0,
        num_selfplay_envs=2 * num_games,
        ai2s=[],
        map_paths=[map_path] * (2 * num_games),
        player=0,
        partial_obs=cfg.agent_config.get("partial_obs", False),
        max_steps=max_steps,
        reward_weight=np.array(cfg.agent_config["reward_weight"]),
        jvm_args=JVM_ARGS,
        alternate_players=False,
        filtered_masks=(
            cfg.agent_config.get("filtered_masks", False)
            or cfg.opponent_config.get("filtered_masks", False)
        ),
        extended_obs=env_extended_obs,
        **pad,
    )
    suppress_java_output()
    base_env = _get_base_env(env)

    p0_adapter = _ObsAdapter(p0_config, num_games)
    p1_adapter = _ObsAdapter(p1_config, num_games)

    raw_obs = env.reset()
    obs_p0, obs_p1 = _process_obs_rl_vs_rl(
        env,
        raw_obs,
        p0_adapter,
        p1_adapter,
        cfg.device,
        base_env=base_env,
        env_extended_obs=env_extended_obs,
    )

    # Create per-game recorders (render P0 side of each pair)
    recorders = [None] * num_games
    if rec_dir:
        mname = map_short(map_path)
        label = "P1" if swap else "P0"
        for i in range(num_games):
            vid_base = os.path.join(rec_dir, f"{mname}_{label}_game{i + 1:02d}")
            _switch_render_client(env, 2 * i, MODE_RL_VS_RL)  # P0 side of pair i
            recorders[i] = VideoRecorder(env, base_path=vid_base)
            recorders[i].frames_per_sec = 150
            recorders[i].capture_frame()

    completed = [False] * num_games
    results = [None] * num_games
    step_counts = [0] * num_games
    global_step = 0

    # Per-model mask handling: if one agent needs unfiltered masks
    p0_needs_unfiltered = not p0_config.get("filtered_masks", False)
    p1_needs_unfiltered = not p1_config.get("filtered_masks", False)
    env_uses_filtered = cfg.agent_config.get("filtered_masks", False) or cfg.opponent_config.get(
        "filtered_masks", False
    )

    while not all(completed) and global_step < _MAX_BATCH_STEPS:
        with torch.no_grad():
            masks = torch.tensor(env.get_action_mask()).to(cfg.device)
            # Override masks for agents that need unfiltered (standard) masks
            if env_uses_filtered and (p0_needs_unfiltered or p1_needs_unfiltered):
                unfiltered = torch.tensor(_get_base_env(env).get_unfiltered_action_mask()).to(
                    cfg.device
                )
                masks_p0 = unfiltered[0::2] if p0_needs_unfiltered else masks[0::2]
                masks_p1 = unfiltered[1::2] if p1_needs_unfiltered else masks[1::2]
            else:
                masks_p0 = masks[0::2]
                masks_p1 = masks[1::2]
            a_p0 = get_actions(p0_model, obs_p0, masks_p0, cfg.deterministic)
            a_p1 = get_actions(p1_model, obs_p1, masks_p1, cfg.deterministic)
            all_actions = torch.zeros(
                env.num_envs, *a_p0.shape[1:], dtype=a_p0.dtype, device=cfg.device
            )
            all_actions[0::2] = a_p0
            all_actions[1::2] = a_p1
        obs_np, _, dones, infos = env.step(all_actions.cpu().numpy().reshape(env.num_envs, -1))
        obs_p0, obs_p1 = _process_obs_rl_vs_rl(
            env,
            obs_np,
            p0_adapter,
            p1_adapter,
            cfg.device,
            base_env=base_env,
            env_extended_obs=env_extended_obs,
        )
        global_step += 1

        for k in range(num_games):
            if completed[k]:
                continue
            step_counts[k] += 1
            if recorders[k] is not None:
                _switch_render_client(env, 2 * k, MODE_RL_VS_RL)
                recorders[k].capture_frame()
            if dones[2 * k]:
                completed[k] = True
                wl = infos[2 * k]["raw_rewards"][0]
                results[k] = (wl_to_result(wl, player=0), step_counts[k])
                if recorders[k] is not None:
                    _close_recorder(recorders[k])
                if on_done:
                    on_done(k, results[k])

    # Mark unfinished games as DRAW
    for k in range(num_games):
        if not completed[k]:
            print(
                f"  WARNING: game {k} did not finish within {_MAX_BATCH_STEPS} steps, treating as DRAW"
            )
            results[k] = ("DRAW", step_counts[k])
            if recorders[k] is not None:
                _close_recorder(recorders[k])
            if on_done:
                on_done(k, results[k])

    # Don't close vec_client — JVM cleanup is slow and a new env will be created for next position
    return results


def play_batch(cfg, map_path, max_steps, swap, num_games, rec_dir=None, on_done=None):
    """Dispatch to the right batch function based on mode."""
    if cfg.mode == MODE_BOT_VS_BOT:
        return play_batch_bot_vs_bot(cfg, map_path, max_steps, swap, num_games, rec_dir, on_done)
    if cfg.mode == MODE_RL_VS_RL:
        return play_batch_rl_vs_rl(cfg, map_path, max_steps, swap, num_games, rec_dir, on_done)
    return play_batch_rl_vs_bot(cfg, map_path, max_steps, swap, num_games, rec_dir, on_done)


# ── Main ──────────────────────────────────────────────────────────────────


def main():
    cfg = parse_config()
    cfg.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    load_models(cfg)
    print_header(cfg)

    # UtsImass needs its Python server running (auto-start, cleanup on exit)
    uts_proc = None
    if needs_uts_imass([cfg.agent_name, cfg.opponent_name]):
        uts_proc = start_uts_imass_server()

    rec_dir = None
    if cfg.record:
        rec_dir = os.path.join(str(RECORDINGS_DIR), f"{cfg.agent_name}_vs_{cfg.opponent_name}")
        os.makedirs(rec_dir, exist_ok=True)
        print(f"\nVideos: {rec_dir}/")

    pos_stats = {label: PositionStats(label=label) for _, _, label in cfg.positions}

    for map_idx, map_path in enumerate(cfg.maps):
        mname = map_short(map_path)
        max_steps = cfg.max_steps_per_map[map_idx]
        print(f"\n  Map {map_idx + 1}/{len(cfg.maps)}: {mname}  (max_steps={max_steps})")
        print(f"  {'-' * 54}")

        for swap, num_games, label in cfg.positions:
            if num_games == 0:
                continue
            stats = pos_stats[label]

            # Print + record as each game finishes (live)
            def on_done(
                game_idx,
                result_tuple,
                _label=label,
                _swap=swap,
                _stats=stats,
                _mname=mname,
                _num=num_games,
            ):
                raw_result, steps = result_tuple
                result = _interpret_result(raw_result, _swap)
                _record_result(_stats, result, steps)
                vid_tag = ""
                if rec_dir:
                    vid_tag = f"  -> {_mname}_{_label}_game{game_idx + 1:02d}.mp4"
                print(
                    f"    [{_label}] Game {game_idx + 1:>3d}/{_num}: "
                    f"{result:>4s}  len={steps:>5d}{vid_tag}"
                )

            play_batch(cfg, map_path, max_steps, swap, num_games, rec_dir=rec_dir, on_done=on_done)

    all_stats = [pos_stats[label] for _, _, label in cfg.positions if pos_stats[label].games > 0]
    print_summary(all_stats, cfg)

    stop_uts_imass_server(uts_proc)

    try:
        from multiprocessing.resource_tracker import _resource_tracker

        if _resource_tracker._pid is not None:
            os.kill(_resource_tracker._pid, 9)
    except Exception:
        pass

    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
