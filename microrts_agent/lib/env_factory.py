"""Environment factory. All env creation goes through here.
make_agent_env() — RL agent vs bot/self-play (training, eval)
make_bot_env()   — bot vs bot (tournaments)
"""

import re
import subprocess
from typing import Callable

import numpy as np


# ── JVM ──────────────────────────────────────────────────────────────────────
def _detect_jvm_args():
    """Return JVM args appropriate for the installed Java version.

    --enable-native-access=ALL-UNNAMED is required for JPype on Java 17+
    but crashes Java 11 and earlier.
    """
    try:
        out = subprocess.check_output(["java", "-version"], stderr=subprocess.STDOUT, text=True)
        match = re.search(r'"(\d+)', out)
        if match and int(match.group(1)) >= 17:
            return ["--enable-native-access=ALL-UNNAMED"]
    except Exception:
        pass
    return []


JVM_ARGS = _detect_jvm_args()


def make_agent_env(
    *,  # force keyword-only
    num_bot_envs: int,
    num_selfplay_envs: int,
    ai2s: list[Callable],
    map_paths: list[str],
    player: int,
    partial_obs: bool,
    max_steps: int,
    reward_weight: np.ndarray,
    jvm_args: list[str],
    padded: bool,
    max_height: int,
    max_width: int,
    alternate_players: bool,
    filtered_masks: bool,
    extended_obs: bool,
):
    """Create a MicroRTSRLVecEnv (or Padded variant).
    All parameters are required — no hidden defaults.
    """
    if not isinstance(reward_weight, np.ndarray):
        reward_weight = np.array(reward_weight)

    # Broadcast single map to all envs
    num_envs = num_selfplay_envs + num_bot_envs
    if len(map_paths) == 1:
        map_paths = map_paths * num_envs
    assert len(map_paths) == num_envs, (
        f"map_paths must be 1 (broadcast) or {num_envs} (one per env), got {len(map_paths)}"
    )

    kwargs = {
        "num_selfplay_envs": num_selfplay_envs,
        "num_bot_envs": num_bot_envs,
        "partial_obs": partial_obs,
        "max_steps": max_steps,
        "ai2s": ai2s,
        "map_paths": map_paths,
        "reward_weight": reward_weight,
        "cycle_maps": [],
        "jvm_args": jvm_args,
        "player": player,
        "alternate_players": alternate_players,
        "filtered_masks": filtered_masks,
        "extended_obs": extended_obs,
    }

    if padded:
        from microrts_agent.lib.envs.padded_rl_vec_env import PaddedMicroRTSRLVecEnv

        return PaddedMicroRTSRLVecEnv(max_height=max_height, max_width=max_width, **kwargs)

    from microrts_agent.lib.envs.rl_vec_env import MicroRTSRLVecEnv

    return MicroRTSRLVecEnv(**kwargs)


def make_bot_env(
    *,  # force keyword-only
    ai1s: list[Callable],
    ai2s: list[Callable],
    map_paths: list[str],
    partial_obs: bool,
    max_steps: int,
    jvm_args: list[str],
    full_rewards: bool = False,
):
    """Create a MicroRTSBotVecEnv. All parameters required."""
    from microrts_agent.lib.envs.bot_vec_env import MicroRTSBotVecEnv

    return MicroRTSBotVecEnv(
        ai1s=ai1s,
        ai2s=ai2s,
        partial_obs=partial_obs,
        max_steps=max_steps,
        map_paths=map_paths,
        jvm_args=jvm_args,
        full_rewards=full_rewards,
    )
