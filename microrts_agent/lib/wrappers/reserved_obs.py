"""Adds +1 binary obs channel: reserved cells (pending move/produce destination).
Lets the agent SEE which cells are blocked, not just lose mask options.
obs (B, H, W, C) -> (B, H, W, C+1). Requires --filtered-masks.
"""

import gymnasium as gym  # type: ignore[import-not-found]
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper  # type: ignore[import-not-found]

from microrts_agent.lib.envs.base_vec_env import get_base_env


class ReservedPositionObs(VecEnvWrapper):
    def __init__(self, env):
        super().__init__(env)
        h, w, c = env.observation_space.shape
        self.obs_height = h  # may be padded (max_h)
        self.obs_width = w  # may be padded (max_w)

        self.observation_space = gym.spaces.Box(  # C -> C+1
            low=0.0,
            high=1.0,
            shape=(h, w, c + 1),
            dtype=env.observation_space.dtype,
        )

        from ts import FilteredMaskClient  # type: ignore[import]

        self._fmc = FilteredMaskClient  # Java helper for reserved positions

        self._base_env = get_base_env(self)  # need direct access to vec_client

    def _get_reserved_planes(self):
        """Query Java for reserved positions -> (B, H, W, 1) binary float32."""
        planes = np.zeros(
            (self.num_envs, self.obs_height, self.obs_width, 1), dtype=np.float32
        )  # (B, H, W, 1) — all zeros, fill actual region below
        base_env = self._base_env
        n_sp = base_env.num_selfplay_envs

        # Self-play: both P0 and P1 share the same game state -> same reserved positions
        for i in range(len(base_env.vec_client.selfPlayClients)):
            gs = base_env.vec_client.selfPlayClients[i].gs
            grid = np.array(self._fmc.getReservedPositions(gs))  # (actual_h, actual_w)
            gh, gw = grid.shape
            planes[i * 2, :gh, :gw, 0] = grid  # P0 env
            planes[i * 2 + 1, :gh, :gw, 0] = grid  # P1 env (same game)

        # Bot envs: each has its own client/game state
        for j in range(base_env.num_bot_envs):
            gs = base_env.vec_client.clients[j].gs
            grid = np.array(self._fmc.getReservedPositions(gs))  # (actual_h, actual_w)
            gh, gw = grid.shape
            planes[n_sp + j, :gh, :gw, 0] = grid

        return planes

    def _append_reserved(self, obs):
        """(B, H, W, C) + (B, H, W, 1) -> (B, H, W, C+1)"""
        return np.concatenate([obs, self._get_reserved_planes()], axis=-1)

    def reset(self):
        return self._append_reserved(self.venv.reset())

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        return self._append_reserved(obs), rewards, dones, infos
