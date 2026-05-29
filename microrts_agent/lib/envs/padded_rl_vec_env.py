"""Padded env for multi-map training with different map sizes.
Pads obs/masks to (max_h, max_w) for the agent, crops actions back to actual size for Java.
switch_map() changes map mid-training (for PLR) without changing network input size.
"""

import os
import xml.etree.ElementTree as ET

import gymnasium as gym  # type: ignore[import-not-found]
import numpy as np
from gymnasium.spaces import MultiDiscrete  # type: ignore[import-not-found]

from .rl_vec_env import MicroRTSRLVecEnv


class PaddedMicroRTSRLVecEnv(MicroRTSRLVecEnv):
    def __init__(
        self,
        max_height,  # int: padded height (agent sees this, never changes)
        max_width,  # int: padded width  (agent sees this, never changes)
        **kwargs,  # all MicroRTSRLVecEnv params
    ):
        super().__init__(**kwargs)  # self.height/width = actual map size (changes on switch_map)
        self.max_height = max_height
        self.max_width = max_width

        # Override obs/action spaces from parent to use padded dimensions
        n_channels = self.observation_space.shape[-1]  # C from parent (29 or 73)
        self.observation_space = gym.spaces.Box(
            low=0.0,
            high=1.0,
            shape=(max_height, max_width, n_channels),
            dtype=self.observation_space.dtype,
        )

        # Override action space to use padded dimensions
        self.action_space = MultiDiscrete(
            np.array([self.action_space_dims] * max_height * max_width).flatten()
        )

    # ------------------------------------------------------------------
    # Padding / cropping helpers
    # ------------------------------------------------------------------

    def _pad_obs(self, obs):
        """(B, h, w, C) → (B, max_h, max_w, C). Zero-fill for empty cells."""
        B, h, w, C = obs.shape
        if h == self.max_height and w == self.max_width:
            return obs  # no-op if same size
        padded = np.zeros((B, self.max_height, self.max_width, C), dtype=obs.dtype)
        padded[:, :h, :w, :] = obs  # top-left = actual obs
        return padded

    def _pad_mask(self, mask):
        """(B, h*w, 78) → (B, max_h*max_w, 78). Padded cells = all-zero (can't act)."""
        B, _, D = mask.shape
        if self.height == self.max_height and self.width == self.max_width:
            return mask  # no-op if same size
        spatial = mask.reshape(B, self.height, self.width, D)  # (B, H, W, 78)
        padded = np.zeros((B, self.max_height, self.max_width, D), dtype=mask.dtype)
        padded[:, : self.height, : self.width, :] = spatial  # top-left = actual mask
        return padded.reshape(B, self.max_height * self.max_width, D)  # (B, max_H * max_W, 78)

    def _crop_actions(self, actions):
        """(B, max_h*max_w*7) → (B, h*w*7). Discard padded cell actions."""
        if self.height == self.max_height and self.width == self.max_width:
            return actions  # no-op if same size
        B = actions.shape[0]
        spatial = actions.reshape(B, self.max_height, self.max_width, -1)  # (B, max_H, max_W, 7)
        cropped = spatial[:, : self.height, : self.width, :]  # (B, H, W, 7)
        return cropped.reshape(B, -1)  # (B, H * W * 7)

    # ------------------------------------------------------------------
    # Override env interface methods
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs and return padded observations."""
        obs = super().reset()
        return self._pad_obs(obs)

    def get_action_mask(self):
        """Return padded action mask. source_unit_mask stays at actual size internally."""
        mask = super().get_action_mask()
        return self._pad_mask(mask)

    def step_async(self, actions):
        """Crop padded actions to actual map size, then pass to base class."""
        cropped = self._crop_actions(actions)
        super().step_async(cropped)

    def step_wait(self):
        """Execute game step and pad returned observations."""
        obs, reward, done, infos = super().step_wait()
        return self._pad_obs(obs), reward, done, infos

    # ------------------------------------------------------------------
    # Map switching
    # ------------------------------------------------------------------

    def switch_map(self, map_path):
        """Switch all envs to a new map mid-training (called by PLR in train).
        Updates actual dims, max_steps, recreates Java client. JVM stays alive.
        """
        from microrts_agent.lib.mappings.maps import get_max_cycles

        # 1. Parse new map dimensions
        root = ET.parse(os.path.join(self.microrts_path, map_path)).getroot()
        new_height = int(root.get("height"))
        new_width = int(root.get("width"))

        assert new_height <= self.max_height and new_width <= self.max_width, (
            f"Map {map_path} is {new_width}x{new_height}, "
            f"exceeds max_size {self.max_width}x{self.max_height}"
        )

        # 2. Update actual dimensions and max_steps for this map
        self.height = new_height
        self.width = new_width
        self.max_steps = get_max_cycles(map_path)

        # 3. Update map paths for all envs
        self.map_paths = [map_path] * self.num_envs

        # 4. Close old Java client (NOT the JVM) and recreate
        self.vec_client.close()
        self.start_client()

        # 5. Recompute precomputed cell indices for new actual size
        self.source_unit_idxs = np.tile(
            np.arange(self.height * self.width), (self.num_envs, 1)
        ).reshape((self.num_envs, self.height * self.width, 1))
