"""Stack last N obs along channel axis: (B, H, W, C) → (B, H, W, C*N).
Gives the CNN temporal context (movement, buildup) without recurrent layers.
On reset/done: buffer filled with N copies of initial obs. Masks are NOT stacked.
"""

from collections import deque

import gymnasium as gym  # type: ignore[import-not-found]
import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper  # type: ignore[import-not-found]


class FrameStack(VecEnvWrapper):
    def __init__(self, env, num_stack=4):
        super().__init__(env)
        self.num_stack = num_stack

        h, w, c = env.observation_space.shape
        self._h, self._w, self._c = h, w, c  # C = channels per single frame

        # One deque per env, auto-drops oldest when full
        self._frames = [deque(maxlen=num_stack) for _ in range(self.num_envs)]

        self.observation_space = gym.spaces.Box(  # C → C*N
            low=0.0,
            high=1.0,
            shape=(h, w, c * num_stack),
            dtype=env.observation_space.dtype,
        )

    def _stack(self):
        """Concat buffered frames → (B, H, W, C*N)."""
        stacked = np.empty(
            (self.num_envs, self._h, self._w, self._c * self.num_stack),
            dtype=self.observation_space.dtype,
        )  # (B, H, W, C*N)
        for i in range(self.num_envs):
            # deque[0]=(H,W,C), deque[1]=(H,W,C), ... → concat on axis=-1 → (H,W,C*N)
            np.concatenate(list(self._frames[i]), axis=-1, out=stacked[i])
        return stacked

    def _fill(self, env_idx, obs_single):
        """Fill buffer with N copies of same obs (no history yet)."""
        buf = self._frames[env_idx]
        buf.clear()
        for _ in range(self.num_stack):
            buf.append(obs_single)  # (H, W, C) × N

    def reset(self):
        obs = self.venv.reset()  # (B, H, W, C)
        for i in range(self.num_envs):
            self._fill(i, obs[i])  # no history → N copies of first obs
        return self._stack()  # (B, H, W, C*N)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()  # (B, H, W, C)
        for i in range(self.num_envs):
            if dones[i]:
                self._fill(i, obs[i])  # episode ended → reset buffer with new obs
            else:
                self._frames[i].append(obs[i])  # push new frame, oldest auto-dropped by deque
        return self._stack(), rewards, dones, infos  # (B, H, W, C*N)
