"""Passthrough wrapper that records per-episode raw + discounted returns.
On done: sums all step rewards into info["microrts_stats"] for TensorBoard logging.
Does NOT change game behaviour: only observes.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper  # type: ignore[import-not-found]


class StatsRecorder(VecEnvWrapper):
    def __init__(
        self,
        env,
        gamma: float = 0.99,  # discount factor for discounted returns
    ):
        super().__init__(env)
        self.gamma = gamma
        # self.rfs comes from the base env via VecEnvWrapper.__getattr__
        self._raw_names = [
            str(rf) for rf in self.rfs
        ]  # ["WinDrawLoss...", "ResourceGather...", ...]
        self._raw_discount_names = ["discounted_" + n for n in self._raw_names] + [
            "discounted"
        ]  # +1 for total

    def reset(self):
        obs = self.venv.reset()
        self.raw_rewards = [[] for _ in range(self.num_envs)]  # per-env list of (10,) arrays
        self.timestep = np.zeros(self.num_envs, dtype=np.float32)  # t for γ^t discounting
        self.raw_discount_rewards = [
            [] for _ in range(self.num_envs)
        ]  # per-env list of (11,) arrays
        return obs

    def step_wait(self):
        obs, rews, dones, infos = self.venv.step_wait()
        newinfos = list(infos)
        for i in range(len(dones)):
            raw_r = infos[i]["raw_rewards"]  # (10,) from rl_vec_env
            self.raw_rewards[i].append(raw_r)
            # Discounted: γ^t × [10 components + their sum] → (11,)
            self.raw_discount_rewards[i].append(
                (self.gamma ** self.timestep[i])
                * np.concatenate((raw_r, raw_r.sum()), axis=None)  # append total as 11th element
            )
            self.timestep[i] += 1
            if dones[i]:
                # Episode ended → sum all step rewards into cumulative returns
                info = infos[i].copy()
                raw_returns = np.array(self.raw_rewards[i]).sum(0)  # (10,) total per component
                raw_discount_returns = np.array(self.raw_discount_rewards[i]).sum(
                    0
                )  # (11,) discounted
                info["microrts_stats"] = dict(zip(self._raw_names, raw_returns, strict=False))
                info["microrts_stats"].update(
                    dict(zip(self._raw_discount_names, raw_discount_returns, strict=False))
                )
                # Reset buffers for next episode
                self.raw_rewards[i] = []
                self.raw_discount_rewards[i] = []
                self.timestep[i] = 0
                newinfos[i] = info

        return obs, rews, dones, newinfos
