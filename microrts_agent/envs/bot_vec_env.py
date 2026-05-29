"""Bot-vs-bot environment: both sides are Java AIs, no RL agent.
Used for tournaments and recordings. Only tracks WinDrawLoss result.
"""

import json
import os

import numpy as np
from jpype.types import JArray, JInt

from .base_vec_env import BaseMicroRTSVecEnv


class MicroRTSBotVecEnv(BaseMicroRTSVecEnv):
    def __init__(
        self,
        ai1s,  # list[callable]: Java AI factories for player 0
        ai2s,  # list[callable]: Java AI factories for player 1
        partial_obs,  # bool: enable fog of war
        max_steps,  # int: max game steps before forced draw
        map_paths,  # list[str]: map XML paths relative to microrts/
        jvm_args,  # list[str]: extra JVM arguments (e.g. ["-Xmx4g"])
        full_rewards=False,  # bool: use all 10 reward functions (for BC value fitting)
    ):

        self.ai1s = ai1s
        self.ai2s = ai2s
        self.full_rewards = full_rewards
        assert len(ai1s) == len(ai2s), "for each environment, a microrts ai should be provided"
        assert len(ai1s) > 0, "at least one environment is required"

        # Base class: num_envs, partial_obs, max_steps, map_paths, microrts_path, JVM, UnitTypeTable
        super().__init__(
            num_envs=len(ai1s),
            partial_obs=partial_obs,
            max_steps=max_steps,
            map_paths=map_paths,
            jvm_args=jvm_args,
        )

        from ai.reward import (  # type: ignore[import]
            RewardFunctionInterface,
            WinDrawLossRewardFunction,
        )

        if full_rewards:
            from ai.reward import (  # type: ignore[import]
                AttackRewardFunction,
                MilitaryScoreRewardFunction,
                ProduceBarracksRewardFunction,
                ProduceBaseRewardFunction,
                ProduceHeavyUnitRewardFunction,
                ProduceLightUnitRewardFunction,
                ProduceRangedUnitRewardFunction,
                ProduceWorkerRewardFunction,
                ResourceGatherRewardFunction,
            )

            self.rfs = JArray(RewardFunctionInterface)(
                [
                    WinDrawLossRewardFunction(),  # idx 0
                    ResourceGatherRewardFunction(),  # idx 1
                    ProduceWorkerRewardFunction(),  # idx 2
                    ProduceBaseRewardFunction(),  # idx 3
                    ProduceBarracksRewardFunction(),  # idx 4
                    AttackRewardFunction(),  # idx 5
                    ProduceLightUnitRewardFunction(),  # idx 6
                    ProduceHeavyUnitRewardFunction(),  # idx 7
                    ProduceRangedUnitRewardFunction(),  # idx 8
                    MilitaryScoreRewardFunction(),  # idx 9
                ]
            )
        else:
            self.rfs = JArray(RewardFunctionInterface)([WinDrawLossRewardFunction()])
        self.start_client()

    # ------------------------------------------------------------------
    # Java client management
    # ------------------------------------------------------------------

    def start_client(self):
        """Create Java client with BOTH sides controlled by bots (no RL agent)."""
        from ai.core import AI  # type: ignore[import]
        from ts import JNIGridnetVecClient as Client  # type: ignore[import]

        self.vec_client = Client(
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai1(self.real_utt) for ai1 in self.ai1s]),  # player 0 bots
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),  # player 1 bots
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = self.vec_client.botClients[0]
        self.utt = json.loads(str(self.render_client.sendUTT()))

    # ------------------------------------------------------------------
    # Gym VecEnv interface
    # ------------------------------------------------------------------

    def reset(self):
        """Reset all envs.
        Returns None — bot envs have no observations (both sides are Java AIs).
        Tournament code resets clients individually via client.reset() for
        per-client setup (AI assignment, traces, preanalysis).
        """
        self.vec_client.reset([0 for _ in range(self.num_envs)])
        return None

    def step_async(self, actions):
        """No-op: bots compute their own actions internally in Java."""
        pass

    def step_wait(self):
        """Advance one game step. Bots act automatically; returns (obs, reward, done, info)."""
        all_winloss_reward = []
        all_dones = []
        for i in range(self.num_envs):
            response = self.vec_client.botClients[i].gameStep(JInt(0))
            self.vec_client.envSteps[i] += 1
            winloss_reward = float(response.reward[0])  # WinLoss: +1 win, -1 loss, 0 draw
            done = bool(response.done[0])
            if done or self.vec_client.envSteps[i] >= self.vec_client.maxSteps:
                done = True
                self.vec_client.botClients[i].reset(JInt(0))
                self.vec_client.envSteps[i] = 0
            all_winloss_reward.append(winloss_reward)
            all_dones.append(done)
        infos = [{"raw_rewards": np.array([winloss])} for winloss in all_winloss_reward]
        return None, None, np.array(all_dones), infos
