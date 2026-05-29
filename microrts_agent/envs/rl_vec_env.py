"""Main RL environment: N parallel MicroRTS games via Java JNI.
Agent vs bot (bot envs) or agent vs itself (self-play envs).

Per step: get_action_mask() -> agent acts -> step_async() -> step_wait() -> (obs, reward, done, info)
Obs: standard 29ch one-hot or extended 73ch (--extended-obs).
Actions: 78 per cell [type(6), move(4), harvest(4), return(4), produce_dir(4), unit_type(z), target(49)].
Subclassed by PaddedMicroRTSRLVecEnv for multi-map with different sizes.
"""

import json
import os
import xml.etree.ElementTree as ET
from itertools import cycle

import gymnasium as gym  # type: ignore[import-not-found]
import numpy as np
from jpype.types import JArray, JInt

from .base_vec_env import BaseMicroRTSVecEnv


class MicroRTSRLVecEnv(BaseMicroRTSVecEnv):
    def __init__(
        self,
        num_selfplay_envs,  # int: number of self-play envs (agent vs itself)
        num_bot_envs,  # int: number of bot envs (agent vs Java AI)
        partial_obs,  # bool: enable fog of war (+2 visibility planes)
        max_steps,  # int: max game steps before forced draw
        ai2s,  # list[callable]: Java AI factories, one per bot env
        map_paths,  # list[str]: map XML paths relative to microrts/
        reward_weight,  # np.array(10,): weights for the 10 reward functions
        cycle_maps,  # list[str]: maps to rotate on episode end ([] = disabled)
        jvm_args,  # list[str]: extra JVM arguments (e.g. ["-Xmx4g"])
        player,  # int: 0 or 1 — which side the RL agent controls
        alternate_players=False,  # bool: flip player on episode end (P0↔P1)
        filtered_masks=False,  # bool: destination-aware action mask filtering
        extended_obs=False,  # bool: 13-feature extended obs (action params + ETA)
    ):

        self.player = player
        self.alternate_players = alternate_players
        self.filtered_masks = filtered_masks
        self.extended_obs = extended_obs
        self.num_selfplay_envs = num_selfplay_envs
        self.num_bot_envs = num_bot_envs
        num_envs = num_selfplay_envs + num_bot_envs

        assert num_bot_envs == len(ai2s), "for each environment, a microrts ai should be provided"
        assert num_selfplay_envs % 2 == 0, "self-play envs must be even (paired P0/P1)"
        assert len(map_paths) == num_envs, (
            "map_paths must have one entry per env (use env_factory for broadcasting)"
        )

        super().__init__(
            num_envs=num_envs,
            partial_obs=partial_obs,
            max_steps=max_steps,
            map_paths=map_paths,
            jvm_args=jvm_args,
        )

        self.ai2s = ai2s
        self.reward_weight = reward_weight

        self._last_raw_obs_int = None

        self._init_players()
        self._init_map_cycling(cycle_maps)
        self._init_reward_functions()
        self.start_client()
        self._init_spaces()

    # ==================================================================
    # Init helpers
    # ==================================================================

    def _init_players(self):
        """Set up per-env player tracking.
        Env ordering matches Java: [selfplay..., bots...].
        Self-play envs are fixed pairs: even=P0, odd=P1.
        Bot envs start at self.player.
        """
        self.players = np.full(self.num_envs, self.player, dtype=np.int32)
        for i in range(self.num_selfplay_envs // 2):
            self.players[i * 2] = 0
            self.players[i * 2 + 1] = 1

    def _init_map_cycling(self, cycle_maps):
        """Set up map rotation on episode end."""
        self.cycle_maps = [os.path.join(self.microrts_path, m) for m in cycle_maps]
        self.next_map = cycle(self.cycle_maps)
        root = ET.parse(os.path.join(self.microrts_path, self.map_paths[0])).getroot()
        self.height, self.width = int(root.get("height")), int(root.get("width"))

    def _init_reward_functions(self):
        """Instantiate the 10 Java reward functions."""
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
            RewardFunctionInterface,
            WinDrawLossRewardFunction,
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

    def _init_spaces(self):
        """Set up observation and action spaces from the UTT."""
        nUnitTypes = len(self.utt["unitTypes"])
        if self.extended_obs:
            # Mixed encoding: continuous + one-hot + thresholds
            self._extended_n_ut = nUnitTypes + 2  # +empty +pending
            self._extended_per_cell = (
                1 + 2 + 3 + self._extended_n_ut + 6 + 5 + 5 + 5 + 5 + 8 + 6 + 3 + 1
            )
            self._extended_global = 14  # 7 per player × 2
            total_channels = self._extended_per_cell + self._extended_global
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.height, self.width, total_channels),
                dtype=np.float32,
            )
            self.num_planes = []  # unused for extended obs path
        else:
            # Standard one-hot encoding (baseline)
            if self.partial_obs:
                self.num_planes = [5, 5, 3, nUnitTypes + 1, 6, 2, 1, 1]
            else:
                self.num_planes = [5, 5, 3, nUnitTypes + 1, 6, 2]
            self.observation_space = gym.spaces.Box(
                low=0.0,
                high=1.0,
                shape=(self.height, self.width, sum(self.num_planes)),
                dtype=np.float32,
            )

        # Prefix sums for fast one-hot indexing in _encode_obs
        self.num_planes_len = len(self.num_planes)
        self.num_planes_prefix_sum = [0]
        for p in self.num_planes:
            self.num_planes_prefix_sum.append(self.num_planes_prefix_sum[-1] + p)

        # Per-cell action: [type(6), move(4), harvest(4), return(4),
        #                    produce_dir(4), unit_type(z), target(49)]
        self.action_space_dims = [6, 4, 4, 4, 4, nUnitTypes, 7 * 7]
        self.action_space = gym.spaces.MultiDiscrete(
            np.array([self.action_space_dims] * self.height * self.width).flatten()
        )
        self.action_plane_space = gym.spaces.MultiDiscrete(self.action_space_dims)

        # Precompute cell indices for prepending to actions in step_async
        self.source_unit_idxs = np.tile(np.arange(self.height * self.width), (self.num_envs, 1))
        self.source_unit_idxs = self.source_unit_idxs.reshape(self.source_unit_idxs.shape + (1,))

    # ==================================================================
    # Java client
    # ==================================================================

    def start_client(self):
        """Create the Java-side vectorized game client and instantiate bot AIs."""
        from ai.core import AI  # type: ignore[import]
        from ts import JNIGridnetVecClient as Client  # type: ignore[import]

        self.vec_client = Client(
            self.num_selfplay_envs,
            self.num_bot_envs,
            self.max_steps,
            self.rfs,
            os.path.expanduser(self.microrts_path),
            self.map_paths,
            JArray(AI)([ai2(self.real_utt) for ai2 in self.ai2s]),
            self.real_utt,
            self.partial_obs,
        )
        self.render_client = (
            self.vec_client.selfPlayClients[0]
            if len(self.vec_client.selfPlayClients) > 0
            else self.vec_client.clients[0]
        )
        self.utt = json.loads(str(self.render_client.sendUTT()))

        if self.filtered_masks:
            from ts import FilteredMaskClient  # type: ignore[import]

            self._filtered_mask_client = FilteredMaskClient

        if self.extended_obs:
            from ai.wrapper import GameStateWrapper  # type: ignore[import]

            self._GameStateWrapper = GameStateWrapper

    # ==================================================================
    # Observation encoding
    # ==================================================================

    # Resource threshold values for global resource planes.
    # Encodes "can afford worker (1), light/ranged (2), heavy (3),
    # barracks (5), base (10), high-resource (32)".
    _RES_THRESHOLDS = (1, 2, 3, 5, 10, 32)

    def _get_player_list(self):
        """Get player list for Java calls (same order as env indices)."""
        return list(self.players.astype(int))

    def encode_obs(self, raw, my_res=None, opp_res=None):
        """Unified observation encoding. Handles standard/extended and batch/single.
        Standard: raw (N, F, H, W) int → (N, H, W, 29) float32
        Extended: raw (N, 13, H, W) + my_res (N,) + opp_res (N,) → (N, H, W, 73) float32
        Single env: pass raw with leading dim 1, result[0] to get (H, W, C).
        """
        if self.extended_obs:
            return self._encode_extended_obs_batch(raw, my_res, opp_res)
        return self._encode_obs_batch(raw)

    def _encode_obs_batch(self, all_obs_raw):
        """Vectorised version of _encode_obs for ALL envs at once.
        Input:  (N, num_features, H, W) int   Output: (N, H, W, total_ch) float32
        """
        N = all_obs_raw.shape[0]
        n_cells = self.height * self.width
        total_ch = self.num_planes_prefix_sum[-1]

        all_obs = all_obs_raw.reshape(N, all_obs_raw.shape[1], n_cells)  # (N, F, H*W)
        all_obs = all_obs.clip(0, np.array(self.num_planes).reshape(1, -1, 1) - 1)

        result = np.zeros((N, n_cells, total_ch), dtype=np.float32)  # (N, H*W, C)
        env_idx = np.arange(N)[:, None]  # (N, 1) broadcast
        cell_idx = np.arange(n_cells)[None, :]  # (1, H*W) broadcast
        # Same fancy indexing as _encode_obs but with env dimension
        result[env_idx, cell_idx, all_obs[:, 0, :]] = 1
        for i in range(1, self.num_planes_len):
            result[env_idx, cell_idx, all_obs[:, i, :] + self.num_planes_prefix_sum[i]] = 1

        return result.reshape(N, self.height, self.width, -1)  # (N, H, W, C)

    def _encode_extended_obs_batch(self, all_raw, all_my_res, all_opp_res):
        """Vectorised version of _encode_extended_obs for ALL envs at once.
        Input:  all_raw (N, 13, H, W) int, all_my/opp_res (N,) int
        Output: (N, H, W, 73) float32
        """
        N = all_raw.shape[0]
        n_cells = self.height * self.width
        total_ch = self._extended_per_cell + self._extended_global

        raw = all_raw.reshape(N, 13, n_cells)  # (N, 13, H*W)
        result = np.zeros((N, n_cells, total_ch), dtype=np.float32)  # (N, H*W, 73)
        env_idx = np.arange(N)[:, None]  # (N, 1)
        cell_idx = np.arange(n_cells)[None, :]  # (1, H*W)
        col = 0

        # 0: HP -> 1 continuous plane (value / 10)
        result[:, :, col] = np.clip(raw[:, 0].astype(np.float32) / 10.0, 0.0, 1.0)
        col += 1

        # 1: Resources -> 2 planes (value / 40, threshold >= 1)
        result[:, :, col] = np.clip(raw[:, 1].astype(np.float32) / 40.0, 0.0, 1.0)
        col += 1
        result[:, :, col] = (raw[:, 1] >= 1).astype(np.float32)
        col += 1

        # 2: Owner -> 3 one-hot (0=none, 1=self, 2=enemy)
        result[env_idx, cell_idx, col + raw[:, 2].clip(0, 2)] = 1.0
        col += 3

        # 3: Unit type -> n_ut one-hot
        n_ut = self._extended_n_ut
        result[env_idx, cell_idx, col + raw[:, 3].clip(0, n_ut - 1)] = 1.0
        col += n_ut

        # 4: Action -> 6 one-hot
        result[env_idx, cell_idx, col + raw[:, 4].clip(0, 5)] = 1.0
        col += 6

        # 5-8: Direction params (move, harvest, return, produce_dir) -> 5 one-hot each
        for f in range(5, 9):
            result[env_idx, cell_idx, col + raw[:, f].clip(0, 4)] = 1.0
            col += 5

        # 9: Produce type -> 8 one-hot
        result[env_idx, cell_idx, col + raw[:, 9].clip(0, 7)] = 1.0
        col += 8

        # 10: Attack dir -> 6 one-hot
        result[env_idx, cell_idx, col + raw[:, 10].clip(0, 5)] = 1.0
        col += 6

        # 11: ETA -> 3 planes (continuous + 2 thresholds)
        eta_shifted = raw[:, 11].astype(np.int32) + 128
        result[:, :, col] = np.clip(eta_shifted.astype(np.float32) / 255.0, 0.0, 1.0)
        col += 1
        result[:, :, col] = (eta_shifted >= 5).astype(np.float32)
        col += 1
        result[:, :, col] = (eta_shifted >= 10).astype(np.float32)
        col += 1

        # 12: Terrain -> 1 identity plane (1=empty, 0=wall)
        result[:, :, col] = raw[:, 12].astype(np.float32)
        col += 1

        # Global resource planes: 7 per player × 2 (my, opponent)
        for res_arr in (all_my_res, all_opp_res):
            result[:, :, col] = np.clip(res_arr.astype(np.float32) / 32.0, 0.0, 1.0)[:, None]
            col += 1
            for t in self._RES_THRESHOLDS:
                result[:, :, col] = (res_arr >= t).astype(np.float32)[:, None]
                col += 1

        return result.reshape(N, self.height, self.width, -1)

    def _encode_single_client(self, client, player):
        """Encode obs for one Java client (used at episode-end for re-encoding).
        Handles both standard and extended via encode_obs.
        Input: Java client + player index   Output: (H, W, C) float32
        """
        if self.extended_obs:
            gsw = self._GameStateWrapper(client.gs)
            raw = np.array(gsw.getVectorObservation(player))  # (13, H, W)
            resources = gsw.getPlayerResources(player)
            return self.encode_obs(
                raw[np.newaxis],  # (1, 13, H, W)
                np.array([int(resources[0])]),  # (1,)
                np.array([int(resources[1])]),  # (1,)
            )[0]  # (H, W, 73)
        else:
            if self.partial_obs:
                from rts import PartiallyObservableGameState  # type: ignore[import]

                view = PartiallyObservableGameState(client.gs, player)
            else:
                view = client.gs
            raw = np.array(client.ai1.getObservation(player, view))  # (F, H*W)
            return self.encode_obs(raw[np.newaxis])[0]  # (H, W, 29)

    def _get_all_extended_obs(self):
        """Collect raw obs from each Java client (N JNI calls) then batch-encode.
        Returns (N, H, W, 73) float32. Needed because extended obs has no bulk JNI API.
        """
        GSW = self._GameStateWrapper
        raw_list = [None] * self.num_envs
        my_res = np.empty(self.num_envs, dtype=np.int32)
        opp_res = np.empty(self.num_envs, dtype=np.int32)

        # Self-play envs (indices 0..num_selfplay_envs-1)
        for i, sp in enumerate(self.vec_client.selfPlayClients):
            gsw = GSW(sp.gs)
            raw_list[i * 2] = np.array(gsw.getVectorObservation(0))
            res0 = gsw.getPlayerResources(0)
            my_res[i * 2], opp_res[i * 2] = int(res0[0]), int(res0[1])

            raw_list[i * 2 + 1] = np.array(gsw.getVectorObservation(1))
            res1 = gsw.getPlayerResources(1)
            my_res[i * 2 + 1], opp_res[i * 2 + 1] = int(res1[0]), int(res1[1])

        # Bot envs (indices num_selfplay_envs..num_envs-1)
        n_sp = self.num_selfplay_envs
        java_players = self._get_player_list()
        for j, client in enumerate(self.vec_client.clients):
            p = java_players[n_sp + j]
            gsw = GSW(client.gs)
            raw_list[n_sp + j] = np.array(gsw.getVectorObservation(p))
            res = gsw.getPlayerResources(p)
            my_res[n_sp + j], opp_res[n_sp + j] = int(res[0]), int(res[1])

        return self._encode_extended_obs_batch(np.stack(raw_list), my_res, opp_res)

    def encode_standard_from_raw(self, raw_obs_int=None):
        """Re-encode cached raw obs as standard 29ch (for cross-encoding in rl_vs_rl).
        When env runs in extended mode but a model needs standard obs, this converts
        without an extra Java call. Uses _last_raw_obs_int if no arg provided.
        Input: (N, num_features, H*W) int   Output: (N, H, W, 29) float32
        """
        if raw_obs_int is None:
            raw_obs_int = self._last_raw_obs_int

        if self.num_planes:
            # Standard obs path already initialized — reuse existing encoding
            return self._encode_obs_batch(raw_obs_int)

        # Extended-obs env: num_planes is empty -> compute standard encoding
        # on the fly from the UTT (same as the standard path would use).
        num_planes = [5, 5, 3, len(self.utt["unitTypes"]) + 1, 6, 2]
        prefix_sum = [0]
        for p in num_planes:
            prefix_sum.append(prefix_sum[-1] + p)

        N = raw_obs_int.shape[0]
        n_cells = self.height * self.width
        total_ch = prefix_sum[-1]

        all_obs = raw_obs_int.reshape(N, raw_obs_int.shape[1], n_cells)
        planes_max = np.array(num_planes).reshape(1, -1, 1)
        all_obs = all_obs[:, : len(num_planes), :].clip(0, planes_max - 1)

        result = np.zeros((N, n_cells, total_ch), dtype=np.float32)
        env_idx = np.arange(N)[:, None]
        cell_idx = np.arange(n_cells)[None, :]

        result[env_idx, cell_idx, all_obs[:, 0, :]] = 1
        for i in range(1, len(num_planes)):
            result[env_idx, cell_idx, all_obs[:, i, :] + prefix_sum[i]] = 1

        return result.reshape(N, self.height, self.width, -1)

    # ==================================================================
    # Episode-end handling
    # ==================================================================

    def _cycle_maps_on_done(self, obs, done):
        """On episode end, swap to next map in the rotation cycle."""
        if not self.cycle_maps:
            return
        for i, d in enumerate(done[:, 0]):  # done is (num_envs, 1); squeeze to 1-D
            if not d:
                continue
            if i < self.num_selfplay_envs:
                # Self-play env: only even (P0) triggers the pair reset
                if i % 2 == 0:
                    pair_idx = i // 2
                    sp = self.vec_client.selfPlayClients[pair_idx]
                    sp.mapPath = next(self.next_map)
                    sp.reset()
                    obs[i] = self._encode_single_client(sp, 0)
                    obs[i + 1] = self._encode_single_client(sp, 1)
            else:
                # Bot env: reset single client
                bot_j = i - self.num_selfplay_envs
                client = self.vec_client.clients[bot_j]
                client.mapPath = next(self.next_map)
                client.reset(self.players[i])
                obs[i] = self._encode_single_client(client, int(self.players[i]))

    def _alternate_players_on_done(self, done):
        """Flip P0↔P1 for bot envs on episode end."""
        if self.alternate_players:
            for i in range(self.num_selfplay_envs, self.num_envs):
                if done[i, 0]:
                    self.players[i] = 1 - self.players[i]

    def _sync_selfplay_done(self, done_flat):
        """Force P1 done to mirror P0 done for each self-play pair.
        Java selfPlayClients only report done reliably for the P0 env
        (even index). P1's done flag may contain stale values.
        """
        for i in range(self.num_selfplay_envs // 2):
            p0 = i * 2
            done_flat[p0 + 1] = done_flat[p0]

    def _re_encode_on_player_flip(self, obs, done):
        """Re-encode obs for bot envs whose player just flipped.
        After _alternate_players_on_done flips self.players[i], the obs from
        gameStep is still from the OLD player's perspective. This re-encodes
        from the NEW player's perspective so the first obs of the new episode is correct.
        """
        if self.alternate_players:
            for i in range(self.num_selfplay_envs, self.num_envs):
                if done[i, 0]:
                    client = self.vec_client.clients[i - self.num_selfplay_envs]
                    obs[i] = self._encode_single_client(client, int(self.players[i]))

    # ==================================================================
    # Gym VecEnv interface
    # ==================================================================

    def reset(self):
        """Reset all envs and return encoded observations. (N, H, W, C) float32"""
        responses = self.vec_client.reset(self._get_player_list())
        raw = np.array(responses.observation)  # (N, F, H, W) int
        self._last_raw_obs_int = raw
        if self.extended_obs:
            return self._get_all_extended_obs()  # extended needs per-client JNI calls
        return self.encode_obs(raw)  # standard: direct batch encode

    def step_async(self, actions):
        """Prepare actions for Java: prepend cell idx, keep only cells with units.
        Input:  actions (num_envs, H*W*7) flat from agent
        Output: self.actions = Java int[num_envs][num_units][8] (cell_idx + 7 sub-actions)
        """
        actions = actions.reshape((self.num_envs, self.height * self.width, -1))  # (N, H*W, 7)
        actions = np.concatenate(
            (self.source_unit_idxs, actions), 2
        )  # (N, H*W, 8) — prepend cell index
        actions = actions[
            np.where(self.source_unit_mask == 1)
        ]  # (~total_units, 8) — keep only cells with units
        action_counts_per_env = self.source_unit_mask.sum(1)  # (N,) — how many units per env

        # Convert numpy -> Java 3D array: int[env][unit][8]
        py_actions = [None] * len(action_counts_per_env)
        action_idx = 0
        for outer_idx, action_count in enumerate(action_counts_per_env):
            java_valid_action = [None] * action_count
            for idx in range(action_count):
                java_valid_action[idx] = JArray(JInt)(
                    actions[action_idx]
                )  # numpy row -> Java int[]
                action_idx += 1
            py_actions[outer_idx] = JArray(JArray(JInt))(java_valid_action)  # -> Java int[][]

        self.actions = JArray(JArray(JArray(JInt)))(py_actions)  # -> Java int[][][]

    def step_wait(self):
        """Execute one game step.
        Returns: (obs, reward, done, infos)
            obs:    (N, H, W, C)  float32 — encoded observations
            reward: (N,)          float32 — weighted scalar reward per env
            done:   (N,)          bool    — episode ended?
            infos:  list[dict]            — raw_rewards(10,) + player per env
        """
        # 1. Java advances all games by one tick
        responses = self.vec_client.gameStep(self.actions, self._get_player_list())
        reward = np.array(responses.reward)  # (N, 10) — 10 raw reward components
        done = np.array(responses.done)  # (N, 1) — Java uses 2D
        raw_obs = np.array(responses.observation)  # (N, num_features, H, W) int
        self._last_raw_obs_int = raw_obs  # cache for encode_standard_from_raw()
        # extended obs needs per-client JNI calls; standard is a direct batch encode
        obs = self._get_all_extended_obs() if self.extended_obs else self.encode_obs(raw_obs)

        # 2. Build infos with raw (unweighted) reward components
        infos = [
            {"raw_rewards": item, "player": int(self.players[i])} for i, item in enumerate(reward)
        ]

        # 3. Episode-end handling (order matters: cycle map → flip player → re-encode obs)
        self._cycle_maps_on_done(obs, done)
        self._alternate_players_on_done(done)
        self._re_encode_on_player_flip(obs, done)

        done_flat = done[:, 0]  # (N, 1) → (N,)
        self._sync_selfplay_done(done_flat)

        # 4. Weighted scalar reward: (N, 10) @ (10,) -> (N,)
        return np.array(obs), reward @ self.reward_weight, done_flat, infos

    # ==================================================================
    # Action masking
    # ==================================================================

    def get_action_mask(self):
        """Get valid action masks for all envs. Called once per step BEFORE agent acts.
        Returns: (N, H*W, 78) int — per-cell mask of valid sub-actions
        Side effect: sets self.source_unit_mask (N, H*W) — 1 where cell has a unit
        """
        if self.filtered_masks:
            # Filtered path: destination-aware masks, 1 JNI call per client (slower)
            mask_dim = 1 + sum(self.action_space_dims)  # 1 (source) + 78 = 79
            action_mask = np.zeros(
                (self.num_envs, self.height, self.width, mask_dim), dtype=np.int32
            )
            fmc = self._filtered_mask_client
            for i, sp in enumerate(self.vec_client.selfPlayClients):
                m0 = np.array(fmc.getMasksFiltered(sp, 0))  # (H, W, 79) P0 perspective
                m1 = np.array(fmc.getMasksFiltered(sp, 1))  # (H, W, 79) P1 perspective
                idx0, idx1 = i * 2, i * 2 + 1
                action_mask[idx0] = m0 if self.players[idx0] == 0 else m1
                action_mask[idx1] = m1 if self.players[idx1] == 1 else m0
            for j, client in enumerate(self.vec_client.clients):
                bot_idx = self.num_selfplay_envs + j
                action_mask[bot_idx] = np.array(
                    fmc.getMasksFiltered(client, int(self.players[bot_idx]))
                )
        else:
            # Standard path: 1-2 bulk JNI calls for all envs (fast)
            # getMasks(0) returns correct masks for selfplay (Java handles P0/P1 pairs)
            action_mask = np.array(self.vec_client.getMasks(0))  # (N, H, W, 79)
            # Fix bot envs that play as P1 (getMasks(0) gave them wrong masks)
            bot_players = self.players[self.num_selfplay_envs :]
            p1_local = np.where(bot_players == 1)[0]
            if len(p1_local) > 0:
                mask_p1 = np.array(self.vec_client.getMasks(1))  # 2nd call for P1
                p1_bot_idx = p1_local + self.num_selfplay_envs
                action_mask[p1_bot_idx] = mask_p1[p1_bot_idx]  # patch P1 bots

        # Split: index 0 = has_unit flag, index 1-78 = sub-action masks
        self.source_unit_mask = action_mask[:, :, :, 0].reshape(self.num_envs, -1)  # (N, H*W)
        action_type_and_parameter_mask = action_mask[:, :, :, 1:].reshape(
            self.num_envs, self.height * self.width, -1
        )  # (N, H*W, 78)
        return action_type_and_parameter_mask

    def get_unfiltered_action_mask(self):
        """Get standard (non-filtered) masks regardless of self.filtered_masks setting.
        Used by ObsAdapter to give unfiltered masks to agents trained without filtered_masks.
        """
        action_mask = np.array(self.vec_client.getMasks(0))  # (N, H, W, 79)
        bot_players = self.players[self.num_selfplay_envs :]
        p1_local = np.where(bot_players == 1)[0]
        if len(p1_local) > 0:
            mask_p1 = np.array(self.vec_client.getMasks(1))
            p1_bot_idx = p1_local + self.num_selfplay_envs
            action_mask[p1_bot_idx] = mask_p1[p1_bot_idx]
        return action_mask[:, :, :, 1:].reshape(self.num_envs, self.height * self.width, -1)
