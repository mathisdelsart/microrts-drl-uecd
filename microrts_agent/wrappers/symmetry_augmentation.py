"""Random spatial flip augmentation (4 modes: identity, H-flip, V-flip, both).
Java runs unflipped → obs/masks flipped for agent → actions unflipped back for Java.
Padding-aware: only actual map region flipped, padding stays zero.
Directions (0=N,1=E,2=S,3=W) and 7x7 attack targets are permuted consistently.
"""

import numpy as np
from stable_baselines3.common.vec_env import VecEnvWrapper  # type: ignore[import-not-found]


class SymmetryAugmentation(VecEnvWrapper):
    def __init__(self, env):

        super().__init__(env)
        # Padded (observation) dimensions — fixed for the agent's lifetime
        self.pad_height, self.pad_width = env.observation_space.shape[:2]
        self._obs_channels = env.observation_space.shape[2]

        from microrts_agent.envs.base_vec_env import get_base_env

        self._base_env = get_base_env(self)  # for dynamic actual dimensions (switch_map)

        # action_nvec per cell: [6, 4, 4, 4, 4, z, 49]
        self.action_nvec = env.action_plane_space.nvec.tolist()
        self.num_action_params = len(self.action_nvec)

        # Per-env flip state
        self.flip_state = np.zeros(self.num_envs, dtype=np.int32)

        self._precompute_tables()
        self._precompute_obs_dir_channels()

    def _precompute_tables(self):
        """Precompute direction and attack-target permutation tables.

        Cell remap tables are NOT precomputed because actual map dims change
        with switch_map(). Instead, spatial flips use numpy slicing.
        """
        # Direction permutations [N=0, E=1, S=2, W=3]
        self.dir_perm = {
            0: np.array([0, 1, 2, 3]),  # identity
            1: np.array([0, 3, 2, 1]),  # H-flip: E<->W
            2: np.array([2, 1, 0, 3]),  # V-flip: N<->S
            3: np.array([2, 3, 0, 1]),  # both
        }

        # Attack target permutations (7x7 grid)
        tgt = np.arange(49).reshape(7, 7)
        self.target_perm = {
            0: np.arange(49),
            1: tgt[:, ::-1].flatten().copy(),
            2: tgt[::-1, :].flatten().copy(),
            3: tgt[::-1, ::-1].flatten().copy(),
        }

        # Cumulative offsets for action mask components
        self.mask_offsets = np.cumsum([0] + self.action_nvec)

        # Precompute mask slices for directional and target components
        off = self.mask_offsets
        self._dir_slices = [
            slice(off[1], off[2]),  # move_dir (4)
            slice(off[2], off[3]),  # harvest_dir (4)
            slice(off[3], off[4]),  # return_dir (4)
            slice(off[4], off[5]),  # produce_dir (4)
        ]
        self._tgt_slice = slice(off[-2], off[-1])  # attack_target (49)

    def _precompute_obs_dir_channels(self):
        """Precompute observation channel indices that encode directions.

        Extended obs has 4 direction groups (move, harvest, return, produce)
        each encoded as 5 one-hot channels [N, E, S, W, none].
        These must be permuted when the observation is spatially flipped.

        Standard obs has no directional channels (action is type-only).

        Handles frame_stack: if active, direction channels are replicated
        for each stacked frame.
        """
        self._obs_dir_groups = []  # list of lists, each inner list = 5 channel indices

        extended = getattr(self._base_env, "extended_obs", False)
        if not extended:
            return  # standard obs has no directional channels

        n_ut = self._base_env._extended_n_ut
        # Direction groups start at: hp(1) + res(2) + owner(3) + ut(n_ut) + action(6)
        dir_start = 1 + 2 + 3 + n_ut + 6
        # 4 groups × 5 one-hot each: move_dir, harvest_dir, return_dir, produce_dir
        base_groups = []
        for g in range(4):
            start = dir_start + g * 5
            base_groups.append([start + k for k in range(5)])

        # Detect frame_stack: if obs_channels > single-frame channels, frames are stacked
        # Single frame = per_cell(59) + global(14) = 73
        single_frame_ch = self._base_env._extended_per_cell + self._base_env._extended_global
        # ReservedObs adds 1 extra channel ONCE after stacking (not per frame)
        reserved = False
        tmp = self.venv
        while hasattr(tmp, "venv"):
            if type(tmp).__name__ == "ReservedPositionObs":
                reserved = True
                break
            tmp = tmp.venv
        reserved_extra = 1 if reserved else 0

        effective_ch = self._obs_channels - reserved_extra
        if single_frame_ch > 0 and effective_ch > single_frame_ch:
            n_frames = effective_ch // single_frame_ch
        else:
            n_frames = 1

        for f in range(n_frames):
            offset = f * single_frame_ch
            for grp in base_groups:
                self._obs_dir_groups.append([c + offset for c in grp])

        # Channel permutation for 5-channel direction one-hot [N, E, S, W, none]
        self._obs_ch_perm = {
            0: [0, 1, 2, 3, 4],  # identity
            1: [0, 3, 2, 1, 4],  # H-flip: E↔W
            2: [2, 1, 0, 3, 4],  # V-flip: N↔S
            3: [2, 3, 0, 1, 4],  # both: E↔W + N↔S
        }

    def _actual_dims(self):
        """Current actual map dimensions (may change after switch_map)."""
        return self._base_env.height, self._base_env.width

    @staticmethod
    def _flip_region(region, fs):
        """Flip a spatial region according to flip state fs."""
        if fs == 1:
            return region[:, ::-1].copy()
        elif fs == 2:
            return region[::-1, :].copy()
        else:  # fs == 3
            return region[::-1, ::-1].copy()

    def _flip_obs(self, obs):
        """Flip actual map region in observations. obs: (B, pad_H, pad_W, C).

        After spatial flip, also permutes directional channels in extended obs
        so that e.g. 'moving East' becomes 'moving West' after a horizontal flip.
        """
        result = obs.copy()
        h, w = self._actual_dims()
        for i in range(self.num_envs):
            fs = self.flip_state[i]
            if fs == 0:
                continue
            # Flip only the actual region, padding stays zero
            region = obs[i, :h, :w, :]
            result[i, :h, :w, :] = self._flip_region(region, fs)

            # Permute directional one-hot channels in extended obs
            if self._obs_dir_groups:
                perm = self._obs_ch_perm[fs]
                region = result[i, :h, :w, :]  # (h, w, C) — basic indexing only
                for grp in self._obs_dir_groups:
                    orig = region[:, :, grp].copy()  # (h, w, 5)
                    region[:, :, grp] = orig[:, :, perm]
        return result

    def _flip_mask(self, mask):
        """Flip actual map region in action masks. mask: (B, pad_H*pad_W, D)."""
        h, w = self._actual_dims()
        pH, pW = self.pad_height, self.pad_width
        D = mask.shape[-1]
        result = mask.copy()

        for i in range(self.num_envs):
            fs = self.flip_state[i]
            if fs == 0:
                continue

            # Reshape to spatial, extract actual region
            spatial = mask[i].reshape(pH, pW, D)
            region = self._flip_region(spatial[:h, :w, :], fs)

            # Remap directional bits within flipped region
            dp = self.dir_perm[fs]
            for s in self._dir_slices:
                region[:, :, s] = region[:, :, s][:, :, dp]

            # Remap attack target bits
            tp = self.target_perm[fs]
            region[:, :, self._tgt_slice] = region[:, :, self._tgt_slice][:, :, tp]

            # Write back: only actual region changes, padding stays zero
            out = spatial.copy()
            out[:h, :w, :] = region
            result[i] = out.reshape(pH * pW, D)
        return result

    def _unflip_actions(self, actions):
        """Unflip actual map region in actions before sending to base env.

        actions shape: (B, pad_H * pad_W * num_action_params) — flattened.
        """
        h, w = self._actual_dims()
        pH, pW = self.pad_height, self.pad_width
        n_params = self.num_action_params
        actions = actions.reshape(self.num_envs, pH, pW, n_params)
        result = actions.copy()

        for i in range(self.num_envs):
            fs = self.flip_state[i]
            if fs == 0:
                continue

            region = result[i, :h, :w, :].copy()

            # Unflip directional action values (cols 1-4)
            dp = self.dir_perm[fs]
            for col in [1, 2, 3, 4]:
                region[:, :, col] = dp[region[:, :, col]]

            # Unflip attack target index (last column)
            tp = self.target_perm[fs]
            region[:, :, -1] = tp[region[:, :, -1]]

            # Unflip cell positions (flip is self-inverse)
            result[i, :h, :w, :] = self._flip_region(region, fs)

        return result.reshape(self.num_envs, -1)

    # ------------------------------------------------------------------
    # SB3 compatibility: allow intentional shadowing of get_action_mask
    # ------------------------------------------------------------------

    def getattr_depth_check(self, name, already_found):
        """Override SB3 ambiguity check for get_action_mask.

        We intentionally shadow the base env's get_action_mask to flip masks.
        Without this, VecMonitor.__getattr__ raises an ambiguity error because
        the attribute exists on both this wrapper and the base env.
        """
        if name == "get_action_mask":
            if already_found:
                return f"{type(self).__module__}.{type(self).__name__}"
            return None  # No ambiguity — this wrapper owns get_action_mask
        return super().getattr_depth_check(name, already_found)

    # ------------------------------------------------------------------
    # VecEnvWrapper interface
    # ------------------------------------------------------------------

    def reset(self):
        obs = self.venv.reset()
        self.flip_state = np.random.randint(0, 4, size=self.num_envs)
        return self._flip_obs(obs)

    def step_async(self, actions):
        unflipped = self._unflip_actions(actions)
        self.venv.step_async(unflipped)

    def step_wait(self):
        obs, rewards, dones, infos = self.venv.step_wait()
        # Re-randomize flip for envs that just ended an episode
        for i in range(self.num_envs):
            if dones[i]:
                self.flip_state[i] = np.random.randint(0, 4)
        return self._flip_obs(obs), rewards, dones, infos

    def get_action_mask(self):
        raw_mask = self.venv.get_action_mask()
        return self._flip_mask(raw_mask)
