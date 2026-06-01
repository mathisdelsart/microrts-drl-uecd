"""Per-model observation adapter for mismatched configs (rl_vs_rl).

Handles frame_stack, reserved_obs, extended_obs, and multi_map padding
independently per model so two agents with different configs can share one env.
Used by evaluate.py and tournament game_loops. Also exposes is_agent_dir().
"""

import os

import numpy as np


def is_agent_dir(path):
    """True if path contains config.json (= RL run directory)."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))


class ObsAdapter:
    """Transform raw env obs into model-expected obs.

    Each model gets its own adapter. Handles:
      - multi_map: crop padded obs to real map size (for single-map agents on padded env)
      - frame_stack: stack last N frames along channel axis
      - reserved_obs: append reserved-position plane
      - extended_obs: flag for obs re-encoding (handled externally)
    """

    def __init__(self, config, num_envs):
        self.fs = config.get("frame_stack", 0) if config else 0
        self.reserved = config.get("reserved_obs", False) if config else False
        self.extended_obs = config.get("extended_obs", False) if config else False
        self.multi_map = config.get("multi_map", False) if config else False
        self.max_map_size = config.get("max_map_size", 64) if config else 64
        self.num_envs = num_envs
        self._buf = None

    def reset(self):
        """Clear frame buffer (call between games)."""
        self._buf = None

    def __call__(self, obs, reserved_planes=None, raw_obs_override=None, real_h=None, real_w=None):
        """Process raw obs -> model-expected obs.

        obs:              (N, H, W, C) raw env output (possibly padded)
        reserved_planes:  (N, H, W, 1) or None
        raw_obs_override: (N, H, W, C) pre-encoded obs (for extended_obs mismatch)
        real_h, real_w:   actual map size (for cropping padded obs for single-map agents)
        """
        if raw_obs_override is not None:
            obs = raw_obs_override

        # Crop padded obs for single-map agents (they expect real map size, not padded)
        if (
            not self.multi_map
            and real_h is not None
            and real_w is not None
            and (obs.shape[1] != real_h or obs.shape[2] != real_w)
        ):
            obs = obs[:, :real_h, :real_w, :]
            if reserved_planes is not None:
                reserved_planes = reserved_planes[:, :real_h, :real_w, :]

        # Multi-map agents expect obs padded to their max_map_size
        if self.multi_map:
            target = self.max_map_size
            h, w = obs.shape[1], obs.shape[2]
            if h != target or w != target:
                # Crop if env is larger, pad if env is smaller
                if h > target or w > target:
                    obs = obs[:, :target, :target, :]
                    if reserved_planes is not None:
                        reserved_planes = reserved_planes[:, :target, :target, :]
                if h < target or w < target:
                    pad_h, pad_w = target - obs.shape[1], target - obs.shape[2]
                    obs = np.pad(obs, ((0, 0), (0, pad_h), (0, pad_w), (0, 0)))
                    if reserved_planes is not None:
                        reserved_planes = np.pad(
                            reserved_planes, ((0, 0), (0, pad_h), (0, pad_w), (0, 0))
                        )

        # Frame stacking: maintain sliding window of last N frames
        if self.fs > 1:
            if self._buf is None:
                self._buf = np.tile(obs, (1, 1, 1, self.fs)).copy()
            else:
                C = obs.shape[-1]
                self._buf[:, :, :, :-C] = self._buf[:, :, :, C:]
                self._buf[:, :, :, -C:] = obs
            obs = self._buf.copy()

        # Append reserved-position channel
        if self.reserved and reserved_planes is not None:
            # reserved_planes are built at the real map size, but obs may have been
            # padded/cropped to max_map_size (multi_map) or cropped to the real size
            # (single_map). When the env already emits obs padded to max_map_size, the
            # blocks above leave reserved_planes untouched, so align spatial dims here.
            oh, ow = obs.shape[1], obs.shape[2]
            rh, rw = reserved_planes.shape[1], reserved_planes.shape[2]
            if (rh, rw) != (oh, ow):
                reserved_planes = np.pad(
                    reserved_planes,
                    ((0, 0), (0, max(0, oh - rh)), (0, max(0, ow - rw)), (0, 0)),
                )[:, :oh, :ow, :]
            obs = np.concatenate([obs, reserved_planes], axis=-1)

        return obs


def get_reserved_planes(env):
    """Compute reserved-position planes from Java game state (no wrapper needed).

    Accepts either a wrapped env or a raw base env.
    Returns (num_envs, H, W, 1) float32 array.
    """
    from ts import FilteredMaskClient  # type: ignore[import]

    from microrts_agent.envs.base_vec_env import get_base_env

    # Accept both wrapped and unwrapped envs
    if hasattr(env, "venv") or hasattr(env, "vec_client"):
        base = env if hasattr(env, "vec_client") else get_base_env(env)
    else:
        base = get_base_env(env)
    h, w = base.height, base.width
    num_envs = base.num_selfplay_envs + base.num_bot_envs
    planes = np.zeros((num_envs, h, w, 1), dtype=np.float32)

    for i in range(len(base.vec_client.selfPlayClients)):
        gs = base.vec_client.selfPlayClients[i].gs
        grid = np.array(FilteredMaskClient.getReservedPositions(gs))
        gh, gw = grid.shape
        planes[i * 2, :gh, :gw, 0] = grid
        planes[i * 2 + 1, :gh, :gw, 0] = grid

    n_sp = base.num_selfplay_envs
    for j in range(base.num_bot_envs):
        gs = base.vec_client.clients[j].gs
        grid = np.array(FilteredMaskClient.getReservedPositions(gs))
        gh, gw = grid.shape
        planes[n_sp + j, :gh, :gw, 0] = grid

    return planes


def process_obs_rl_vs_rl(
    env, raw_obs, p0_adapter, p1_adapter, device, base_env=None, env_extended_obs=False
):
    """Split raw obs into per-model tensors, applying adapters for mismatched configs.

    Handles extended_obs re-encoding, frame_stack, reserved_obs, and multi_map cropping.
    Used by evaluate.py and tournament game_loops.py.
    """
    import torch

    from microrts_agent.envs.base_vec_env import get_base_env

    if base_env is None:
        base_env = get_base_env(env) if hasattr(env, "venv") else env

    reserved = None
    if p0_adapter.reserved or p1_adapter.reserved:
        reserved = get_reserved_planes(env)

    # Re-encode obs when model's extended_obs differs from env's
    p0_override, p1_override = None, None
    if p0_adapter.extended_obs != env_extended_obs:
        if p0_adapter.extended_obs:
            p0_override = base_env._get_all_extended_obs()[0::2]
        else:
            p0_override = base_env.encode_standard_from_raw()[0::2]
    if p1_adapter.extended_obs != env_extended_obs:
        if p1_adapter.extended_obs:
            p1_override = base_env._get_all_extended_obs()[1::2]
        else:
            p1_override = base_env.encode_standard_from_raw()[1::2]

    real_h = base_env.height
    real_w = base_env.width

    p0_obs = p0_adapter(
        raw_obs[0::2],
        reserved[0::2] if reserved is not None else None,
        raw_obs_override=p0_override,
        real_h=real_h,
        real_w=real_w,
    )
    p1_obs = p1_adapter(
        raw_obs[1::2],
        reserved[1::2] if reserved is not None else None,
        raw_obs_override=p1_override,
        real_h=real_h,
        real_w=real_w,
    )

    return torch.as_tensor(p0_obs).to(device), torch.as_tensor(p1_obs).to(device)
