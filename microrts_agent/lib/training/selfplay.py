"""Self-play state management: side alternation, opponent model, checkpoint pool.

Handles two types of self-play during training:
  - Latest SP: opponent uses the same weights as the training agent (no lag)
  - Pool SP:   opponent loads historical checkpoints from a saved pool

The SelfPlayManager encapsulates all self-play state that would otherwise
be 15+ local variables in the training loop.
"""

import os
from collections import deque

import numpy as np
import torch

from microrts_agent.lib.arch_factory import create_agent
from microrts_agent.lib.training.scheduling import CheckpointPool


class SelfPlayManager:
    """Manages self-play state: sides, opponent model, checkpoint pool.

    Env layout (all SP envs come before bot envs):
        [latest_SP_pair0, latest_SP_pair1, ... | pool_SP_pair0, ... | bot_envs...]

    Each pair = 2 envs playing the same game. One env is controlled by the
    training agent, the other by the opponent (latest weights or pool checkpoint).
    Which env is agent vs opponent alternates every episode to avoid P0/P1 bias.

    Adaptive mode: pool creation is delayed until bot opponents are mastered
    (checked via AdaptiveScheduler). Before activation, pool envs are idle
    (advantages zeroed out).
    """

    # ── Initialization ────────────────────────────────────────────────────

    def __init__(self, args, agent, obs_channels, action_nvec, device, run_dir):
        self.args = args
        self.device = device
        self.run_dir = run_dir

        # ── Env partition ──
        # n_selfplay = n_latest + n_pool, all in pairs of 2
        n_sp = args.num_selfplay_envs
        n_latest = args.num_latest_selfplay_envs
        n_pool = n_sp - n_latest

        self.n_selfplay = n_sp
        self.n_latest = n_latest
        self.n_pool = n_pool
        self.enabled = n_sp > 0

        # ── Mode flags ──
        # Adaptive: delay pool until AdaptiveScheduler signals mastery
        self.adaptive = args.adaptive_selfplay and args.adaptive_opponents and n_sp > 0
        # Historical: pool SP needs a separate opponent network to load checkpoints into
        self.use_historical = n_pool > 0 and (args.selfplay_pool_size > 0 or self.adaptive)

        # ── Opponent model + checkpoint pool ──
        self.opponent = None
        self.pool = None
        self.pool_active = False

        if self.use_historical:
            # Create a clone of the agent as the initial pool opponent
            self.opponent = create_agent(args, obs_channels, action_nvec, device)
            self.opponent.load_state_dict(agent.state_dict())
            self.opponent.eval()

            # Non-adaptive: pool ready immediately; adaptive: wait for mastery
            if not self.adaptive:
                self._create_pool(agent, seed_step=0)

        # ── Side alternation ──
        # _agent_side[i] = 0 means agent plays P0 in pair i, 1 means P1
        # Flipped every episode end to ensure ~50/50 P0/P1 experience
        self.num_pairs = n_sp // 2
        self.num_latest_pairs = n_latest // 2
        self._agent_side = np.zeros(self.num_pairs, dtype=np.int32)
        self.opp_indices, self.agent_indices, self.pool_opp_indices = self._compute_indices()

        # Rolling win rate for monitoring (last 100 SP games)
        self.win_buffer = deque(maxlen=100)

    # ── Pool creation ─────────────────────────────────────────────────────

    def _create_pool(self, agent, seed_step):
        """Create checkpoint pool and seed it with current weights.

        Pool is split 50/50 between recent checkpoints (FIFO) and
        sampled checkpoints (kept based on PFSP win rate diversity).
        """
        args = self.args
        pfsp_mode = args.pfsp_mode if args.pfsp else "uniform"
        self.pool = CheckpointPool(
            pool_dir=os.path.join(self.run_dir, "selfplay_pool"),
            max_recent=args.selfplay_pool_size // 2,
            max_sampled=args.selfplay_pool_size - args.selfplay_pool_size // 2,
            pfsp_mode=pfsp_mode,
            pfsp_p=args.pfsp_p,
        )
        self.pool.save(agent, seed_step)
        self.pool_active = True

    # ── Index computation ─────────────────────────────────────────────────

    def _compute_indices(self):
        """Compute which env indices belong to agent vs opponent.

        For pair i with envs (2i, 2i+1):
          _agent_side[i] == 0  →  agent=2i (P0), opponent=2i+1 (P1)
          _agent_side[i] == 1  →  agent=2i+1 (P1), opponent=2i (P0)

        Returns (opp_indices, agent_indices, pool_opp_indices).
        """
        opp_idx, agent_idx, pool_opp_idx = [], [], []

        for pair in range(self.num_pairs):
            p0, p1 = pair * 2, pair * 2 + 1

            if self._agent_side[pair] == 0:
                agent_idx.append(p0)
                opp_idx.append(p1)
                opp_env = p1
            else:
                agent_idx.append(p1)
                opp_idx.append(p0)
                opp_env = p0

            # Only pool pairs (not latest pairs) need the opponent model
            if pair >= self.num_latest_pairs:
                pool_opp_idx.append(opp_env)

        return opp_idx, agent_idx, pool_opp_idx

    # ── Per-step methods (called inside the rollout loop) ─────────────────

    def get_opponent_actions(self, next_obs, invalid_masks):
        """Compute actions for pool-SP opponent envs using the historical model.

        Latest-SP envs don't need this — they use the training agent's actions
        directly (both sides see the same weights).

        Returns (env_indices, actions) or ([], None) if no pool SP.
        """
        if not self.use_historical or not self.pool_opp_indices:
            return [], None

        opp_obs = next_obs[self.pool_opp_indices]
        opp_masks = invalid_masks[self.pool_opp_indices]
        with torch.no_grad():
            actions = self.opponent.forward(opp_obs, invalid_action_masks=opp_masks)["action"]
        return self.pool_opp_indices, actions

    def record_step_mask(self, sp_agent_mask, step):
        """Mark which envs are agent (1.0) vs opponent (0.0) at this step.

        This mask is later used by mask_advantages to zero out opponent
        transitions so they don't contribute to the policy gradient.
        """
        if sp_agent_mask is None or not self.enabled:
            return
        sp_agent_mask[step] = 1.0
        for opp_idx in self.opp_indices:
            sp_agent_mask[step, opp_idx] = 0.0

    def on_episode_end(self, dones):
        """Flip agent/opponent sides for pairs whose episode just ended.

        Both envs in a pair share the same game, so checking dones[p0]
        is sufficient (p1 will also be done). After flipping, recompute
        all env index lists.
        """
        if self.num_pairs == 0:
            return
        for pair in range(self.num_pairs):
            p0_idx = pair * 2
            if dones[p0_idx]:
                self._agent_side[pair] = 1 - self._agent_side[pair]
        self.opp_indices, self.agent_indices, self.pool_opp_indices = self._compute_indices()

    # ── Per-update methods (called after rollout, before PPO) ─────────────

    def mask_advantages(self, advantages, sp_agent_mask):
        """Zero out advantages for opponent envs so only agent transitions train.

        Works for both (T, N) single-head advantages and (T, N, H) multi-head
        advantages. In adaptive mode before pool activation, also zeros
        pool-assigned envs entirely (indices n_latest..n_selfplay) since those
        games are meaningless.
        """
        if sp_agent_mask is None or not self.enabled:
            return

        # Element-wise: agent envs keep their advantages, opponent envs → 0
        # Broadcast (T, N) mask over the head dimension if present.
        if advantages.dim() == 3:
            advantages *= sp_agent_mask.unsqueeze(-1)
        else:
            advantages *= sp_agent_mask

        # Adaptive pre-activation: pool envs play but don't train
        if self.adaptive and not self.pool_active:
            for idx in range(self.n_latest, self.n_selfplay):
                advantages[:, idx] = 0.0

    def manage_pool(self, agent, update, global_step, adaptive_scheduler=None):
        """Handle checkpoint pool lifecycle: creation, saving, and opponent swapping.

        Three scenarios:
          1. Adaptive, pool not yet active: check mastery → create pool or sync opponent
          2. Pool active: periodically save snapshots + swap active opponent
          3. No historical SP: no-op
        """
        if not self.use_historical:
            return
        args = self.args

        # ── Scenario 1: adaptive, waiting for mastery ──
        if self.adaptive and not self.pool_active:
            if adaptive_scheduler is not None and adaptive_scheduler.is_selfplay_ready(
                args.adaptive_selfplay_threshold
            ):
                # Mastery achieved → create pool and seed with current weights
                self._create_pool(agent, global_step)
                print(
                    f"  [ADAPT-SP] Terminal tier mastered! Self-play pool activated "
                    f"(seeded at step {global_step:,})"
                )
            else:
                # Not ready yet → keep opponent = current agent (latest-vs-latest)
                if update % args.selfplay_swap_interval == 0:
                    self.opponent.load_state_dict(agent.state_dict())
            return

        # ── Scenario 2: pool active, normal management ──
        if self.pool_active and self.pool is not None:
            # Save current weights as a new checkpoint in the pool
            if update % args.selfplay_save_interval == 0:
                self.pool.save(agent, global_step)
                print(f"  [SP] Saved checkpoint (pool={len(self.pool.all_checkpoints)})")

            # Swap opponent to a (possibly different) checkpoint from the pool
            if update % args.selfplay_swap_interval == 0:
                ckpt_path = self.pool.sample()
                if ckpt_path is not None:
                    self.opponent.load_state_dict(
                        torch.load(ckpt_path, map_location=self.device, weights_only=False)
                    )
                    wr_str = f" (WR={self.pool.get_win_rate(ckpt_path):.2f})" if args.pfsp else ""
                    print(f"  [SP] Swapped opponent -> {os.path.basename(ckpt_path)}{wr_str}")

    # ── Result tracking + logging ─────────────────────────────────────────

    def record_result(self, win):
        """Record a self-play game result for monitoring + PFSP win rate tracking."""
        self.win_buffer.append(win)
        if self.pool is not None:
            self.pool.record(win)

    def log_stats(self, writer, global_step):
        """Log self-play metrics to TensorBoard."""
        if self.win_buffer:
            # Should hover around 50% if SP is balanced
            writer.add_scalar("charts/selfplay_win_rate", np.mean(self.win_buffer), global_step)

        if self.pool is not None and self.args.pfsp and self.pool.current_ckpt:
            # PFSP: WR against the currently loaded opponent checkpoint
            writer.add_scalar(
                "pfsp/current_opponent_wr",
                self.pool.get_win_rate(self.pool.current_ckpt),
                global_step,
            )
            writer.add_scalar("pfsp/pool_size", len(self.pool.all_checkpoints), global_step)

        if self.adaptive:
            # Binary flag: useful to see on the TB curve when pool activates
            writer.add_scalar("adaptive/selfplay_pool_active", int(self.pool_active), global_step)
