"""Training scheduling: opponent tracking, checkpoint pooling, reward/HP interpolation.

This module contains everything that *changes during training*:
  - OpponentTracker: per-opponent WR tracking + prioritized sampling weights
  - AdaptiveOpponentScheduler: automatic difficulty escalation
  - CheckpointPool: historical model storage for self-play (PFSP)
  - MapPLR: prioritized level replay for multi-map training
  - compute_schedules: central scheduler for LR, entropy, adv weights, reward weights, vf coefs
  - build_importance_weights: per-transition PPO weights from opponent difficulty
"""

import os
import random
from collections import deque
from typing import Optional

import numpy as np
import torch

# ═══════════════════════════════════════════════════════════════════════════
# Opponent Tracking — per-bot WR, returns, lengths, importance weights
# ═══════════════════════════════════════════════════════════════════════════


class OpponentTracker:
    """Track per-opponent win rates, episode stats, and optional importance weights.

    Always provides per-opponent win rates for logging (used by log_update).
    When ``prioritized=True``, also computes importance weights for PPO:
    opponents near 50% WR get higher weight in training (most learning signal).

    Two separate windows:
      - results (default 50): used for logging and display
      - priority_results (default 100): used for weight computation (larger = more stable)
    """

    def __init__(
        self,
        opponent_names: list[str],
        window: int = 50,
        prioritized: bool = False,
        priority_window: int = 100,
    ):
        self.names = sorted(set(opponent_names))
        self.window = window
        self.prioritized = prioritized

        # Per-opponent sliding windows for win/loss tracking
        self.results: dict[str, deque] = {name: deque(maxlen=window) for name in self.names}
        # Per-opponent episode stats for logging (return, length, total count)
        self.returns: dict[str, deque] = {name: deque(maxlen=window) for name in self.names}
        self.lengths: dict[str, deque] = {name: deque(maxlen=window) for name in self.names}
        self.episode_counts: dict[str, int] = dict.fromkeys(self.names, 0)

        # Per-(opponent, map) tracking for multi-map logging
        self.map_results: dict[tuple, deque] = {}

        # Separate (larger) window for priority weights to reduce variance
        if prioritized:
            self.priority_window = priority_window
            self.priority_results: dict[str, deque] = {
                name: deque(maxlen=priority_window) for name in self.names
            }

    def record(
        self,
        opponent_name: str,
        win: bool,
        map_path: Optional[str] = None,
        ep_return: float = 0.0,
        ep_length: int = 0,
    ) -> None:
        """Record a game result with optional episode stats.

        Auto-registers unknown opponents (can appear from adaptive scheduling
        promoting envs to a new bot type mid-training).
        """
        if opponent_name not in self.results:
            self.results[opponent_name] = deque(maxlen=self.window)
            self.returns[opponent_name] = deque(maxlen=self.window)
            self.lengths[opponent_name] = deque(maxlen=self.window)
            self.episode_counts[opponent_name] = 0
            self.names = sorted(self.results.keys())
            if self.prioritized:
                self.priority_results[opponent_name] = deque(maxlen=self.priority_window)

        self.results[opponent_name].append(float(win))
        self.returns[opponent_name].append(ep_return)
        self.lengths[opponent_name].append(ep_length)
        self.episode_counts[opponent_name] += 1

        if self.prioritized:
            self.priority_results[opponent_name].append(float(win))

        # Per-(opponent, map) for multi-map logging
        if map_path is not None:
            key = (opponent_name, map_path)
            if key not in self.map_results:
                self.map_results[key] = deque(maxlen=self.window)
            self.map_results[key].append(float(win))

    def get_win_rates(self) -> dict[str, float]:
        """Per-opponent win rates from the logging window (0.5 if no data)."""
        return {
            name: float(np.mean(results)) if results else 0.5
            for name, results in self.results.items()
        }

    def is_window_full(self, opponent_name: str) -> bool:
        """True if the logging window is saturated (enough data to be meaningful)."""
        buf = self.results.get(opponent_name)
        return buf is not None and len(buf) >= buf.maxlen

    def get_avg_returns(self) -> dict[str, float]:
        """Average episode return per opponent (sliding window)."""
        return {name: float(np.mean(buf)) if buf else 0.0 for name, buf in self.returns.items()}

    def get_avg_lengths(self) -> dict[str, float]:
        """Average episode length per opponent (sliding window)."""
        return {name: float(np.mean(buf)) if buf else 0.0 for name, buf in self.lengths.items()}

    def get_sampling_weights(self) -> dict[str, float]:
        """Importance weights for prioritized sampling in PPO.

        Weight formula: w_i = max(1 - |WR_i - 0.5|, 0.1)
          - WR = 50% → w = 1.0 (hardest opponent, most learning signal)
          - WR = 0% or 100% → w = 0.6, floored to 0.1 (trivial, but not ignored)

        Uses the priority window (larger, more stable) for variance reduction.
        Returns uniform weights if prioritized=False.
        """
        if not self.prioritized:
            n = len(self.names)
            return dict.fromkeys(self.names, 1.0 / n)

        # Win rates from the larger priority window
        wr = {
            name: float(np.mean(results)) if results else 0.5
            for name, results in self.priority_results.items()
        }
        # Floor at 0.1 so no opponent is completely ignored
        weights = {name: max(1.0 - abs(rate - 0.5), 0.1) for name, rate in wr.items()}
        total = sum(weights.values())
        return {name: w / total for name, w in weights.items()}


# ═══════════════════════════════════════════════════════════════════════════
# Adaptive Opponent Scheduling — automatic difficulty escalation
# ═══════════════════════════════════════════════════════════════════════════


class AdaptiveOpponentScheduler:
    """Promotes bot envs to harder opponents when a bot type is mastered.

    Difficulty tiers (ordered):
        Tier 0: RandomBiasedAI                   (random with heuristic bias)
        Tier 1: POWorkerRush                     (basic rush strategy)
        Tier 2: POLightRush                      (more effective rush)
        Tier 3: CoacAI / Mayari                  (competition-level, co-terminal)
        Tier 4: TMA / ObiBotKenobi / RAISocketAI (top tournament bots; terminal
                                                  tier — self-play pool activates
                                                  alongside once mastered)

    When WR against a bot type exceeds ``threshold`` over ``window`` games,
    most envs running that type are promoted to the next tier.
    ``MIN_RETAIN`` envs are kept per mastered tier to prevent catastrophic
    forgetting (the agent could forget how to beat easy bots).

    Co-terminal tiers split promoted envs via round-robin.

    Promotion is one-way: once promoted, envs never revert to easier types.

    Multi-map mastery criteria (via ``criteria`` parameter):
      - 'strict':  WR >= threshold on EVERY map individually (original behavior).
      - 'average': global mean WR >= threshold (ignores per-map breakdown).
      - 'hybrid':  global mean WR >= threshold AND min(per-map WR) >= hybrid_min
                   (default: 0.4). Recommended for multi-map runs with
                   pathological maps where strict would block promotion.
    """

    DIFFICULTY_TIERS = [
        ["RandomBiasedAI"],
        ["POWorkerRush"],
        ["POLightRush"],
        ["CoacAI", "Mayari"],
        ["TMA", "ObiBotKenobi", "RAISocketAI"],
    ]

    MIN_RETAIN = 2  # envs kept per mastered tier to avoid forgetting

    def __init__(
        self,
        env_labels: list[str],
        threshold: float = 0.7,
        window: int = 50,
        map_pool: Optional[list[str]] = None,
        criteria: str = "strict",
        hybrid_min: float = 0.4,
    ):
        assert criteria in ("strict", "average", "hybrid"), (
            f"criteria must be strict|average|hybrid, got {criteria!r}"
        )
        self.env_labels = list(env_labels)
        self.threshold = threshold
        self.window = window
        self.criteria = criteria
        self.hybrid_min = hybrid_min

        # Per-bot-type sliding window of results
        self.results: dict[str, deque] = {t: deque(maxlen=window) for t in set(env_labels)}
        self.promoted: set = set()  # bot types already promoted (one-way flag)
        self._promote_counter = 0  # round-robin index for co-terminal tier

        # Multi-map: per-(bot, map) tracking for stricter promotion criteria
        self.map_pool = map_pool
        self._per_map_results: dict[str, dict[str, deque]] = {}
        if map_pool:
            for bot in set(env_labels):
                self._per_map_results[bot] = {m: deque(maxlen=window) for m in map_pool}

    def record(self, opponent_type: str, win: bool, map_name: Optional[str] = None) -> None:
        """Record a game result for adaptive scheduling."""
        if opponent_type not in self.results:
            self.results[opponent_type] = deque(maxlen=self.window)
        self.results[opponent_type].append(float(win))

        # Per-map tracking (multi-map only)
        if map_name is not None and self.map_pool:
            if opponent_type not in self._per_map_results:
                self._per_map_results[opponent_type] = {
                    m: deque(maxlen=self.window) for m in self.map_pool
                }
            per_map = self._per_map_results[opponent_type]
            if map_name not in per_map:
                per_map[map_name] = deque(maxlen=self.window)
            per_map[map_name].append(float(win))

    def is_selfplay_ready(self, selfplay_threshold: float) -> bool:
        """Check if the terminal tier is mastered → self-play pool can activate.

        Returns True when EVERY terminal-tier bot type that has active envs
        has WR >= selfplay_threshold over a full window of games.
        In multi-map mode, requires mastery on every map individually.
        """
        terminal_tier = self.DIFFICULTY_TIERS[-1]
        active_terminal = [t for t in terminal_tier if t in set(self.env_labels)]
        if not active_terminal:
            return False

        for bot_type in active_terminal:
            results = self.results.get(bot_type)
            if not results or len(results) < self.window:
                return False
            global_wr = float(np.mean(results))

            if not self.map_pool:
                if global_wr < selfplay_threshold:
                    return False
                continue

            per_map = self._per_map_results.get(bot_type, {})
            for map_name in self.map_pool:
                map_results = per_map.get(map_name)
                if not map_results or len(map_results) < self.window:
                    return False
            per_map_wrs = [float(np.mean(per_map[m])) for m in self.map_pool]

            if self.criteria == "strict":
                if any(wr < selfplay_threshold for wr in per_map_wrs):
                    return False
            elif self.criteria == "average":
                if global_wr < selfplay_threshold:
                    return False
            else:  # hybrid
                if global_wr < selfplay_threshold or min(per_map_wrs) < self.hybrid_min:
                    return False
        return True

    def _is_mastered(self, bot_type: str) -> tuple:
        """Check if a bot type is mastered. Returns (mastered: bool, wr: float).

        Single-map: WR >= threshold over a full window (always strict).
        Multi-map: behavior depends on self.criteria:
          - 'strict':  WR >= threshold on EVERY map in the pool
          - 'average': global mean WR >= threshold (ignores per-map)
          - 'hybrid':  global mean WR >= threshold AND min(per-map) >= hybrid_min
        """
        results = self.results.get(bot_type)
        if not results or len(results) < self.window:
            return False, 0.0
        global_wr = float(np.mean(results))

        # Single-map: always strict on the single global window
        if not self.map_pool:
            return global_wr >= self.threshold, global_wr

        per_map = self._per_map_results.get(bot_type, {})

        # Require full windows on all maps before evaluating any criterion
        for map_name in self.map_pool:
            map_results = per_map.get(map_name)
            if not map_results or len(map_results) < self.window:
                return False, global_wr

        per_map_wrs = [float(np.mean(per_map[m])) for m in self.map_pool]

        if self.criteria == "strict":
            return all(wr >= self.threshold for wr in per_map_wrs), global_wr
        if self.criteria == "average":
            return global_wr >= self.threshold, global_wr
        # hybrid
        return (global_wr >= self.threshold and min(per_map_wrs) >= self.hybrid_min), global_wr

    def check_promotions(self) -> list[tuple]:
        """Check if any bot type is mastered and return promotions.

        Scans tiers 0..N-2 (skips the terminal tier — nothing to promote to).
        For each mastered type, keeps MIN_RETAIN envs on that type and
        promotes the rest to the next tier via round-robin.

        Returns list of (env_idx, new_type, old_type, wr) tuples.
        Empty list if no promotions needed.
        """
        promotions = []

        # Only check non-terminal tiers (terminal has nowhere to promote to)
        for tier_idx, tier in enumerate(self.DIFFICULTY_TIERS[:-1]):
            for bot_type in tier:
                if bot_type in self.promoted:
                    continue

                mastered, wr = self._is_mastered(bot_type)
                if not mastered:
                    continue

                next_tier = self.DIFFICULTY_TIERS[tier_idx + 1]
                envs_with_type = [i for i, label in enumerate(self.env_labels) if label == bot_type]

                # Keep MIN_RETAIN envs on this tier to avoid forgetting
                to_promote = envs_with_type[self.MIN_RETAIN :]
                if not to_promote:
                    continue

                # Promote each env to the next tier (round-robin for co-terminal)
                for idx in to_promote:
                    new_type = next_tier[self._promote_counter % len(next_tier)]
                    self._promote_counter += 1
                    promotions.append((idx, new_type, bot_type, wr))
                    self.env_labels[idx] = new_type

                self.promoted.add(bot_type)

                # Register next-tier types in results if they're new
                for t in next_tier:
                    if t not in self.results:
                        self.results[t] = deque(maxlen=self.window)

        return promotions


# ═══════════════════════════════════════════════════════════════════════════
# Checkpoint Pool — historical model storage for self-play (PFSP)
# ═══════════════════════════════════════════════════════════════════════════


class CheckpointPool:
    """Manages a pool of historical model checkpoints for self-play.

    Two-tier storage:
      - recent: ring buffer (FIFO) of the last ``max_recent`` checkpoints
      - all_checkpoints: full list of every checkpoint ever saved

    When sampling, candidates = all recent + random sample of ``max_sampled``
    from the older historical checkpoints. This balances recency with diversity.

    Supports Prioritized Fictitious Self-Play (PFSP) sampling modes:
      - 'uniform': random selection (default)
      - 'hard':    P(B) ∝ (1 - WR)^p — focus on opponents we lose against
      - 'var':     P(B) ∝ WR*(1 - WR) + ε — focus on 50/50 matchups

    Reference: Vinyals et al. (2019), "Grandmaster level in StarCraft II"
    """

    def __init__(
        self,
        pool_dir: str,
        max_recent: int = 10,
        max_sampled: int = 10,
        pfsp_mode: str = "uniform",
        pfsp_p: float = 1.0,
        pfsp_window: int = 50,
    ):
        self.pool_dir = pool_dir
        os.makedirs(pool_dir, exist_ok=True)
        self.max_recent = max_recent
        self.max_sampled = max_sampled
        self.recent: deque = deque(maxlen=max_recent)
        self.all_checkpoints: list[str] = []

        # PFSP state: per-checkpoint win/loss history for weighted sampling
        self.pfsp_mode = pfsp_mode
        self.pfsp_p = pfsp_p
        self.pfsp_window = pfsp_window
        self.ckpt_results: dict[str, deque] = {}
        self.current_ckpt: Optional[str] = None  # currently loaded opponent

        # Resume support: rehydrate from disk if pool_*.pt files already exist.
        # Sort by the numeric step embedded in the filename so recent/historical
        # ordering matches a fresh training run.
        existing = []
        try:
            for fname in os.listdir(pool_dir):
                if fname.startswith("pool_") and fname.endswith(".pt"):
                    try:
                        step = int(fname[len("pool_") : -len(".pt")])
                    except ValueError:
                        continue
                    existing.append((step, os.path.join(pool_dir, fname)))
        except FileNotFoundError:
            pass
        existing.sort(key=lambda x: x[0])
        for _step, path in existing:
            self.all_checkpoints.append(path)
            self.recent.append(path)  # deque(maxlen) auto-drops old entries
            self.ckpt_results[path] = deque(maxlen=pfsp_window)
        if existing:
            print(
                f"  [SP pool] Rehydrated {len(existing)} checkpoints from {pool_dir} "
                f"(recent={len(self.recent)}/{max_recent})"
            )

    def save(self, agent: torch.nn.Module, step: int) -> None:
        """Save current agent weights as a new checkpoint in the pool."""
        path = os.path.join(self.pool_dir, f"pool_{step}.pt")
        torch.save(agent.state_dict(), path)
        self.recent.append(path)
        self.all_checkpoints.append(path)
        self.ckpt_results[path] = deque(maxlen=self.pfsp_window)

    def record(self, win: bool) -> None:
        """Record a game result against the currently loaded checkpoint."""
        if self.current_ckpt is not None and self.current_ckpt in self.ckpt_results:
            self.ckpt_results[self.current_ckpt].append(float(win))

    def get_win_rate(self, ckpt_path: str) -> float:
        """Win rate against a specific checkpoint (0.5 default if no data)."""
        results = self.ckpt_results.get(ckpt_path)
        if not results:
            return 0.5
        return float(np.mean(results))

    def _build_candidate_pool(self) -> list[str]:
        """Build candidate pool: all recent + random sample of older historical."""
        pool = list(self.recent)
        if len(self.all_checkpoints) > self.max_recent:
            recent_set = set(self.recent)
            historical = [c for c in self.all_checkpoints if c not in recent_set]
            n_sample = min(self.max_sampled, len(historical))
            pool += random.sample(historical, n_sample)
        return pool

    def sample(self) -> Optional[str]:
        """Sample a checkpoint from the pool with optional PFSP weighting.

        PFSP modes:
          uniform: equal probability (backward-compatible default)
          hard:    P(B) ∝ (1 - WR)^p — focus on opponents we lose against
          var:     P(B) ∝ WR*(1 - WR) + 0.1 — focus on 50/50 matchups
        """
        pool = self._build_candidate_pool()
        if not pool:
            return None

        if self.pfsp_mode == "uniform":
            chosen = random.choice(pool)
        else:
            win_rates = [self.get_win_rate(p) for p in pool]
            if self.pfsp_mode == "hard":
                weights = [(1.0 - wr) ** self.pfsp_p for wr in win_rates]
            else:  # 'var'
                weights = [wr * (1.0 - wr) + 0.1 for wr in win_rates]
            chosen = random.choices(pool, weights=weights, k=1)[0]

        self.current_ckpt = chosen
        return chosen


# ═══════════════════════════════════════════════════════════════════════════
# Prioritized Level Replay — sample harder maps more often
# ═══════════════════════════════════════════════════════════════════════════


class MapPLR:
    """Simplified Prioritized Level Replay for multi-map training.

    Instead of cycling through maps uniformly (round-robin), samples the
    next map proportional to learning signal: maps where the agent struggles
    (low win rate) get more training time.

    Sampling weight for map i:  w_i = (1 - WR_i) + epsilon

    The epsilon (default 0.1) ensures even mastered maps get occasional
    visits to prevent catastrophic forgetting.

    Falls back to uniform random if no data for any map yet (WR defaults to 0.5).
    """

    def __init__(self, map_pool: list[str], window: int = 50, epsilon: float = 0.1):
        self.map_pool = list(map_pool)
        self.epsilon = epsilon
        self.outcomes: dict[str, deque] = {m: deque(maxlen=window) for m in self.map_pool}

    def record(self, map_path: str, win: int):
        """Record a game outcome (1=win, 0=loss) for the given map."""
        if map_path in self.outcomes:
            self.outcomes[map_path].append(win)

    def get_win_rates(self) -> dict[str, float]:
        """Per-map win rates (0.5 default if no data)."""
        return {m: float(np.mean(buf)) if len(buf) > 0 else 0.5 for m, buf in self.outcomes.items()}

    def sample(self) -> str:
        """Sample next map proportional to (1 - WR + epsilon)."""
        wr = self.get_win_rates()
        weights = np.array([1.0 - wr[m] + self.epsilon for m in self.map_pool])
        probs = weights / weights.sum()
        return np.random.choice(self.map_pool, p=probs)

    def get_sampling_probs(self) -> dict[str, float]:
        """Current sampling probabilities per map (for logging)."""
        wr = self.get_win_rates()
        weights = np.array([1.0 - wr[m] + self.epsilon for m in self.map_pool])
        probs = weights / weights.sum()
        return {m: float(p) for m, p in zip(self.map_pool, probs)}


# ═══════════════════════════════════════════════════════════════════════════
# Hyperparameter scheduling — interpolation functions
# ═══════════════════════════════════════════════════════════════════════════


def get_scheduled_reward_weight(progress: float, args) -> np.ndarray:
    """Linearly interpolate reward weights from shaped to target.

    Transitions from args.reward_weight toward a target weight configuration
    between reward_schedule_start and reward_schedule_end.

    Target priority:
      1. If args.reward_weight_end is set, use it as the target.
      2. Otherwise, target = args.reward_weight * reward_schedule_min
         (with WinDrawLoss kept at full weight).

    Before start: full shaped reward (exploration signal).
    After end: target reward.
    """
    if args.reward_schedule == "none":
        return np.array(args.reward_weight)

    start = np.array(args.reward_weight)
    weight_end = getattr(args, "reward_weight_end", None)
    if weight_end is not None:
        end = np.array(weight_end)
        if end.shape != start.shape:
            raise ValueError(
                f"reward_weight_end length {len(end)} != reward_weight length {len(start)}"
            )
    else:
        min_frac = getattr(args, "reward_schedule_min", 0.0)
        end = start * min_frac
        end[0] = 10.0  # WinDrawLoss always at full weight
    t_start = args.reward_schedule_start
    t_end = args.reward_schedule_end

    if progress <= t_start:
        return start
    elif progress >= t_end:
        return end
    else:
        t = (progress - t_start) / (t_end - t_start)
        return (1 - t) * start + t * end


def interpolate_schedule(progress, start, end, t_start, t_end):
    """Generic linear interpolation between two values with phase boundaries.

    Supports scalars and tuples/lists (element-wise interpolation).

        start ─────╲
                     ╲  (linear transition)
                      ╲───── end
        |      |      |      |
        0   t_start  t_end  1.0
    """
    if progress <= t_start:
        return start
    if progress >= t_end:
        return end
    t = (progress - t_start) / (t_end - t_start)
    if isinstance(start, (tuple, list)):
        return tuple(s + t * (e - s) for s, e in zip(start, end))
    return start + t * (end - start)


def multi_phase_interpolate(current_step, phase_steps, phase_values):
    """Piecewise linear interpolation across N training phases.

    Used by RAISocketAI-style multi-phase schedules where hyperparameters
    change in stages (e.g. 3-phase schedule over 300M steps).

    Args:
        current_step: current global training step
        phase_steps: boundary steps [s1, e1, s2, e2, ...].
            Each (s_i, e_i) pair defines a transition region.
            Between transitions = flat plateau.
        phase_values: values at each plateau. Can be scalars or tuples.

    Example (3 phases, 2 transitions):
        phase_steps  = [90M, 150M, 180M, 240M]
        phase_values = [v1, v2, v3]

        0──90M:     v1 (flat)
        90M──150M:  v1 → v2 (linear)
        150M──180M: v2 (flat)
        180M──240M: v2 → v3 (linear)
        240M──end:  v3 (flat)
    """
    n_transitions = len(phase_steps) // 2

    for i in range(n_transitions):
        t_start = phase_steps[2 * i]
        t_end = phase_steps[2 * i + 1]

        # Before this transition: return current plateau value
        if current_step < t_start:
            return phase_values[i]

        # Inside this transition: linearly interpolate
        if current_step <= t_end:
            t = (current_step - t_start) / max(t_end - t_start, 1)
            v_start = phase_values[i]
            v_end = phase_values[i + 1]
            if isinstance(v_start, (tuple, list)):
                return tuple(s + t * (e - s) for s, e in zip(v_start, v_end))
            return v_start + t * (v_end - v_start)

    # Past all transitions: return final plateau
    return phase_values[-1]


# ═══════════════════════════════════════════════════════════════════════════
# Central schedule dispatcher — called once per update
# ═══════════════════════════════════════════════════════════════════════════


def compute_schedules(args, update, global_step):
    """Compute all scheduled hyperparameters for this update.

    Priority order for each parameter:
      1. Multi-phase schedule (--phase-steps + --phase-lr/ent/adv/vf)
      2. Two-point phase schedule (--advantage-weights-end defines transition region)
      3. Classic annealing or constant value

    Returns dict with keys:
      lr, ent_coef, adv_weights, reward_weight, vf_coef, vf_sparse, vf_cost
    """
    progress = update / args.num_updates  # 0.0 → 1.0 over training
    t_start = args.reward_schedule_start
    t_end = args.reward_schedule_end
    use_multi_phase = args.phase_steps is not None
    # Two-point phase: advantage_weights_end signals a start→end transition
    use_phase_schedule = args.advantage_weights_end is not None

    # ── Learning rate ──
    if use_multi_phase and args.phase_lr is not None:
        lr = multi_phase_interpolate(global_step, args.phase_steps, args.phase_lr)
    elif args.anneal_lr:
        lr_end = args.learning_rate_end if args.learning_rate_end is not None else 0.0
        if use_phase_schedule:
            # Phase-aware: only anneals between t_start and t_end
            lr = interpolate_schedule(progress, args.learning_rate, lr_end, t_start, t_end)
        else:
            # Classic linear annealing over the full training
            frac = max(0.0, 1.0 - (update - 1.0) / args.num_updates)
            lr = lr_end + frac * (args.learning_rate - lr_end)
    else:
        lr = args.learning_rate

    # ── LR warmup (overrides above during warmup phase) ──
    warmup = getattr(args, "lr_warmup_updates", 0)
    if warmup > 0 and update <= warmup:
        lr = args.learning_rate * (update / warmup)

    # ── Entropy coefficient ──
    if use_multi_phase and args.phase_ent_coef is not None:
        ent_coef = multi_phase_interpolate(global_step, args.phase_steps, args.phase_ent_coef)
    elif args.ent_coef_end is not None:
        if use_phase_schedule:
            # Phase-aware: transition between t_start and t_end
            ent_coef = interpolate_schedule(
                progress, args.ent_coef, args.ent_coef_end, t_start, t_end
            )
        else:
            # Simple linear decay over the full training
            ent_coef = args.ent_coef + progress * (args.ent_coef_end - args.ent_coef)
    else:
        ent_coef = args.ent_coef

    # ── Advantage weights (shaped/sparse/cost blend) ──
    if use_multi_phase and args.phase_adv_weights is not None:
        adv_weights = multi_phase_interpolate(global_step, args.phase_steps, args.phase_adv_weights)
    elif args.advantage_weights is not None and use_phase_schedule:
        adv_weights = interpolate_schedule(
            progress, args.advantage_weights, args.advantage_weights_end, t_start, t_end
        )
    elif args.advantage_weights is not None:
        adv_weights = args.advantage_weights
    else:
        adv_weights = None

    # ── Reward weights (shaped → sparse transition) ──
    reward_weight = get_scheduled_reward_weight(progress, args)

    # ── Value function loss coefficients ──
    if use_multi_phase and args.phase_vf_coefs is not None:
        vf_tuple = multi_phase_interpolate(global_step, args.phase_steps, args.phase_vf_coefs)
        vf_coef, vf_sparse, vf_cost = vf_tuple[0], vf_tuple[1], vf_tuple[2]
    elif use_phase_schedule and args.dual_value_heads:
        vf_coef = interpolate_schedule(progress, args.vf_coef, args.vf_coef_end, t_start, t_end)
        vf_sparse = interpolate_schedule(
            progress, args.vf_coef_sparse, args.vf_coef_sparse_end, t_start, t_end
        )
        vf_cost = interpolate_schedule(
            progress, args.vf_coef_cost, args.vf_coef_cost_end, t_start, t_end
        )
    else:
        vf_coef = args.vf_coef
        vf_sparse = args.vf_coef_sparse
        vf_cost = args.vf_coef_cost

    return {
        "lr": lr,
        "ent_coef": ent_coef,
        "adv_weights": adv_weights,
        "reward_weight": reward_weight,
        "vf_coef": vf_coef,
        "vf_sparse": vf_sparse,
        "vf_cost": vf_cost,
    }


# ═══════════════════════════════════════════════════════════════════════════
# Importance weights — per-transition PPO weights from opponent difficulty
# ═══════════════════════════════════════════════════════════════════════════


def build_importance_weights(opp_tracker, env_opponent_labels, num_envs, num_steps, device):
    """Build per-transition importance weights from opponent difficulty.

    Each env's transitions are weighted by its opponent's sampling weight
    (from OpponentTracker.get_sampling_weights). The weights are normalized
    so mean = 1.0 (preserving unbiased gradient magnitude).

    Returns flattened tensor of shape (T*N,), or None if prioritized=False.
    """
    if not opp_tracker.prioritized:
        return None

    # Map each env index to its opponent's weight
    weights = opp_tracker.get_sampling_weights()
    env_weights = torch.tensor(
        [weights[env_opponent_labels[i]] for i in range(num_envs)],
        dtype=torch.float32,
        device=device,
    )
    # Normalize: mean weight = 1.0 so total gradient magnitude is preserved
    env_weights = env_weights * num_envs / env_weights.sum()

    # Expand (N,) → (T, N) → (T*N,) to match flattened rollout tensors
    return env_weights.unsqueeze(0).expand(num_steps, -1).reshape(-1)
