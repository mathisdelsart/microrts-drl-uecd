"""
Environment pool for tournament execution.

Manages and reuses MicroRTS environments keyed by (map_path, env_type).
Environments are created once with config.iterations slots and reused
across matchups by swapping bot AIs between games.
"""

import contextlib
import os
import time
from pathlib import Path

import numpy as np
import torch

from microrts_agent.envs.factory import JVM_ARGS

from .config import TournamentConfig
from .servers import UTS_IMASS_TRAINING_DIR


def save_client_trace(
    client, ai1_idx: int, ai2_idx: int, map_idx: int, iteration: int, output_dir: str
):
    """Save trace from a single JNI client to ZIP."""
    if not hasattr(client, "collectTrace") or not bool(client.collectTrace):
        return
    traces_dir = os.path.join(output_dir, "traces")
    os.makedirs(traces_dir, exist_ok=True)
    zip_name = f"{ai1_idx}-vs-{ai2_idx}-{map_idx}-{iteration}.zip"
    zip_path = os.path.join(traces_dir, zip_name)
    with contextlib.suppress(Exception):
        client.saveTraceToZip(zip_path)


class EnvPool:
    """Manages MicroRTS environments, keyed by (map_path, env_type).

    Reuses envs for the same map by swapping bot AIs between games.
    Envs are created with config.iterations slots for vectorized batch play.

    NOTE: bot AIs are NOT cached -- Java clone() is shallow, corrupting
    shared mutable state between clones and originals.
    """

    def __init__(self, config: TournamentConfig, device: torch.device, game_log_path: str = None):
        self.config = config
        self.device = device
        self._bot_envs = {}  # map_path -> MicroRTSBotVecEnv
        self._agent_envs = {}  # (map, player, po, fm, fs, ro, eo) -> env
        self._selfplay_envs = {}  # (map, po, fm, fs, ro, eo) -> env
        self._agents = {}  # run_dir -> (agent, agent_config)
        self._preanalyzed = set()  # (bot_name, map_path) already done
        self._jvm_redirected = False
        self._game_log_path = game_log_path
        self._ai_mapping = None
        self._java_classes = None

    def _get_ai_mapping(self):
        """Lazy-load AI_MAPPING (requires JVM to be started)."""
        if self._ai_mapping is None:
            from microrts_agent.registries import ai as microrts_ai

            self._ai_mapping = microrts_ai.AI_MAPPING
        return self._ai_mapping

    def _get_java_classes(self):
        """Lazy-load Java classes for AI manipulation."""
        if self._java_classes is None:
            import jpype

            self._java_classes = {
                "AIWithComputationBudget": jpype.JClass("ai.core.AIWithComputationBudget"),
                "InterruptibleAI": jpype.JClass("ai.core.InterruptibleAI"),
                "ContinuingAI": jpype.JClass("ai.core.ContinuingAI"),
                "System": jpype.JClass("java.lang.System"),
            }
        return self._java_classes

    def _redirect_java_output(self):
        """Redirect Java stdout/stderr to file or /dev/null."""
        if self._jvm_redirected:
            return
        import jpype

        if jpype._jpype.isStarted():
            if self._game_log_path:
                fos = jpype.JClass("java.io.FileOutputStream")(str(self._game_log_path), True)
                ps = jpype.JClass("java.io.PrintStream")(fos)
            else:
                ps = jpype.JClass("java.io.PrintStream")(
                    jpype.JClass("java.io.OutputStream").nullOutputStream()
                )
            jpype.JClass("java.lang.System").setOut(ps)
            jpype.JClass("java.lang.System").setErr(ps)
            self._jvm_redirected = True

    def configure_cloned_ai(self, ai):
        """Set budgets and wrap ContinuingAI on a cloned AI instance.

        Matches Java Tournament behavior (clone -> setBudgets -> wrap -> reset).
        Called AFTER JNI reset() which already cloned and reset the AI.
        """
        jc = self._get_java_classes()
        if isinstance(ai, jc["AIWithComputationBudget"]):
            ai.setTimeBudget(self.config.time_budget)
            ai.setIterationsBudget(self.config.iterations_budget)
        if isinstance(ai, jc["InterruptibleAI"]):
            ai = jc["ContinuingAI"](ai)
        return ai

    def run_java_gc(self):
        """Trigger Java garbage collection."""
        jc = self._get_java_classes()
        jc["System"].gc()

    def get_bot_env(self, map_path: str, max_steps: int):
        """Get or create a bot-vs-bot env with config.iterations slots."""
        if map_path in self._bot_envs:
            return self._bot_envs[map_path]

        from microrts_agent.envs.factory import make_bot_env

        N = self.config.iterations
        ai_mapping = self._get_ai_mapping()
        env = make_bot_env(
            ai1s=[ai_mapping["RandomBiasedAI"]] * N,
            ai2s=[ai_mapping["RandomBiasedAI"]] * N,
            partial_obs=not self.config.full_observability,
            max_steps=max_steps,
            map_paths=[map_path] * N,
            jvm_args=JVM_ARGS,
        )
        self._redirect_java_output()
        self._bot_envs[map_path] = env
        return env

    def _apply_wrappers(self, env, agent_cfg):
        """Apply observation wrappers matching the agent's training config.

        Delegates to setup.py:apply_env_wrappers for canonical wrapper order.
        Note: augment_symmetry is intentionally NOT applied during tournament.
        """
        if not agent_cfg:
            return env
        from microrts_agent.wrappers.factory import apply_env_wrappers

        return apply_env_wrappers(
            env,
            frame_stack=agent_cfg.get("frame_stack", 0),
            reserved_obs=agent_cfg.get("reserved_obs", False),
        )

    def get_agent_env(self, map_path: str, max_steps: int, player: int, agent_cfg: dict = None):
        """Get or create an agent-vs-bot env with config.iterations slots.

        Uses agent_cfg to apply the correct filtered_masks and observation
        wrappers (frame_stack, reserved_obs) matching training.
        """
        partial_obs = agent_cfg.get("partial_obs", False) if agent_cfg else False
        filtered_masks = agent_cfg.get("filtered_masks", False) if agent_cfg else False
        frame_stack = agent_cfg.get("frame_stack", 0) if agent_cfg else 0
        reserved_obs = agent_cfg.get("reserved_obs", False) if agent_cfg else False
        extended_obs = agent_cfg.get("extended_obs", False) if agent_cfg else False

        key = (
            map_path,
            player,
            partial_obs,
            filtered_masks,
            frame_stack,
            reserved_obs,
            extended_obs,
        )
        if key in self._agent_envs:
            return self._agent_envs[key]

        from microrts_agent.envs.factory import make_agent_env

        N = self.config.iterations
        ai_mapping = self._get_ai_mapping()
        env = make_agent_env(
            num_bot_envs=N,
            num_selfplay_envs=0,
            ai2s=[ai_mapping["RandomBiasedAI"]] * N,
            map_paths=[map_path] * N,
            player=player,
            partial_obs=partial_obs,
            max_steps=max_steps,
            # Get reward_weight from agent config; unused by bots, required by Java
            reward_weight=np.array(agent_cfg["reward_weight"])
            if agent_cfg and "reward_weight" in agent_cfg
            else np.array([10.0, 1.0, 1.0, 0.2, 0.2, 1.0, 4.0, 4.0, 4.0, 0.0]),
            jvm_args=JVM_ARGS,
            padded=False,
            max_height=0,
            max_width=0,
            alternate_players=False,
            filtered_masks=filtered_masks,
            extended_obs=extended_obs,
        )
        env = self._apply_wrappers(env, agent_cfg)
        self._redirect_java_output()
        self._agent_envs[key] = env
        return env

    def get_selfplay_env(
        self, map_path: str, max_steps: int, agent_cfg: dict = None, agent_cfg2: dict = None
    ):
        """Get or create a self-play env for agent-vs-agent matches.

        Uses num_selfplay_envs = 2 * iterations (each game occupies 2 slots:
        even = P0 obs/actions, odd = P1 obs/actions).

        When agent_cfg2 is provided, uses the richest obs config from both
        agents (extended_obs OR, filtered_masks OR). Wrappers are NOT applied
        since per-model adapters in game_loops handle obs mismatches.
        """
        partial_obs = agent_cfg.get("partial_obs", False) if agent_cfg else False

        if agent_cfg2 is not None:
            # Use richest config from both agents — adapters handle per-model differences
            fm1 = agent_cfg.get("filtered_masks", False) if agent_cfg else False
            fm2 = agent_cfg2.get("filtered_masks", False)
            filtered_masks = fm1 or fm2

            eo1 = agent_cfg.get("extended_obs", False) if agent_cfg else False
            eo2 = agent_cfg2.get("extended_obs", False)
            extended_obs = eo1 or eo2

            # Do NOT apply wrappers — adapters handle frame_stack/reserved_obs per-model
            frame_stack = 0
            reserved_obs = False
        else:
            filtered_masks = agent_cfg.get("filtered_masks", False) if agent_cfg else False
            extended_obs = agent_cfg.get("extended_obs", False) if agent_cfg else False
            frame_stack = agent_cfg.get("frame_stack", 0) if agent_cfg else 0
            reserved_obs = agent_cfg.get("reserved_obs", False) if agent_cfg else False

        key = (map_path, partial_obs, filtered_masks, frame_stack, reserved_obs, extended_obs)
        if key in self._selfplay_envs:
            return self._selfplay_envs[key]

        from microrts_agent.envs.factory import make_agent_env

        N = self.config.iterations
        num_slots = 2 * N
        env = make_agent_env(
            num_bot_envs=0,
            num_selfplay_envs=num_slots,
            ai2s=[],
            map_paths=[map_path] * num_slots,
            player=0,
            partial_obs=partial_obs,
            max_steps=max_steps,
            # unused by bots, required by Java
            reward_weight=np.array(agent_cfg["reward_weight"])
            if agent_cfg and "reward_weight" in agent_cfg
            else np.array([10.0, 1.0, 1.0, 0.2, 0.2, 1.0, 4.0, 4.0, 4.0, 0.0]),
            jvm_args=JVM_ARGS,
            padded=False,
            max_height=0,
            max_width=0,
            alternate_players=False,
            filtered_masks=filtered_masks,
            extended_obs=extended_obs,
        )
        if agent_cfg2 is None:
            # Single-agent config: apply wrappers as before
            env = self._apply_wrappers(env, agent_cfg)
        # When agent_cfg2 is provided, skip wrappers — adapters handle them
        self._redirect_java_output()
        self._selfplay_envs[key] = env
        return env

    def get_agent(self, run_dir: str):
        """Load and cache a trained agent."""
        if run_dir not in self._agents:
            from microrts_agent.architectures.factory import load_agent_from_config

            agent, cfg = load_agent_from_config(run_dir, device=str(self.device))
            self._agents[run_dir] = (agent, cfg)
        return self._agents[run_dir]

    def get_bot_ai(self, bot_name: str, map_path: str, utt):
        """Create a fresh Java AI instance (never cached -- clone() is shallow)."""
        ai_mapping = self._get_ai_mapping()
        return ai_mapping[bot_name](utt)

    def run_preanalysis(self, ai, bot_name: str, map_path: str, gs):
        """Run pre-game analysis for a bot on a map (once per tournament run)."""
        budget = self.config.pre_analysis_budget
        if budget <= 0:
            return

        key = (bot_name, map_path)
        from jpype.types import JLong

        map_stem = Path(map_path).stem
        is_uts = bot_name.lower().replace("_", "") == "utsimass"

        # UtsImass: each match spawns a new Java AI instance -> new socket -> new
        # server thread with no loaded data. Must call preGameAnalysis every match.
        # Do NOT early-return on _preanalyzed cache for UtsImass.
        if not is_uts and key in self._preanalyzed:
            return

        UTS_MIN_BUDGET_MS = 300_000
        if is_uts:
            map_dir = UTS_IMASS_TRAINING_DIR / map_stem
            cached = map_dir.is_dir() and any(
                f.name.endswith("_config.json") for f in map_dir.iterdir()
            )
            os.makedirs(map_dir, exist_ok=True)
            # UtsImass accepts either a relative or absolute folder path; prefer relative
            # for log readability, fall back to absolute when CWD isn't a repo ancestor
            # (e.g. SLURM $TMPDIR), where `relative_to(Path.cwd())` would raise ValueError.
            abs_folder = map_dir.resolve()
            try:
                folder = str(abs_folder.relative_to(Path.cwd()))
            except ValueError:
                folder = str(abs_folder)
            if cached:
                # Load cached data into fresh server thread (fast ~1s)
                effective_budget = 1000
                label = f"cached ({map_dir.name}/)"
            elif budget < UTS_MIN_BUDGET_MS:
                print(
                    f"    Pre-analysis: {bot_name} on {map_stem} "
                    f"-- skipped (budget {budget / 1000:.0f}s < 5min minimum)"
                )
                return
            else:
                effective_budget = int(budget)
                label = f"{budget / 1000:.0f}s budget"
            print(f"    Pre-analysis: {bot_name} on {map_stem} -- {label}")
            t0 = time.time()
            ai.preGameAnalysis(gs, JLong(effective_budget), folder)
            elapsed = time.time() - t0
            print(f"    Pre-analysis done ({elapsed:.1f}s)")
            return

        print(f"    Pre-analysis: {bot_name} on {map_stem} ({budget / 1000:.0f}s budget)...")
        t0 = time.time()
        ai.preGameAnalysis(gs, JLong(budget))
        elapsed = time.time() - t0
        self._preanalyzed.add(key)
        print(f"    Pre-analysis done ({elapsed:.1f}s)")

    @staticmethod
    def _get_base_env(env):
        from microrts_agent.envs.base_vec_env import get_base_env

        return get_base_env(env)

    def close_all(self):
        """Close all env vec_clients (keep JVM alive)."""
        for env in self._bot_envs.values():
            with contextlib.suppress(Exception):
                env.vec_client.close()
        for env in self._agent_envs.values():
            with contextlib.suppress(Exception):
                self._get_base_env(env).vec_client.close()
        for env in self._selfplay_envs.values():
            with contextlib.suppress(Exception):
                self._get_base_env(env).vec_client.close()
