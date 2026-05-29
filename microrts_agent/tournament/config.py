"""
Tournament configuration, matchup generation, and chunk scheduling.
"""

import heapq
import json
import os
import random
from collections import OrderedDict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from microrts_agent.paths import AGENT_DIR

# ── Data classes ─────────────────────────────────────────────────────────────


@dataclass
class TournamentConfig:
    """Tournament configuration loaded from JSON."""

    maps: list[str]
    ais: list[str]  # raw entries (bot names or "agent:path")
    iterations: int
    max_game_lengths: list[int]  # one per map
    time_budget: int
    iterations_budget: int  # -1 = unlimited
    full_observability: bool
    self_matches: bool
    timeout_check: bool
    run_gc: bool
    save_traces: bool
    save_game_logs: bool
    slow_ais: list[str]
    pre_analysis_budget: int  # ms, 0 = disabled

    # Derived (filled by load())
    ai_names: list[str] = field(default_factory=list)
    ai_is_agent: list[bool] = field(default_factory=list)
    ai_run_dirs: list[Optional[str]] = field(default_factory=list)

    config_name: str = ""
    config_path: str = ""

    @classmethod
    def load(cls, path: str) -> "TournamentConfig":
        with open(path) as f:
            d = json.load(f)

        config = cls(
            maps=d["maps"],
            ais=d["ais"],
            iterations=d.get("iterations", 1),
            max_game_lengths=d.get("maxGameLengths", [3000] * len(d["maps"])),
            time_budget=d.get("timeBudget", 100),
            iterations_budget=d.get("iterationsBudget", -1),
            full_observability=d.get("fullObservability", True),
            self_matches=d.get("selfMatches", False),
            timeout_check=d.get("timeoutCheck", False),
            run_gc=d.get("runGC", False),
            save_traces=d.get("saveTraces", False),
            save_game_logs=d.get("saveGameLogs", False),
            slow_ais=d.get("slowAIs", []),
            pre_analysis_budget=d.get("preAnalysisBudget", 0),
        )
        config.config_path = os.path.abspath(path)
        config.config_name = Path(path).stem

        for ai_entry in config.ais:
            if ai_entry.startswith("agent:"):
                run_dir = ai_entry[len("agent:") :]
                if not os.path.isabs(run_dir):
                    run_dir = str(AGENT_DIR / run_dir)
                cfg_file = os.path.join(run_dir, "config.json")
                pt_file = os.path.join(run_dir, "agent.pt")
                if not os.path.exists(cfg_file):
                    raise FileNotFoundError(f"Agent config not found: {cfg_file}")
                if not os.path.exists(pt_file):
                    raise FileNotFoundError(f"Agent weights not found: {pt_file}")
                config.ai_names.append(os.path.basename(run_dir.rstrip("/")))
                config.ai_is_agent.append(True)
                config.ai_run_dirs.append(run_dir)
            else:
                config.ai_names.append(ai_entry)
                config.ai_is_agent.append(False)
                config.ai_run_dirs.append(None)

        assert len(config.maps) == len(config.max_game_lengths), (
            f"maps ({len(config.maps)}) and maxGameLengths ({len(config.max_game_lengths)}) must match"
        )
        assert len(config.ais) >= 2, "Need at least 2 AIs"

        return config


@dataclass
class Matchup:
    iteration: int
    map_idx: int
    ai1_idx: int
    ai2_idx: int
    matchup_type: str  # "bot_vs_bot", "agent_vs_bot", "bot_vs_agent", "agent_vs_agent"


@dataclass
class MatchupGroup:
    """A group of matchups sharing (map, ai1, ai2, type) for vectorized play."""

    map_idx: int
    ai1_idx: int
    ai2_idx: int
    matchup_type: str
    iterations: list[int]


# ── Matchup generation ───────────────────────────────────────────────────────


def generate_matchups(config: TournamentConfig) -> list[Matchup]:
    """Generate full round-robin schedule."""
    matchups = []
    for iteration in range(config.iterations):
        for map_idx in range(len(config.maps)):
            for ai1_idx in range(len(config.ais)):
                for ai2_idx in range(len(config.ais)):
                    if not config.self_matches and ai1_idx == ai2_idx:
                        continue

                    a1_agent = config.ai_is_agent[ai1_idx]
                    a2_agent = config.ai_is_agent[ai2_idx]
                    if a1_agent and a2_agent:
                        mtype = "agent_vs_agent"
                    elif a1_agent:
                        mtype = "agent_vs_bot"
                    elif a2_agent:
                        mtype = "bot_vs_agent"
                    else:
                        mtype = "bot_vs_bot"

                    matchups.append(
                        Matchup(
                            iteration=iteration,
                            map_idx=map_idx,
                            ai1_idx=ai1_idx,
                            ai2_idx=ai2_idx,
                            matchup_type=mtype,
                        )
                    )
    return matchups


def group_matchups(matchups: list[Matchup]) -> list[MatchupGroup]:
    """Group matchups by (map, ai1, ai2, type) for vectorized batch play.

    All iterations of the same matchup are grouped together so they can
    be played in parallel using vectorized environments.
    """
    groups = OrderedDict()
    for m in matchups:
        key = (m.map_idx, m.ai1_idx, m.ai2_idx, m.matchup_type)
        if key not in groups:
            groups[key] = MatchupGroup(
                map_idx=m.map_idx,
                ai1_idx=m.ai1_idx,
                ai2_idx=m.ai2_idx,
                matchup_type=m.matchup_type,
                iterations=[],
            )
        groups[key].iterations.append(m.iteration)
    return list(groups.values())


# ── Chunking (LPT bin-packing on groups) ─────────────────────────────────────


def select_chunk_groups(
    groups: list[MatchupGroup], config: TournamentConfig, chunk: int, total_chunks: int
) -> list[MatchupGroup]:
    """Select a subset of matchup groups for the given chunk using LPT scheduling.

    Distributes complete groups (not individual matchups) so that each chunk
    gets full batches of iterations, maximizing vectorized env utilization.
    """
    slow_set = set(config.slow_ais)
    costs = []
    for g in groups:
        base = config.max_game_lengths[g.map_idx]
        slow_count = sum(1 for idx in [g.ai1_idx, g.ai2_idx] if config.ai_names[idx] in slow_set)
        if g.matchup_type == "agent_vs_agent":
            agent_factor = 5
        elif g.matchup_type != "bot_vs_bot":
            agent_factor = 3
        else:
            agent_factor = 1
        costs.append(base * (1 + slow_count) * agent_factor)

    indexed = sorted(enumerate(groups), key=lambda x: -costs[x[0]])

    heap = [(0, c) for c in range(total_chunks)]
    heapq.heapify(heap)
    assignments = {}
    for orig_idx, _ in indexed:
        load, cid = heapq.heappop(heap)
        assignments[orig_idx] = cid
        heapq.heappush(heap, (load + costs[orig_idx], cid))

    chunk_groups = [g for i, g in enumerate(groups) if assignments[i] == chunk]
    rng = random.Random(42 + chunk)
    rng.shuffle(chunk_groups)

    return chunk_groups
