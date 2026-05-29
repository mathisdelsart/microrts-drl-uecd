"""
Tournament Data Models
Shared data structures for tournament parsing and visualization
"""

from dataclasses import dataclass


@dataclass
class ParsedTournamentConfig:
    """Tournament configuration parsed from CSV results (for visualization)."""

    tournament_type: str  # Type of tournament (e.g. "round-robin", "all-combinations")
    ais: list[str]  # List of AI class names participating in the tournament
    maps: list[str]  # List of map file paths used in the tournament
    iterations: int  # Number of times each matchup is repeated
    max_game_lengths: list[int]  # Max game duration (in frames) per map
    time_budget: int  # CPU time budget per action (in ms) for each AI
    pregame_analysis_budget_ai1: int = 1000  # Pre-game thinking time (ms) for AI 1
    pregame_analysis_budget_ai2: int = 1000  # Pre-game thinking time (ms) for AI 2
    pre_analysis: bool = True  # Whether AIs get a pre-game analysis phase
    full_observability: bool = True  # Whether the game state is fully observable (vs fog of war)
    timeout_check: bool = True  # Whether to enforce action time limits
    run_gc: bool = False  # Whether to run Java garbage collection between games
    self_matches: bool = False  # Whether AIs play against themselves


@dataclass
class GameResult:
    """Individual game result from CSV parsing"""

    iteration: int  # Repetition index for this matchup (0-based)
    map_id: int  # Index of the map in the tournament map list
    ai1_id: int  # Index of player 1 AI in the tournament AI list
    ai2_id: int  # Index of player 2 AI in the tournament AI list
    time: int  # Game duration in frames
    winner: int  # 0 = player 1 wins, 1 = player 2 wins, -1 = draw
    crashed: int  # Whether the game ended due to a crash (0 or 1)
    timedout: int  # Whether the game ended due to timeout (0 or 1)
    ai1_time_ns: int = 0  # Total CPU time used by AI 1 (in nanoseconds)
    ai2_time_ns: int = 0  # Total CPU time used by AI 2 (in nanoseconds)


@dataclass
class GameData:
    """Individual game result for visualization (with full names)"""

    iteration: int  # Repetition index for this matchup (0-based)
    map_id: int  # Index of the map in the tournament map list
    map_name: str  # Human-readable map name (e.g. "basesWorkers16x16")
    ai1_id: int  # Index of player 1 AI in the tournament AI list
    ai1_name: str  # Human-readable name of player 1 AI (e.g. "RAISocketAI")
    ai2_id: int  # Index of player 2 AI in the tournament AI list
    ai2_name: str  # Human-readable name of player 2 AI
    time: int  # Game duration in frames
    winner: int  # 0 = player 1 wins, 1 = player 2 wins, -1 = draw
    crashed: int  # Whether the game ended due to a crash (0 or 1)
    timedout: int  # Whether the game ended due to timeout (0 or 1)
    ai1_time_ns: int = 0  # Total CPU time used by AI 1 (in nanoseconds)
    ai2_time_ns: int = 0  # Total CPU time used by AI 2 (in nanoseconds)
