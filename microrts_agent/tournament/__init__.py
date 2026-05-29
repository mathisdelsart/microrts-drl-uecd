"""
Tournament infrastructure: configuration, execution, and analysis.

  - config:     TournamentConfig, matchup generation, chunk scheduling
  - env_pool:   EnvPool (environment reuse across matchups)
  - game_loops: Vectorized game loops (bot-vs-bot, agent-vs-bot, agent-vs-agent)
  - csv_io:     CSV read/write, chunk merging, results directory resolution
  - servers:     External server management (UTS_Imass)
  - runner:      Tournament orchestration
  - parser:      Post-hoc CSV parsing (TournamentParser)
  - visualizer:  Result visualisation (TournamentVisualizer, TournamentData) + plots/
  - game_theory: Game-theoretic metrics (Nash, alpha-rank, ...)
"""

from .config import TournamentConfig, generate_matchups, group_matchups, select_chunk_groups
from .csv_io import ensure_tournament_csv, resolve_results_dir
from .parser import TournamentParser
from .runner import run_tournament
from .servers import needs_uts_imass, start_uts_imass_server, stop_uts_imass_server
from .visualizer import TournamentData, TournamentVisualizer

__all__ = [
    "TournamentConfig",
    "generate_matchups",
    "group_matchups",
    "select_chunk_groups",
    "ensure_tournament_csv",
    "resolve_results_dir",
    "run_tournament",
    "needs_uts_imass",
    "start_uts_imass_server",
    "stop_uts_imass_server",
    "TournamentParser",
    "TournamentData",
    "TournamentVisualizer",
]
