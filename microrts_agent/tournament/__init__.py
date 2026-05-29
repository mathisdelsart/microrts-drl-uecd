"""
Tournament infrastructure: configuration, execution, and analysis.

  - config:     TournamentConfig, matchup generation, chunk scheduling
  - env_pool:   EnvPool (environment reuse across matchups)
  - game_loops: Vectorized game loops (bot-vs-bot, agent-vs-bot, agent-vs-agent)
  - csv_io:     CSV read/write, chunk merging, results directory resolution
  - servers:    External server management (UTS_Imass)
  - runner:     Tournament orchestration
  - viz:        Post-hoc analysis — CSV parsing, visualization, game-theoretic metrics
"""

from .config import TournamentConfig, generate_matchups, group_matchups, select_chunk_groups
from .csv_io import ensure_tournament_csv, resolve_results_dir
from .runner import run_tournament
from .servers import needs_uts_imass, start_uts_imass_server, stop_uts_imass_server

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
]
