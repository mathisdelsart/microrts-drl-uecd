"""
Tournament visualization plot modules.

Each module exports a single public function with signature:
    plot_xxx(data, console, output_path, ...)
"""

from .final_standings import plot_final_standings
from .game_length_boxplot import plot_game_length_distribution
from .game_theory_metrics import generate_game_theory
from .head_to_head import plot_head_to_head_matrix
from .winrate_heatmap import plot_per_map_winrates

__all__ = [
    "plot_final_standings",
    "plot_head_to_head_matrix",
    "plot_game_length_distribution",
    "plot_per_map_winrates",
    "generate_game_theory",
]
