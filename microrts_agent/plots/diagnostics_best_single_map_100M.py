"""Training diagnostics (return, length, per-bot WR) for best_single_map_100M_s1.

Output: figures/diagnostics_best_single_map_100M_s1.pdf
"""

from _style import apply_style, plot_training_diagnostics

apply_style()
plot_training_diagnostics("best_single_map_100M_s1")
