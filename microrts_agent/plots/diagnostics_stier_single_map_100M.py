"""Training diagnostics (return, length, per-bot WR) for stier_single_map_100M_s1.

Output: figures/diagnostics_stier_single_map_100M_s1.pdf
"""

from _style import apply_style, plot_training_diagnostics

apply_style()
plot_training_diagnostics("stier_single_map_100M_s1")
