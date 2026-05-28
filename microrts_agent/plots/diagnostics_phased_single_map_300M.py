"""Training diagnostics (return, length, per-bot WR) for phased_single_map_300M_s1.

Output: figures/diagnostics_phased_single_map_300M_s1.pdf
"""

from _style import apply_style, plot_training_diagnostics

apply_style()
plot_training_diagnostics("phased_single_map_300M_s1")
