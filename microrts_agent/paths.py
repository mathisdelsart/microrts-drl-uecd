"""
Centralized path constants for the project.

All output directories and key paths defined once. Entry-point scripts
import these instead of computing paths from SCRIPT_DIR.
"""

from pathlib import Path

AGENT_DIR = Path(__file__).parent.resolve()  # microrts_agent/

PROJECT_ROOT = AGENT_DIR.parent  # microrts-drl-uecd/
OUTPUTS_DIR = PROJECT_ROOT / "outputs"
MICRORTS_DIR = AGENT_DIR / "microrts"

RUNS_DIR = OUTPUTS_DIR / "runs"
RECORDINGS_DIR = OUTPUTS_DIR / "recordings"
BENCHMARKS_DIR = OUTPUTS_DIR / "inference_bench"
COMPARISONS_DIR = OUTPUTS_DIR / "comparisons"
TOURNAMENT_RESULTS_DIR = OUTPUTS_DIR / "tournament_results"

TOURNAMENT_CONFIGS_DIR = AGENT_DIR / "tournament_configs"
