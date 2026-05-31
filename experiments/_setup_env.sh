# Shared SLURM environment preamble for every experiments/*.slurm script.
#
# Each SLURM script sources this file right after its #SBATCH block:
#
#     source "$(dirname "$0")/../_setup_env.sh"
#
# It loads the Python / Java / CUDA cluster modules, activates the project's
# venv at $HOME/microrts-drl-uecd/cluster_venv, fixes PYTHONUNBUFFERED, and
# cds to the repo root. Any missing module aborts the SLURM job with a clear
# message: preferable to letting the python process crash with a stale
# library later.

module load Python/3.11 2>/dev/null || module load Python/3.9 2>/dev/null || {
    echo "ERROR: No Python module found"; exit 1; }
module load Java/17 2>/dev/null || module load Java/11 2>/dev/null || {
    echo "ERROR: No Java module found"; exit 1; }
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1 2>/dev/null || {
    echo "WARNING: No CUDA module found"; }

VENV_DIR="$HOME/microrts-drl-uecd/cluster_venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: cluster_venv not found. Run setup/cluster.sh first."
    exit 1
fi

export PYTHONUNBUFFERED=1
cd "$HOME/microrts-drl-uecd"
