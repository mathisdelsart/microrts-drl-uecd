#!/bin/bash
#SBATCH --job-name=bc_gen_extra
#SBATCH --output=slurm-bc-gen-%j.out
#SBATCH --error=slurm-bc-gen-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Generate extra BC data: 500 games RAI vs CoacAI + 500 games RAI vs Mayari
#
# Usage:
#   sbatch experiments/bc/bc_generate_extra.slurm
#

echo "============================================================"
echo "  BC Extra Data Generation — CoacAI + Mayari"
echo "============================================================"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Node:     $SLURM_NODELIST"
echo "  Started:  $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

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
    echo "ERROR: cluster_venv not found."
    exit 1
fi

export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python -m microrts_agent.bc generate \
    --bot RAISocketAI \
    --opponents CoacAI Mayari \
    --games-per-opponent 1000 1000 \
    --output outputs/bc_data/RAISocketAI_vs_CoacAI_Mayari_2000g.npz

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
