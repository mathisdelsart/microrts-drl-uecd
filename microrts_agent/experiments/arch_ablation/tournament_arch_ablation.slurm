#!/bin/bash
#SBATCH --job-name=tourn_arch
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-6
#
# ARCHITECTURE TOURNAMENT — 12 AIs, 264 matchups, 7 chunks
#
#   6 RL agents + 5 scripted bots, round-robin, 2 iterations
#
# Usage:
#   sbatch microrts_agent/experiments/ablation/tournament_arch_ablation.slurm
#
# After all chunks finish:
#   python microrts_agent/run_tournament.py parse arch_ablation_16x16
#   python microrts_agent/run_tournament.py viz arch_ablation_16x16
#   python microrts_agent/run_tournament.py analyze arch_ablation_16x16

TOTAL_CHUNKS=7

echo "============================================================"
echo "  TOURNAMENT — arch_ablation_16x16 (chunk $SLURM_ARRAY_TASK_ID/$TOTAL_CHUNKS)"
echo "============================================================"
echo "  Job ID:   $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
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

# ── Run tournament chunk ─────────────────────────────────────────────────

python microrts_agent/run_tournament.py run arch_ablation_16x16 \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks $TOTAL_CHUNKS

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  CHUNK $SLURM_ARRAY_TASK_ID COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
