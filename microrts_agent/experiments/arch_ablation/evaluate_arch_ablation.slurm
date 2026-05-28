#!/bin/bash
#SBATCH --job-name=eval_arch
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-34
#
# ARCHITECTURE EVALUATION — 500 games per position (1000 total) per bot
#
#   7 architectures × 5 bots = 35 jobs (array 0-34)
#   Each job runs: python evaluate.py --agent <arch> --opponent <bot> --nb_games 500
#   Results saved to outputs/eval_arch_ablation/
#
# Usage:
#   sbatch microrts_agent/experiments/ablation/evaluate_arch_ablation.slurm
#

SEED=${SEED:-1}

ARCHS=(
    "outputs/runs/arch_ablation/arch_ablation_gridnet_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_impala_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_unet_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_impala_entity_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_unet_entity_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_unet_entity_cbam_s${SEED}"
    "outputs/runs/arch_ablation/arch_ablation_unet_entity_cbam_deep_s${SEED}"
)
ARCH_NAMES=("gridnet" "impala" "unet" "impala_entity" "unet_entity" "unet_entity_cbam" "unet_entity_cbam_deep")

BOTS=("RandomBiasedAI" "POWorkerRush" "POLightRush" "CoacAI" "Mayari")

# Decode array task ID: arch_idx = task / 5, bot_idx = task % 5
ARCH_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
BOT_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

AGENT_DIR=${ARCHS[$ARCH_IDX]}
ARCH_NAME=${ARCH_NAMES[$ARCH_IDX]}
BOT=${BOTS[$BOT_IDX]}

echo "============================================================"
echo "  ARCH EVALUATION — 1000 games (500 P0 + 500 P1)"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Architecture: $ARCH_NAME"
echo "  Agent dir:    $AGENT_DIR"
echo "  Opponent:     $BOT"
echo "  Node:         $SLURM_NODELIST"
echo "  Started:      $(date)"
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

# ── Run evaluation ─────────────────────────────────────────────────────────

OUTDIR="outputs/eval_arch_ablation/s${SEED}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/${ARCH_NAME}_vs_${BOT}.txt"

python microrts_agent/evaluate.py \
    --agent "$AGENT_DIR" \
    --opponent "$BOT" \
    --nb_games 500 \
    --maps maps/16x16/basesWorkers16x16A.xml \
    --max-steps 4000 \
    2>&1 | tee "$OUTFILE"

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  EVAL COMPLETE: $ARCH_NAME vs $BOT"
echo "  Output: $OUTFILE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
