#!/bin/bash
#SBATCH --job-name=evaluate_arch_ablation
#SBATCH --output=slurm-evaluate_arch_ablation-%j_%a.out
#SBATCH --error=slurm-evaluate_arch_ablation-%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-34
#
# ARCHITECTURE EVALUATION: 500 games per position (1000 total) per bot
#
#   7 architectures × 5 bots = 35 jobs (array 0-34)
#   Each job runs: microrts-agent evaluate --agent <arch> --opponent <bot> --nb_games 500
#   Results saved to outputs/eval_arch_ablation/
#
# Usage:
#   sbatch experiments/ablation/evaluate_arch_ablation.slurm
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

BOTS=("RandomBiasedAI" "WorkerRush" "LightRush" "CoacAI" "Mayari")

# Decode array task ID: arch_idx = task / 5, bot_idx = task % 5
ARCH_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
BOT_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

AGENT_DIR=${ARCHS[$ARCH_IDX]}
ARCH_NAME=${ARCH_NAMES[$ARCH_IDX]}
BOT=${BOTS[$BOT_IDX]}

echo "============================================================"
echo "  ARCH EVALUATION: 1000 games (500 P0 + 500 P1)"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Architecture: $ARCH_NAME"
echo "  Agent dir:    $AGENT_DIR"
echo "  Opponent:     $BOT"
echo "  Node:         $SLURM_NODELIST"
echo "  Started:      $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

source "$(dirname "$0")/../_setup_env.sh"

# ── Run evaluation ─────────────────────────────────────────────────────────

OUTDIR="outputs/eval_arch_ablation/s${SEED}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/${ARCH_NAME}_vs_${BOT}.txt"

microrts-agent evaluate \
    --agent "$AGENT_DIR" \
    --opponent "$BOT" \
    --nb_games 500 \
    --maps maps/open_competition/basesWorkers16x16A.xml \
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
