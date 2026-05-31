#!/bin/bash
#SBATCH --job-name=evaluate_feats_ablation
#SBATCH --output=slurm-evaluate_feats_ablation-%j_%a.out
#SBATCH --error=slurm-evaluate_feats_ablation-%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-109
#
# FEATURE EVALUATION: 500 games per position (1000 total) per bot
#
#   22 features × 5 bots = 110 jobs (array 0-109)
#   Each job runs: microrts-agent evaluate --agent <feat_dir> --opponent <bot> --nb_games 500
#
# Usage:
#   sbatch experiments/ablation/evaluate_feats_ablation.slurm
#

SEED=${SEED:-1}

FEATS=(
    "outputs/runs/feat_ablation/baseline_s${SEED}"
    "outputs/runs/feat_ablation/extended_obs_s${SEED}"
    "outputs/runs/feat_ablation/triple_heads_s${SEED}"
    "outputs/runs/feat_ablation/filtmask_resobs_s${SEED}"
    "outputs/runs/feat_ablation/bots_selfplay_pfsp_s${SEED}"
    "outputs/runs/feat_ablation/aux_spatial_s${SEED}"
    "outputs/runs/feat_ablation/aux_contrastive_s${SEED}"
    "outputs/runs/feat_ablation/aux_unit_count_s${SEED}"
    "outputs/runs/feat_ablation/framestack4_s${SEED}"
    "outputs/runs/feat_ablation/autoregressive_s${SEED}"
    "outputs/runs/feat_ablation/hl_gauss_s${SEED}"
    "outputs/runs/feat_ablation/gelu_s${SEED}"
    "outputs/runs/feat_ablation/adaptive_s${SEED}"
    "outputs/runs/feat_ablation/priori_samp_s${SEED}"
    "outputs/runs/feat_ablation/pae_s${SEED}"
    "outputs/runs/feat_ablation/spp_critic_s${SEED}"
    "outputs/runs/feat_ablation/opponent_modeling_s${SEED}"
    "outputs/runs/feat_ablation/popart_s${SEED}"
    "outputs/runs/feat_ablation/augment_symmetry_s${SEED}"
    "outputs/runs/feat_ablation/buildtime_rewards_s${SEED}"
    "outputs/runs/feat_ablation/hierarchical_mask_s${SEED}"
    "outputs/runs/feat_ablation/autoregressive_hmask_s${SEED}"
)
FEAT_NAMES=(
    "baseline" "extended_obs" "triple_heads" "filtmask_resobs"
    "bots_selfplay_pfsp" "aux_spatial" "aux_contrastive" "aux_unit_count"
    "framestack4" "autoregressive" "hl_gauss" "gelu"
    "adaptive" "priori_samp" "pae" "spp_critic"
    "opponent_modeling" "popart" "augment_symmetry"
    "buildtime_rewards" "hierarchical_mask" "autoregressive_hmask"
)

BOTS=("RandomBiasedAI" "WorkerRush" "LightRush" "CoacAI" "Mayari")

# Decode array task ID: feat_idx = task / 5, bot_idx = task % 5
FEAT_IDX=$(( SLURM_ARRAY_TASK_ID / 5 ))
BOT_IDX=$(( SLURM_ARRAY_TASK_ID % 5 ))

AGENT_DIR=${FEATS[$FEAT_IDX]}
FEAT_NAME=${FEAT_NAMES[$FEAT_IDX]}
BOT=${BOTS[$BOT_IDX]}

echo "============================================================"
echo "  FEATURE EVALUATION: 1000 games (500 P0 + 500 P1)"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Feature:      $FEAT_NAME"
echo "  Agent dir:    $AGENT_DIR"
echo "  Opponent:     $BOT"
echo "  Node:         $SLURM_NODELIST"
echo "  Started:      $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

source "$(dirname "$0")/../_setup_env.sh"

# ── Run evaluation ─────────────────────────────────────────────────────────

OUTDIR="outputs/runs/feat_ablation/EVAL/s${SEED}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/${FEAT_NAME}_vs_${BOT}.txt"

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
echo "  EVAL COMPLETE: $FEAT_NAME vs $BOT"
echo "  Output: $OUTFILE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
