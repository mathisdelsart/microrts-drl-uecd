#!/bin/bash
#SBATCH --job-name=eval_feats
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-109
#
# FEATURE EVALUATION — 500 games per position (1000 total) per bot
#
#   22 features × 5 bots = 110 jobs (array 0-109)
#   Each job runs: python evaluate.py --agent <feat_dir> --opponent <bot> --nb_games 500
#
# Usage:
#   sbatch experiments/ablation/evaluate_feats_ablation.slurm
#

SEED=${SEED:-1}

FEATS=(
    "outputs/runs/feat_ablation/feats_baseline_s${SEED}"
    "outputs/runs/feat_ablation/feats_extended_obs_s${SEED}"
    "outputs/runs/feat_ablation/feats_triple_heads_s${SEED}"
    "outputs/runs/feat_ablation/feats_filtmask_resobs_s${SEED}"
    "outputs/runs/feat_ablation/feats_bots_selfplay_pfsp_s${SEED}"
    "outputs/runs/feat_ablation/feats_aux_spatial_s${SEED}"
    "outputs/runs/feat_ablation/feats_aux_contrastive_s${SEED}"
    "outputs/runs/feat_ablation/feats_aux_unit_count_s${SEED}"
    "outputs/runs/feat_ablation/feats_framestack4_s${SEED}"
    "outputs/runs/feat_ablation/feats_autoregressive_s${SEED}"
    "outputs/runs/feat_ablation/feats_hl_gauss_s${SEED}"
    "outputs/runs/feat_ablation/feats_gelu_s${SEED}"
    "outputs/runs/feat_ablation/feats_adaptive_s${SEED}"
    "outputs/runs/feat_ablation/feats_priori_samp_s${SEED}"
    "outputs/runs/feat_ablation/feats_pae_s${SEED}"
    "outputs/runs/feat_ablation/feats_spp_critic_s${SEED}"
    "outputs/runs/feat_ablation/feats_opponent_modeling_s${SEED}"
    "outputs/runs/feat_ablation/feats_popart_s${SEED}"
    "outputs/runs/feat_ablation/feats_augment_symmetry_s${SEED}"
    "outputs/runs/feat_ablation/feats_buildtime_rewards_s${SEED}"
    "outputs/runs/feat_ablation/feats_hierarchical_mask_s${SEED}"
    "outputs/runs/feat_ablation/feats_autoregressive_hmask_s${SEED}"
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
echo "  FEATURE EVALUATION — 1000 games (500 P0 + 500 P1)"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Feature:      $FEAT_NAME"
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

OUTDIR="outputs/runs/feat_ablation/EVAL/s${SEED}"
mkdir -p "$OUTDIR"
OUTFILE="$OUTDIR/${FEAT_NAME}_vs_${BOT}.txt"

python microrts_agent/evaluate.py \
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
