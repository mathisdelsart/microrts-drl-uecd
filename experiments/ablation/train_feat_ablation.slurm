#!/bin/bash
#SBATCH --job-name=feats_abl
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err
#SBATCH --time=1-12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-21
#
# FEATURE ABLATION — 50M steps, UNet-Entity-CBAM-Deep, single map (basesWorkers16x16A)
#
#   0  = BASELINE (no extra features)
#   1  = + extended obs (73ch)
#   2  = + triple value heads
#   3  = + filtered masks + reserved obs
#   4  = + bots + self-play + PFSP (12 bots + 12 SP: 6 pool + 6 latest)
#   5  = + aux spatial (ownership prediction)
#   6  = + aux contrastive (temporal InfoNCE)
#   7  = + aux unit count
#   8  = + frame stacking (4 frames)
#   9  = + autoregressive sub-actions
#   10 = + HL-Gauss (value classification, 255 bins)
#   11 = + GELU activations
#   12 = + adaptive opponents
#   13 = + prioritized sampling
#   14 = + PAE (Partial Advantage Estimator, keep=90%)
#   15 = + SPP critic (spatial pyramid pooling)
#   16 = + opponent modeling (predict enemy actions, focal loss)
#   17 = + PopArt (adaptive value normalization)
#   18 = + symmetry augmentation (random H/V flip)
#   19 = + build-time scaled rewards (Light=4, Ranged=5.25, Heavy=6)
#   20 = + hierarchical sub-action masking (RAISocketAI-style)
#   21 = + autoregressive + hierarchical mask (combined)
#
# Usage:
#   sbatch experiments/ablation/train_feat_ablation.slurm
#

SEED=${SEED:-1}

# ── Shared config ────────────────────────────────────────────────────────

ARCH="unet_entity_cbam_deep"
STEPS=50000000

COMMON_ARGS="
    --architecture $ARCH
    --arch-channels 48
    --total-timesteps $STEPS
    --num-bot-envs 24
    --num-selfplay-envs 0
    --map maps/open_competition/basesWorkers16x16A.xml
    --diverse-opponents True
    --num-steps 512
    --update-epochs 2
    --n-minibatch 3
    --gamma 0.99
    --gae-lambda 0.95
    --clip-coef 0.1
    --max-steps 4000
    --learning-rate 2.5e-4
    --anneal-lr True
    --ent-coef 0.01
    --ent-coef-end 0.001
    --vf-coef 0.5
    --alternate-players True
    --with-eval False
    --num-models 2
    --seed $SEED
"

# ── Per-feature config ───────────────────────────────────────────────────

case $SLURM_ARRAY_TASK_ID in
    0)
        NAME="baseline_s${SEED}"
        EXTRA=""
        ;;
    1)
        NAME="extended_obs_s${SEED}"
        EXTRA="--extended-obs True"
        ;;
    2)
        NAME="triple_heads_s${SEED}"
        EXTRA="--triple-value-heads True --gamma-sparse 0.999 --gae-lambda-sparse 0.99 --gamma-cost 0.999 --gae-lambda-cost 0.99 --vf-coef-cost 0.2 --advantage-weights 0.8 0.01 0.19"
        ;;
    3)
        NAME="filtmask_resobs_s${SEED}"
        EXTRA="--filtered-masks True --reserved-obs True"
        ;;
    4)
        NAME="bots_selfplay_pfsp_s${SEED}"
        EXTRA="--num-bot-envs 12 --num-selfplay-envs 12 --num-latest-selfplay-envs 6 --pfsp True"
        ;;
    5)
        NAME="aux_spatial_s${SEED}"
        EXTRA="--aux-spatial True"
        ;;
    6)
        NAME="aux_contrastive_s${SEED}"
        EXTRA="--aux-contrastive True"
        ;;
    7)
        NAME="aux_unit_count_s${SEED}"
        EXTRA="--aux-unit-count True"
        ;;
    8)
        NAME="framestack4_s${SEED}"
        EXTRA="--frame-stack 4"
        ;;
    9)
        NAME="autoregressive_s${SEED}"
        EXTRA="--autoregressive True"
        ;;
    10)
        NAME="hl_gauss_s${SEED}"
        EXTRA="--hl-gauss True"
        ;;
    11)
        NAME="gelu_s${SEED}"
        EXTRA="--gelu True"
        ;;
    12)
        NAME="adaptive_s${SEED}"
        EXTRA="--adaptive-opponents True --opponent-list RandomBiasedAI:24 --diverse-opponents False"
        ;;
    13)
        NAME="priori_samp_s${SEED}"
        EXTRA="--prioritized-sampling True"
        ;;
    14)
        NAME="pae_s${SEED}"
        EXTRA="--pae-keep 0.95"
        ;;
    15)
        NAME="spp_critic_s${SEED}"
        EXTRA="--spp-critic True"
        ;;
    16)
        NAME="opponent_modeling_s${SEED}"
        EXTRA="--aux-opponent-modeling True --aux-opponent-modeling-coef 0.01"
        ;;
    17)
        NAME="popart_s${SEED}"
        EXTRA="--popart True"
        ;;
    18)
        NAME="augment_symmetry_s${SEED}"
        EXTRA="--augment-symmetry True"
        ;;
    19)
        NAME="buildtime_rewards_s${SEED}"
        EXTRA="--reward-unit-build-time True"
        ;;
    20)
        NAME="hierarchical_mask_s${SEED}"
        EXTRA="--hierarchical-mask True"
        ;;
    21)
        NAME="autoregressive_hmask_s${SEED}"
        EXTRA="--autoregressive True --hierarchical-mask True"
        ;;
esac

echo "============================================================"
echo "  FEATURE ABLATION — 50M steps, basesWorkers16x16A"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Experiment:   $NAME"
echo "  Extra flags:  $EXTRA"
echo "  Architecture: $ARCH"
echo "  Node:         $SLURM_NODELIST"
echo "  Started:      $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

source "$(dirname "$0")/../_setup_env.sh"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Training command ───────────────────────────────────────────────────────

RESUME_FLAG=""
RUN_DIR="outputs/runs/${NAME}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

python -m microrts_agent train \
    $COMMON_ARGS \
    $EXTRA \
    --exp-name $NAME \
    $RESUME_FLAG

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Experiment: $NAME"
echo "  Exit code:  $EXIT_CODE"
echo "  Finished:   $(date)"
echo "============================================================"

exit $EXIT_CODE
