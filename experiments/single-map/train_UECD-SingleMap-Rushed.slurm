#!/bin/bash
#SBATCH --job-name=train_UECD-SingleMap-Rushed
#SBATCH --output=slurm-train_UECD-SingleMap-Rushed-%j.out
#SBATCH --error=slurm-train_UECD-SingleMap-Rushed-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Phased training: 300M steps, single map, RAISocketAI-inspired schedule
#
# Phase 1 (0-100M):   24 bots, shaped rewards dominant, high LR, high entropy
#                      → learn fundamentals (gather, build, rush defence)
# Transition 1 (100M-150M): linear blend to Phase 2
# Phase 2 (150M-250M): 12 bots + 12 self-play + PFSP, mixed rewards, normal LR
#                      → diversify strategies against self-play opponents
# Transition 2 (250M-280M): linear blend to Phase 3
# Phase 3 (280M-300M): 12 bots + 12 self-play, sparse dominant, low LR, low entropy
#                      → polish for win/loss only
#
# S-tier features only (robust across 3 seeds):
#   extended-obs, filtered-masks, reserved-obs, prioritized-sampling,
#   aux-opponent-modeling, gelu
#
# Usage:
#   sbatch experiments/single-map/train_UECD-SingleMap-Rushed.slurm
#   SEED=2 sbatch experiments/single-map/train_UECD-SingleMap-Rushed.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  Phased Training: 300M steps, single map"
echo "============================================================"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Seed:     $SEED"
echo "  Node:     $SLURM_NODELIST"
echo "  GPUs:     $CUDA_VISIBLE_DEVICES"
echo "  Started:  $(date)"
echo "============================================================"

source "$(dirname "$0")/../_setup_env.sh"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Resume logic ──────────────────────────────────────────────────────────

RESUME_FLAG=""
RUN_DIR="outputs/runs/UECD-SingleMap-Rushed_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

# ── Training command ───────────────────────────────────────────────────────

microrts-agent train \
    $RESUME_FLAG \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 300000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    \
    --num-bot-envs 12 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 6 \
    --pfsp True \
    --diverse-opponents True \
    --alternate-players True \
    --prioritized-sampling True \
    \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --vf-coef 0.5 \
    \
    --reward-schedule linear \
    --reward-schedule-start 0.3 \
    --reward-schedule-end 0.7 \
    \
    --phase-steps 100000000 150000000 250000000 280000000 \
    --phase-ent-coef 0.01 0.005 0.001 \
    --phase-lr 2.5e-4 2.5e-4 5e-5 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name UECD-SingleMap-Rushed_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
