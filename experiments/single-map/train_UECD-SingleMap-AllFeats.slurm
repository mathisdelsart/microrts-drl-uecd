#!/bin/bash
#SBATCH --job-name=best_100M
#SBATCH --output=slurm-best100M-%j.out
#SBATCH --error=slurm-best100M-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BEST agent — 100M steps, single map, all positive features from 3-seed ablation
#
# Architecture: UNet-Entity-CBAM-Deep (4.72M params)
# Features (from 3-seed ablation summary, sorted by mean delta):
#   S-tier: Extended Obs, Filt. Masks + Res. Obs, Prioritized Sampling, Opponent Modeling
#   A-tier: PAE, PopArt, GELU, Frame Stack 4
#   B-tier: Autoregressive, SPP Critic
#
# Usage:
#   sbatch experiments/single-map/train_UECD-SingleMap-AllFeats.slurm
#   SEED=2 sbatch experiments/single-map/train_UECD-SingleMap-AllFeats.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BEST Agent — 100M steps, all positive features"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Seed:         $SEED"
echo "  Node:         $SLURM_NODELIST"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
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

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Resume logic ──────────────────────────────────────────────────────────

RESUME_FLAG=""
RUN_DIR="outputs/runs/best_single_map_100M_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

# ── Training command ───────────────────────────────────────────────────────

python microrts_agent/train.py \
    $RESUME_FLAG \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    \
    --num-bot-envs 24 \
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
    --learning-rate 2.5e-4 \
    --anneal-lr True \
    --ent-coef 0.01 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --pae-keep 0.95 \
    --popart True \
    --gelu True \
    --frame-stack 4 \
    --autoregressive True \
    --spp-critic True \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name best_single_map_100M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
