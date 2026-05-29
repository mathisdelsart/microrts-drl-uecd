#!/bin/bash
#SBATCH --job-name=BEST_300M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BEST AGENT — 300M steps, single-map, robust features only (3-seed ablation)
#
# Architecture: UNet-Entity-CBAM-Deep (4.7M params)
# Features enabled (from 3-seed feat ablation, mean ± std):
#   S-tier (robust, all seeds positive):
#     - Extended Obs 73ch           (84.3 ± 1.7, +26.3%)
#     - Filtered Masks + Res Obs    (80.6 ± 5.6, +22.6%)
#     - Prioritized Sampling        (78.4 ± 0.4, +20.4%)
#     - Opponent Modeling            (77.8 ± 4.0, +19.8%)
#   A-tier (positive mean, moderate variance):
#     - PAE keep=95%                (71.3 ± 8.1, +13.3%)
#     - GELU                        (66.3 ± 8.2, +8.3%)
#     - Frame Stack 4               (65.3 ± 4.5, +7.3%)
#   Self-play (positive mean, high variance):
#     - Bots + Self-Play + PFSP     (66.9 ± 9.1, +8.9%)
#   REMOVED (negative or unstable cross-seed):
#     - HL-Gauss, Adaptive Opponents, Augment Symmetry, Triple Value Heads
#     - Aux Contrastive, Aux Spatial, SPP Critic, Autoregressive
#
# Usage:
#   sbatch experiments/BEST_agent_300M.slurm
#   SEED=2 sbatch experiments/BEST_agent_300M.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BEST AGENT — 300M steps, all best features"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Seed:         $SEED"
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

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Resume logic ──────────────────────────────────────────────────────────

RESUME_FLAG=""
RUN_DIR="outputs/runs/BEST_agent_300M_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

# ── Training command ───────────────────────────────────────────────────────

python microrts_agent/train.py \
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
    --gelu True \
    --frame-stack 4 \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name BEST_agent_300M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
