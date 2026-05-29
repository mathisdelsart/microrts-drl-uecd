#!/bin/bash
#SBATCH --job-name=BEST_MM_300M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BEST AGENT — 300M steps, MULTI-MAP (5 maps <= 16x16), all best features
#
# Architecture: UNet-Entity-CBAM-Deep (4.7M params, C=48)
# Map pool (5 maps, padded to 16x16):
#   - basesWorkers8x8A        (8x8)
#   - FourBasesWorkers8x8     (8x8)
#   - NoWhereToRun9x8         (9x8)
#   - basesWorkers16x16A      (16x16)
#   - TwoBasesBarracks16x16   (16x16)
#
# Multi-map features:
#   - PLR (prioritized level replay) for smart map sampling
#   - HL-Gauss + symlog for scale-invariant value estimation across maps
#
# Same 12 features as single-map BEST agent.
#
# Usage:
#   sbatch experiments/multi-map/BEST_agent_multi_map_300M.slurm
#   SEED=2 sbatch experiments/multi-map/BEST_agent_multi_map_300M.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BEST AGENT — 300M steps, MULTI-MAP (<= 16x16)"
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

# ── Training command ───────────────────────────────────────────────────────

python microrts_agent/train.py \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 300000000 \
    --max-steps 4000 \
    \
    --multi-map True \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/NoWhereToRun9x8.xml \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
    --max-map-size 16 \
    --map-switch-freq 50 \
    --plr True \
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
    --frame-stack 4 \
    --spp-critic True \
    --gelu True \
    --autoregressive True \
    --aux-contrastive True \
    --aux-spatial True \
    --adaptive-opponents True \
    --hl-gauss True \
    --aux-unit-count True \
    --pae-keep 0.95 \
    --augment-symmetry True \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name BEST_agent_multi_map_300M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
