#!/bin/bash
#SBATCH --job-name=BEST_PH_MM
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BEST AGENT (PHASED) — 300M steps, MULTI-MAP, RAISocketAI-style 3-phase schedule
#
# Same features as BEST_agent_multi_map_300M.slurm, plus:
#   - Triple value heads (shaped + sparse/WL + cost)
#   - 3-phase schedule (RAISocketAI-style):
#       Phase 1 (0-90M):   shaped-dominant, high LR, high entropy
#       Phase 2 (90M-150M): transition
#       Phase 3 (150M-300M): sparse-dominant, low LR, low entropy
#   - PopArt value normalization (stabilizes critic during reward shift)
#   - Reward schedule: shaped -> pure WinLoss (30%-70% of training)
#
# Usage:
#   sbatch microrts_agent/experiments/BEST_agent_multi_map_300M_phased.slurm
#   SEED=2 sbatch microrts_agent/experiments/BEST_agent_multi_map_300M_phased.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BEST AGENT (PHASED) — 300M steps, MULTI-MAP"
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
#
# 3-phase schedule (RAISocketAI-style):
#   Phase 1 (0 - 90M):     shaped-dominant  adv=[0.8, 0.01, 0.19]  lr=2.5e-4  ent=0.01
#   Transition (90M-150M):  linear interpolation
#   Phase 2 (150M - 180M):  balanced         adv=[0.0, 0.5, 0.5]   lr=2.5e-4  ent=0.01
#   Transition (180M-240M): linear interpolation
#   Phase 3 (240M - 300M):  sparse-dominant  adv=[0.0, 0.99, 0.01] lr=5e-5    ent=0.001
#

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
    --gamma-sparse 0.999 \
    --gamma-cost 0.999 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --vf-coef 0.5 \
    \
    --triple-value-heads True \
    --popart True \
    --reward-schedule linear \
    --reward-schedule-start 0.3 \
    --reward-schedule-end 0.7 \
    \
    --phase-steps 90000000 150000000 180000000 240000000 \
    --phase-adv-weights "0.8,0.01,0.19" "0.0,0.5,0.5" "0.0,0.99,0.01" \
    --phase-vf-coefs "0.5,0.1,0.002" "0.0,0.4,0.4" "0.0,0.5,0.1" \
    --phase-ent-coef 0.01 0.01 0.001 \
    --phase-lr 2.5e-4 2.5e-4 5e-5 \
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
    --aux-unit-count True \
    --adaptive-opponents True \
    \
    --pae-keep 0.95 \
    --augment-symmetry True \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name BEST_agent_phased_MM_300M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
