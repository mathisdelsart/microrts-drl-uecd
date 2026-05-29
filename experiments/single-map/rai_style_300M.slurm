#!/bin/bash
#SBATCH --job-name=rai300M
#SBATCH --output=slurm-rai-%j.out
#SBATCH --error=slurm-rai-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# RAISocketAI-style training — 300M steps, single map
#
# Replicates RAISocketAI's training schedule and stacks best-of-ablation features:
#   - Triple value heads (shaped, sparse γ=0.999, cost γ=0.999)
#   - 3-phase advantage weight schedule: shaped→mixed→sparse
#   - Self-play + PFSP + prioritized sampling
#   - S-tier features (extended obs, filtered masks, reserved obs, GELU)
#   - Autoregressive sub-actions + hierarchical mask (auto-enabled)
#   - PAE (keep 95%), PopArt, SPP critic
#   - Opponent modeling aux task
#
# RAI-matched durations ([0.3, 0.2, 0.1, 0.2, 0.2] of 300M):
#
# Phase 1 (0→90M):    adv=(0.8,0.01,0.19), vf=(0.5,0.1,0.2), LR=1e-4, ent=0.01
#   → shaped reward dominates, learn fundamentals
# Transition (90M→150M): linear blend
# Phase 2 (150M→180M): adv=(0.0,0.5,0.5), vf=(0.0,0.4,0.4), LR=1e-4, ent=0.01
#   → mixed sparse/cost, short bridge phase
# Transition (180M→240M): linear blend
# Phase 3 (240M→300M): adv=(0.0,0.99,0.01), vf=(0.0,0.5,0.1), LR=5e-5, ent=0.001
#   → pure sparse, polish for winning
#
SEED=${SEED:-1}

echo "============================================================"
echo "  RAISocketAI-style — 300M, triple heads, phased schedule"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1

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

RESUME_FLAG=""
RUN_DIR="outputs/runs/rai_style_v3_300M_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

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
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
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
    --triple-value-heads True \
    --gamma-sparse 0.999 \
    --gae-lambda-sparse 0.99 \
    --gamma-cost 0.99 \
    --gae-lambda-cost 0.99 \
    \
    --phase-steps 90000000 150000000 180000000 240000000 \
    --phase-adv-weights "0.8,0.01,0.19" "0.0,0.5,0.5" "0.0,0.99,0.01" \
    --phase-vf-coefs "0.5,0.1,0.2" "0.0,0.4,0.4" "0.0,0.5,0.1" \
    --phase-ent-coef 0.01 0.01 0.001 \
    --phase-lr 1e-4 1e-4 5e-5 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --gelu True \
    --reward-unit-build-time True \
    \
    --pae-keep 0.95 \
    --autoregressive True \
    --spp-critic True \
    --popart True \
    \
    --aux-opponent-modeling True \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name rai_style_v3_300M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
