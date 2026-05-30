#!/bin/bash
#SBATCH --job-name=mm_ft_v2
#SBATCH --output=slurm-mmftv2-%j.out
#SBATCH --error=slurm-mmftv2-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Multi-map fine-tuning V2 — resumes from UECD-MultiMap-Best @ ~325M
# Target: fix worker-rush mode collapse observed in V1.
#
# Diagnosis from V1 (320M eval, 0% vs RAISocketAI):
#   - ProduceBarracks=0.2 too low vs ProduceWorker=1.0 → no incentive
#   - Adaptive curriculum (threshold 0.7) kept agent on weak bots
#   - Result: worker-only policy, never produces light, barely heavy/ranged
#
# V2 changes (vs V1):
#   1. Reward weights aggressive start → standard end (linear blend):
#        Start: ProduceBarracks=5.0, ProduceLight=8.0  (force build chain)
#        End:   ProduceBarracks=1.5, ProduceLight=4.0  (sustainable)
#   2. Opponent pool: drop Random + RAI (saturated/hopeless), add TMA + ObiBot
#        16 bot envs: POWork:2, POLight:2, CoacAI:3, Mayari:3, TMA:3, ObiBot:3
#        16 self-play envs: 8 latest + 8 PFSP-hard pool
#   3. Adaptive opponents DISABLED (force exposure to hard bots)
#   4. ent-coef-end raised: 0.001 → 0.005 (preserve exploration)
#   5. Triple value heads kept (shaped + sparse + cost γ=0.999)
#
# Phase schedule (325M → 615M = 290M new steps):
#   Phase 1 (325→425M): adv=(0.3,0.5,0.2)  vf=(0.3,0.4,0.3)  lr=8e-5  ent=0.01
#   Trans  (425→525M):  linear blend (also reward weight transition)
#   Phase 2 (525→615M): adv=(0.05,0.7,0.25) vf=(0.05,0.5,0.2) lr=2e-5  ent=0.005
#
SEED=${SEED:-1}

echo "============================================================"
echo "  MM Fine-tuning V2 — seed $SEED | Started: $(date)"
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

OLD_RUN_DIR="outputs/runs/best_multimap_400M_s${SEED}"
NEW_RUN_NAME="best_multimap_finetune_v2_s${SEED}"
NEW_RUN_DIR="outputs/runs/${NEW_RUN_NAME}"

# If V2 already started (re-launch / 5d timeout retry), resume from V2.
# Otherwise: initial launch, resume weights+optim+step from V1.
if [ -f "${NEW_RUN_DIR}/checkpoint.pt" ]; then
    RESUME_FLAG="--resume ${NEW_RUN_DIR}"
    echo "  Re-launch: resuming from V2 dir = $NEW_RUN_DIR"
elif [ -f "${OLD_RUN_DIR}/checkpoint.pt" ]; then
    RESUME_FLAG="--resume ${OLD_RUN_DIR}"
    echo "  Initial launch: resuming V1 checkpoint = $OLD_RUN_DIR"
    echo "  Will write new logs/configs to = $NEW_RUN_DIR"
else
    echo "ERROR: no checkpoint.pt found in $OLD_RUN_DIR or $NEW_RUN_DIR"
    exit 1
fi

python microrts_agent/train.py \
    $RESUME_FLAG \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 615000000 \
    --max-steps 4000 \
    \
    --multi-map True \
    --max-map-size 16 \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/NoWhereToRun9x8.xml \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
    --map-switch-freq 50 \
    --plr True \
    --plr-window 50 \
    \
    --num-bot-envs 16 \
    --num-selfplay-envs 16 \
    --num-latest-selfplay-envs 8 \
    --opponent-list "WorkerRush:2,LightRush:2,CoacAI:3,Mayari:3,TMA:3,ObiBotKenobi:3" \
    --diverse-opponents False \
    --alternate-players True \
    --prioritized-sampling True \
    \
    --adaptive-opponents False \
    --adaptive-selfplay False \
    \
    --pfsp True \
    --pfsp-mode hard \
    --selfplay-pool-size 30 \
    --selfplay-save-interval 150 \
    --selfplay-swap-interval 25 \
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
    --gamma-cost 0.999 \
    --gae-lambda-cost 0.99 \
    \
    --phase-steps 425000000 525000000 \
    --phase-adv-weights "0.3,0.5,0.2" "0.05,0.7,0.25" \
    --phase-vf-coefs "0.3,0.4,0.3" "0.05,0.5,0.2" \
    --phase-ent-coef 0.01 0.005 \
    --phase-lr 8e-5 2e-5 \
    \
    --reward-schedule linear \
    --reward-schedule-start 0.69 \
    --reward-schedule-end 0.85 \
    --reward-weight     10.0 1.0 1.0 0.2 5.0 1.0 8.0 6.0 5.25 0.0 \
    --reward-weight-end 10.0 1.0 1.0 0.2 1.5 1.0 4.0 6.0 5.25 0.0 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --frame-stack 4 \
    --gelu True \
    \
    --autoregressive True \
    --spp-critic True \
    --popart True \
    --pae-keep 0.95 \
    \
    --aux-opponent-modeling True \
    \
    --with-eval True \
    --eval-bots RandomBiasedAI WorkerRush LightRush CoacAI Mayari ObiBotKenobi TMA RAISocketAI \
    --eval-games 10 \
    --num-models 24 \
    --seed $SEED \
    --exp-name ${NEW_RUN_NAME}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
