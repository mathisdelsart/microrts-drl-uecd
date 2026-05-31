#!/bin/bash
#SBATCH --job-name=train_UECD-SingleMap-Best_phase1
#SBATCH --output=slurm-train_UECD-SingleMap-Best_phase1-%j.out
#SBATCH --error=slurm-train_UECD-SingleMap-Best_phase1-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Fine-tune from the UECD-SingleMap-Rushed checkpoint at 150M.
# NO reward annealing (keep shaped rewards).
# More self-play for diversity.
# Low LR to refine, not destroy.
#
SEED=${SEED:-1}

source "$(dirname "$0")/../_setup_env.sh"

echo "============================================================"
echo "  Fine-tune from 150M: no reward annealing, self-play"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

microrts-agent train \
    --resume outputs/runs/UECD-SingleMap-Best_phase1_s1 \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 300000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 8 \
    --num-selfplay-envs 16 \
    --num-latest-selfplay-envs 8 \
    --pfsp True \
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
    --opponent-list "CoacAI:2,Mayari:2,ObiBotKenobi:2,TMA:2" \
    --alternate-players True \
    --prioritized-sampling True \
    --reward-schedule linear \
    --reward-schedule-start 0.5 \
    --reward-schedule-end 0.9 \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --learning-rate 1e-4 \
    --anneal-lr True \
    --ent-coef 0.005 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --num-models 5 \
    --seed $SEED \
    --exp-name UECD-SingleMap-Best_phase1_s${SEED}
