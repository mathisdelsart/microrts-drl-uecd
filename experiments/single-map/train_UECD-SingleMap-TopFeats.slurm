#!/bin/bash
#SBATCH --job-name=stier100M
#SBATCH --output=slurm-stier-%j.out
#SBATCH --error=slurm-stier-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:TeslaA40:1
#SBATCH --partition=batch
#
# S-tier only agent: 100M steps, only the 4 robust features
#
SEED=${SEED:-1}

echo "============================================================"
echo "  S-tier Agent: 100M steps, 4 robust features only"
echo "  Seed: $SEED"
echo "  Started: $(date)"
echo "============================================================"

source "$(dirname "$0")/../_setup_env.sh"

microrts-agent train \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 24 \
    --diverse-opponents True \
    --alternate-players True \
    --prioritized-sampling True \
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
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name UECD-SingleMap-TopFeats_s${SEED}
