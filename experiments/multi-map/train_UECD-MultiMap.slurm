#!/bin/bash
#SBATCH --job-name=train_UECD-MultiMap
#SBATCH --output=slurm-train_UECD-MultiMap-%j.out
#SBATCH --error=slurm-train_UECD-MultiMap-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Multi-map training on small maps (<=16x16): 200M steps
# S-tier features + padded observations
#
SEED=${SEED:-1}

source "$(dirname "$0")/../_setup_env.sh"

echo "============================================================"
echo "  Multi-map small (<=16x16): 100M steps"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

microrts-agent train \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 200000000 \
    --multi-map True \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/NoWhereToRun9x8.xml \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
    --max-map-size 16 \
    --map-switch-freq 50 \
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
    --pae-keep 0.95 \
    --popart True \
    --gelu True \
    --frame-stack 4 \
    --autoregressive True \
    --spp-critic True \
    --plr True \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name UECD-MultiMap_s${SEED}
