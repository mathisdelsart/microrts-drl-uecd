#!/bin/bash
#SBATCH --job-name=mmap_sp200M
#SBATCH --output=slurm-mmap-sp-%j.out
#SBATCH --error=slurm-mmap-sp-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Multi-map + self-play on small maps (<=16x16) — 200M steps
# S-tier features + padded observations + PFSP self-play
#
SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Multi-map + Self-play (<=16x16) — 200M steps"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
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
    --exp-name multimap_selfplay_200M_s${SEED}
