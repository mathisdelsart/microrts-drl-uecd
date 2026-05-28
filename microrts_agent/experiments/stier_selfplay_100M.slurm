#!/bin/bash
#SBATCH --job-name=stier_sp
#SBATCH --output=slurm-stier-sp-%j.out
#SBATCH --error=slurm-stier-sp-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# S-tier features + self-play + PFSP — 100M steps, single map
# 12 bot envs + 12 self-play envs (6 latest + 6 pool via PFSP)
#
SEED=${SEED:-1}

echo "============================================================"
echo "  S-tier + Self-Play — 100M steps"
echo "  Seed: $SEED"
echo "  Started: $(date)"
echo "============================================================"

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python microrts_agent/train.py \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 12 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 6 \
    --pfsp True \
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
    --exp-name stier_selfplay_100M_s${SEED}
