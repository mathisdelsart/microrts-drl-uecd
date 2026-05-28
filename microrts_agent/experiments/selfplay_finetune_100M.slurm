#!/bin/bash
#SBATCH --job-name=sp_ft100M
#SBATCH --output=slurm-sp-ft-%j.out
#SBATCH --error=slurm-sp-ft-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Self-play fine-tuning: load best_single_map_100M, continue 100M with self-play
# Phase 1 done (bots only, 100M) → Phase 2 (bots + self-play, 100M more)
#
SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Self-play fine-tune — 100M steps on top of best_100M"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/best_single_map_100M_s1/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 8 \
    --num-selfplay-envs 16 \
    --num-latest-selfplay-envs 8 \
    --pfsp True \
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
    --opponent-list "CoacAI:4,Mayari:4" \
    --alternate-players True \
    --prioritized-sampling True \
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
    --pae-keep 0.95 \
    --popart True \
    --gelu True \
    --frame-stack 4 \
    --autoregressive True \
    --spp-critic True \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name selfplay_finetune_100M_s${SEED}
