#!/bin/bash
#SBATCH --job-name=bc_kl100M
#SBATCH --output=slurm-bc-kl-%j.out
#SBATCH --error=slurm-bc-kl-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:TeslaA40:1
#SBATCH --partition=batch
#
# BC→PPO with KL penalty (AlphaStar-style)
# Loads BC+VF weights AND keeps a frozen teacher copy
# KL(teacher || agent) anneals from 1.0 to 0.0 over 50M steps
# Prevents the initial performance dip observed with naive BC→PPO
#
SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  BC+KL Fine-tuning — 100M steps"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

BC_MODEL="outputs/runs/bc_v3_s1/agent.pt"

python microrts_agent/train.py \
    --load-model "$BC_MODEL" \
    --bc-teacher-model "$BC_MODEL" \
    --bc-kl-coef 1.0 \
    --bc-kl-anneal-steps 50000000 \
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
    --with-eval True \
    --num-models 5 \
    --seed $SEED \
    --exp-name bc_kl_finetune_100M_s${SEED}
