#!/bin/bash
#SBATCH --job-name=ft_map
#SBATCH --output=slurm-ft-map-%j_%a.out
#SBATCH --error=slurm-ft-map-%j_%a.err
#SBATCH --time=2-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-1
#
# Map-specific fine-tuning from multimap base model
#   0 = basesWorkers8x8A (50M steps)
#   1 = NoWhereToRun9x8 (50M steps)
#
# Usage:
#   sbatch experiments/multi-map/finetune_map_specific.slurm
#   sbatch --array=0 experiments/multi-map/finetune_map_specific.slurm  # 8x8 only
#

SEED=${SEED:-1}
BASE_MODEL="outputs/runs/multimap_small_200M_s1/agent.pt"

MAPS=("maps/open_competition/basesWorkers8x8A.xml" "maps/open_competition/NoWhereToRun9x8.xml")
MAP_NAMES=("8x8A" "NoWhereToRun9x8")
MAX_STEPS=(3000 4000)

MAP=${MAPS[$SLURM_ARRAY_TASK_ID]}
MAP_NAME=${MAP_NAMES[$SLURM_ARRAY_TASK_ID]}
MSTEPS=${MAX_STEPS[$SLURM_ARRAY_TASK_ID]}

echo "============================================================"
echo "  Map-specific fine-tune: ${MAP_NAME}"
echo "  Base model: ${BASE_MODEL}"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python microrts_agent/train.py \
    --load-model "$BASE_MODEL" \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 50000000 \
    --map "$MAP" \
    --max-steps $MSTEPS \
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
    --exp-name finetune_${MAP_NAME}_s${SEED}
