#!/bin/bash
#SBATCH --job-name=ft_rai160M
#SBATCH --output=slurm-ft-rai-%j.out
#SBATCH --error=slurm-ft-rai-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Fine-tune from 240M checkpoint with RAISocketAI + stronger pool
# Resume from 240M → 400M total. Reward: 100% shaped [240M-260M], anneal to 10% [260M-360M], 10% constant [360M-400M]
# Total: 160M steps, 16 checkpoints (every 10M)
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

RUN_DIR="outputs/runs/finetune_raisocket_160M_s1"
CKPT_SRC="outputs/runs/FineTuneStronger-300M/239972352.pt"

# Set up new run dir with the 240M checkpoint as checkpoint.pt for --resume
mkdir -p "$RUN_DIR"
if [ ! -f "${RUN_DIR}/checkpoint.pt" ]; then
    cp "$CKPT_SRC" "${RUN_DIR}/checkpoint.pt"
    echo "Copied 240M checkpoint to ${RUN_DIR}/checkpoint.pt"
fi

echo "============================================================"
echo "  Fine-tune with RAISocketAI — resume from 240M → 400M"
echo "  Started: $(date)"
echo "============================================================"

python -m microrts_agent train \
    --resume "$RUN_DIR" \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 400000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --opponent-list "RAISocketAI:4,CoacAI:1,Mayari:1,ObiBotKenobi:1,TMA:1" \
    --num-bot-envs 8 \
    --num-selfplay-envs 16 \
    --num-latest-selfplay-envs 8 \
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
    --learning-rate 2e-5 \
    --learning-rate-end 5e-6 \
    --anneal-lr True \
    --ent-coef 0.002 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --reward-schedule linear \
    --reward-schedule-start 0.65 \
    --reward-schedule-end 0.90 \
    --reward-schedule-min 0.1 \
    --with-eval True \
    --eval-bots RAISocketAI CoacAI Mayari ObiBotKenobi TMA \
    --num-models 16 \
    --seed 1 \
    --exp-name finetune_raisocket_160M_s1
