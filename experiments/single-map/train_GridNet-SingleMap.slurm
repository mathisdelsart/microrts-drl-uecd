#!/bin/bash
#SBATCH --job-name=gridnet_baseline_300M
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=48:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Baseline — GridNet encoder-decoder (0.84M params) — 300M steps
#
# Matches the best GridNet config from Huang et al. 2021:
#   - 24 envs: 18 CoacAI + 2 RandomBiased + 2 WorkerRush + 2 LightRush
#   - 512 steps/rollout, action masking, encoder-decoder architecture
#
# Usage:
#   cd ~/microrts-drl-uecd
#   sbatch experiments/single-map/train_GridNet-SingleMap.slurm
#
#   # With a specific seed:
#   SEED=42 sbatch experiments/single-map/train_GridNet-SingleMap.slurm
#
#   # Resume from checkpoint (the run dir name on disk is unchanged):
#   RESUME=outputs/runs/gridnet_baseline_300M sbatch experiments/single-map/train_GridNet-SingleMap.slurm

echo "=== Job Info ==="
echo "Job ID:      $SLURM_JOB_ID"
echo "Node:        $SLURM_NODELIST"
echo "GPUs:        $CUDA_VISIBLE_DEVICES"
echo "CPUs:        $SLURM_CPUS_PER_TASK"
echo "Seed:        ${SEED:=1}"
echo "Resume:      ${RESUME:-none}"
echo "Started:     $(date)"
echo "================"

# Load modules (fail early if critical ones missing)
source "$(dirname "$0")/../_setup_env.sh"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# Build training command
CMD="python -m microrts_agent train \
    --architecture gridnet \
    --baseline \
    --total-timesteps 300000000 \
    --num-bot-envs 24 \
    --num-steps 512 \
    --update-epochs 4 \
    --n-minibatch 4 \
    --diverse-opponents True \
    --alternate-players False \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --learning-rate 2.5e-4 \
    --anneal-lr True \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --ent-coef 0.01 \
    --vf-coef 0.5 \
    --max-grad-norm 0.5 \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name gridnet_baseline_300M_s${SEED}"

# Resume from previous run if provided
if [ -n "${RESUME:-}" ]; then
    CMD="$CMD --resume $RESUME"
fi

echo ""
echo "Running: $CMD"
echo ""

eval $CMD

echo ""
echo "=== Job Complete ==="
echo "Finished: $(date)"
