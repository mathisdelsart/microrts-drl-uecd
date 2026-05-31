#!/bin/bash
#SBATCH --job-name=arch_ablation_100M
#SBATCH --output=slurm-%j_%a.out
#SBATCH --error=slurm-%j_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-6
#
# ARCHITECTURE ABLATION: 100M steps each: basesWorkers16x16A
#
#   Array job: launches 7 architectures in parallel (1 GPU each).
#   0=gridnet, 1=impala, 2=impala_entity, 3=unet,
#   4=unet_entity, 5=unet_entity_cbam, 6=unet_entity_cbam_deep
#   Baseline hyperparams only (no advanced features).
#
# Usage:
#   sbatch experiments/ablation/train_arch_ablation.slurm
#   SEED=2 sbatch experiments/ablation/train_arch_ablation.slurm
#

SEED=${SEED:-1}
ARCHS=("gridnet" "impala" "impala_entity" "unet" "unet_entity" "unet_entity_cbam" "unet_entity_cbam_deep")
CURRENT_ARCH=${ARCHS[$SLURM_ARRAY_TASK_ID]}
EXP_NAME="arch_ablation_${CURRENT_ARCH}_s${SEED}"

echo "============================================================"
echo "  ARCHITECTURE ABLATION: 100M steps, basesWorkers16x16A"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID (array task $SLURM_ARRAY_TASK_ID)"
echo "  Architecture: $CURRENT_ARCH"
echo "  Exp name:     $EXP_NAME"
echo "  Seed:         $SEED"
echo "  Node:         $SLURM_NODELIST"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Started:      $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

source "$(dirname "$0")/../_setup_env.sh"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Training command ───────────────────────────────────────────────────────

microrts-agent train \
    --architecture $CURRENT_ARCH \
    --total-timesteps 100000000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --diverse-opponents True \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-steps 4000 \
    --learning-rate 2.5e-4 \
    --anneal-lr True \
    --ent-coef 0.01 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --alternate-players True \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name $EXP_NAME

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Architecture: $CURRENT_ARCH"
echo "  Exp name:     $EXP_NAME"
echo "  Exit code:    $EXIT_CODE"
echo "  Finished:     $(date)"
echo "============================================================"

exit $EXIT_CODE
