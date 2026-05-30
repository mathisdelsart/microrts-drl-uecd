#!/bin/bash
#SBATCH --job-name=bc_ft_100M
#SBATCH --output=slurm-bc-ft-%j.out
#SBATCH --error=slurm-bc-ft-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:TeslaA40:1
#SBATCH --partition=batch
#
# BC-PPO: Load BC warmstart, fine-tune with RL for 100M steps
#
# Compare with from-scratch training to measure BC warmstart benefit.
# Same feature stack as the single-map UECD agents but 100M steps.
#
# Usage:
#   sbatch experiments/BC/train_UECD-BC-PPO.slurm
#   SEED=2 sbatch experiments/BC/train_UECD-BC-PPO.slurm
#

SEED=${SEED:-1}
BC_MODEL="outputs/runs/bc/UECD-BC/agent.pt"

echo "============================================================"
echo "  BC-PPO Fine-tuning — 100M steps, single map"
echo "============================================================"
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Seed:         $SEED"
echo "  BC model:     $BC_MODEL"
echo "  Node:         $SLURM_NODELIST"
echo "  GPUs:         $CUDA_VISIBLE_DEVICES"
echo "  Started:      $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

module load Python/3.11 2>/dev/null || module load Python/3.9 2>/dev/null || {
    echo "ERROR: No Python module found"; exit 1; }
module load Java/17 2>/dev/null || module load Java/11 2>/dev/null || {
    echo "ERROR: No Java module found"; exit 1; }
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1 2>/dev/null || {
    echo "WARNING: No CUDA module found"; }

VENV_DIR="$HOME/microrts-drl-uecd/cluster_venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: cluster_venv not found."
    exit 1
fi

export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Check BC model exists ─────────────────────────────────────────────────

if [ ! -f "$BC_MODEL" ]; then
    echo "ERROR: BC model not found: $BC_MODEL"
    echo "Run bc_pipeline.slurm first."
    exit 1
fi

# ── Resume logic ──────────────────────────────────────────────────────────

RESUME_FLAG=""
RUN_DIR="outputs/runs/bc_v3_finetune_100M_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

# ── Training command ───────────────────────────────────────────────────────

python microrts_agent/train.py \
    $RESUME_FLAG \
    --load-model "$BC_MODEL" \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 24 \
    --diverse-opponents True \
    --alternate-players True \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --learning-rate 2.5e-4 \
    --anneal-lr True \
    --lr-warmup-updates 100 \
    --ent-coef 0.01 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name bc_v3_finetune_100M_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
