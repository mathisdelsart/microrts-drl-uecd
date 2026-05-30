#!/bin/bash
#SBATCH --job-name=mmtop4_bestrl_A
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# XP1a Phase A — Multi-map (top-4 maps <= 16x16), BestRL-style lean feature pack.
# 250M steps, 12 bot envs + 12 self-play envs (PFSP).
#
# Map pool (4 — NoWhereToRun9x8 excluded as 'the special one'):
#   - basesWorkers8x8A        (8x8)
#   - FourBasesWorkers8x8     (8x8)
#   - basesWorkers16x16A      (16x16)
#   - TwoBasesBarracks16x16   (16x16)
#
# Features mirror single-map UECD-Best config.json:
#   extended-obs, filtered-masks + reserved-obs, gelu, aux-opponent-modeling,
#   prioritized-sampling, pfsp, alternate-players, anneal-lr, reward annealing.
#
# Chained by:
#   sbatch --dependency=afterok:$JOB --export=ALL,VARIANT=BestRL \
#       experiments/multimap_top4_phaseB_150M.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  XP1a — Multi-map top-4 (<=16x16), BestRL features, 250M"
echo "  Job ID: $SLURM_JOB_ID | Seed: $SEED | Node: $SLURM_NODELIST"
echo "  Started: $(date)"
echo "============================================================"

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
    echo "ERROR: cluster_venv not found."; exit 1
fi

export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

python microrts_agent/train.py \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 250000000 \
    --max-steps 4000 \
    \
    --multi-map True \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
    --max-map-size 16 \
    --map-switch-freq 50 \
    \
    --num-bot-envs 12 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 6 \
    --pfsp True \
    --diverse-opponents True \
    --alternate-players True \
    --prioritized-sampling True \
    \
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
    \
    --reward-schedule linear \
    --reward-schedule-start 0.3 \
    --reward-schedule-end 0.7 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --gelu True \
    --aux-opponent-modeling True \
    --aux-opponent-modeling-coef 0.01 \
    \
    --with-eval True \
    --num-models 2 \
    --seed $SEED \
    --exp-name MultiMapTop4_BestRL_PhaseA_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE | Exit code: $EXIT_CODE | $(date)"
echo "============================================================"
exit $EXIT_CODE
