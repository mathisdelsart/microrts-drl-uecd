#!/bin/bash
#SBATCH --job-name=mmtop4_phaseB
#SBATCH --output=slurm-%j.out
#SBATCH --error=slurm-%j.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# XP1 Phase B — Hardening 150M steps against strong bots only.
# Resumes a Phase A checkpoint (BestRL or AllFeats variant); feature flags
# are reloaded from the saved config — only training-side knobs change here.
#
# VARIANT must be exported when sbatch-ing:
#   sbatch --dependency=afterok:<JOB_A> --export=ALL,VARIANT=BestRL  <this>.slurm
#   sbatch --dependency=afterok:<JOB_A> --export=ALL,VARIANT=AllFeats <this>.slurm
#
# Phase B changes vs Phase A:
#   - Restrict opponent pool to strong bots only (CoacAI, Mayari, ObiBot,
#     TMA, RAISocketAI). Drops weak rushes (LightRush, WorkerRush, etc.).
#   - Drop reward annealing (250M was the slope; B holds the line).
#   - Lower LR (2.5e-4 -> 1e-4 -> 5e-5) and entropy (0.005 -> 0.001).
#   - Cumulative --total-timesteps 400M (250M Phase A + 150M Phase B).
#

SEED=${SEED:-1}
VARIANT=${VARIANT:?ERROR: VARIANT must be set to BestRL or AllFeats}

RUN_DIR="outputs/runs/MultiMapTop4_${VARIANT}_PhaseA_s${SEED}"
if [ ! -d "$RUN_DIR" ]; then
    echo "ERROR: Phase A run dir not found: $RUN_DIR"
    exit 1
fi
if [ ! -f "$RUN_DIR/agent.pt" ] && [ ! -f "$RUN_DIR/checkpoint.pt" ]; then
    echo "ERROR: No checkpoint to resume from in $RUN_DIR"
    exit 1
fi

echo "============================================================"
echo "  XP1 Phase B — Hardening 150M, variant=$VARIANT"
echo "  Job ID: $SLURM_JOB_ID | Seed: $SEED | Node: $SLURM_NODELIST"
echo "  Resuming: $RUN_DIR"
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
    --resume $RUN_DIR \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 400000000 \
    --max-steps 4000 \
    \
    --num-bot-envs 12 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 6 \
    --pfsp True \
    --opponent-list "CoacAI:2,Mayari:2,ObiBotKenobi:2,TMA:2,RAISocketAI:2" \
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
    --learning-rate 1e-4 \
    --learning-rate-end 5e-5 \
    --anneal-lr True \
    --ent-coef 0.005 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    \
    --with-eval True \
    --num-models 4 \
    --seed $SEED \
    --exp-name MultiMapTop4_${VARIANT}_PhaseB_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE | Variant: $VARIANT | Exit code: $EXIT_CODE | $(date)"
echo "============================================================"
exit $EXIT_CODE
