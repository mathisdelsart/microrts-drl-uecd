#!/bin/bash
#SBATCH --job-name=eval_rush
#SBATCH --output=slurm-eval-rush-%A_%a.out
#SBATCH --error=slurm-eval-rush-%A_%a.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-7
#
# Evaluate phased_single_map_300M_s1 at 150M (balanced) and 300M (rush)
# against TMA, ObiBotKenobi, CoacAI, Mayari — 500 games per position.
#
# Array layout (8 jobs):
#   0-3: 150M checkpoint vs [TMA, ObiBotKenobi, CoacAI, Mayari]
#   4-7: 300M checkpoint vs [TMA, ObiBotKenobi, CoacAI, Mayari]
#
# Usage:
#   sbatch experiments/eval/eval_rush_collapse.slurm
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

RUN_DIR="outputs/runs/phased_single_map_300M_s1"
BOTS=("TMA" "ObiBotKenobi" "CoacAI" "Mayari")
NB_GAMES=500
MAP="maps/16x16/basesWorkers16x16A.xml"
MAX_STEPS=4000

# Determine checkpoint and bot from array index
BOT_IDX=$(( SLURM_ARRAY_TASK_ID % 4 ))
CKPT_IDX=$(( SLURM_ARRAY_TASK_ID / 4 ))
BOT=${BOTS[$BOT_IDX]}

CKPTS=("150011904.pt" "299999232.pt")
LABELS=("150M" "300M")

CKPT_NAME=${CKPTS[$CKPT_IDX]}
LABEL=${LABELS[$CKPT_IDX]}

EVAL_DIR="${RUN_DIR}/EVAL_${LABEL}"
mkdir -p "$EVAL_DIR"

# Create a temporary run dir with the chosen checkpoint
# load_agent_from_config auto-detects full checkpoints (model key)
TMPDIR=$(mktemp -d)
cp "${RUN_DIR}/config.json" "$TMPDIR/"
cp "${RUN_DIR}/${CKPT_NAME}" "$TMPDIR/agent.pt"

OUTFILE="${EVAL_DIR}/phased_${LABEL}_vs_${BOT}.txt"

echo "============================================================"
echo "  Eval: ${LABEL} checkpoint vs ${BOT} (${NB_GAMES} games/position)"
echo "  Array task: ${SLURM_ARRAY_TASK_ID} | $(date)"
echo "============================================================"

python microrts_agent/evaluate.py \
    --agent "$TMPDIR" \
    --opponent "$BOT" \
    --nb_games "$NB_GAMES" \
    --maps "$MAP" \
    --max-steps "$MAX_STEPS" \
    2>&1 | tee "$OUTFILE"

rm -rf "$TMPDIR"
echo "Done: $OUTFILE"
