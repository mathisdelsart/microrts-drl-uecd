#!/bin/bash
#SBATCH --job-name=eval_best
#SBATCH --output=slurm-eval-best-%j.out
#SBATCH --error=slurm-eval-best-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-4
#
# Evaluate best_single_map_100M_s1 — 500 games per position per bot
# Array: 0=Random, 1=POWorker, 2=POLight, 3=CoacAI, 4=Mayari
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

AGENT="outputs/runs/best_single_map_100M_s1"
BOTS=("RandomBiasedAI" "POWorkerRush" "POLightRush" "CoacAI" "Mayari")
BOT=${BOTS[$SLURM_ARRAY_TASK_ID]}

EVAL_DIR="${AGENT}/EVAL"
mkdir -p "$EVAL_DIR"
OUTFILE="${EVAL_DIR}/best_vs_${BOT}.txt"

echo "============================================================"
echo "  Eval: best_single_map_100M_s1 vs $BOT (1000 games)"
echo "  $(date)"
echo "============================================================"

python microrts_agent/evaluate.py \
    --agent "$AGENT" \
    --opponent "$BOT" \
    --nb_games 500 \
    --maps maps/16x16/basesWorkers16x16A.xml \
    --max-steps 4000 \
    2>&1 | tee "$OUTFILE"

echo "Done: $OUTFILE"
