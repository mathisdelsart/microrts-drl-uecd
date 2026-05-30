#!/bin/bash
#SBATCH --job-name=eval_bestrl_vs_rai
#SBATCH --output=slurm-evalrai-%j.out
#SBATCH --error=slurm-evalrai-%j.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:TeslaA40:1
#SBATCH --partition=batch
#
# Evaluation: UECD-Best vs RAISocketAI on basesWorkers16x16A
# 500 games as P0 + 500 games as P1 = 1000 stochastic games total.
# Cluster: Hercules2 (A40).
#
# Usage:
#   sbatch experiments/eval/eval_bestrl_vs_rai_500g.slurm
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

AGENT="outputs/runs/single_map/UECD-SingleMap-Best"
OPPONENT="RAISocketAI"
MAP="maps/16x16/basesWorkers16x16A.xml"
NB_GAMES=500
MAX_STEPS=4000

EVAL_DIR="${AGENT}/EVAL_vs_RAI_basesWorkers16x16A_${NB_GAMES}g"
mkdir -p "$EVAL_DIR"
OUTFILE="${EVAL_DIR}/bestrl_vs_rai.txt"

echo "============================================================"
echo "  UECD-Best vs RAISocketAI"
echo "  Map:     $MAP"
echo "  Games:   ${NB_GAMES} P0 + ${NB_GAMES} P1 = $((NB_GAMES * 2)) stochastic"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
echo "  Started: $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

python microrts_agent/evaluate.py \
    --agent "$AGENT" \
    --opponent "$OPPONENT" \
    --nb_games $NB_GAMES \
    --maps "$MAP" \
    --max-steps $MAX_STEPS \
    2>&1 | tee "$OUTFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  Done. Exit code: $EXIT_CODE"
echo "  Results: $OUTFILE"
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
