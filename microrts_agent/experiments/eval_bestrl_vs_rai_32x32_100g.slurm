#!/bin/bash
#SBATCH --job-name=eval_bestrl_vs_rai_32x32_record
#SBATCH --output=slurm-evalrai32rec-%j.out
#SBATCH --error=slurm-evalrai32rec-%j.err
#SBATCH --time=02:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:TeslaA40:1
#SBATCH --partition=batch
#
# Evaluation: BestRL-350M vs RAISocketAI on basesWorkers32x32A
# 1 game as P0 + 1 game as P1 = 2 stochastic games, with video recording.
# Cluster: Hercules2 (A40).
#
# Usage:
#   sbatch microrts_agent/experiments/eval_bestrl_vs_rai_32x32_100g.slurm
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
module load Xvfb/21.1.8-GCCcore-12.3.0 2>/dev/null || module load Xvfb/21.1.6-GCCcore-12.2.0 2>/dev/null || module load Xvfb
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

AGENT="outputs/runs/single_map/BestRL-350M"
OPPONENT="RAISocketAI"
MAP="maps/32x32/basesWorkers32x32A.xml"
NB_GAMES=1
MAX_STEPS=4000

EVAL_DIR="${AGENT}/EVAL_vs_RAI_basesWorkers32x32A_record"
mkdir -p "$EVAL_DIR"
OUTFILE="${EVAL_DIR}/bestrl_vs_rai.txt"

echo "============================================================"
echo "  BestRL-350M vs RAISocketAI (RECORD)"
echo "  Map:     $MAP"
echo "  Games:   ${NB_GAMES} P0 + ${NB_GAMES} P1 = $((NB_GAMES * 2)) stochastic"
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Node:    $SLURM_NODELIST"
echo "  GPUs:    $CUDA_VISIBLE_DEVICES"
echo "  Started: $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Headless rendering: clear any inherited X11 forwarding and use a virtual framebuffer
unset DISPLAY

xvfb-run -a python microrts_agent/evaluate.py \
    --agent "$AGENT" \
    --opponent "$OPPONENT" \
    --nb_games $NB_GAMES \
    --maps "$MAP" \
    --max-steps $MAX_STEPS \
    --record \
    2>&1 | tee "$OUTFILE"

EXIT_CODE=${PIPESTATUS[0]}

echo ""
echo "============================================================"
echo "  Done. Exit code: $EXIT_CODE"
echo "  Results: $OUTFILE"
echo "  Finished: $(date)"
echo "============================================================"

exit $EXIT_CODE
