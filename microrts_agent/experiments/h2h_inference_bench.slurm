#!/bin/bash
#SBATCH --job-name=h2h-infer
#SBATCH --output=slurm-h2h-infer-%j.out
#SBATCH --error=slurm-h2h-infer-%j.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Head-to-head per-tick decision time benchmark:
# BestRL-350M (P0) vs RAISocketAI (P1), same game sequence.
# Gives fair apples-to-apples comparison (both sides see the same game states).
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

AGENT_PATH="outputs/runs/single_map/BestRL-350M"
MAP="maps/16x16/basesWorkers16x16A.xml"
GAMES=10
MAX_STEPS=4000
TIME_BUDGET=100
OUTPUT_DIR="outputs/inference_bench/h2h_bestrl_vs_rai_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "  H2H Inference Benchmark: BestRL-350M (P0) vs RAISocketAI (P1)"
echo "  Map:       $MAP"
echo "  Games:     $GAMES (same sequence for both sides)"
echo "  Max steps: $MAX_STEPS"
echo "  Output:    $OUTPUT_DIR"
echo "  Started:   $(date)"
echo "============================================================"

nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

python microrts_agent/h2h_inference_bench.py \
    --agent "$AGENT_PATH" \
    --opponent RAISocketAI \
    --games $GAMES \
    --map "$MAP" \
    --max-steps $MAX_STEPS \
    --time-budget $TIME_BUDGET \
    --output "$OUTPUT_DIR"

echo ""
echo "============================================================"
echo "  Done: $(date)"
echo "  Results in: $OUTPUT_DIR"
echo "============================================================"
