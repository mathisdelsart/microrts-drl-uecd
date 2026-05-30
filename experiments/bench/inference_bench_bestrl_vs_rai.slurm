#!/bin/bash
#SBATCH --job-name=infer-bench
#SBATCH --output=slurm-infer-bench-%j.out
#SBATCH --error=slurm-infer-bench-%j.err
#SBATCH --time=06:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#
# Per-tick decision time benchmark: UECD-Best vs RAISocketAI
# Runs self-play for each agent on basesWorkers16x16A, on one GPU.
# Produces CSV + PDF (boxplot + CDF).
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

# ── CONFIG ────────────────────────────────────────────────────────
# EDIT THIS: path to your UECD-Best agent (relative to microrts-drl-uecd root)
AGENT_PATH="outputs/runs/single_map/UECD-Best"

MAP="maps/16x16/basesWorkers16x16A.xml"
GAMES=5            # games per agent (self-play), ~4000 ticks/game = ~20k ticks total each
MAX_STEPS=3000
TIME_BUDGET=100    # ms — standard competition budget for Java bots
OUTPUT_DIR="outputs/inference_bench/bestrl_vs_rai_$(date +%Y%m%d_%H%M%S)"

echo "============================================================"
echo "  Inference Time Benchmark: UECD-Best vs RAISocketAI"
echo "  Map:       $MAP"
echo "  Games:     $GAMES per agent"
echo "  Max steps: $MAX_STEPS"
echo "  Output:    $OUTPUT_DIR"
echo "  Started:   $(date)"
echo "============================================================"

# GPU info
echo ""
echo "GPU:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo ""

# Benchmark — both agents in one call so CSV / PDF are produced together
python -m microrts_agent.bench inference \
    --agents "$AGENT_PATH" RAISocketAI \
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
