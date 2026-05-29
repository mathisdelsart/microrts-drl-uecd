#!/bin/bash
#SBATCH --job-name=tourney-5maps-mm
#SBATCH --output=slurm-tourney-5maps-mm-%A_%a.out
#SBATCH --error=slurm-tourney-5maps-mm-%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-15%16
#
# Round-robin tournament on the 5 open-competition maps <= 16x16,
# INCLUDING multimap_small_200M_s1 (the 78.6% multi-map agent) on top of
# every other RL agent + scripted bot we have. 20 AIs total.
#
# 20 AIs × 19 others / 2 = 190 pairs × 5 maps × 2 iterations × 2 (P0/P1)
#   ≈ 3800 games total, ~475 games per chunk.
#
# Sharded into 8 chunks to fit walltime; chunks merge automatically on parse.
#
# Submit:
#   sbatch experiments/tournament/tournament_5maps_with_multimap.slurm
#
# Parse + viz once all chunks done:
#   python microrts_agent/run_tournament.py analyze final_open_5maps_with_multimap

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Tournament: final_open_5maps_with_multimap (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "  20 AIs × 5 maps × 2 iterations ≈ 3800 games total"
echo "  ~475 games per chunk"
echo "  Started: $(date)"
echo "============================================================"

python microrts_agent/run_tournament.py run final_open_5maps_with_multimap \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks 8

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
