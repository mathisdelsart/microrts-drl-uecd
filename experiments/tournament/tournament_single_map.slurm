#!/bin/bash
#SBATCH --job-name=tourney-single
#SBATCH --output=slurm-tourney-single-%A_%a.out
#SBATCH --error=slurm-tourney-single-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-7%8
#
# Round-robin tournament on basesWorkers16x16A: single-map headline of the
# dissertation. Reproduces data/tournaments/single_map/.
#
# 19 AIs (5 RL + 14 scripted) x 1 map x 5 iterations = 1710 games total,
# sharded into 8 chunks (~213 games per chunk). Chunks merge automatically
# on parse.
#
# Config: microrts_agent/tournament_configs/single_map.json
#
# Submit:
#   sbatch experiments/tournament/tournament_single_map.slurm
#
# Parse + visualise once all chunks are done:
#   microrts-agent tournament analyze single_map

source "$(dirname "$0")/../_setup_env.sh"

echo "============================================================"
echo "  Tournament: single_map (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "  19 AIs x 1 map x 5 iterations = 1710 games total"
echo "  ~213 games per chunk"
echo "  Started: $(date)"
echo "============================================================"

microrts-agent tournament run single_map \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks 8

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
