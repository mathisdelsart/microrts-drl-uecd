#!/bin/bash
#SBATCH --job-name=tournament_multi_map
#SBATCH --output=slurm-tournament_multi_map-%A_%a.out
#SBATCH --error=slurm-tournament_multi_map-%A_%a.err
#SBATCH --time=3-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-7%8
#
# Round-robin tournament on the 5 open-competition maps <= 16x16. Includes
# UECD-MultiMap and UECD-SingleMap-Best alongside every scripted bot.
# Reproduces data/tournaments/multi_map/.
#
# 16 AIs (2 RL + 14 scripted) x 5 maps x 5 iterations = 6000 games total,
# sharded into 8 chunks (~750 games per chunk). Chunks merge automatically
# on parse.
#
# Config: microrts_agent/tournament_configs/multi_map.json
#
# Submit:
#   sbatch experiments/tournament/tournament_multi_map.slurm
#
# Parse + visualise once all chunks are done:
#   microrts-agent tournament analyze multi_map

source "$(dirname "$0")/../_setup_env.sh"

echo "============================================================"
echo "  Tournament: multi_map (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "  16 AIs x 5 maps x 5 iterations = 6000 games total"
echo "  ~750 games per chunk"
echo "  Started: $(date)"
echo "============================================================"

microrts-agent tournament run multi_map \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks 8

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
