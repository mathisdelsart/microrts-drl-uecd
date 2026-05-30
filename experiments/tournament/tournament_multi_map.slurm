#!/bin/bash
#SBATCH --job-name=tourney-multi
#SBATCH --output=slurm-tourney-multi-%A_%a.out
#SBATCH --error=slurm-tourney-multi-%A_%a.err
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
#   python -m microrts_agent tournament analyze multi_map

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Tournament: multi_map (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "  16 AIs x 5 maps x 5 iterations = 6000 games total"
echo "  ~750 games per chunk"
echo "  Started: $(date)"
echo "============================================================"

python -m microrts_agent tournament run multi_map \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks 8

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
