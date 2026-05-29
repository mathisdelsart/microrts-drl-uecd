#!/bin/bash
#SBATCH --job-name=tourney
#SBATCH --output=slurm-tourney-%j.out
#SBATCH --error=slurm-tourney-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-4

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "=== Tournament: single_map_tournament (chunk ${SLURM_ARRAY_TASK_ID}/5) ==="
echo "Started: $(date)"

python microrts_agent/run_tournament.py run single_map_tournament --chunk $SLURM_ARRAY_TASK_ID --total-chunks 5

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
