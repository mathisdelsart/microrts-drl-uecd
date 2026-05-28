#!/bin/bash
#SBATCH --job-name=tourney-16x16
#SBATCH --output=slurm-tourney-16x16-%A_%a.out
#SBATCH --error=slurm-tourney-16x16-%A_%a.err
#SBATCH --time=1-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=gpu
#SBATCH --array=0-7%8

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Tournament: final_single_map_16x16 (chunk ${SLURM_ARRAY_TASK_ID}/8)"
echo "  19 AIs x 1 map x 5 iterations = 1710 games total"
echo "  ~213 games per chunk"
echo "  Started: $(date)"
echo "============================================================"

python microrts_agent/run_tournament.py run final_single_map_16x16 \
    --chunk $SLURM_ARRAY_TASK_ID \
    --total-chunks 8

echo "=== Chunk ${SLURM_ARRAY_TASK_ID} done: $(date) ==="
