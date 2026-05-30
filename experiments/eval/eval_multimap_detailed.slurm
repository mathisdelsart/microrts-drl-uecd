#!/bin/bash
#SBATCH --job-name=eval_mmap
#SBATCH --output=slurm-eval-mmap-%j.out
#SBATCH --error=slurm-eval-mmap-%j.err
#SBATCH --time=12:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Detailed evaluation of multimap agent: 10 games per position per bot per map
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

AGENT="outputs/runs/UECD-MultiMap"
EVAL_DIR="${AGENT}/EVAL_detailed"
mkdir -p "$EVAL_DIR"

MAPS=(
    "maps/open_competition/basesWorkers8x8A.xml"
    "maps/open_competition/FourBasesWorkers8x8.xml"
    "maps/open_competition/NoWhereToRun9x8.xml"
    "maps/open_competition/basesWorkers16x16A.xml"
    "maps/open_competition/TwoBasesBarracks16x16.xml"
)

BOTS=("RandomBiasedAI" "WorkerRush" "LightRush" "CoacAI" "Mayari")

for MAP in "${MAPS[@]}"; do
    MAP_NAME=$(basename "$MAP" .xml)
    for BOT in "${BOTS[@]}"; do
        OUTFILE="${EVAL_DIR}/${MAP_NAME}_vs_${BOT}.txt"
        if [ -f "$OUTFILE" ]; then
            echo "SKIP: ${MAP_NAME} vs ${BOT} — already done"
            continue
        fi
        echo "=== ${MAP_NAME} vs ${BOT} (20 games) ==="
        python microrts_agent/evaluate.py \
            --agent "$AGENT" \
            --opponent "$BOT" \
            --nb_games 10 \
            --maps "$MAP" \
            --max-steps 4000 \
            2>&1 | tee "$OUTFILE"
    done
done

echo "=== ALL DONE === $(date)"
