#!/bin/bash
#SBATCH --job-name=eval_final
#SBATCH --output=slurm-eval-final-%j.out
#SBATCH --error=slurm-eval-final-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --dependency=singleton
#
# Evaluate all final agents — 500 games per position per bot
# Run after training jobs complete.
# --dependency=singleton ensures only one eval job runs at a time.
#
# Usage:
#   sbatch experiments/eval/eval_final_agents.slurm
#

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

BOTS=("RandomBiasedAI" "POWorkerRush" "POLightRush" "CoacAI" "Mayari")

# List of agents to evaluate (add paths as runs finish)
AGENTS=(
    "outputs/runs/best_single_map_100M_s1"
    "outputs/runs/stier_single_map_100M_s1"
    "outputs/runs/stier_selfplay_100M_s1"
    "outputs/runs/phased_single_map_300M_s1"
    "outputs/runs/phased_single_map_300M_s2"
    "outputs/runs/top2_features_100M_s1"
)

for AGENT_DIR in "${AGENTS[@]}"; do
    AGENT_NAME=$(basename "$AGENT_DIR")

    # Skip if agent not finished (no agent.pt)
    if [ ! -f "${AGENT_DIR}/agent.pt" ]; then
        echo "SKIP: ${AGENT_NAME} — no agent.pt yet"
        continue
    fi

    EVAL_DIR="${AGENT_DIR}/EVAL"
    mkdir -p "$EVAL_DIR"

    # Skip if already evaluated
    if [ -f "${EVAL_DIR}/eval_done" ]; then
        echo "SKIP: ${AGENT_NAME} — already evaluated"
        continue
    fi

    echo "============================================================"
    echo "  Evaluating: ${AGENT_NAME}"
    echo "  $(date)"
    echo "============================================================"

    for BOT in "${BOTS[@]}"; do
        OUTFILE="${EVAL_DIR}/${AGENT_NAME}_vs_${BOT}.txt"
        if [ -f "$OUTFILE" ]; then
            echo "  SKIP: ${BOT} — already done"
            continue
        fi
        echo "  vs ${BOT} (500 games per position)..."
        python microrts_agent/evaluate.py \
            --agent "$AGENT_DIR" \
            --opponent "$BOT" \
            --nb_games 500 \
            --maps maps/16x16/basesWorkers16x16A.xml \
            --max-steps 4000 \
            2>&1 | tee "$OUTFILE"
    done

    touch "${EVAL_DIR}/eval_done"
    echo "  Done: ${AGENT_NAME}"
done

echo ""
echo "============================================================"
echo "  ALL EVALUATIONS COMPLETE — $(date)"
echo "============================================================"
