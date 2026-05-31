#!/bin/bash
#SBATCH --job-name=eval_generalization_probes
#SBATCH --output=slurm-eval_generalization_probes-%A_%a.out
#SBATCH --error=slurm-eval_generalization_probes-%A_%a.err
#SBATCH --time=04:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#SBATCH --array=0-7
#
# Generalisation probe: UECD-SingleMap-Best on TwoBasesBarracks16x16 (held-out 16x16 map)
# vs 8 opponents, ordered weakest -> strongest.
# 50 P0 + 50 P1 = 100 stochastic games per bot.
# Array: 0=RandomBiasedAI, 1=WorkerRush, 2=LightRush, 3=CoacAI,
#        4=Mayari, 5=ObiBotKenobi, 6=TMA, 7=RAISocketAI
#

source "$(dirname "$0")/../_setup_env.sh"

AGENT="outputs/runs/single_map/UECD-SingleMap-Best"
MAP="maps/open_competition/TwoBasesBarracks16x16.xml"
BOTS=("RandomBiasedAI" "WorkerRush" "LightRush" "CoacAI" "Mayari" "ObiBotKenobi" "TMA" "RAISocketAI")
BOT=${BOTS[$SLURM_ARRAY_TASK_ID]}

EVAL_DIR="${AGENT}/EVAL_genprobe_TwoBasesBarracks"
mkdir -p "$EVAL_DIR"
OUTFILE="${EVAL_DIR}/$(printf '%d' $((SLURM_ARRAY_TASK_ID + 1)))_uecdbest_vs_${BOT}.txt"

echo "============================================================"
echo "  Generalisation probe: UECD-SingleMap-Best vs $BOT"
echo "  Map:   $MAP"
echo "  Games: 50 P0 + 50 P1 = 100 stochastic"
echo "  $(date)"
echo "============================================================"

microrts-agent evaluate \
    --agent "$AGENT" \
    --opponent "$BOT" \
    --nb_games 50 \
    --maps "$MAP" \
    --max-steps 4000 \
    2>&1 | tee "$OUTFILE"

echo "Done: $OUTFILE"
