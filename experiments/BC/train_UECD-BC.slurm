#!/bin/bash
#SBATCH --job-name=bc_te
#SBATCH --output=slurm-bc-te-%j.out
#SBATCH --error=slurm-bc-te-%j.err
#SBATCH --time=2:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch

source "$(dirname "$0")/../_setup_env.sh"

SEED=${SEED:-1}
BC_RUN_DIR="outputs/runs/UECD-BC_s${SEED}"
mkdir -p "${BC_RUN_DIR}/EVAL"

echo "=== BC+VF Train ==="
python -m microrts_agent.bc train \
    --data outputs/bc_data/bc_chunk_RAISocketAI_*.npz outputs/bc_data/bc_chunk_CoacAI_*.npz outputs/bc_data/bc_chunk_Mayari_*.npz \
    --architecture unet_entity_cbam_deep \
    --gelu --epochs 30 --batch-size 128 --lr 1e-4 --vf-coef 0.5 --gamma 0.99 \
    --seed $SEED --output "$BC_RUN_DIR"

[ $? -ne 0 ] && { echo "ERROR: Train failed"; exit 1; }

echo "=== Eval ==="
for BOT in RandomBiasedAI WorkerRush LightRush CoacAI Mayari; do
    echo "  vs $BOT..."
    python -m microrts_agent evaluate \
        --agent "$BC_RUN_DIR" --opponent "$BOT" \
        --nb_games 10 --maps maps/open_competition/basesWorkers16x16A.xml --max-steps 4000 \
        2>&1 | tee "${BC_RUN_DIR}/EVAL/bc_vs_${BOT}.txt"
done

echo "=== DONE ==="
