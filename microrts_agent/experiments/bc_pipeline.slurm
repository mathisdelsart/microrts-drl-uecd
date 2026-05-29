#!/bin/bash
#SBATCH --job-name=bc_pipeline
#SBATCH --output=slurm-bc-%j.out
#SBATCH --error=slurm-bc-%j.err
#SBATCH --time=5:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BC Pipeline: generate replays → train BC → evaluate
#
#   Step 1: Record 3000 games of RAISocketAI vs 5 opponents
#   Step 2: Train BC on the recorded data (30 epochs)
#   Step 3: Evaluate the BC agent vs 4 bots (10 games each)
#
# Usage:
#   sbatch microrts_agent/experiments/bc_pipeline.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BC Pipeline — Generate + Train + Evaluate"
echo "============================================================"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Seed:     $SEED"
echo "  Node:     $SLURM_NODELIST"
echo "  GPUs:     $CUDA_VISIBLE_DEVICES"
echo "  Started:  $(date)"
echo "============================================================"

# ── Environment setup ──────────────────────────────────────────────────────

module load Python/3.11 2>/dev/null || module load Python/3.9 2>/dev/null || {
    echo "ERROR: No Python module found"; exit 1; }
module load Java/17 2>/dev/null || module load Java/11 2>/dev/null || {
    echo "ERROR: No Java module found"; exit 1; }
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1 2>/dev/null || {
    echo "WARNING: No CUDA module found"; }

VENV_DIR="$HOME/microrts-drl-uecd/cluster_venv"
if [ -d "$VENV_DIR" ]; then
    source "$VENV_DIR/bin/activate"
else
    echo "ERROR: cluster_venv not found."
    exit 1
fi

export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

# ── Paths ──────────────────────────────────────────────────────────────────

BC_DATA="outputs/bc_data/RAISocketAI_vs_5opps_basesWorkers16x16A_250g.npz"
BC_RUN_DIR="outputs/runs/bc_unet_entity_cbam_deep_s${SEED}"
BC_EVAL_DIR="${BC_RUN_DIR}/EVAL"
MAP="maps/open_competition/basesWorkers16x16A.xml"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate BC data (3000 games)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 1/3 — Generating BC replay data"
echo "============================================================"

mkdir -p "$(dirname "$BC_DATA")"

if [ -f "$BC_DATA" ]; then
    echo "  BC data already exists: $BC_DATA — skipping generation"
else
    python -m microrts_agent.bc generate \
        --bot RAISocketAI \
        --opponents RAISocketAI CoacAI Mayari POWorkerRush POLightRush \
        --games-per-opponent 50 50 50 50 50 \
        --map "$MAP" \
        --output "$BC_DATA"

    if [ $? -ne 0 ]; then
        echo "ERROR: BC data generation failed"
        exit 1
    fi
fi

echo ""
echo "  Step 1 complete: $BC_DATA"
echo "  $(date)"

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Train BC (30 epochs, batch_size=128)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 2/3 — Training behavior cloning"
echo "============================================================"

python -m microrts_agent.bc train \
    --data "$BC_DATA" \
    --architecture unet_entity_cbam_deep \
    --gelu \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-4 \
    --seed $SEED \
    --output "$BC_RUN_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: BC training failed"
    exit 1
fi

echo ""
echo "  Step 2 complete: ${BC_RUN_DIR}/agent.pt"
echo "  $(date)"

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate BC agent (10 games per bot per position = 20 total each)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 3/3 — Evaluating BC agent"
echo "============================================================"

mkdir -p "$BC_EVAL_DIR"

BOTS=("RandomBiasedAI" "POWorkerRush" "POLightRush" "CoacAI" "Mayari")

for BOT in "${BOTS[@]}"; do
    echo ""
    echo "  Evaluating vs $BOT (10 games per position)..."
    OUTFILE="${BC_EVAL_DIR}/bc_vs_${BOT}.txt"

    python microrts_agent/evaluate.py \
        --agent "$BC_RUN_DIR" \
        --opponent "$BOT" \
        --nb_games 10 \
        --maps maps/16x16/basesWorkers16x16A.xml \
        --max-steps 4000 \
        2>&1 | tee "$OUTFILE"

    echo "  -> Saved: $OUTFILE"
done

# ══════════════════════════════════════════════════════════════════════════
# Done
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  BC PIPELINE COMPLETE"
echo "============================================================"
echo "  BC data:   $BC_DATA"
echo "  BC model:  ${BC_RUN_DIR}/agent.pt"
echo "  Eval logs: $BC_EVAL_DIR"
echo "  Finished:  $(date)"
echo "============================================================"
