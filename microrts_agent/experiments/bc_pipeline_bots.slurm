#!/bin/bash
#SBATCH --job-name=bc_bots
#SBATCH --output=slurm-bc-bots-%j.out
#SBATCH --error=slurm-bc-bots-%j.err
#SBATCH --time=24:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BC Pipeline (bot replays): clone CoacAI and Mayari from their own games
#
#   CoacAI vs CoacAI:   2000 games (learn CoacAI's style)
#   Mayari vs Mayari:    2000 games (learn Mayari's style)
#   CoacAI vs Mayari:    1000 games (learn CoacAI against a strong opponent)
#   Mayari vs CoacAI:    1000 games (learn Mayari against a strong opponent)
#   Total: 6000 games
#
# Usage:
#   sbatch microrts_agent/experiments/bc_pipeline_bots.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BC Pipeline (Bot Replays) — Generate + Train + Evaluate"
echo "============================================================"
echo "  Job ID:   $SLURM_JOB_ID"
echo "  Seed:     $SEED"
echo "  Node:     $SLURM_NODELIST"
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

BC_DATA_DIR="outputs/bc_data"
BC_RUN_DIR="outputs/runs/bc_bots_s${SEED}"
BC_EVAL_DIR="${BC_RUN_DIR}/EVAL"
MAP="maps/open_competition/basesWorkers16x16A.xml"

DATA_COAC_SELF="${BC_DATA_DIR}/CoacAI_vs_CoacAI_500g.npz"
DATA_MAYA_SELF="${BC_DATA_DIR}/Mayari_vs_Mayari_500g.npz"
DATA_COAC_MAYA="${BC_DATA_DIR}/CoacAI_vs_Mayari_250g.npz"
DATA_MAYA_COAC="${BC_DATA_DIR}/Mayari_vs_CoacAI_250g.npz"

mkdir -p "$BC_DATA_DIR"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate BC data (6000 games total)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 1/3 — Generating BC replay data"
echo "============================================================"

# CoacAI vs CoacAI (2000 games, learn CoacAI's perspective)
if [ ! -f "$DATA_COAC_SELF" ]; then
    echo "  [1/4] CoacAI vs CoacAI (500 games)..."
    python -m microrts_agent.bc generate \
        --bot CoacAI --opponents CoacAI \
        --games-per-opponent 500 \
        --map "$MAP" --output "$DATA_COAC_SELF"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [1/4] Skipping: $DATA_COAC_SELF exists"
fi

# Mayari vs Mayari (2000 games, learn Mayari's perspective)
if [ ! -f "$DATA_MAYA_SELF" ]; then
    echo "  [2/4] Mayari vs Mayari (500 games)..."
    python -m microrts_agent.bc generate \
        --bot Mayari --opponents Mayari \
        --games-per-opponent 500 \
        --map "$MAP" --output "$DATA_MAYA_SELF"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [2/4] Skipping: $DATA_MAYA_SELF exists"
fi

# CoacAI vs Mayari (1000 games, learn CoacAI's perspective against Mayari)
if [ ! -f "$DATA_COAC_MAYA" ]; then
    echo "  [3/4] CoacAI vs Mayari (250 games)..."
    python -m microrts_agent.bc generate \
        --bot CoacAI --opponents Mayari \
        --games-per-opponent 250 \
        --map "$MAP" --output "$DATA_COAC_MAYA"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [3/4] Skipping: $DATA_COAC_MAYA exists"
fi

# Mayari vs CoacAI (250 games, learn Mayari's perspective against CoacAI)
if [ ! -f "$DATA_MAYA_COAC" ]; then
    echo "  [4/4] Mayari vs CoacAI (250 games)..."
    python -m microrts_agent.bc generate \
        --bot Mayari --opponents CoacAI \
        --games-per-opponent 250 \
        --map "$MAP" --output "$DATA_MAYA_COAC"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [4/4] Skipping: $DATA_MAYA_COAC exists"
fi

echo ""
echo "  Step 1 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Train BC on all bot data (early stopping, patience=5)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 2/3 — Training behavior cloning"
echo "============================================================"

python -m microrts_agent.bc train \
    --data "$DATA_COAC_SELF" "$DATA_MAYA_SELF" "$DATA_COAC_MAYA" "$DATA_MAYA_COAC" \
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
# STEP 3: Evaluate BC agent
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
echo "  BC PIPELINE (BOT REPLAYS) COMPLETE"
echo "============================================================"
echo "  Data:      $DATA_COAC_SELF"
echo "             $DATA_MAYA_SELF"
echo "             $DATA_COAC_MAYA"
echo "             $DATA_MAYA_COAC"
echo "  BC model:  ${BC_RUN_DIR}/agent.pt"
echo "  Eval logs: $BC_EVAL_DIR"
echo "  Finished:  $(date)"
echo "============================================================"
