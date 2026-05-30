#!/bin/bash
#SBATCH --job-name=bc_v2
#SBATCH --output=slurm-bc-v2-%j.out
#SBATCH --error=slurm-bc-v2-%j.err
#SBATCH --time=4:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BC+VF Pipeline v2: full 10-reward training (matching PPO critic)
#
#   Step 1: Generate BC data with 10 reward components
#   Step 2: Train BC+VF (actor + critic together)
#   Step 3: Evaluate
#
# Usage:
#   sbatch experiments/bc/bc_pipeline_v2.slurm
#

SEED=${SEED:-1}

echo "============================================================"
echo "  BC+VF Pipeline v2 — Full rewards, critic pre-training"
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
BC_RUN_DIR="outputs/runs/bc_v3_s${SEED}"
BC_EVAL_DIR="${BC_RUN_DIR}/EVAL"
MAP="maps/open_competition/basesWorkers16x16A.xml"

DATA_RR="${BC_DATA_DIR}/v3_RAI_vs_RAI_100g.npz"
DATA_RC="${BC_DATA_DIR}/v3_RAI_vs_CoacAI_100g.npz"
DATA_RM="${BC_DATA_DIR}/v3_RAI_vs_Mayari_100g.npz"

mkdir -p "$BC_DATA_DIR"

# ══════════════════════════════════════════════════════════════════════════
# STEP 1: Generate BC data with full 10-component rewards
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 1/3 — Generating BC replay data (full rewards, RAI perspective)"
echo "============================================================"

if [ ! -f "$DATA_RR" ]; then
    echo "  [1/3] RAISocketAI vs RAISocketAI (100 games)..."
    python -m microrts_agent.bc generate \
        --bot RAISocketAI --opponents RAISocketAI \
        --games-per-opponent 100 \
        --map "$MAP" --output "$DATA_RR"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [1/3] Skipping: $DATA_RR exists"
fi

if [ ! -f "$DATA_RC" ]; then
    echo "  [2/3] RAISocketAI vs CoacAI (100 games)..."
    python -m microrts_agent.bc generate \
        --bot RAISocketAI --opponents CoacAI \
        --games-per-opponent 100 \
        --map "$MAP" --output "$DATA_RC"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [2/3] Skipping: $DATA_RC exists"
fi

if [ ! -f "$DATA_RM" ]; then
    echo "  [3/3] RAISocketAI vs Mayari (100 games)..."
    python -m microrts_agent.bc generate \
        --bot RAISocketAI --opponents Mayari \
        --games-per-opponent 100 \
        --map "$MAP" --output "$DATA_RM"
    [ $? -ne 0 ] && { echo "ERROR: Generation failed"; exit 1; }
else
    echo "  [3/3] Skipping: $DATA_RM exists"
fi

echo ""
echo "  Step 1 complete: $(date)"

# ══════════════════════════════════════════════════════════════════════════
# STEP 2: Train BC+VF (actor + critic together)
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 2/3 — Training BC+VF"
echo "============================================================"

python -m microrts_agent.bc train \
    --data ${BC_DATA_DIR}/bc_chunk_RAISocketAI_*.npz ${BC_DATA_DIR}/bc_chunk_CoacAI_*.npz ${BC_DATA_DIR}/bc_chunk_Mayari_*.npz \
    --architecture unet_entity_cbam_deep \
    --gelu \
    --epochs 30 \
    --batch-size 128 \
    --lr 1e-4 \
    --vf-coef 0.5 \
    --gamma 0.99 \
    --seed $SEED \
    --output "$BC_RUN_DIR"

if [ $? -ne 0 ]; then
    echo "ERROR: BC+VF training failed"
    exit 1
fi

echo ""
echo "  Step 2 complete: ${BC_RUN_DIR}/agent.pt"
echo "  $(date)"

# ══════════════════════════════════════════════════════════════════════════
# STEP 3: Evaluate BC+VF agent
# ══════════════════════════════════════════════════════════════════════════

echo ""
echo "============================================================"
echo "  STEP 3/3 — Evaluating BC+VF agent"
echo "============================================================"

mkdir -p "$BC_EVAL_DIR"

BOTS=("RandomBiasedAI" "WorkerRush" "LightRush" "CoacAI" "Mayari")

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

echo ""
echo "============================================================"
echo "  BC+VF PIPELINE v2 COMPLETE"
echo "============================================================"
echo "  Data:      $DATA_RR $DATA_RC $DATA_RM"
echo "  BC model:  ${BC_RUN_DIR}/agent.pt"
echo "  Eval logs: $BC_EVAL_DIR"
echo "  Finished:  $(date)"
echo "============================================================"
