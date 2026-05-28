#!/bin/bash
#SBATCH --job-name=ft24x24
#SBATCH --output=slurm-ft24x24-%j.out
#SBATCH --error=slurm-ft24x24-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Cross-map fine-tune: BestRL-350M (basesWorkers16x16A) -> DoubleGame24x24
# ---------------------------------------------------------------------------
# Objective:
#   Re-purpose BestRL-350M on a NEW map of a different size (24x24).
#   The unet_entity_cbam_deep backbone is fully convolutional + attention
#   over entities, so size-agnostic at the architecture level; the agent
#   has never seen 24x24 spatial statistics, however, so this is a
#   genuine re-learning (not a polish).
#
# Pool (24 bot envs, no self-play until the agent is decent on this map):
#   POWorkerRush:3, POLightRush:3, CoacAI:3, Mayari:3,
#   TMA:4, ObiBotKenobi:4, RAISocketAI:4
#   -- RandomBiased skipped: BestRL already beats it trivially.
#   -- Balanced: premium bots (TMA / Obi / RAI) get 50% of the envs so the
#      learned competitive priors are reinforced from step 1, not eroded
#      by over-exposure to scripted rushes.
#   -- --prioritized-sampling: bots where the WR is low are over-sampled.
#      (No --adaptive-opponents to match BestRL-350M's exact feature set.)
#
# Reward schedule (full restart: this is relearning, not polishing):
#   reward_weight       = 1.00 x default (back to full shaped bootstrap)
#   reward_weight_end   = 0.10 x default (10% floor, our validated setting)
#   schedule            = hold at 100% for first 30M, linear anneal to 10%
#                         between 30-80M, hold at 10% after.
#   Rationale: 24x24 + 8000-step games = long-horizon credit assignment
#   is hard on sparse signal alone. BestRL's value head was exposed to
#   the full 100%->10% range during its own training, so it handles the
#   restart without a dangerous distribution shift.
#
# Other choices:
#   - Load weights only (--load-model), start fresh optimizer/steps
#   - Higher LR (1e-4 -> 5e-5) than the polish fine-tune: real relearning
#   - Higher entropy (0.01 -> 0.005) to encourage exploration on new map
#   - 5000 max steps per game (24x24 games are longer than 16x16)
#   - 9 checkpoints (every ~25M) via --num-models 8
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Fine-tune BestRL-350M on DoubleGame24x24 (cross-map)"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/BestRL-350M/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 200000000 \
    --map maps/open_competition/DoubleGame24x24.xml \
    --max-steps 5000 \
    --num-bot-envs 24 \
    --num-selfplay-envs 0 \
    --opponent-list "POWorkerRush:3,POLightRush:3,CoacAI:3,Mayari:3,TMA:4,ObiBotKenobi:4,RAISocketAI:4" \
    --alternate-players True \
    --prioritized-sampling True \
    --reward-weight 10.0 1.0 1.0 0.2 0.2 1.0 4.0 6.0 5.25 0.0 \
    --reward-weight-end 10.0 0.1 0.1 0.02 0.02 0.1 0.4 0.6 0.525 0.0 \
    --reward-schedule linear \
    --reward-schedule-start 0.3 \
    --reward-schedule-end 0.8 \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --learning-rate 1e-4 \
    --learning-rate-end 5e-5 \
    --anneal-lr True \
    --ent-coef 0.01 \
    --ent-coef-end 0.005 \
    --vf-coef 0.5 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --num-models 8 \
    --seed $SEED \
    --exp-name finetune_bestrl_24x24_s${SEED}
