#!/bin/bash
#SBATCH --job-name=ft24x24_v2
#SBATCH --output=slurm-ft24x24-v2-%j.out
#SBATCH --error=slurm-ft24x24-v2-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Cross-map fine-tune: UECD-Best (basesWorkers16x16A) -> DoubleGame24x24
# V2 — OOM mitigations applied
# ---------------------------------------------------------------------------
# Changes vs v1 (which OOM'd):
#   --mem=32G              -> --mem=64G       (system RAM doubled)
#   --num-bot-envs 24      -> --num-bot-envs 12   (halve parallel envs)
#   (kept) --num-steps 512                       (preserve PPO stability)
#   (kept) --arch-channels 48                    (must match BestRL checkpoint)
#
# Memory math:
#   24x24 = 576 cells (2.25x larger than 16x16's 256 cells).
#   Halving num-bot-envs (24 -> 12) compensates for ~2x of that increase
#   on both system RAM (12 fewer JVMs) and GPU activations.
#
# If V2 still OOMs, fallback options:
#   1. --num-bot-envs 8 + --num-steps 384  (V3 — more aggressive)
#   2. Add bf16 mixed precision in train.py (V4 — code change)
#   3. Gradient checkpointing on U-Net blocks (V5 — code change)
#
# Strategy unchanged from v1:
#   - Load weights only (--load-model), fresh optimizer/steps
#   - Reward 100% -> 10% over [30M, 80M], hold at 10% after
#   - Pool: 24 bot envs (now 12), no self-play
#   - 5000 max steps per game (24x24 games are longer)
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Fine-tune UECD-Best on DoubleGame24x24 (V2 — OOM fix)"
echo "  Seed: $SEED | Started: $(date)"
echo "  Mem: 64G | Envs: 12 | num-steps: 512"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/UECD-Best/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 200000000 \
    --map maps/open_competition/DoubleGame24x24.xml \
    --max-steps 5000 \
    --num-bot-envs 12 \
    --num-selfplay-envs 0 \
    --opponent-list "WorkerRush:2,LightRush:2,CoacAI:1,Mayari:1,TMA:2,ObiBotKenobi:2,RAISocketAI:2" \
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
    --exp-name finetune_bestrl_24x24_v2_s${SEED}
