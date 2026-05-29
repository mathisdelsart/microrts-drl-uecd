#!/bin/bash
#SBATCH --job-name=ft24x24_v3
#SBATCH --output=slurm-ft24x24-v3-%j.out
#SBATCH --error=slurm-ft24x24-v3-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=64G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Cross-map fine-tune: BestRL-350M (basesWorkers16x16A) -> DoubleGame24x24
# V3 — Aggressive OOM mitigation (keeps arch-channels=48 to load checkpoint)
# ---------------------------------------------------------------------------
# v1 crash root cause:
#   torch.OutOfMemoryError during PPO update on entropy computation:
#       entropy = torch.stack([cat.entropy() for cat in multi_categoricals])
#       # shape (7, B * mapsize) where mapsize=576 (24x24) and B=4096
#   GPU was at 41/48 GB, OOM tried to alloc 442 MiB -> only 248 MiB free.
#
# v3 mitigations (compounded):
#   1) PYTORCH_ALLOC_CONF=expandable_segments:True
#      Free trick. Reduces memory fragmentation, often reclaims 1-3 GiB.
#   2) --num-bot-envs 12  (was 24)
#      Halves both system RAM (12 fewer JVMs) and rollout buffer.
#   3) --num-steps 384    (was 512)
#      Reduces rollout buffer further: 12 * 384 = 4608 samples (vs 12288 in v1).
#   4) --n-minibatch 6    (was 3)
#      Splits PPO update into 6 chunks: 768 samples per minibatch (vs 4096).
#      Directly shrinks the (7, B*576) entropy tensor by 5.3x.
#
# Net effect: peak GPU memory during PPO update drops by ~5x. Should give
# ample headroom to also accommodate value-head + aux-opponent-modeling backward.
#
# arch-channels=48 is REQUIRED to load BestRL-350M weights (cannot change).
#
# If V3 still OOMs (very unlikely):
#   V4 — drop to --num-bot-envs 8 + --num-steps 256 + --n-minibatch 8
#   V5 — code change: bf16 mixed precision in train.py
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1

# Memory management hint to PyTorch — reduces fragmentation
export PYTORCH_ALLOC_CONF=expandable_segments:True

cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Fine-tune BestRL-350M on DoubleGame24x24 (V3)"
echo "  Seed: $SEED | Started: $(date)"
echo "  Mem: 64G | Bot envs: 10 | Self-play: 2 | num-steps: 384 | n-minibatch: 6"
echo "  PYTORCH_ALLOC_CONF=$PYTORCH_ALLOC_CONF"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/BestRL-350M/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 150000000 \
    --map maps/open_competition/DoubleGame24x24.xml \
    --max-steps 5000 \
    --num-bot-envs 10 \
    --num-selfplay-envs 2 \
    --num-latest-selfplay-envs 2 \
    --pfsp True \
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
    --diverse-opponents True \
    --opponent-list "POWorkerRush:1,POLightRush:1,CoacAI:1,Mayari:1,TMA:2,ObiBotKenobi:2,RAISocketAI:2" \
    --alternate-players True \
    --prioritized-sampling True \
    --reward-weight 10.0 1.0 1.0 0.2 0.2 1.0 4.0 6.0 5.25 0.0 \
    --reward-weight-end 10.0 0.1 0.1 0.02 0.02 0.1 0.4 0.6 0.525 0.0 \
    --reward-schedule linear \
    --reward-schedule-start 0.3 \
    --reward-schedule-end 0.8 \
    --num-steps 384 \
    --update-epochs 2 \
    --n-minibatch 6 \
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
    --exp-name finetune_bestrl_24x24_v3_s${SEED}
