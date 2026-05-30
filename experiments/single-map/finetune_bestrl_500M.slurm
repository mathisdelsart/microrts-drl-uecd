#!/bin/bash
#SBATCH --job-name=ft500M
#SBATCH --output=slurm-ft500M-%j.out
#SBATCH --error=slurm-ft500M-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Fine-tune UECD-Best (single-map specialist) to 500M total steps
# on basesWorkers16x16A.
# ---------------------------------------------------------------------------
# Goals:
#   - Fix the WorkerRush residual weakness (~50% WR on single-map)
#   - Cover competition-bot coverage gaps (adds WorkerRush, LightRush,
#     and doubles RAISocketAI exposure)
#   - Polish toward Condorcet-winner status on the 16x16 tournament
#
# Pool (8 bot envs + 16 self-play = 24 envs):
#   CoacAI:1, Mayari:1, RAISocketAI:2, ObiBotKenobi:1, TMA:1,
#   WorkerRush:1, LightRush:1
#   (UtsImass excluded -- server/preanalysis fragile on cluster)
#
# Safeguards:
#   - Load weights only (--load-model), start fresh optimizer/steps
#   - Small LR (5e-5 -> 1e-5) to refine, not forget
#   - Low entropy (0.005 -> 0.001) to stay close to learned policy
#   - Shaped reward continues smoothly from BestRL's state at step 350M:
#     default_weight * 0.19 (where BestRL ended) -> default_weight * 0.10
#     (the 10% floor), linearly over the 150M fine-tune.
#   - PFSP hard self-play + prioritized opponent sampling
#   - 7 checkpoints (every ~25M steps) via --num-models 6
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Fine-tune UECD-Best -> 500M (single-map, 10% shaped floor)"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/UECD-SingleMap-Best/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 150000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --max-steps 4000 \
    --num-bot-envs 8 \
    --num-selfplay-envs 16 \
    --num-latest-selfplay-envs 8 \
    --pfsp True \
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
    --opponent-list "CoacAI:1,Mayari:1,RAISocketAI:2,ObiBotKenobi:1,TMA:1,WorkerRush:1,LightRush:1" \
    --alternate-players True \
    --prioritized-sampling True \
    --reward-weight 10.0 0.19 0.19 0.038 0.038 0.19 0.76 1.14 0.9975 0.0 \
    --reward-weight-end 10.0 0.1 0.1 0.02 0.02 0.1 0.4 0.6 0.525 0.0 \
    --reward-schedule linear \
    --reward-schedule-start 0.0 \
    --reward-schedule-end 1.0 \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --learning-rate 5e-5 \
    --learning-rate-end 1e-5 \
    --anneal-lr True \
    --ent-coef 0.005 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --num-models 6 \
    --seed $SEED \
    --exp-name finetune_bestrl_500M_s${SEED}
