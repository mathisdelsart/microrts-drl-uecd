#!/bin/bash
#SBATCH --job-name=agentA_8x8
#SBATCH --output=slurm-agentA-8x8-%j.out
#SBATCH --error=slurm-agentA-8x8-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Agent A — 8x8/9x8 CLUSTER cross-map fine-tune
# ---------------------------------------------------------------------------
# Objective:
#   Adapt BestRL-350M (16x16 specialist) to the small-map cluster. Although
#   strategy regimes differ (rush dominates here vs army-building on 16x16),
#   the agent already has the basics: movement, harvesting, combat, building.
#   We re-purpose those fundamentals via aggressive fine-tuning rather than
#   throwing them away with a from-scratch run.
#
# Zero-shot diagnostic of BestRL-350M on small maps:
#   basesWorkers8x8A       : 16%  (poor — builds army instead of rushing)
#   FourBasesWorkers8x8    : 53%  (coin flip — agent does *something*)
#   NoWhereToRun9x8        : 29%  (poor — wall confuses it)
#
# The 53% on FourBasesWorkers8x8 confirms baseline transfer. We just need to
# unlearn the army-build bias and develop rush behaviour.
#
# Strategy:
#   - Load BestRL-350M weights (--load-model)
#   - 3 small maps: basesWorkers8x8A + FourBasesWorkers8x8 + NoWhereToRun9x8
#   - Relearning-style hyperparams: higher LR (1e-4 -> 1e-5) than polish,
#     standard entropy, full reward shaping range
#   - 100M steps total (vs 150M from scratch — leveraging baseline knowledge)
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Agent A — 8x8/9x8 cluster fine-tune from BestRL-350M"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/BestRL-350M/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --multi-map True \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/NoWhereToRun9x8.xml \
    --max-map-size 9 \
    --map-switch-freq 50 \
    --max-steps 2000 \
    --num-bot-envs 12 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 6 \
    --pfsp True \
    --selfplay-save-interval 200 \
    --selfplay-swap-interval 25 \
    --selfplay-pool-size 50 \
    --diverse-opponents True \
    --opponent-list "WorkerRush:1,LightRush:1,CoacAI:2,Mayari:2,TMA:2,ObiBotKenobi:2,RAISocketAI:2" \
    --alternate-players True \
    --prioritized-sampling True \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --learning-rate 1e-4 \
    --learning-rate-end 1e-5 \
    --anneal-lr True \
    --ent-coef 0.01 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --reward-weight 10.0 1.0 1.0 0.2 0.2 1.0 4.0 4.0 4.0 0.0 \
    --reward-weight-end 10.0 0.1 0.1 0.02 0.02 0.1 0.4 0.4 0.4 0.0 \
    --reward-schedule linear \
    --reward-schedule-start 0.30 \
    --reward-schedule-end 0.70 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --eval-bots WorkerRush LightRush CoacAI Mayari TMA ObiBotKenobi RAISocketAI \
    --num-models 8 \
    --seed $SEED \
    --exp-name agent_a_8x8_cluster_100M_s${SEED}
