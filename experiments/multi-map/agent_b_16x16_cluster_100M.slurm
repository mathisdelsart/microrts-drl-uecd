#!/bin/bash
#SBATCH --job-name=agentB_16x16
#SBATCH --output=slurm-agentB-16x16-%j.out
#SBATCH --error=slurm-agentB-16x16-%j.err
#SBATCH --time=4-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=24
#SBATCH --mem=32G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# Agent B — 16x16 CLUSTER fine-tune
# ---------------------------------------------------------------------------
# Objective:
#   Take UECD-Best (single-map, basesWorkers16x16A) and turn it into a
#   16x16-cluster generalist by fine-tuning on a multi-map mix of 16x16-class
#   layouts.
#
# Zero-shot diagnostic (before this run):
#   basesWorkers16x16A      : 92%  (training map)
#   TwoBasesBarracks16x16   : 52%  (overfit to original layout)
#   NoWhereToRun9x8         : 29%  (different topology — wall in middle)
#
# Goal: lift TwoBasesBarracks16x16 + EightBasesWorkers16x16 above 80%
# without losing basesWorkers16x16A.
# NoWhereToRun9x8 is left out of training (separate strategic regime due to
# the wall); we evaluate on it but don't optimize for it here.
#
# Strategy:
#   - Start from UECD-Best weights (--load-model)
#   - Multi-map mix (3 layouts): basesWorkers16x16A + TwoBasesBarracks16x16
#                               + EightBasesWorkers16x16
#   - Polish-style hyperparams: low LR, low entropy, shaped-weight floor at 0.10
#   - Track winrate on the original map continuously to catch forgetting
#   - 100M steps total (~33M per map — adapted for 3-map mix)
# ---------------------------------------------------------------------------

SEED=${SEED:-1}

module load Python/3.11 2>/dev/null || module load Python/3.9
module load Java/17 2>/dev/null || module load Java/11
module load CUDA/11.8 2>/dev/null || module load CUDA/12.1
source $HOME/microrts-drl-uecd/cluster_venv/bin/activate
export PYTHONUNBUFFERED=1
cd $HOME/microrts-drl-uecd

echo "============================================================"
echo "  Agent B — 16x16 cluster fine-tune from UECD-Best"
echo "  Seed: $SEED | Started: $(date)"
echo "============================================================"

python microrts_agent/train.py \
    --load-model outputs/runs/single_map/UECD-Best/agent.pt \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 100000000 \
    --multi-map True \
    --map-pool \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
        maps/16x16/EightBasesWorkers16x16.xml \
    --max-map-size 16 \
    --map-switch-freq 50 \
    --max-steps 4000 \
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
    --learning-rate 5e-5 \
    --learning-rate-end 1e-5 \
    --anneal-lr True \
    --ent-coef 0.005 \
    --ent-coef-end 0.001 \
    --vf-coef 0.5 \
    --reward-weight 10.0 1.0 1.0 0.2 0.2 1.0 4.0 6.0 5.25 0.0 \
    --reward-weight-end 10.0 0.1 0.1 0.02 0.02 0.1 0.4 0.6 0.525 0.0 \
    --reward-schedule linear \
    --reward-schedule-start 0.0 \
    --reward-schedule-end 0.6 \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --aux-opponent-modeling True \
    --gelu True \
    --with-eval True \
    --eval-bots WorkerRush LightRush CoacAI Mayari TMA ObiBotKenobi RAISocketAI \
    --num-models 8 \
    --seed $SEED \
    --exp-name agent_b_16x16_cluster_100M_s${SEED}
