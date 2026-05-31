#!/bin/bash
#SBATCH --job-name=best_mm400M
#SBATCH --output=slurm-bestmm-%j.out
#SBATCH --error=slurm-bestmm-%j.err
#SBATCH --time=5-00:00:00
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=48G
#SBATCH --gres=gpu:1
#SBATCH --partition=batch
#
# BEST multi-map agent: 400M steps, 5 maps (<= 16x16)
#
# Stacks every ablation-positive feature with a RAI-matched phased schedule
# and adaptive opponent curriculum that escalates from RandomBiasedAI all
# the way to TMA/ObiBotKenobi, with dormant self-play pool activating on
# terminal mastery.
#
# Features activated:
#   Architecture:   unet_entity_cbam_deep (ch=48)
#   Obs encoding:   extended_obs (73ch) + filtered_masks + reserved_obs
#                   + frame_stack 4
#   Activations:    GELU
#   Policy head:    autoregressive + hierarchical_mask (RAISocketAI-style)
#   Critic:         SPP critic (scale-invariant across map sizes)
#   Value heads:    triple (shaped + sparse γ=0.999 + cost γ=0.999)
#                   with RAI-matched phased advantage/vf schedule
#   Normalization:  PopArt (per-head) + PAE 0.95
#   Aux tasks:      opponent_modeling
#   Rewards:        build-time weighted (Heavy=6, Ranged=5.25)
#   Curriculum:     Map PLR + Adaptive opponents (hybrid criteria, 5 tiers
#                   R→POW→POL→[Coac,Mayari]→[TMA,ObiBot,RAISocketAI])
#   Self-play:      dormant 12 envs, activates on terminal mastery
#                   + PFSP hard sampling, prioritized sampling on bots
#   Map pool:       5 maps <= 16x16 (max_map_size=16, padded)
#
# Phased schedule (RAI-matched proportions, 400M):
#   Phase 1 (0→120M):  adv=(0.8,0.01,0.19) vf=(0.5,0.1,0.2) lr=1e-4 ent=0.01
#   Trans  (120→200M): linear blend
#   Phase 2 (200→240M): adv=(0.0,0.5,0.5)  vf=(0.0,0.4,0.4)  lr=1e-4 ent=0.01
#   Trans  (240→320M): linear blend
#   Phase 3 (320→400M): adv=(0.0,0.99,0.01) vf=(0.0,0.5,0.1) lr=5e-5 ent=0.001
#
SEED=${SEED:-1}

echo "============================================================"
echo "  BEST multi-map 400M: seed $SEED | Started: $(date)"
echo "============================================================"

source "$(dirname "$0")/../_setup_env.sh"

python -c "import torch; print(f'CUDA: {torch.cuda.is_available()}, GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"N/A\"}')"

RESUME_FLAG=""
RUN_DIR="outputs/runs/UECD-MultiMap-Best_s${SEED}"
if [ -f "${RUN_DIR}/checkpoint.pt" ]; then
    echo "  Resuming from $RUN_DIR"
    RESUME_FLAG="--resume ${RUN_DIR}"
fi

microrts-agent train \
    $RESUME_FLAG \
    --architecture unet_entity_cbam_deep \
    --arch-channels 48 \
    --total-timesteps 400000000 \
    --max-steps 4000 \
    \
    --multi-map True \
    --max-map-size 16 \
    --map-pool \
        maps/open_competition/basesWorkers8x8A.xml \
        maps/open_competition/FourBasesWorkers8x8.xml \
        maps/open_competition/NoWhereToRun9x8.xml \
        maps/open_competition/basesWorkers16x16A.xml \
        maps/open_competition/TwoBasesBarracks16x16.xml \
    --map-switch-freq 50 \
    --plr True \
    --plr-window 50 \
    \
    --num-bot-envs 24 \
    --num-selfplay-envs 12 \
    --num-latest-selfplay-envs 0 \
    --opponent-list RandomBiasedAI:24 \
    --diverse-opponents False \
    --alternate-players True \
    --prioritized-sampling True \
    \
    --adaptive-opponents True \
    --adaptive-threshold 0.7 \
    --adaptive-window 50 \
    --adaptive-criteria hybrid \
    --adaptive-hybrid-min 0.4 \
    --adaptive-selfplay True \
    --adaptive-selfplay-threshold 0.7 \
    \
    --pfsp True \
    --pfsp-mode hard \
    --selfplay-pool-size 30 \
    --selfplay-save-interval 150 \
    --selfplay-swap-interval 25 \
    \
    --num-steps 512 \
    --update-epochs 2 \
    --n-minibatch 3 \
    --gamma 0.99 \
    --gae-lambda 0.95 \
    --clip-coef 0.1 \
    --max-grad-norm 0.5 \
    --vf-coef 0.5 \
    \
    --triple-value-heads True \
    --gamma-sparse 0.999 \
    --gae-lambda-sparse 0.99 \
    --gamma-cost 0.999 \
    --gae-lambda-cost 0.99 \
    \
    --phase-steps 120000000 200000000 240000000 320000000 \
    --phase-adv-weights "0.8,0.01,0.19" "0.0,0.5,0.5" "0.0,0.99,0.01" \
    --phase-vf-coefs "0.5,0.1,0.2" "0.0,0.4,0.4" "0.0,0.5,0.1" \
    --phase-ent-coef 0.01 0.01 0.001 \
    --phase-lr 1e-4 1e-4 5e-5 \
    \
    --extended-obs True \
    --filtered-masks True \
    --reserved-obs True \
    --frame-stack 4 \
    --gelu True \
    --reward-unit-build-time True \
    \
    --autoregressive True \
    --spp-critic True \
    --popart True \
    --pae-keep 0.95 \
    \
    --aux-opponent-modeling True \
    \
    --with-eval True \
    --eval-bots RandomBiasedAI WorkerRush LightRush CoacAI Mayari ObiBotKenobi TMA RAISocketAI \
    --eval-games 10 \
    --num-models 16 \
    --seed $SEED \
    --exp-name UECD-MultiMap-Best_s${SEED}

EXIT_CODE=$?

echo ""
echo "============================================================"
echo "  JOB COMPLETE"
echo "  Exit code: $EXIT_CODE"
echo "  Finished:  $(date)"
echo "============================================================"

exit $EXIT_CODE
