# `microrts_agent/bc/`

Behaviour cloning: dataset generation (rolling out a teacher bot and
saving its transitions) + supervised BC training that consumes the
generated NPZ chunks.

## Files

| File | Role |
|---|---|
| [`__main__.py`](__main__.py) | Sub-subcommand dispatcher: `microrts-agent bc {generate\|train}`. |
| [`generate.py`](generate.py) | `microrts-agent bc generate`: rolls out a teacher (`RAISocketAI` by default) against an opponent list, saves per-game transitions to NPZ chunks. |
| [`train.py`](train.py) | `microrts-agent bc train`: supervised cross-entropy on the saved transitions. Optionally also fits a value head via Monte-Carlo returns. |

## Dataset format (NPZ chunks)

Each chunk written by [`generate.py`](generate.py) contains the
following arrays (one row per transition):

| Key | Shape | dtype | Meaning |
|---|---|---|---|
| `obs` | `(N, C, H, W)` | float32 | Observation tensor (default 27 channels, 16x16). |
| `actions` | `(N, n_units, action_nvec)` | int64 | Per-unit categorical actions. |
| `rewards` | `(N,)` | float32 | Reward at step t (after weighting). |
| `dones` | `(N,)` | bool | Episode boundary flag (`True` on last step of each game). |

The shipped teacher dataset under [`../../data/BC/training/`](../../data/BC/training/)
is 6 chunks (~65 MB total, 313 394 transitions; RAISocketAI playing
100 games each against RAISocketAI, CoacAI and Mayari on
basesWorkers16x16A).

## BC training

The `bc train` command:

1. Loads all NPZ chunks matching `--data <glob>`.
2. Optionally runs `compute_returns(rewards, dones=dones)` if a value
   head is requested (`--vf-coef > 0`). For legacy chunks **without**
   the `dones` key (shipped pre-#79), it falls back to a
   `|reward| > 5` heuristic and prints a warning.
3. Optimises cross-entropy on the per-unit action logits, plus optional
   MSE on the Monte-Carlo returns for the value head.
4. Writes `agent.pt` / `config.json` / `train.log` to
   `outputs/runs/<exp-name>_s<seed>/` (same layout as PPO training).

## Pipeline

```bash
# 1. Generate the dataset
microrts-agent bc generate \
    --teacher RAISocketAI \
    --opponents RAISocketAI,CoacAI,Mayari \
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --games-per-opponent 100 \
    --output outputs/bc_data

# 2. Train the BC policy (UECD-BC family)
microrts-agent bc train \
    --data "outputs/bc_data/*.npz" \
    --architecture unet_entity_cbam_deep \
    --epochs 5 \
    --exp-name UECD-BC \
    --seed 1

# 3. (Optional) Warm-start a PPO run from the BC checkpoint
microrts-agent train \
    --load-model outputs/runs/UECD-BC_s1/agent.pt \
    ...
```

## Notes

- The teacher is invoked through `RAISocketAI` (or any bot in
  `AI_MAPPING`); the rollout uses the same JNI bridge as training so
  per-step latency stays under the bot's time budget.
- The shipped `UECD-BC-PPO` agent at
  [`../../data/agents/UECD-BC-PPO/`](../../data/agents/UECD-BC-PPO/) is
  the 3-stage product: dataset (`data/BC/training/`) -> BC train
  (`UECD-BC`) -> PPO fine-tune (`UECD-BC-PPO`).
- The BC-only baseline at [`../../data/BC/baseline/`](../../data/BC/baseline/)
  is the horizontal line of the BC+VF -> PPO figure in the
  dissertation.
