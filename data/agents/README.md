# Trained agents — multi-map family

Trained checkpoints of the two multi-map agents discussed in the thesis,
shipped in *medium* form: enough to **load and re-use** an agent (`agent.pt`),
enough to **resume training** (`checkpoint.pt`, includes optimiser state),
and enough to **audit what was trained** (`config.json`, `eval_results.csv`,
`train.log`). TensorBoard event files were not shipped — they live under
`outputs/runs/multiple_map/<run>/` locally and are git-ignored.

## Contents

| Run | Steps | Final pool WR | Self-play | Adaptive opp. | Size |
|---|---:|---:|---|---|---:|
| `UECD-MultiMap`      | 200M | 80% | none                          | none                           | 108 MB |
| `UECD-MultiMap-Best` | 330M / 400M (partial) | 84% | 12 envs pool-only, pool=30, adaptive | yes (hybrid criterion, `min`=0.4) | 116 MB |

Both share the same architecture and observation stack:

- `unet_entity_cbam_deep`, `channels=48`, no constant-channel mode
- `extended_obs=True`, 293-channel observation, `reserved_obs=True`
- 10 reward components, no partial observability
- Trained on the 5-map IEEE-CoG open-competition pool with `map_switch_freq=50`
- Auxiliary opponent modelling head active (`aux_opponent_modeling=True`, coef 0.05)

`UECD-MultiMap` is the headline multi-map agent reported in the dissertation
and used in `data/tournaments/multi_map/`. `UECD-MultiMap-Best` is a longer
run with self-play + adaptive curricula that beats it on `CoacAI` and
`Mayari` but stops short of its full 400M budget.

## Per-file layout

```
data/agents/<run>/
├── agent.pt           # inference-ready policy state-dict (20 MB)
├── checkpoint.pt      # full training checkpoint with optimiser state (~60 MB) — for resume
├── config.json        # every CLI / hyperparameter the run was launched with
├── eval_results.csv   # formal final eval: WR per (map, opponent) at the last eval step
└── train.log          # end-to-end textual training log (loss curves, eval rollups, ...)
```

## Loading an agent

```python
import torch
from microrts_agent.architectures.factory import build_architecture

# Match how the run was trained
config = json.load(open("data/agents/UECD-MultiMap/config.json"))
model = build_architecture(config["architecture"], config)
state = torch.load("data/agents/UECD-MultiMap/agent.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

`microrts_agent/evaluate.py` already does this end-to-end and points to
the on-disk run directory:

```bash
python -m microrts_agent evaluate \
    --agent data/agents/UECD-MultiMap \
    --opponent CoacAI \
    --map basesWorkers16x16A.xml \
    --num-games 100
```

## Resuming training

```bash
python -m microrts_agent train \
    --resume data/agents/UECD-MultiMap-Best/checkpoint.pt \
    --total-timesteps 400000000
```

## Reproducing from scratch

The SLURM job scripts that generated each run live under `experiments/multi-map/`:

| Run | Generator |
|---|---|
| `UECD-MultiMap`      | `experiments/multi-map/multimap_small_200M.slurm` |
| `UECD-MultiMap-Best` | `experiments/multi-map/best_multimap_400M.slurm`  |

Both share the same `data/bc_training/` warm-start lineage (BC pre-training
against `RAISocketAI` demos, followed by PPO fine-tune) discussed in the
training-system chapter of the dissertation.

## See also

- 📊 **Headline tournament results** using these agents: [`../tournaments/`](../tournaments/)
- 🎯 **Generalisation probes** of the single-map family: [`../probes/`](../probes/)
- 🎓 **Teacher dataset** behind the BC warm-start: [`../bc_training/`](../bc_training/)
