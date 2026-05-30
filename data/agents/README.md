# Trained agents — UECD family + GridNet baseline

Trained checkpoints of every agent the thesis published or used as a baseline,
shipped in *medium* form: enough to **load and re-use** an agent (`agent.pt`),
enough to **resume training** (`checkpoint.pt`, includes optimiser state),
and enough to **audit what was trained** (`config.json`, `eval_results.csv`,
`train.log`). TensorBoard event files were not shipped — they live under
`outputs/runs/<family>/<run>/` locally and are git-ignored.

## Single-map agents (trained on `basesWorkers16x16A`)

| Run | Steps | Pool WR | Self-play | Key role |
|---|---:|---:|---|---|
| [`UECD-SingleMap-Best`](UECD-SingleMap-Best/)         | 360M / 400M  | 100% | yes (16 envs, adaptive) | Final agent of the two-phase fine-tuning of the dissertation; **96.67% tournament WR**. |
| [`UECD-SingleMap-Rushed`](UECD-SingleMap-Rushed/)     | 300M ✅      | 100% | 12 envs                 | Demonstrates the *rush-collapse* pathology (Section *Rush Collapse* in the dissertation). |
| [`UECD-SingleMap-TopFeats`](UECD-SingleMap-TopFeats/) | 100M ✅      | 100% | none                    | Top-5 feature ablation — validates the feature cutoff at 100M. |
| [`UECD-SingleMap-AllFeats`](UECD-SingleMap-AllFeats/) | 100M ✅      | 100% | none                    | All-features ablation — confirms the conditional features don't help at the same budget. |
| [`GridNet-SingleMap`](GridNet-SingleMap/)             | 300M ✅      |  50% | none                    | Published baseline from Huang et al. (2021); the gap UECD must beat. |

## Multi-map agents (trained on the 5-map open-competition pool)

| Run | Steps | Pool WR | Self-play | Key role |
|---|---:|---:|---|---|
| [`UECD-MultiMap`](UECD-MultiMap/)           | 200M ✅          | 80% | none                                              | Headline multi-map agent of the dissertation. |
| [`UECD-MultiMap-Best`](UECD-MultiMap-Best/) | 330M / 400M     | 84% | 12 envs pool-only, pool=30, adaptive (hybrid)     | Longer run with stronger curricula; beats `UECD-MultiMap` on `CoacAI` / `Mayari`. |

## Lineage — `UECD-SingleMap-Best`

`UECD-SingleMap-Best` is the only agent in this folder whose `train.log`
does not begin at step 0 — it is a fine-tune that resumes from a checkpoint
of an earlier run. The full training trace from step 0 is reconstructed by
chaining three segments:

```
Step      0M → 150M         150M → 240M               240M → 360M
Segment   UECD-SingleMap-    lineage/phase1-           UECD-SingleMap-Best
          Rushed             pool-broadening           (this directory)
What      Phased training    Phase 1 — broaden the    Phase 2 — harden
          from scratch       opponent pool             against RAISocketAI
Source    The same run was   Original run name was    Original run name was
          continued to 300M  FineTuneStronger-300M    BestRL-350M
          for the rush-      (recovered from local
          collapse demo;     backup, not regenerated)
          its 0-150M segment
          IS this segment.
```

`UECD-SingleMap-Best/lineage/phase1-pool-broadening/` contains the medium-tier
files of the Phase 1 segment (`train.log`, `config.json`, `eval_results.csv`)
plus the **exact handoff checkpoint** that started Phase 2
(`checkpoint_240M.pt`, copied as `checkpoint.pt` in the Phase 2 run dir at
training time). To rebuild the full training history from step 0:

1. Read `data/agents/UECD-SingleMap-Rushed/train.log` up to step 150M.
2. Read `data/agents/UECD-SingleMap-Best/lineage/phase1-pool-broadening/train.log` (covers 150M → 240M).
3. Read `data/agents/UECD-SingleMap-Best/train.log` (covers 240M → 360M).

Every other agent in this folder trains from scratch and is self-contained.

## Per-agent file layout

```
data/agents/<run>/
├── agent.pt           # inference-ready policy state-dict (~20 MB)
├── checkpoint.pt      # full training checkpoint with optimiser state (~60 MB) — for resume
├── config.json        # every CLI / hyperparameter the run was launched with
├── eval_results.csv   # formal final eval (WR per map × opponent at the last eval step)
└── train.log          # end-to-end textual training log
```

`UECD-SingleMap-Best` additionally carries the `lineage/` subdirectory
described above.

## Loading an agent

```python
import torch, json
from microrts_agent.architectures.factory import build_architecture

agent_dir = "data/agents/UECD-SingleMap-Best"
config = json.load(open(f"{agent_dir}/config.json"))
model = build_architecture(config["architecture"], config)
state = torch.load(f"{agent_dir}/agent.pt", map_location="cpu")
model.load_state_dict(state)
model.eval()
```

End-to-end via the CLI:

```bash
python -m microrts_agent evaluate \
    --agent data/agents/UECD-SingleMap-Best \
    --opponent CoacAI \
    --map basesWorkers16x16A.xml \
    --num-games 100
```

## Resuming training

```bash
python -m microrts_agent train \
    --resume data/agents/UECD-SingleMap-Best/checkpoint.pt \
    --total-timesteps 400000000
```

For `UECD-SingleMap-Best`, the resume points are: `checkpoint.pt` (= 360M,
end of Phase 2) or `lineage/phase1-pool-broadening/checkpoint_240M.pt`
(= 240M, end of Phase 1 / start of Phase 2).

## See also

- 📊 **Headline tournament results** using these agents: [`../tournaments/`](../tournaments/)
- 🎯 **Generalisation probes** of `UECD-SingleMap-Best`: [`../probes/`](../probes/)
- 🎓 **BC teacher dataset** (separate research line from the two-phase
  fine-tuning of `UECD-SingleMap-Best`): [`../bc_training/`](../bc_training/)
- 🎬 **Showcase recordings** of `UECD-SingleMap-Best` vs the field:
  [`../recordings/`](../recordings/)
