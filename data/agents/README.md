# Trained agents — UECD family + GridNet baseline

Trained checkpoints of every agent the thesis published or used as a baseline,
shipped in *medium* form: enough to **load and re-use** an agent (`agent.pt`),
enough to **resume training** (`checkpoint.pt`, includes optimiser state),
and enough to **audit what was trained** (`config.json`, `eval_results.csv`,
`train.log`). TensorBoard event files are too large for git (~200 MB to
1.2 GB per agent) and are shipped separately as
[release assets](#tensorboard-archives).

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

## Behaviour-cloning family (warm-started from `data/bc_training/`)

| Run | Steps | Pool mean WR | Pipeline | Key role |
|---|---:|---:|---|---|
| [`UECD-BC`](UECD-BC/)         | n/a (supervised) | **78.0%** | BC-only (no RL phase) | The 78% horizontal reference line of the dissertation's BC+VF→PPO vs from-scratch figure. Numerical proof under [`../bc_baseline/`](../bc_baseline/). |
| [`UECD-BC-PPO`](UECD-BC-PPO/) | 100M ✅          | **96%**    | BC → PPO fine-tune    | The BC+VF→PPO curve+markers of the same figure. Lifts the BC-only 78% pool baseline to near-perfect after 100M of PPO. |

`UECD-BC` is the only agent here without `agent.pt + checkpoint.pt +
train.log + eval_results.csv` — the BC training script writes only
`agent.pt` and `config.json`. The 78% pool win-rate proof lives at
[`../bc_baseline/`](../bc_baseline/) instead of `eval_results.csv`.
`UECD-BC-PPO` is shipped in the standard medium-tier layout.

## Lineage — `UECD-SingleMap-Best`

`UECD-SingleMap-Best` is the only agent in this folder whose `train.log`
does not begin at step 0 — it is a fine-tune that resumes from a checkpoint
of an earlier run. The full training trace from step 0 is reconstructed by
chaining three segments, each shipped under `lineage/` as a self-contained
subdirectory so a reader never has to leave `UECD-SingleMap-Best/`:

```
Step      0M → 150M                    150M → 240M               240M → 360M
Segment   lineage/phase0-from-scratch  lineage/phase1-           UECD-SingleMap-Best
                                       pool-broadening           (this directory)
What      Training from scratch with   Phase 1 — broaden the    Phase 2 — harden
          phased learning rate /       opponent pool             against RAISocketAI
          entropy / shaped reward      (CoacAI/Mayari/Tiamat +
                                       RandomBiased/PORushes)
Original  PhasedRL-300M (same          FineTuneStronger-300M    BestRL-350M
run name  physical run as              (recovered from local     (still the on-disk
          UECD-SingleMap-Rushed,       backup; not regenerated   name of this folder
          but truncated here at        and not present in        in outputs/)
          step 150M to exclude the     outputs/ today)
          150M-300M continuation
          that becomes the rush-
          collapse demo)
```

Each `lineage/<phase>/` directory carries the medium-tier files of that
segment (`train.log`, `config.json`, `eval_results.csv`) **plus the exact
handoff checkpoint that ends the segment**:

- `lineage/phase0-from-scratch/checkpoint_150M.pt` — the 150M snapshot of
  the original `PhasedRL-300M` run that started Phase 1.
- `lineage/phase1-pool-broadening/checkpoint_240M.pt` — the 240M snapshot
  of `FineTuneStronger-300M` that started Phase 2 (copied as `checkpoint.pt`
  in the Phase 2 run dir at training time).

The `train.log` of `phase0-from-scratch` was truncated at the first line
whose `step=` is at or past the resume cutoff (`150,028,584`) so the file
contains only the lineage segment and not the 150M-300M continuation.
`eval_results.csv` was filtered to rows with `global_step <= 150,028,584`
for the same reason. `config.json` is unchanged — it describes the
`PhasedRL-300M` run as launched, before any fork.

To rebuild the full training history from step 0:

1. Read `lineage/phase0-from-scratch/train.log` (covers 0 → 150M).
2. Read `lineage/phase1-pool-broadening/train.log` (covers 150M → 240M).
3. Read `UECD-SingleMap-Best/train.log` (covers 240M → 360M).

`UECD-BC-PPO` is the second resume in this folder, but its lineage is
trivial: it warm-starts from `UECD-BC/agent.pt` at PPO step 0, then
runs PPO for 100M steps. There is no intermediate phase to ship.
The BC training trajectory that produced `UECD-BC/agent.pt` is not
preserved (the BC script writes no train.log or tfevents) — only the
resulting model and its evaluation under [`../bc_baseline/`](../bc_baseline/)
exist.

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
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --nb_games 100
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

## TensorBoard archives

The `events.out.tfevents.*` log of every agent in this folder lives as
release assets on the [`tfevents-agent-archive`](https://github.com/mathisdelsart/microrts-drl-uecd/releases/tag/tfevents-agent-archive)
tag. Each asset is named `<agent-dir-name>_<step-range>.tfevents` so a
file lines up with its data/agents/ subdirectory at a glance.

| Asset | Size | Agent |
|---|---:|---|
| `UECD-SingleMap-Best_0-360M.tfevents`     | 1.2 GB | [`UECD-SingleMap-Best/`](UECD-SingleMap-Best/) — three-phase fine-tune log (phase 0 + 1 + 2 cat-merged into one continuous TFRecord stream) |
| `UECD-SingleMap-Rushed_0-300M.tfevents`   | 704 MB | [`UECD-SingleMap-Rushed/`](UECD-SingleMap-Rushed/) |
| `UECD-MultiMap-Best_0-330M.tfevents`      | 711 MB | [`UECD-MultiMap-Best/`](UECD-MultiMap-Best/) |
| `GridNet-SingleMap_0-300M.tfevents`       | 588 MB | [`GridNet-SingleMap/`](GridNet-SingleMap/) |
| `UECD-MultiMap_0-200M.tfevents`           | 580 MB | [`UECD-MultiMap/`](UECD-MultiMap/) |
| `UECD-BC-PPO_0-100M.tfevents`             | 229 MB | [`UECD-BC-PPO/`](UECD-BC-PPO/) |
| `UECD-SingleMap-TopFeats_0-100M.tfevents` | 207 MB | [`UECD-SingleMap-TopFeats/`](UECD-SingleMap-TopFeats/) |
| `UECD-SingleMap-AllFeats_0-100M.tfevents` | 206 MB | [`UECD-SingleMap-AllFeats/`](UECD-SingleMap-AllFeats/) |

`UECD-BC` has no tfevents entry — the BC training script logs to stdout
only, no TensorBoard events are written.

### Download one agent's tfevents

```bash
gh release download tfevents-agent-archive -p UECD-SingleMap-Best_0-360M.tfevents
tensorboard --logdir .
```

### Download all of them

```bash
gh release download tfevents-agent-archive --dir tfevents_archive
tensorboard --logdir tfevents_archive
```

The release grows incrementally: when new agents are triaged in (the
`outputs/runs/arch_ablation/` and `outputs/runs/feat_ablation/` triage is
still pending), their tfevents will be appended with `gh release upload
tfevents-agent-archive <new files>`. The tag does **not** change, no
existing URL breaks.

## See also

- 📊 **Headline tournament results** using these agents: [`../tournaments/`](../tournaments/)
- 🎯 **Generalisation probes** of `UECD-SingleMap-Best`: [`../generalization_probes/`](../generalization_probes/)
- 🎓 **BC teacher dataset** behind `UECD-BC` / `UECD-BC-PPO`: [`../bc_training/`](../bc_training/)
- 🧪 **BC-only WR baseline** (the 78% line of the BC+VF→PPO figure): [`../bc_baseline/`](../bc_baseline/)
- 🎬 **Showcase recordings** of `UECD-SingleMap-Best` vs the field:
  [`../recordings/`](../recordings/)
