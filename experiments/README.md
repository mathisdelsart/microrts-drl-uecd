# `experiments/`

SLURM batch scripts that drive every reproducible thesis run on the
CECI HPC clusters. Each script is named after the artefact it
produces; if a file under [`../data/agents/`](../data/agents/) or
[`../data/ablation/`](../data/ablation/) was trained on the cluster,
its driver lives here.

## Layout

```
experiments/
├── _setup_env.sh           # shared module-load + venv-activate preamble
├── single-map/             # 16x16 PPO training (UECD family + GridNet baseline)
├── multi-map/              # 5-map PPO training (UECD-MultiMap, UECD-MultiMap-Best)
├── BC/                     # behaviour cloning: dataset generation + 2 BC trainings
├── ablation/               # architecture + feature ablations (training + eval)
├── eval/                   # generalisation probes + rush-collapse eval
└── tournament/             # round-robin tournament drivers (single + multi map)
```

## Scripts

### `_setup_env.sh` (shared)

Sourced by every other script. Loads Python (3.10 or 3.11), Java
(17 or 11), CUDA (12.1 or 11.8) from the cluster module system,
activates `$HOME/microrts-drl-uecd/cluster_venv/`, exports
`PYTHONUNBUFFERED=1`, and `cd`s to the repo root. Missing Python or
Java aborts the job; missing CUDA prints a warning (so CPU-only smoke
runs still work).

The repo root is hardcoded as `$HOME/microrts-drl-uecd`. Edit that line
if you clone the repo elsewhere on the cluster.

### `single-map/`

6 scripts driving the UECD-SingleMap family (basesWorkers16x16A):

| File | Output (`data/agents/<name>`) | Steps |
|---|---|---|
| [`single-map/train_GridNet-SingleMap.slurm`](single-map/train_GridNet-SingleMap.slurm) | `GridNet-SingleMap` | 300M |
| [`single-map/train_UECD-SingleMap-AllFeats.slurm`](single-map/train_UECD-SingleMap-AllFeats.slurm) | `UECD-SingleMap-AllFeats` | 100M |
| [`single-map/train_UECD-SingleMap-TopFeats.slurm`](single-map/train_UECD-SingleMap-TopFeats.slurm) | `UECD-SingleMap-TopFeats` | 100M |
| [`single-map/train_UECD-SingleMap-Rushed.slurm`](single-map/train_UECD-SingleMap-Rushed.slurm) | `UECD-SingleMap-Rushed` (phase 0) | 150M |
| [`single-map/train_UECD-SingleMap-Best_phase1.slurm`](single-map/train_UECD-SingleMap-Best_phase1.slurm) | `UECD-SingleMap-Best/lineage/phase_1` | 150M -> 300M |
| [`single-map/train_UECD-SingleMap-Best_phase2.slurm`](single-map/train_UECD-SingleMap-Best_phase2.slurm) | `UECD-SingleMap-Best` (final, 350M) | 300M -> 350M |

`UECD-SingleMap-Best` is a 3-phase agent: Rushed (0-150M)
-> phase 1 (150M-300M, self-play diversification) -> phase 2
(300M-350M, RAISocketAI fine-tune). The three phase logs are merged in
[`../data/agents/UECD-SingleMap-Best/lineage/`](../data/agents/UECD-SingleMap-Best/lineage/).

### `multi-map/`

| File | Output |
|---|---|
| [`multi-map/train_UECD-MultiMap.slurm`](multi-map/train_UECD-MultiMap.slurm) | `data/agents/UECD-MultiMap` (200M) |
| [`multi-map/train_UECD-MultiMap-Best.slurm`](multi-map/train_UECD-MultiMap-Best.slurm) | `data/agents/UECD-MultiMap-Best` (325M) |

### `BC/`

| File | Output |
|---|---|
| [`BC/generate_BC_dataset.slurm`](BC/generate_BC_dataset.slurm) | The 6 NPZ chunks under `data/BC/training/`. |
| [`BC/train_UECD-BC.slurm`](BC/train_UECD-BC.slurm) | `data/agents/UECD-BC` (BC-only). |
| [`BC/train_UECD-BC-PPO.slurm`](BC/train_UECD-BC-PPO.slurm) | `data/agents/UECD-BC-PPO` (BC warm-start + PPO). |

### `ablation/`

| File | Output |
|---|---|
| [`ablation/train_arch_ablation.slurm`](ablation/train_arch_ablation.slurm) | 21 runs under `data/ablation/arch/agent/` (7 archs x 3 seeds, 100M each). |
| [`ablation/train_feat_ablation.slurm`](ablation/train_feat_ablation.slurm) | 64 runs under `data/ablation/feat/agent/`. |
| [`ablation/evaluate_arch_ablation.slurm`](ablation/evaluate_arch_ablation.slurm) | 21 eval grids -> `data/ablation/arch/eval/`. |
| [`ablation/evaluate_feats_ablation.slurm`](ablation/evaluate_feats_ablation.slurm) | Eval grids -> `data/ablation/feat/eval/`. |

### `eval/`

| File | Output |
|---|---|
| [`eval/eval_generalization_probes.slurm`](eval/eval_generalization_probes.slurm) | `data/generalization_probes/` (UECD-SingleMap-Best vs the field on non-training maps). |
| [`eval/eval_rush_collapse.slurm`](eval/eval_rush_collapse.slurm) | `data/rush_collapse/` (PhasedRL 150M vs 300M head-to-head). |

### `tournament/`

| File | Output |
|---|---|
| [`tournament/tournament_single_map.slurm`](tournament/tournament_single_map.slurm) | `data/tournaments/single_map/` (10 AI round-robin, 16x16). |
| [`tournament/tournament_multi_map.slurm`](tournament/tournament_multi_map.slurm) | `data/tournaments/multi_map/` (round-robin across 5 maps). |

## How to launch on CECI

```bash
# From the cluster, repo cloned at $HOME/microrts-drl-uecd
cd $HOME/microrts-drl-uecd

# Make sure the venv is built (one-shot)
bash setup/cluster.sh
source cluster_venv/bin/activate

# Submit
sbatch experiments/single-map/train_UECD-SingleMap-Best_phase1.slurm
sbatch experiments/ablation/train_arch_ablation.slurm

# With a custom seed (every training script honours SEED=N)
SEED=2 sbatch experiments/single-map/train_UECD-SingleMap-Rushed.slurm
```

`squeue --me`, `sacct -X --format=JobID,JobName,State,Elapsed`, etc.,
follow standard SLURM conventions. Logs land in
`slurm-<job>-<id>.{out,err}` next to the submitting directory.

## Conventions

- **`--exp-name`** values inside the scripts match the directory name
  shipped under [`../data/agents/`](../data/agents/) /
  [`../data/ablation/`](../data/ablation/). Re-running a script
  produces `outputs/runs/<name>_s<SEED>/` that maps 1:1 to the shipped
  artefact.
- The repo's CLI is invoked as **`microrts-agent <cmd>`** (the
  console-script entry point), not the longer `python -m
  microrts_agent <cmd>` form.
- Module-load names vary across CECI sites. The fallback chain in
  [`_setup_env.sh`](_setup_env.sh) handles Lyra, Manneback and
  Hercules as of 2026; new sites may need a third fallback.
- `SLURM partition`, `time`, `mem` and `cpus-per-task` were tuned for
  Lyra. Adjust the `#SBATCH` header if the target cluster has
  different queue limits.
