# `examples/`

Numbered Jupyter notebooks that walk through every CLI subcommand of
`microrts-agent`. Designed to be read + run **in order**: 00 is the
entry point (install + sanity check + package tour), 01-06 each focus
on one subcommand and assume the install is good.

## Files

| File | Subject | Subcommand exercised |
|---|---|---|
| [`00_navigate.ipynb`](00_navigate.ipynb) | Install + sanity check + CLI tour + repo layout | _(none, intro)_ |
| [`01_evaluate.ipynb`](01_evaluate.ipynb) | Load `UECD-SingleMap-Best` + play 1 game vs `RandomBiasedAI` + 1 game vs `CoacAI` | `evaluate` |
| [`02_train.ipynb`](02_train.ipynb) | Train a small PPO agent from scratch (1 M steps, GridNet), inspect outputs | `train` |
| [`03_tournament.ipynb`](03_tournament.ipynb) | Write a config (1 RL + 2 bots), run, parse CSV, generate PDF visualisations | `tournament {run,parse,viz,analyze}` |
| [`04_bc.ipynb`](04_bc.ipynb) | Tour the BC teacher dataset + 3-epoch BC train + evaluate | `bc {generate,train}` |
| [`05_bench.ipynb`](05_bench.ipynb) | Self-play decision-time benchmark + head-to-head benchmark | `bench {inference,head2head}` |
| [`06_analysis.ipynb`](06_analysis.ipynb) | PDF plots + audit + param-count table + inline train.log curves | `analysis {metrics,audit,params}` |

## Prerequisites

Every notebook starts with a markdown cell saying "Prereq:
[`00_navigate.ipynb`](00_navigate.ipynb) must be green". This is
intentional: the sanity check (Java + bridge.jar + `microrts-agent` on
PATH) is centralised in notebook 00, and 01-06 dive straight into the
task.

Before launching Jupyter, run one of:

```bash
bash setup/local.sh && conda activate microrts_agent     # macOS / Linux dev
bash setup/cluster.sh && source cluster_venv/bin/activate # CECI HPC
```

Then `jupyter lab` from the repo root so the relative paths used in the
notebooks (`../setup/...`, `data/agents/...`, etc.) resolve correctly.

## Conventions

All seven notebooks share the same style:

- **Canonical CLI**: `microrts-agent <cmd>` everywhere (the
  `[project.scripts]` console-script alias). Never `python -m
  microrts_agent`.
- **Canonical flags** match the actual argparser: `--max-steps`
  (hyphen), `--nb_games` (underscore), `--load-model`, etc.
- **Canonical paths**: shipped agents are at `data/agents/<name>` with
  no `_sN` seed suffix (only ablation runs under `data/ablation/` have
  it).
- **0 em-dash, 0 emoji** in any markdown or code cell.
- **`subprocess.run(cmd, cwd=str(PROJECT_ROOT), capture_output=True,
  text=True, timeout=...)`** for every CLI invocation; the result is
  printed via `print(result.stdout[-1500:])` to keep the cell output
  bounded.
- Each notebook ends with a "Next steps" section pointing to related
  notebooks and the SLURM scripts under
  [`../experiments/`](../experiments/) for production-scale
  reproduction.

## What is intentionally NOT here

- **No Docker walkthrough notebook**: Docker is best driven from the
  terminal (the notebook would just shell out to `docker build` / `docker
  run`). Use the [`Dockerfile`](../Dockerfile) header recipes or
  [`bash setup/docker.sh`](../setup/docker.sh) instead.
- **No thesis-scale reproduction**: each notebook uses tiny budgets
  (1 game eval, 1 M-step train, 3-AI tournament) so a single run fits
  in minutes. The SLURM scripts under
  [`../experiments/`](../experiments/) reproduce the thesis numbers.
- **No saved cell outputs**: notebooks ship with outputs cleared by
  convention. The CI smoke tests don't execute notebooks; running them
  manually after a fresh `setup/local.sh` is the verification path.

## Outputs land in `outputs/`

Anything the notebooks produce (trained agents, BC checkpoints, mini
tournament results) goes to `outputs/runs/` or `outputs/tournaments/`,
both gitignored. Curated shipped artefacts (the 9 agents in
[`data/agents/`](../data/agents/), the 19-AI tournament in
[`data/tournaments/single_map/`](../data/tournaments/single_map/), etc.)
are never modified by the notebooks.
