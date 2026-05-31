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

## How to run

Every notebook starts with a markdown cell saying "Prereq:
[`00_navigate.ipynb`](00_navigate.ipynb) must be green": the sanity check
(Java + bridge.jar + `microrts-agent` on PATH) is centralised there, and
01-06 dive straight into the task.

Before launching Jupyter, run one of:

```bash
bash setup/local.sh && conda activate microrts_agent     # macOS / Linux dev
bash setup/cluster.sh && source cluster_venv/bin/activate # CECI HPC
```

Then `jupyter lab` from the repo root so the relative paths used in the
notebooks (`../setup/...`, `data/agents/...`, etc.) resolve correctly.

## Where outputs land

Anything the notebooks produce (trained agents, BC checkpoints, mini
tournament results) goes to `outputs/runs/` or `outputs/tournaments/`,
both gitignored. Shipped artefacts under
[`../data/`](../data/) are the curated, hand-picked snapshots.
