# `microrts_agent/analysis/`

Post-hoc analysis of training runs. Reads from a run directory
(`outputs/runs/<name>/` or `data/agents/<name>/`) and writes PDF plots,
audit summaries, or parameter tables. No JVM, no training, just
inspection.

## Files

| File | Role |
|---|---|
| [`__main__.py`](__main__.py) | Sub-subcommand dispatcher: `microrts-agent analysis {metrics\|audit\|params}`. |
| [`metrics.py`](metrics.py) | `microrts-agent analysis metrics`: parses `train.log` and TensorBoard events, generates PDF plots (loss curves, eval win-rates, opponent sampling, ...). |
| [`audit.py`](audit.py) | `microrts-agent analysis audit`: textual audit of a run (config, key TB scalars at fixed milestones, sanity checks on the eval CSV). |
| [`params.py`](params.py) | `microrts-agent analysis params`: instantiates each ablation feature on top of the UECD baseline and reports parameter count + delta. |

## Output paths

`metrics` writes PDFs under:
- `<run_dir>/analysis/` for runs that live under `outputs/runs/`.
- `outputs/analysis/<exp-name>/` for runs that live under `data/`
  (because `data/` is curated by hand and must not be touched by
  automated tools).

`audit` and `params` print to stdout only.

## Usage examples

```bash
# Metrics PDFs for one of the shipped agents
microrts-agent analysis metrics --run-dir data/agents/UECD-SingleMap-Best

# Textual audit of a freshly-trained run
microrts-agent analysis audit --run-dir outputs/runs/my_experiment_s1

# Parameter-count table (each feature on top of the UECD baseline)
microrts-agent analysis params
```

The notebook [`../../examples/06_analysis.ipynb`](../../examples/06_analysis.ipynb)
walks through all three sub-subcommands end-to-end.

## See also

- Run layout: see [`../training/README.md`](../training/README.md).
- Figure scripts that generate the dissertation plots (read the same
  `train.log` / `eval_results.csv` files):
  [`../../dissertation/figs/figs-python/`](../../dissertation/figs/figs-python/).
