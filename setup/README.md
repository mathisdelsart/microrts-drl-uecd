# `setup/`

Install + provision scripts for the two contexts the project targets
(laptop dev and CECI HPC). Both share the same helpers in
[`_common.sh`](_common.sh): rebuild the JNI bridge, install Python deps,
fetch the RAISocketAI wheel, verify the install.

## Files

| File | When to use | What it does |
|---|---|---|
| [`local.sh`](local.sh) | Laptop dev (macOS or Linux) | Creates a conda env `microrts_agent` with Python 3.10, installs the project + `[dev,tournament]` extras, fetches the RAISocketAI wheel, activates pre-commit hooks. |
| [`cluster.sh`](cluster.sh) | HPC (CECI: Lyra, Manneback, Hercules, Dragon1) | Creates a venv at `cluster_venv/` (inside the repo), installs the project + extras, also bootstraps a separate Python 3.6 venv for the UTS_Imass bot's BL_JPS pathfinding. Uses the cu121 torch wheel by default (matches typical CECI GPU drivers). |
| [`_common.sh`](_common.sh) | sourced by the two above | Constants (wheel URL + SHA-256) + helpers (`check_java`, `build_bridge`, `install_python_deps`, `install_raisocketai_wheel`, `verify_install`, `install_pre_commit_hooks`). Never executed directly. |

## Quick-start by context

### Local laptop

```bash
bash setup/local.sh
conda activate microrts_agent
microrts-agent --help
```

The script refuses to run if another Python venv is already active in your
shell (e.g. a leftover `cluster_venv`) because `conda activate` does not
cleanly override an active venv; doing so silently installs into the wrong
Python and breaks the RAISocketAI install. The error message tells you to
`deactivate` first.

### CECI HPC cluster

```bash
# Load the right modules FIRST (the script doesn't load modules itself).
# Names vary across CECI sites; cross-check with `module avail`.
module load Java/17.0.6
module load Python/3.11.3-GCCcore-12.3.0     # 3.10 or 3.11; <3.12 (wheel constraint)
module load CUDA/12.1.1                       # only for GPU training

bash setup/cluster.sh
source cluster_venv/bin/activate
microrts-agent --help
```

The script prints a module-load cookbook at the very top (with a working
example for Lyra). Then it checks Java + Python version, builds the
JNI bridge, creates the venv, installs deps + the RAISocketAI wheel, and
also boostraps `uts_imass_env/` (Python 3.6) for the UTS_Imass bot.

## Override knobs

| Env var | Effect |
|---|---|
| `SKIP_RAISOCKETAI=1` | Skip fetching + installing the RAISocketAI wheel. The bot won't work as an opponent in `tournament` / `evaluate`, but every other feature does. Required if the compute node has no outbound network. |
| `TORCH_INDEX_URL=...` | Override the torch wheel index URL on `cluster.sh`. Default `cu121`. Use `cu118` for older drivers, or empty (`TORCH_INDEX_URL= bash setup/cluster.sh`) for PyPI's default wheel. |

## Sanity check after setup

```bash
microrts-agent --help                          # CLI tour
pytest tests/                                  # 124 smoke tests (~90s)
microrts-agent evaluate --agent data/agents/UECD-SingleMap-Best \
    --opponent CoacAI --maps maps/open_competition/basesWorkers16x16A.xml \
    --nb_games 1 --max-steps 4000
```

If those three pass, the install is solid. See
[`examples/00_navigate.ipynb`](../examples/00_navigate.ipynb) for a
notebook walkthrough of the install + CLI tour.

## Reproducible installs (lock file)

[`../requirements-lock.txt`](../requirements-lock.txt) pins every transitive
dependency with SHA-256 hashes (generated via `uv pip compile
--generate-hashes`). For a byte-reproducible install:

```bash
pip install -r requirements-lock.txt --require-hashes
pip install -e . --no-deps
```

The plain `pip install -e ".[dev,tournament]"` path (via `local.sh` /
`cluster.sh` above) stays the recommended one for day-to-day development;
the lock file is for archival reproducibility (the CoG paper, future
researchers reproducing thesis results).
