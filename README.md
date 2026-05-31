<div align="center">

# MicroRTS Deep-RL Agent

### A competitive real-time-strategy agent via deep reinforcement learning

[![CI](https://github.com/mathisdelsart/microrts-drl-uecd/actions/workflows/ci.yml/badge.svg)](https://github.com/mathisdelsart/microrts-drl-uecd/actions/workflows/ci.yml)
[![Python](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-green.svg)](https://arxiv.org/abs/1707.06347)
[![Built on MicroRTS](https://img.shields.io/badge/Built%20on-MicroRTS-orange.svg)](https://github.com/santiontanon/microrts)
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Master's thesis (UCLouvain): building a MicroRTS agent that targets and surpasses the competition winner RAISocketAI.*

</div>

---

## About

This project trains a deep-RL agent for [MicroRTS](https://github.com/santiontanon/microrts), a
minimalist real-time-strategy game used as a research benchmark. The agent is optimised with PPO
over a hand-written **Java↔Python bridge**, a curriculum of scripted opponents and self-play, and a
family of convolutional / U-Net / entity-attention policies. The goal is to beat **RAISocketAI**,
the winner of the IEEE-CoG MicroRTS competition.

- 📄 **Dissertation:** [`dissertation/dissertation.pdf`](dissertation/dissertation.pdf)
- 📝 **Short paper (CoG 2026, under review):** [`cog-2026-paper/`](cog-2026-paper/)
- 🌐 **Supplementary site** (tournament results, game recordings, analyses): <https://mathisdelsart.github.io/microrts-drl-uecd-website/>

## Highlights

- **Architectures**: GridNet baseline, IMPALA-CNN, and U-Net policies with SE / CBAM attention,
  an entity Transformer, and a map-size-invariant critic (Spatial Pyramid Pooling).
- **Environment stack**: vectorised MicroRTS via JNI: standard + extended observations, action
  masking (incl. destination-aware filtering), self-play, multi-map training with Prioritised
  Level Replay, and 10 shaped reward components.
- **Training**: PPO with dual/triple value heads, PopArt normalisation, HL-Gauss value
  classification, auxiliary tasks, adaptive opponent curricula, and a behaviour-cloning warm-start
  (BC+VF then PPO with a KL teacher penalty).
- **Evaluation**: round-robin tournament engine against the competition bots, game-theoretic
  robustness metrics, generalisation probes, and inference benchmarks.

## Repository structure

```
microrts-drl-uecd/
├── microrts_agent/               # Importable package, unified CLI: microrts-agent <command>
│   ├── envs/                     # Vectorised MicroRTS envs (base, rl, padded, bot) + factory
│   ├── architectures/            # Policies (gridnet, impala, unet, *_entity, *_cbam) + features/ + factory
│   ├── training/                 # PPO core, schedules, self-play, in-training eval, logging
│   ├── tournament/               # Round-robin engine, parser/visualizer, ranking/ (game-theory metrics), plots/
│   ├── registries/               # Bot registry (ai.py) + map registry (maps.py)
│   ├── wrappers/                 # Composable VecEnv wrappers (frame-stack, symmetry, ...) + factory
│   ├── obs_adapter.py            # ObsAdapter for agent-vs-agent evaluation
│   ├── paths.py                  # Canonical output paths
│   ├── bc/                       # Behaviour cloning:    microrts-agent bc {train|generate}
│   ├── bench/                    # Inference benchmarks: microrts-agent bench {inference|head2head}
│   ├── analysis/                 # Run analysis tools:   microrts-agent analysis {metrics|audit|params}
│   ├── microrts/                 # Java engine + Java<->Python bridge
│   │   ├── microrts.jar          # MicroRTS engine (vendored)
│   │   ├── lib/{bots/*.jar, bridge.jar}
│   │   ├── src/                  # Bridge sources (JNI, reward functions, game wrapper)
│   │   └── build_bridge.sh       # Recompiles src/ -> lib/bridge.jar
│   ├── bots/                     # Vendored competition bots (sources/builds + RAISocketAI wheel)
│   ├── tournament_configs/       # Tournament setup JSON files
│   └── train.py  evaluate.py                      # subcommand implementations dispatched by __main__.py
├── data/                         # Curated artefacts shipped with the repo (not regenerated on clone)
│   ├── recordings/               # 36 showcase game clips of UECD-Best vs the field (also served from the supplementary site)
│   ├── tournaments/              # Headline tournament results: single_map/ + multi_map/ (CSV + parsed JSON + PDFs)
│   ├── generalization_probes/    # Generalisation probes (UECD-Best on non-training maps)
│   ├── BC/                       # BC teacher dataset under training/ + 78% BC-only proof under baseline/
│   └── agents/                   # Trained agents in medium form (4 single-map UECD + GridNet baseline + 2 multi-map + UECD-BC + UECD-BC-PPO = 9 agents) with agent.pt + checkpoint.pt + config + eval + log
├── dissertation/                 # LaTeX thesis, figures, figure generators (figs/figs-python/), compiled PDF
├── cog-2026-paper/               # CoG 2026 short-paper submission
├── experiments/                  # SLURM jobs: single-map/ multi-map/ BC/ eval/ tournament/ ablation/ + shared _setup_env.sh
├── setup/                        # env setup scripts (local.sh, cluster.sh, docker.sh)
└── LICENSE  CITATION.cff  ACKNOWLEDGMENTS.md  CREDITS.md  ruff.toml  pyproject.toml  (CONTRIBUTING.md / CODE_OF_CONDUCT.md live under .github/)
```

Python dependencies are declared once in `pyproject.toml` (core stack +
`[tournament]` / `[dev]` extras); the setup scripts install them via
`pip install -e ".[dev,tournament]"`.

Generated runs, recordings, and tournament CSVs all land under `outputs/`
(git-ignored). The subset that backs the dissertation and the CoG paper is
curated under `data/`.

## Quick start

**Requirements:** conda, a JDK 17 (for the bridge + JPype), and CUDA for GPU training.

### Optional: RAISocketAI competition bot

`setup/local.sh` automatically downloads the RAISocketAI bot wheel
(`rl_algo_impls v0.2.1`, ~225 MB) from this repo's
[release assets](https://github.com/mathisdelsart/microrts-drl-uecd/releases/tag/assets-rai-v0.2.1)
on first run, verifies its SHA-256, and installs it with `--no-deps`. Set
`SKIP_RAISOCKETAI=1 bash setup/local.sh` to skip. Every other feature
(training, evaluation against the other competition bots, tournaments that
don't include RAISocketAI) still works.

```bash
# 1. Set up the environment (creates the `microrts_agent` conda env and
#    rebuilds the Java<->Python bridge from source)
bash setup/local.sh
conda activate microrts_agent

# 2. (Re)build the bridge manually if you edit microrts_agent/microrts/src/
bash microrts_agent/microrts/build_bridge.sh

# 3. Train  (list every command with:  microrts-agent --help)
microrts-agent train --total-timesteps 1000000
microrts-agent train --help        # all flags

# 4. Monitor
tensorboard --logdir outputs/runs/

# 5. Evaluate a trained agent against a bot (any of the shipped agents under data/agents/, or your own outputs/runs/<run>)
microrts-agent evaluate --agent data/agents/UECD-SingleMap-Best --opponent CoacAI

# 6. Run a tournament
microrts-agent tournament --help
```

On the HPC cluster, use `setup/cluster.sh` and the job scripts under
`experiments/` (grouped by single-map/ multi-map/ BC/ eval/ tournament/ ablation/, with a shared `_setup_env.sh` preamble each script sources).

### Docker

Two images are shipped. The default `Dockerfile` is CPU-only (~2.7 GB with
the 9 shipped agents baked in) and sufficient for evaluation; `Dockerfile.gpu`
adds CUDA 12.9 + cuDNN (~6.5 GB) for training. Pass
`--build-arg SKIP_RAISOCKETAI=1` to either build to skip the ~225 MB
RAISocketAI wheel; see the Dockerfile header for `docker run -v` examples
that mount the excluded `data/` subtrees (ablation, BC, tournaments, ...)
at runtime when needed.

```bash
# CPU image (Python 3.10 + Java 17 + CPU-only torch + prebuilt JNI bridge)
docker build -t microrts-drl-uecd .
docker run --rm microrts-drl-uecd                              # prints CLI help
docker run --rm microrts-drl-uecd evaluate \
    --agent data/agents/UECD-SingleMap-Best \
    --opponent CoacAI --nb_games 10 --max_steps 1000

# GPU image (CUDA 12.9, requires nvidia-container-toolkit + --gpus all)
docker build -f Dockerfile.gpu -t microrts-drl-uecd:gpu .
docker run --rm --gpus all microrts-drl-uecd:gpu train \
    --exp-name docker-smoke --total-timesteps 100000
```

The CPU image stays the default for evaluation; SLURM-style cluster training
still uses the conda path above.

### Reproducible installs (lock file)

`requirements-lock.txt` pins every transitive dependency with SHA-256 hashes
(generated via `uv pip compile --generate-hashes`). For a byte-reproducible
install:

```bash
pip install -r requirements-lock.txt --require-hashes
pip install -e . --no-deps
```

The plain `pip install -e ".[dev]"` path stays the recommended one for day-to-day
development; the lock file is for archival reproducibility (the CoG paper, future
researchers reproducing thesis results).

### Demo notebook

`examples/load_agent.ipynb` walks through loading `UECD-SingleMap-Best`,
printing its training config, and playing one game against `RandomBiasedAI`
from inside Jupyter. CPU-only, ~30 seconds end-to-end on a laptop.

## Results

Full experiments and analysis are in the dissertation (architecture and feature ablations,
single-map and multi-map agents, generalisation probes, BC warm-start, and head-to-head
tournaments vs the competition bots). See [`dissertation/dissertation.pdf`](dissertation/dissertation.pdf)
and the [supplementary site](https://mathisdelsart.github.io/microrts-drl-uecd-website/).

## Development

Conventions (branch naming, Conventional Commits, PR workflow, squash-merge) are in
[`.github/CONTRIBUTING.md`](.github/CONTRIBUTING.md). Code is linted and formatted with
[ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`), enforced in CI.

```bash
uvx ruff@0.15.14 check .      # lint
uvx ruff@0.15.14 format .     # format
```

## Citation

If you use this code or any of the shipped artefacts in academic work,
please cite the project ([`CITATION.cff`](CITATION.cff) is the source of truth):

```bibtex
@software{delsart_microrts_drl_uecd_2026,
  author  = {Delsart, Mathis},
  title   = {{Deep Reinforcement Learning for Competitive Agents in MicroRTS: Architecture, Training, and Tournament Evaluation}},
  year    = {2026},
  version = {0.1.0},
  url     = {https://github.com/mathisdelsart/microrts-drl-uecd},
  note    = {Master's thesis, UCLouvain},
}
```

## License

Released under the [MIT License](LICENSE) © 2026 Mathis Delsart.

## Author

**Mathis Delsart**, Master's thesis, UCLouvain.

## Acknowledgments

Every experiment ran on the HPC clusters of the **CÉCI** (Consortium des
Équipements de Calcul Intensif). See [`ACKNOWLEDGMENTS.md`](ACKNOWLEDGMENTS.md)
for the full acknowledgment.
