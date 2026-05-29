<div align="center">

# MicroRTS Deep-RL Agent

### A competitive real-time-strategy agent via deep reinforcement learning

[![Python](https://img.shields.io/badge/Python-3.9-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![Algorithm](https://img.shields.io/badge/Algorithm-PPO-green.svg)]()
[![License](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

*Master's thesis (UCLouvain) — building a MicroRTS agent that targets and surpasses the competition winner RAISocketAI.*

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

- **Architectures** — GridNet baseline, IMPALA-CNN, and U-Net policies with SE / CBAM attention,
  an entity Transformer, and a map-size-invariant critic (Spatial Pyramid Pooling).
- **Environment stack** — vectorised MicroRTS via JNI: standard + extended observations, action
  masking (incl. destination-aware filtering), self-play, multi-map training with Prioritised
  Level Replay, and 10 shaped reward components.
- **Training** — PPO with dual/triple value heads, PopArt normalisation, HL-Gauss value
  classification, auxiliary tasks, adaptive opponent curricula, and a behaviour-cloning warm-start
  (BC+VF → PPO with a KL teacher penalty).
- **Evaluation** — round-robin tournament engine against the competition bots, game-theoretic
  robustness metrics, generalisation probes, and inference benchmarks.

## Repository structure

```
microrts-drl-uecd/
├── microrts_agent/               # Importable package — unified CLI: python -m microrts_agent <command>
│   ├── envs/                     # Vectorised MicroRTS envs (base, rl, padded, bot) + factory
│   ├── architectures/            # Policies (gridnet, impala, unet, *_entity, *_cbam) + features/ + factory
│   ├── training/                 # PPO core, schedules, self-play, in-training eval, logging
│   ├── tournament/               # Round-robin engine, parser/visualizer, ranking/ (game-theory metrics), plots/
│   ├── registries/               # Bot registry (ai.py) + map registry (maps.py)
│   ├── wrappers/                 # Composable VecEnv wrappers (frame-stack, symmetry, ...) + factory
│   ├── obs_adapter.py            # ObsAdapter for agent-vs-agent evaluation
│   ├── paths.py                  # Canonical output paths
│   ├── bc/                       # Behaviour cloning:    python -m microrts_agent bc {train|generate}
│   ├── bench/                    # Inference benchmarks: python -m microrts_agent bench {inference|head2head}
│   ├── analysis/                 # Run analysis tools:   python -m microrts_agent analysis {metrics|audit|params}
│   ├── microrts/                 # Java engine + Java↔Python bridge
│   │   ├── microrts.jar          # MicroRTS engine (vendored)
│   │   ├── lib/{bots/*.jar, bridge.jar}
│   │   ├── src/                  # Bridge sources (JNI, reward functions, game wrapper)
│   │   └── build_bridge.sh       # Recompiles src/ -> lib/bridge.jar
│   ├── bots/                     # Vendored competition bots (sources/builds + RAISocketAI wheel)
│   ├── tournament_configs/       # Tournament setup JSON files
│   ├── experiments/              # SLURM job scripts for the HPC runs
│   └── train.py  evaluate.py  run_tournament.py   # simple entry points
├── dissertation/                 # LaTeX thesis, figures, figure generators (figs/figs-python/), compiled PDF
├── cog-2026-paper/               # CoG 2026 short-paper submission
├── setup_local_env.sh  setup_cluster_env.sh
└── LICENSE  CONTRIBUTING.md  ruff.toml  pyproject.toml
```

Python dependencies are declared once in `pyproject.toml` (core stack +
`[tournament]` / `[dev]` extras); the setup scripts install them via
`pip install -e ".[tournament]"`.

Generated runs, recordings, and tournament CSVs live under `outputs/` (git-ignored).

## Quick start

**Requirements:** conda, a JDK 17 (for the bridge + JPype), and — for GPU training — CUDA.

```bash
# 1. Set up the environment (creates the `microrts_agent` conda env and
#    rebuilds the Java<->Python bridge from source)
bash setup_local_env.sh
conda activate microrts_agent

# 2. (Re)build the bridge manually if you edit microrts_agent/microrts/src/
bash microrts_agent/microrts/build_bridge.sh

# 3. Train  (list every command with:  python -m microrts_agent --help)
python -m microrts_agent train --total-timesteps 1000000
python -m microrts_agent train --help        # all flags

# 4. Monitor
tensorboard --logdir outputs/runs/

# 5. Evaluate a trained agent against a bot
python -m microrts_agent evaluate --agent outputs/runs/<run> --opponent CoacAI

# 6. Run a tournament
python -m microrts_agent tournament --help
```

On the HPC cluster, use `setup_cluster_env.sh` and the job scripts under
`microrts_agent/experiments/`.

## Results

Full experiments and analysis are in the dissertation (architecture and feature ablations,
single-map and multi-map agents, generalisation probes, BC warm-start, and head-to-head
tournaments vs the competition bots). See [`dissertation/dissertation.pdf`](dissertation/dissertation.pdf)
and the [supplementary site](https://mathisdelsart.github.io/microrts-drl-uecd-website/).

## Development

Conventions (branch naming, Conventional Commits, PR workflow, squash-merge) are in
[`CONTRIBUTING.md`](CONTRIBUTING.md). Code is linted and formatted with
[ruff](https://docs.astral.sh/ruff/) (config in `ruff.toml`), enforced in CI.

```bash
uvx ruff@0.15.14 check .      # lint
uvx ruff@0.15.14 format .     # format
```

## License

Released under the [MIT License](LICENSE) © 2026 Mathis Delsart.

## Author

**Mathis Delsart** — Master's thesis, UCLouvain.
