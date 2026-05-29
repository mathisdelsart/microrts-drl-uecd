# RAISocketAI

IEEE-CoG MicroRTS competition winner by Scott Goodfriend (2023: PPO, 2024: BC+PPO).

## What's in the wheel

`rl_algo_impls-0.2.1-py3-none-any.whl` contains all pretrained PyTorch models and the `rai_microrts` CLI entry point. Three model sets are available:

| Model set | Description |
|-----------|-------------|
| `ppo`     | Pure PPO (2023 competition winner) |
| `bc`      | Behavioral Cloning only |
| `bcppo`   | BC + PPO fine-tuning (2024 winner, recommended) |

Each set includes general models (16x16, 32x32, 64x64) and map-specific models.

## Installation

```bash
conda create -n microrts39 python=3.9 -y
conda activate microrts39
pip install torch
pip install rl_algo_impls-0.2.1-py3-none-any.whl
```

Verify: `rai_microrts --help`

## How it works

The JAR (`RAISocketAI.jar` or `RAIBCPPOAI.jar`) spawns a Python process running `rai_microrts`, which loads the PyTorch models and communicates via socket. The agent benchmarks each model at startup and selects the largest one that fits within the time budget.

## Usage in tournaments

Activate the conda environment **before** starting the tournament:

```bash
conda activate microrts39
python microrts_agent/run_tournament.py -c microrts_agent/tournaments/configs/default.json
```
