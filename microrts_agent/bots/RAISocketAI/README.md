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

The wheel is too large for git (~225 MB) and is hosted as a [GitHub Release
asset](https://github.com/mathisdelsart/microrts-drl-uecd/releases/tag/assets-rai-v0.2.1).
`setup/local.sh` and `setup/cluster.sh` download it on first run, verify its
SHA-256, and install it into the `microrts_agent` env with `--no-deps`. Skip
with `SKIP_RAISOCKETAI=1` if you do not need to evaluate against this bot.

Verify: `rai_microrts --help`.

## How it works

The JAR (`RAISocketAI.jar` or `RAIBCPPOAI.jar`) spawns a Python process running `rai_microrts`, which loads the PyTorch models and communicates via socket. The agent benchmarks each model at startup and selects the largest one that fits within the time budget.

## Usage in tournaments

Activate the project env (`microrts_agent`) before starting the tournament; the
wheel installs into it directly, so no separate Python is needed.

```bash
conda activate microrts_agent
python -m microrts_agent tournament run single_map
```
