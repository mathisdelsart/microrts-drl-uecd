# `microrts_agent/`

The importable Python package. Every CLI subcommand
(`microrts-agent <cmd> ...`) dispatches into this tree, and every
subdirectory documents its own role in a local README.

## Top-level files

| File | Role |
|---|---|
| [`__init__.py`](__init__.py) | Package init (imports surfaces for `microrts_agent.paths`, etc.). |
| [`__main__.py`](__main__.py) | The unified CLI dispatcher. Resolves `microrts-agent <cmd>` to the right module. |
| [`train.py`](train.py) | `microrts-agent train` implementation (PPO training loop). |
| [`evaluate.py`](evaluate.py) | `microrts-agent evaluate` implementation (agent-vs-bot, agent-vs-agent, optional `--record`). |
| [`obs_adapter.py`](obs_adapter.py) | Observation adapter used during agent-vs-agent (one side's policy reads the other side's obs shape). |
| [`paths.py`](paths.py) | Canonical paths constants (`PROJECT_ROOT`, `OUTPUTS_DIR`, `RUNS_DIR`, `TOURNAMENTS_DIR`, ...). Source of truth. |

## Subdirectories

| Subdir | What it is |
|---|---|
| [`architectures/`](architectures/) | NN policies (`gridnet`, `impala`, `unet_entity_cbam_deep`, ...) + features module + factory. |
| [`training/`](training/) | PPO core (CLI, GAE, opponent scheduling, self-play, aux tasks, eval, logging, checkpoint). |
| [`envs/`](envs/) | Vectorised MicroRTS envs (Python <-> Java via JPype/JNI) + factory. |
| [`wrappers/`](wrappers/) | VecEnv wrappers (frame-stack, reserved-obs, symmetry-augmentation, stats-recorder) + factory. |
| [`registries/`](registries/) | Bot registry (`AI_MAPPING`) and map registry (canonical map paths). |
| [`tournament/`](tournament/) | Round-robin engine (config, env pool, game loops) + ranking (game-theoretic metrics) + plot helpers. |
| [`tournament_configs/`](tournament_configs/) | The two shipped tournament configs (`single_map.json`, `multi_map.json`). |
| [`bc/`](bc/) | Behaviour-cloning: dataset generation (`generate.py`) + BC training (`train.py`). |
| [`bench/`](bench/) | Decision-time benchmarks: `inference.py`, `head_to_head.py`. |
| [`analysis/`](analysis/) | Training-run analysis: `metrics.py`, `audit.py`, `params.py`. |
| [`microrts/`](microrts/) | Vendored MicroRTS Java engine (`microrts.jar`, maps, vendored bot JARs) + our JNI bridge source. |
| [`bots/`](bots/) | Vendored 3rd-party competition bots (CoacAI, Mayari, UTS_Imass, Tiamat, TMA, MixedBot, ObiBotKenobi, StrategyTactics, RAISocketAI). Each has its own README. |

## CLI entry point

The package is invoked through the `microrts-agent` console script
(installed by `pip install -e .` via the `[project.scripts]` entry in
[`../pyproject.toml`](../pyproject.toml)). The dispatcher
[`__main__.py`](__main__.py) routes:

```
microrts-agent train         -> microrts_agent.train.main()
microrts-agent evaluate      -> microrts_agent.evaluate.main()
microrts-agent tournament    -> microrts_agent.tournament.__main__.main()
microrts-agent bc            -> microrts_agent.bc.__main__.main()
microrts-agent bench         -> microrts_agent.bench.__main__.main()
microrts-agent analysis      -> microrts_agent.analysis.__main__.main()
```

Each grouped command (`tournament`, `bc`, `bench`, `analysis`) further
dispatches to sub-subcommands (`tournament run|parse|viz|analyze`,
`bc train|generate`, `bench inference|head2head`,
`analysis metrics|audit|params`).

## Output layout

All commands that produce artefacts write under `outputs/`
(gitignored). Curated, shipped artefacts live under
[`../data/`](../data/) and are never overwritten by the CLI. The
constants in [`paths.py`](paths.py) are the only source of truth for
where things land.

## Style conventions

- The CLI is invoked as `microrts-agent <cmd>` everywhere
  (docstrings, READMEs, SLURM scripts, notebooks). The dotted form
  `python -m microrts_agent.<cmd>` works but is **not** the canonical
  form.
- Architectures, wrappers and envs are wired through their respective
  `factory.py` modules: no module under the package directly
  instantiates a sibling's class without going through the factory.
- The package is import-safe: importing `microrts_agent` does **not**
  start the JVM. The JVM is started lazily on first vec-env construction.
