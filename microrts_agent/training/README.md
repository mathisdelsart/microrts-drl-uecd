# `microrts_agent/training/`

PPO training infrastructure. Every module is hot in
[`../train.py`](../train.py)'s main loop, and every shipped agent under
[`../../data/agents/`](../../data/agents/) was produced by this code.

## Files

| File | Role |
|---|---|
| [`cli.py`](cli.py) | CLI argument definitions, parsing, validation. Single source of truth for every `microrts-agent train ...` flag (~80 flags total). |
| [`config.py`](config.py) | Run configuration serialisation + banner display. Writes `config.json` next to the checkpoint. |
| [`setup.py`](setup.py) | Eval env setup, GAE dispatch, map-pool maybe-switch logic, dynamic-reset helpers. |
| [`checkpoint.py`](checkpoint.py) | Checkpoint save/load, device setup, TensorBoard init, `Tee` log redirect (line-buffered, no fsync stampede). |
| [`opponents.py`](opponents.py) | Opponent selection + configuration; resolves opponent strings into Java bot factories. |
| [`scheduling.py`](scheduling.py) | `OpponentTracker` (adaptive sampling weights), `CheckpointPool` (PFSP pool), reward annealing schedules. |
| [`selfplay.py`](selfplay.py) | `SelfPlayManager`: opponent-model loading, pool save/load, side alternation, PFSP. |
| [`ppo.py`](ppo.py) | GAE computation + the PPO policy-gradient update. The BC-teacher KL is computed here (single-sample importance-weighted score). |
| [`auxiliary.py`](auxiliary.py) | Auxiliary task heads + loss computation (opponent-modeling, unit-count, spatial, contrastive). |
| [`eval.py`](eval.py) | In-training evaluation against the bot pool. Writes `eval_results.csv`. |
| [`logging.py`](logging.py) | TensorBoard logging helpers (curves, histograms, episode metrics). |

## How they connect

```
train.py
  -> cli.parse_args                     (flags, defaults, validation)
  -> config.save_run_config             (writes config.json)
  -> setup.{build_envs, setup_eval}     (vec envs + eval pool)
  -> training loop:
       rollout         -> envs.step + policy forward
       advantages      -> ppo.compute_gae    (or setup.compute_gae for HL-Gauss)
       update          -> ppo.ppo_update
       auxiliary loss  -> auxiliary.compute_aux_loss   (if --aux-tasks)
       eval            -> eval.run_eval                (every --eval-interval)
       selfplay        -> selfplay.maybe_save / swap
       checkpoint      -> checkpoint.save              (every --save-interval)
```

Each module is import-safe (no JVM start), so `from microrts_agent.training
import ppo` is cheap. The JVM is only started when an env is built in
`setup.build_envs`.

## Outputs

Every training run writes under `outputs/runs/<exp-name>_s<seed>/`:

| File | Content |
|---|---|
| `config.json` | The full resolved config (every CLI flag, every default). |
| `agent.pt` | The latest policy + optimizer state. |
| `checkpoint.pt` | Same, suitable for `--resume`. |
| `eval_results.csv` | In-training eval rows (one per `--eval-interval`). |
| `train.log` | Tee'd stdout: banner + per-step `step= rollout= update=` lines + eval rows. |
| `events.out.tfevents.*` | TensorBoard events. |
| `pool/` | (Self-play) saved opponent snapshots. |

The 9 shipped agents at [`../../data/agents/`](../../data/agents/) are
manual snapshots of selected `outputs/runs/` directories.

## Notes

- `cli.py` is the **only** module that touches argparse. Adding a new
  flag means adding it there + threading the resolved value through
  `train.py`'s call sites; do not add ad-hoc argparse calls elsewhere.
- `Tee` (in [`checkpoint.py`](checkpoint.py)) wraps stdout for the
  duration of the training loop. It is line-buffered and proxies
  `encoding`/`fileno`/`isatty` to the underlying stdout so that
  libraries that introspect `sys.stdout` (`moviepy`, `tqdm`) still
  behave.
- The training loop is wrapped in `try/finally` (in `train.py`) so that
  an exception in the loop never leaves the log truncated.
