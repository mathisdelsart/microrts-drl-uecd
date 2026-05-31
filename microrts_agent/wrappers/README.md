# `microrts_agent/wrappers/`

VecEnv wrappers applied on top of the raw MicroRTS vec env. The canonical
stack is composed by [`factory.apply_env_wrappers()`](factory.py), which
every code path that builds an env goes through.

## Files

| File | What it does |
|---|---|
| [`factory.py`](factory.py) | `apply_env_wrappers(env, cfg)`: composes the canonical stack based on the run's config. |
| [`frame_stack.py`](frame_stack.py) | Frame stacking (last N observations concatenated along the channel axis). |
| [`reserved_obs.py`](reserved_obs.py) | Adds a binary "reserved" channel to the observation, used by the auxiliary opponent-modeling task. |
| [`stats_recorder.py`](stats_recorder.py) | Per-episode statistics (length, return, win/lose/draw) surfaced through `info["episode"]`. |
| [`symmetry_augmentation.py`](symmetry_augmentation.py) | Augmentation by 4 board symmetries (rotations + reflections). Doubles effective batch size during training. |

## Composition order

```python
env = make_agent_env(...)               # microrts_agent/envs/factory.py
env = apply_env_wrappers(env, cfg)      # this module
# stack applied inside the factory:
#   StatsRecorder       (innermost)
#   FrameStack          (if cfg.frame_stack > 0)
#   ReservedObs         (if cfg.reserved_obs)
#   SymmetryAugmentation (if cfg.augment_symmetry)
```

Order matters:
- `StatsRecorder` wraps the raw env so it sees the original per-episode
  dynamics before any obs transformation.
- `FrameStack` precedes `ReservedObs` so the reserved binary channel is
  appended once after stacking, not stacked itself.
- `SymmetryAugmentation` is outermost so each base sample expands into 4
  symmetric samples downstream.

## See also

- Env construction: [`../envs/factory.py`](../envs/factory.py)
- Aux tasks that depend on these wrappers: [`../training/auxiliary.py`](../training/auxiliary.py)
