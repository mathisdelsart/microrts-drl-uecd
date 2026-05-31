# `microrts_agent/envs/`

Vectorised MicroRTS environments. The Python side talks to the Java
engine through the JPype-backed JNI bridge that lives in
[`../microrts/src/`](../microrts/src/); see
[`../microrts/src/README.md`](../microrts/src/README.md) for the full
per-field contract.

## Files

| File | What it is |
|---|---|
| [`factory.py`](factory.py) | `make_agent_env(...)`, `make_bot_env(...)`: the only entry points used by the rest of the package. |
| [`base_vec_env.py`](base_vec_env.py) | JVM start (with the right classpath), shared helpers. Importing this module does **not** start the JVM; the first env build does. |
| [`rl_vec_env.py`](rl_vec_env.py) | `RlVecEnv`: standard agent-vs-bot vec env (each sub-env hosts one match). |
| [`bot_vec_env.py`](bot_vec_env.py) | `BotVecEnv`: bot-vs-bot vec env (used by tournament to collect bot-bot games). |
| [`padded_rl_vec_env.py`](padded_rl_vec_env.py) | `PaddedRlVecEnv`: multi-map env where each cell pads the obs to the largest map in the pool. |

## Lifecycle

```
factory.make_agent_env(...)
    -> ensure JVM started (base_vec_env.start_jvm if first call)
    -> RlVecEnv(jni_client, cfg)        # one JNIGridnetVecClient per env
        -> reset()                       # returns obs, info
        -> step(action)                  # returns obs, reward, done, info
                                         # done envs auto-reset (engine contract)
        -> get_action_mask()             # called every step
        -> close()                       # detaches JVM thread
```

Each sub-env holds its own `JNIGridnetVecClient` instance. Inference is
batched on the Python side (one tensor across all sub-envs); the Java
side runs each sub-env on its own thread, and JPype releases the GIL
during the call so they do run in parallel.

## Choosing a factory

- **`make_agent_env`** when training, evaluating an RL agent, or
  collecting BC data with an RL teacher. Returns observations + per-step
  action masks shaped for `microrts_agent.architectures` policies.
- **`make_bot_env`** when running bot-vs-bot matches (tournament). The
  Python side just observes; both action streams are picked by Java
  bots.

The factory honours every relevant flag from the run's config
(map list, max-steps, num envs, reward weights, render flag, ...). Code
that needs to add an env-side knob should plumb it through the factory,
not bypass it.

## Note on map shapes

Multi-map training uses [`padded_rl_vec_env.PaddedRlVecEnv`](padded_rl_vec_env.py),
which pads every sub-env's obs to the **largest** map in the pool. The
policy then sees a uniform tensor shape; the unused cells are masked
out by the action mask. This is what every `data/agents/UECD-MultiMap*`
agent uses.
