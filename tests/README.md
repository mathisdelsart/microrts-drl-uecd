# `tests/`

Pytest smoke suite for the repo. **127 tests** collected, runs in ~90 s in
CI and ~140 s locally on a laptop. The suite is intentionally a smoke
gate, not a unit-test corpus: it answers "does the project import,
dispatch, train one rollout, evaluate one game, and parse the shipped
artefacts?" on every push.

## Files

| File | Role |
|---|---|
| [`test_smoke.py`](test_smoke.py) | The whole suite (single file by design). |
| [`conftest.py`](conftest.py) | Session-scoped `jvm` fixture: starts the JVM once with the same classpath as `microrts_agent/envs/base_vec_env.py`. |

## What is covered

The 127 tests fall into the following groups (see the source for the
exact names; they are grouped by section comment):

- **Imports + dispatcher**: recursive import of `microrts_agent`, every
  grouped sub-subcommand `--help` dispatches (`tournament`, `bc`,
  `bench`, `analysis` each with 2-4 sub-subcommands).
- **Paths**: every constant in `microrts_agent/paths.py` resolves under
  the repo root.
- **Tournament configs**: `single_map.json` + `multi_map.json` load,
  every referenced agent path exists, every bot is in `AI_MAPPING`,
  every map file exists. `tournament parse` succeeds on the shipped
  `data/tournaments/*/tournament_parsed.json` outputs.
- **AI registry**: 18 `AI_MAPPING` keys are routable. 10 vendored bot
  JARs open as ZIP and contain `.class` entries.
- **JNI bridge**: classpath assembly + a real `vec.step()` over the
  `JNIGridnetVecClient` (using the session JVM).
- **Architectures**: every entry in `ARCHITECTURE_REGISTRY`
  (`gridnet`, `impala`, `impala_entity`, `unet`, `unet_entity`,
  `unet_entity_cbam`, `unet_entity_cbam_deep`) instantiates + does a
  forward pass on synthetic obs.
- **Shipped agents**: each `data/agents/<name>/` is reloadable via
  `load_agent_from_config(config_path)`.
- **Wrappers**: the canonical stack composes without errors.
- **End-to-end**: a 1-game `evaluate` (GridNet vs `RandomBiasedAI`,
  max-steps 200) and a 256-step PPO `train` both reach the end of the
  loop and write to `outputs/`.
- **Shell scripts**: bash syntax of `setup/local.sh`, `setup/cluster.sh`,
  `setup/docker.sh`, `setup/_common.sh`, `microrts_agent/microrts/build_bridge.sh`.
  Shebang + executable bit on the three top-level setup scripts.
- **BC dataset**: every `.npz` chunk under
  [`../data/BC/training/`](../data/BC/training/) has the expected
  schema (`obs`/`actions`/`rewards`, 16x16 obs shape).
- **Schema sanity**: every YAML, TOML, JSON in the repo parses;
  `CITATION.cff` has the expected fields; `pyproject.toml` + `ruff.toml`
  parse via `tomllib` (3.11+) or `tomli` (3.10).
- **Vendored content**: every `microrts/maps/**.xml` parses as XML;
  every `data/agents/*/train.log` starts with `step=` lines.
- **Tournament outputs**: shipped CSVs + parsed JSONs have the expected
  shape; `tournament viz` runs on the shipped tree.
- **Analysis**: `analysis metrics` + `analysis params` run on
  `data/agents/GridNet-SingleMap`.
- **Lock file sanity**: `requirements-lock.txt` has the `uv` autogen
  header, >=50 pinned packages, and >=1 `--hash=sha256:` per package.

## Running locally

```bash
# Fast, the whole suite
pytest tests/

# Stop at first failure
pytest tests/ -x

# One test by name
pytest tests/ -k "evaluate_end_to_end" -v

# Run only the helpers that don't need the JVM (fastest, ~10 s)
pytest tests/ -k "not jvm and not end_to_end and not evaluate and not train"
```

## CI

The suite runs in CI as the **`pytest (smoke)`** status check, an
aggregator over a matrix that exercises the suite on Python 3.10, 3.11
and 3.12 in parallel. The matrix is defined in
[`../.github/workflows/ci.yml`](../.github/workflows/ci.yml).
