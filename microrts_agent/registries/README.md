# `microrts_agent/registries/`

Name -> object lookup tables. Two of them: the bot registry
(`AI_MAPPING`) and the map registry (canonical map paths).

## Files

| File | Role |
|---|---|
| [`ai.py`](ai.py) | `AI_MAPPING`: dict of canonical-name -> factory function. Each factory takes a `UnitTypeTable` and returns the Java AI instance. |
| [`maps.py`](maps.py) | Canonical map paths: `COMPETITION_OPEN_MAPS`, `COMPETITION_CLOSED_MAPS`, and helpers to resolve a short name to a full XML path under [`../microrts/maps/`](../microrts/maps/). |

## Bot registry (`AI_MAPPING`)

Every bot the project knows about is listed here under a single
canonical name. The keys are referenced from:

- The tournament configs ([`../tournament_configs/`](../tournament_configs/))
- The training CLI (`--opponent-list "CoacAI:2,Mayari:2,..."`)
- The evaluation CLI (`--opponent CoacAI`)
- The BC dataset generator (`--teacher`, `--opponents`)

Canonical name conventions:

- **Rush bots**: `WorkerRush`, `LightRush`, `HeavyRush`, `RangedRush`
  (the `PO`-prefixed Java classes are no longer used; canonical names
  point at the full-observation variants).
- **Random**: `RandomAI` (uniform), `RandomBiasedAI` (single-unit
  biased baseline), `randomBiasedSingleUnitAI` (renamed in PR #82 for
  truthfulness).
- **Competition bots**: `CoacAI`, `Mayari`, `Tiamat`, `TMA`,
  `ObiBotKenobi`, `StrategyTactics`, `MixedBot`, `UtsImass`,
  `RAISocketAI`.

## Adding a bot

1. Drop the JAR under `microrts_agent/bots/<MyBot>/` with a README
   following the existing bots' style.
2. Add the JAR path to `_JARS` in [`../envs/base_vec_env.py`](../envs/base_vec_env.py)
   so the JVM classpath picks it up.
3. Add a factory `def myBot(utt): return MyBotClass(utt)` in
   [`ai.py`](ai.py).
4. Register it in `AI_MAPPING` with the canonical name.
5. Re-run the smoke tests: `test_ai_mapping_routable` and
   `test_bot_jars_open_as_zip` iterate over the registry + JARs and
   would catch missing wiring.

## Map registry

The map registry is mostly a convenience layer over the canonical map
paths under [`../microrts/maps/open_competition/`](../microrts/maps/open_competition/)
and `maps/closed_competition/`. The lists
`COMPETITION_OPEN_MAPS` / `COMPETITION_CLOSED_MAPS` define the **fixed
order** used by the multi-map training scheduler (and any other code
that needs a deterministic enumeration). Add a new map by appending it
to the relevant list; the order of existing entries must not change so
the shipped agents reproduce.
