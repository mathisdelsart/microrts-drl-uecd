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

Canonical name groups:

- **Rush bots**: `WorkerRush`, `LightRush`, `HeavyRush`, `RangedRush`
  (full-observation variants).
- **Random**: `RandomAI` (uniform), `RandomBiasedAI` (single-unit
  biased baseline), `randomBiasedSingleUnitAI`.
- **Competition bots**: `CoacAI`, `Mayari`, `Tiamat`, `TMA`,
  `ObiBotKenobi`, `StrategyTactics`, `MixedBot`, `UtsImass`,
  `RAISocketAI`.

## Map registry

`COMPETITION_OPEN_MAPS` / `COMPETITION_CLOSED_MAPS` enumerate the
canonical map paths under [`../microrts/maps/open_competition/`](../microrts/maps/open_competition/)
and `maps/closed_competition/`. The order defines the enumeration used
by the multi-map training scheduler.
