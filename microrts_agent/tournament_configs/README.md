# `microrts_agent/tournament_configs/`

The shipped tournament configurations. Two canonical configs, both used
in the thesis and surfaced as the headline tournaments under
[`../../data/tournaments/`](../../data/tournaments/).

## Files

| File | What it runs |
|---|---|
| [`single_map.json`](single_map.json) | 10-AI round-robin on `basesWorkers16x16A`, 20 iterations per pair (180 games per agent). Output: [`data/tournaments/single_map/`](../../data/tournaments/single_map/). |
| [`multi_map.json`](multi_map.json) | Round-robin across 5 maps (mixed open + closed competition). Output: [`data/tournaments/multi_map/`](../../data/tournaments/multi_map/). |

## Anatomy of a config

```json
{
  "name": "single_map",
  "maps": ["maps/open_competition/basesWorkers16x16A.xml"],
  "iterations_per_pair": 20,
  "max_steps": 4000,
  "ais": [
    { "name": "UECD-SingleMap-Best",
      "type": "agent",
      "agent": "data/agents/UECD-SingleMap-Best",
      "device": "cpu" },
    { "name": "CoacAI", "type": "bot", "bot": "CoacAI" },
    ...
  ]
}
```

Required fields:
- `name`: used to name the output directory (`outputs/tournaments/<name>/`).
- `maps`: list of XML paths (relative to repo root).
- `iterations_per_pair`: number of games per ordered (P0, P1) pair per
  map.
- `max_steps`: per-game step cap.
- `ais`: list of participants. `type: "agent"` entries point to a
  `data/agents/<dir>/` (which must contain `config.json` + `agent.pt`);
  `type: "bot"` entries reference a name in `AI_MAPPING` (see
  [`../registries/ai.py`](../registries/ai.py)).

## How to run

```bash
microrts-agent tournament run single_map
microrts-agent tournament analyze single_map     # parse + viz in one
```

See [`../tournament/README.md`](../tournament/README.md) for the full
sub-subcommand reference.

Tournament outputs land under `outputs/tournaments/` (gitignored). The
shipped directories at
[`../../data/tournaments/`](../../data/tournaments/) are curated
snapshots of the thesis tournaments.
