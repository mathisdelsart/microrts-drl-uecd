# `microrts_agent/tournament/`

Round-robin tournament engine. Drives the four sub-subcommands of
`microrts-agent tournament` (`run`, `parse`, `viz`, `analyze`) and
produces every artefact under
[`../../data/tournaments/`](../../data/tournaments/).

## Files

| File | Role |
|---|---|
| [`__main__.py`](__main__.py) | Sub-subcommand dispatcher: `microrts-agent tournament run\|parse\|viz\|analyze`. |
| [`__init__.py`](__init__.py) | Public surface (`TournamentConfig`, `TournamentData`, `TournamentParser`, `TournamentVisualizer`, helpers). |
| [`config.py`](config.py) | `TournamentConfig.load(name)`: reads `microrts_agent/tournament_configs/<name>.json`, builds the matchup set, schedules chunks. |
| [`runner.py`](runner.py) | The actual runner: iterates matchups, dispatches to `game_loops`, writes CSV rows. |
| [`env_pool.py`](env_pool.py) | `EnvPool`: reuses vec envs across matchups when shapes match. Handles UTS_Imass pre-analysis (Python 3.6 sidecar). |
| [`game_loops.py`](game_loops.py) | The per-matchup game loops: `play_bot_vs_bot`, `play_agent_vs_bot`, `play_agent_vs_agent`. Tracks decision-time accounting. |
| [`servers.py`](servers.py) | UTS_Imass Python 3.6 server: start, configure, stop. |
| [`csv_io.py`](csv_io.py) | CSV I/O helpers (writer / atomic-append / per-chunk file). |
| [`parser.py`](parser.py) | `TournamentParser`: turns the raw `tournament.csv` into `tournament_parsed.json` (one entry per game). |
| [`visualizer.py`](visualizer.py) | `TournamentVisualizer`: dispatcher that calls every plot in [`plots/`](plots/). |
| [`ranking/`](ranking/) | Game-theoretic ranking: payoff matrix, Nash equilibrium, regret, exploitability. |
| [`plots/`](plots/) | Per-plot PDF generators (final standings, win-rate heatmap, head-to-head, game length, game-theoretic metrics). |

## Outputs

A complete run writes the following tree under
`outputs/tournaments/<config-name>/`:

```
outputs/tournaments/<config-name>/
├── tournament.csv                      # one row per game
├── tournament_parsed.json              # one entry per game, structured
├── chunks/                             # per-chunk CSV when sharded with --chunk N --total-chunks M
└── visualizations/
    ├── basic-metrics/                  # final standings, game length, etc.
    └── game-theoretic-metrics/         # winrate matrix, regret, Nash mixture, ...
```

For multi-map configs the `visualizations/` tree additionally has a
`global/` (averaged across maps) and `individual/<map>/` (per map)
split.

## Sub-subcommands

```bash
# Run the round-robin (single chunk; or shard with --chunk 0 --total-chunks 10)
microrts-agent tournament run single_map

# Parse the raw CSV into JSON
microrts-agent tournament parse single_map

# Generate the PDF plots
microrts-agent tournament viz single_map

# Shortcut: parse + viz
microrts-agent tournament analyze single_map
```

`<name>` resolves to:
- `microrts_agent/tournament_configs/<name>.json` for `run` (the
  config defines the AIs, maps, games, chunk scheduling).
- `outputs/tournaments/<name>/` for `parse`/`viz`/`analyze`.

The runner is resumable: rows are appended one at a time and the next
invocation skips already-completed matchups by reading
`tournament.csv`.

## See also

- Configs: [`../tournament_configs/`](../tournament_configs/)
- Bot registry: [`../registries/ai.py`](../registries/ai.py)
- Vendored bots: [`../bots/`](../bots/)
- Shipped tournament results: [`../../data/tournaments/`](../../data/tournaments/)
