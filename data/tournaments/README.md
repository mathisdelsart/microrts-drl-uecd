# Tournament results

Round-robin tournament results that constitute the **headline empirical
contribution of the thesis**. Each tournament pits the trained agents against
the full competition field of MicroRTS bots; the analysis distinguishes
basic counters (win rate, head-to-head, game length) from game-theoretic
robustness metrics (Nash, alpha-rank, Copeland, regret).

## The two canonical tournaments

### `single_map/` — single-map evaluation (`basesWorkers16x16A`)

19 AIs (5 UECD agents + 14 bots), iterations = 5, on the 16×16 training map.
This is the primary tournament reported in the dissertation: it isolates
agent strength on the map each agent was actually trained on, separating
quality of play from generalisation.

### `multi_map/` — multi-map evaluation (5 open-competition maps)

16 AIs (2 UECD agents + 14 bots), iterations = 5, on the official IEEE-CoG
open-competition 5-map suite (`basesWorkers8x8A`, `FourBasesWorkers8x8`,
`NoWhereToRun9x8`, `basesWorkers16x16A`, `TwoBasesBarracks16x16`). This is
the head-to-head used in the CoG 2026 short paper and discussed in the
generalisation chapter of the thesis.

## File layout per tournament

```
data/tournaments/<name>/
├── tournament.csv                 # full raw record (one line per game)
├── tournament_parsed.json         # parsed structured form (consumed by viz)
├── chunks/                        # per-SLURM-chunk raw CSVs (reproduction)
│   └── chunk_<i>.csv
└── visualizations/
    Single-map:
        basic-metrics/{final_standings, h2h_matrix, games_length}.pdf
        game-theoretic-metrics/{nash_scores, alpha_rank_sweep,
                                copeland_scores, robustness_score}.pdf
    Multi-map:
        global/basic-metrics/{...4 PDFs incl. winrates_per_map}.pdf
        global/game-theoretic-metrics/{...4 PDFs}.pdf
        individual/<map>/basic-metrics/{...3 PDFs}.pdf
        individual/<map>/game-theoretic-metrics/{...4 PDFs}.pdf
```

## Reproducing

The two configs that generated these tournaments live at
`microrts_agent/tournament_configs/single_map.json` and
`microrts_agent/tournament_configs/multi_map.json`. To reproduce the
single-map tournament:

```bash
# 1. Run the tournament (writes to outputs/tournaments/single_map/)
python -m microrts_agent tournament run single_map

# 2. Parse + visualise
python -m microrts_agent tournament analyze single_map
```

`outputs/tournaments/` is `.gitignore`'d — the runner always writes there,
the shipped subset under `data/tournaments/` is a manual snapshot.

## Where else these results appear

- 📄 **Dissertation:** chapter on Results — final standings, generalisation,
  head-to-head matrices, game-theoretic robustness all draw on these PDFs.
- 📝 **CoG 2026 short paper:** uses the `multi_map/` head-to-head matrix
  and final standings.
- 🌐 **Supplementary site:** [interactive tournament dashboard, recordings, analyses](https://mathisdelsart.github.io/microrts-drl-uecd-website/).
