# Generalisation probes — UECD-Best

How well does the best single-map agent (`UECD-Best`, trained exclusively on
`basesWorkers16x16A`) hold up on maps it has never seen during training? Each
probe plays 100 matches against a fixed opponent on a non-training map — 50
games starting as Player 0, 50 starting as Player 1.

This is the numerical source behind the *Generalisation* chapter of the
dissertation: every win-rate dot on the dumbbell figure
(`dissertation/figs/figs-python/generalization_probes.py`) comes from the
table below.

## Maps tested

| Map | Why |
|---|---|
| `TwoBasesBarracks16x16` | Same size as training (16×16), but layout and starting configuration differ — barracks instead of just bases, different resource layout. |
| `basesWorkers32x32A`    | Double the side length (32×32) — tests whether the convolutional policy scales to a strictly larger map. |

## Results

Sorted by map, then by training-field difficulty (random → rush → competition bots → top bots).

| Map | Opponent | Games | P0 WR | P1 WR | **Total WR** |
|---|---|---:|---:|---:|---:|
| `TwoBasesBarracks16x16` | RandomBiasedAI | 100 | 100% | 100% | **100%** |
| `TwoBasesBarracks16x16` | WorkerRush     | 100 |  94% |  94% |  **94%** |
| `TwoBasesBarracks16x16` | LightRush      | 100 |   0% |   0% |   **0%** |
| `TwoBasesBarracks16x16` | CoacAI         | 100 |  94% |  76% |  **85%** |
| `TwoBasesBarracks16x16` | Mayari         | 100 |  70% |  60% |  **65%** |
| `TwoBasesBarracks16x16` | ObiBotKenobi   | 100 |   6% |   2% |   **4%** |
| `TwoBasesBarracks16x16` | TMA            | 100 |   8% |   0% |   **4%** |
| `TwoBasesBarracks16x16` | RAISocketAI    | 100 |  12% |  32% |  **22%** |
| `basesWorkers32x32A`    | RandomBiasedAI | 100 | 100% | 100% | **100%** |
| `basesWorkers32x32A`    | WorkerRush     | 100 | 100% | 100% | **100%** |
| `basesWorkers32x32A`    | LightRush      | 100 |  92% |  28% |  **60%** |
| `basesWorkers32x32A`    | CoacAI         | 100 | 100% |  92% |  **96%** |
| `basesWorkers32x32A`    | Mayari         | 100 | 100% |  90% |  **95%** |
| `basesWorkers32x32A`    | ObiBotKenobi   | 100 |  52% |  40% |  **46%** |
| `basesWorkers32x32A`    | TMA            | 100 |  16% |   6% |  **11%** |
| `basesWorkers32x32A`    | RAISocketAI    |  —  |   —  |   —  |   **—**  |

> The `basesWorkers32x32A` × RAISocketAI row is intentionally left blank.
> The original probe crashed on the cluster and was re-attempted on a
> local machine that could not honour RAISocketAI's per-move time budget
> (~100 ms), so the bot effectively played at random during long
> stretches and the resulting ~95% win rate against it does not reflect
> a fair matchup. Reporting that figure would inflate the agent's
> measured generalisation. A canonical value will only land here once the
> probe is re-run on the cluster with the bot's full time budget honoured.
> All other 15 rows reproduce the dissertation table exactly.

Full numerical data: [`results.csv`](results.csv).
Per-game logs (W/L/draw, length) for each probe: [`raw/`](raw/).

## Reproducing

Each probe is a single `evaluate` invocation with `--positions both` so
the agent plays both sides of the same matchup:

```bash
python -m microrts_agent evaluate \
    --agent outputs/runs/single_map/BestRL-350M \
    --opponent <opponent> \
    --map maps/open_competition/<map>.xml \
    --num-games 50 \
    --positions both
```

The on-disk run directory still uses the training-time name `BestRL-350M`;
only the public-facing labels here and in the raw logs were renamed to
`UECD-Best` for consistency with the rest of the shipped artefacts.
