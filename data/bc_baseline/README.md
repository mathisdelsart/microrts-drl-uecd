# BC-only baseline (UECD-BC)

Win-rate evaluation of the **BC-only** model (`UECD-BC`) against the 5
base-pool bots on `basesWorkers16x16A`. This is the **78% horizontal
line** in the dissertation's BC+VF→PPO vs *from scratch* figure
(`bc_only_wr = 0.78` hardcoded in
[`dissertation/figs/figs-python/bc_vs_scratch_overall.py`](../../dissertation/figs/figs-python/bc_vs_scratch_overall.py)).

The model was trained by behaviour cloning only: no RL phase, no
value-function head update past the supervised loss. The dataset is the
v3 BC teacher set shipped under [`../bc_training/`](../bc_training/)
(`RAISocketAI` demonstrating against `RAISocketAI`, `CoacAI`, `Mayari`,
100 games each, 313 394 transitions).

The next step in the pipeline, adding 100 M of PPO fine-tuning on top
of these BC weights, produces the **`UECD-BC-PPO`** agent
([`../agents/UECD-BC-PPO/`](../agents/UECD-BC-PPO/)), which lifts the
pool win rate from 78 % to ~96 % on the same five opponents.

## Results

| Opponent       | Games | P0 WR | P1 WR | **Total WR** | P0 avg len | P1 avg len |
|---|---:|---:|---:|---:|---:|---:|
| `RandomBiasedAI` | 20    | 100.0% | 100.0% | **100.0%** | ~662 f | ~668 f |
| `WorkerRush`     | 20    | 100.0% |  40.0% |  **70.0%** | ~815 f | ~1 081 f |
| `LightRush`      | 20    | 100.0% |  80.0% |  **90.0%** | ~676 f | ~1 555 f |
| `CoacAI`         | 20    |  70.0% |  90.0% |  **80.0%** | ~1 066 f | ~838 f |
| `Mayari`         | 20    | 100.0% |   0.0% |  **50.0%** | ~717 f | ~1 054 f |
| **Pool mean**    |       |        |        | **78.0%**  |        |        |

Full numerical data: [`results.csv`](results.csv).
Per-game logs: [`raw/`](raw/).

## How to read this

- **P0 vs P1 asymmetry** is unusually large here (Mayari 100% as P0 vs
  0% as P1, WorkerRush 100% vs 40%). This is a signature of *pure*
  behaviour cloning: the model has internalised the side-specific
  behaviours of the demonstrator (RAISocketAI played both sides during
  data collection, but the network learned them as two distinct
  distributions instead of a side-invariant policy). PPO fine-tuning
  later smooths most of this out.
- **Sample size is small** (20 games / opponent = 10 P0 + 10 P1), so
  individual cells have wide confidence intervals (Wald 95 % CI is
  ±22 pp at 50 %, ±18 pp at 90 %). The aggregate 78 % is a useful
  baseline number but per-opponent rates are noisy.
- **The 78 % is what the dissertation figure pins as the horizontal
  reference line for BC-only**; the BC+PPO curve climbs from there.

## Files

| Path | What |
|---|---|
| [`results.csv`](results.csv) | 5 rows (one per opponent): the table above as machine-readable CSV |
| [`raw/UECD-BC_vs_<opponent>.txt`](raw/) | Full evaluator stdout (per-game W/L + the aggregate `RESULTS` block at the bottom), 5 files |

## Reproducing

The BC-only agent itself lives at
[`../agents/UECD-BC/`](../agents/UECD-BC/). To re-run the 5 evaluations:

```bash
for opp in RandomBiasedAI WorkerRush LightRush CoacAI Mayari; do
    microrts-agent evaluate \
        --agent data/agents/UECD-BC \
        --opponent $opp \
        --maps maps/open_competition/basesWorkers16x16A.xml \
        --nb_games 10
done
```

`evaluate` plays `--nb_games` games as P0 *and* the same number as P1 by
default, so 10 here means 20 games per opponent (matching the 20-games column
in the table above).

## See also

- **Teacher dataset** behind the BC weights: [`../bc_training/`](../bc_training/)
- **The next-step agent** (BC + 100 M of PPO): [`../agents/UECD-BC-PPO/`](../agents/UECD-BC-PPO/)
- **Dissertation figure** comparing BC-only / BC+PPO / from-scratch: [`../../dissertation/figs/figs-python/bc_vs_scratch_overall.py`](../../dissertation/figs/figs-python/bc_vs_scratch_overall.py)
