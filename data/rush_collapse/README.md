# Rush collapse — UECD-SingleMap-Rushed at 150M vs 300M

Empirical evidence for the *rush collapse* pathology described in the
dissertation: the discount factor + near-terminal-only reward + policy
control over episode length together select for a shortcut policy that
ends games as fast as possible. The pre-collapse checkpoint (step
**150M**, balanced policy) and the post-collapse checkpoint (step
**300M**, all-in worker rush) of `UECD-SingleMap-Rushed` are evaluated
against the same four opponents — two **in-training** (`CoacAI`,
`Mayari`) and two **held-out CoG competitors** (`ObiBotKenobi`, `TMA`)
— so the difference attributable to the strategy change is isolated
from learning progress.

Each evaluation plays **1 000 games** (500 as P0, 500 as P1) on
`basesWorkers16x16A.xml` at the stochastic-action protocol used
elsewhere in the thesis.

## Results

| Stage | Opponent | Games | P0 WR | P1 WR | **Total WR** | Avg ep length |
|---|---|---:|---:|---:|---:|---:|
| `150M` *(balanced)* | CoacAI         | 1 000 |  96.8% |  99.2% |  **98.0%** | ~987 f |
| `150M` *(balanced)* | Mayari         | 1 000 | 100.0% |  99.6% |  **99.8%** | ~864 f |
| `150M` *(balanced)* | ObiBotKenobi   | 1 000 |  67.8% |  73.0% |  **70.4%** | ~1 248 f |
| `150M` *(balanced)* | TMA            | 1 000 |  50.8% |  62.2% |  **56.5%** | ~1 375 f |
| `300M` *(rush)*     | CoacAI         | 1 000 |  96.6% |  99.4% |  **98.0%** | ~307 f |
| `300M` *(rush)*     | Mayari         | 1 000 | 100.0% | 100.0% | **100.0%** | ~295 f |
| `300M` *(rush)*     | ObiBotKenobi   | 1 000 |   2.2% |   0.8% |   **1.5%** | ~765 f |
| `300M` *(rush)*     | TMA            | 1 000 |   0.0% |   0.0% |   **0.0%** | ~946 f |

## How to read this

**On in-training opponents (`CoacAI`, `Mayari`):**
the two checkpoints are indistinguishable on win rate (98–100% for both);
the only signature of the strategy change is the **3× drop in episode
length** (~900 → ~300 frames), the fingerprint of a successful worker rush.

**On held-out opponents (`ObiBotKenobi`, `TMA`):**
the picture inverts completely. WR collapses from 56–70% at 150M to near
zero (0–1.5%) at 300M, while episode lengths fall from ~1 300 to
~700–950 frames — well above the ~300 frames of a successful rush. The
rush is attempted but **fails**: the early worker aggression that broke
CoacAI / Mayari is dismantled by opponents with more robust early-game
defences.

This is the empirical signature of the discount-induced shortcut.
Against held-out opponents, neither pillar of the rush incentive holds:
the rush does not produce a fast win, and many attempts terminate in
losses. The shortcut is selected for *by the training distribution* and
collapses against everything else.

## Mitigation

The dissertation's remedy is a **10% shaped-reward floor** maintained
throughout training, preserving a dense per-step gradient strong enough
to anchor the optimiser to balanced play and dominate the discounted-
terminal one. Every subsequent run (`UECD-SingleMap-AllFeats`,
`UECD-SingleMap-TopFeats`, `UECD-SingleMap-Best`, both `UECD-MultiMap`
variants) carries that 10% floor as a result of this experiment.

## Files

| Path | What |
|---|---|
| [`results.csv`](results.csv) | 8 rows, one per (stage, opponent) — the table above as machine-readable CSV |
| [`raw/150M/`](raw/150M/)     | 4 cleaned stdout dumps of the 150M evaluator runs |
| [`raw/300M/`](raw/300M/)     | 4 cleaned stdout dumps of the 300M evaluator runs |

Each raw file is the full evaluator log (per-game W/L/length lines + the
aggregate `RESULTS` block at the bottom). The agent label in the raw
files was `tmp.XXX` at training time (each evaluator spawned the loaded
snapshot under a temporary identifier); it has been replaced with
`UECD-SingleMap-Rushed-at-150M` / `UECD-SingleMap-Rushed-at-300M`
for readability.

## Reproducing

The agent at each stage is a checkpoint of `UECD-SingleMap-Rushed`
(formerly `PhasedRL-300M`). Each opponent eval was launched as:

```bash
python -m microrts_agent evaluate \
    --agent outputs/runs/single_map/UECD-SingleMap-Rushed \
    --checkpoint <step>.pt           # 150M -> 150006272.pt, 300M -> 299962368.pt
    --opponent <opponent> \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --num-games 500 \
    --positions both
```

## See also

- 🎓 **Trained agent** behind these checkpoints: [`../agents/UECD-SingleMap-Rushed/`](../agents/UECD-SingleMap-Rushed/)
- 📊 **Tournament context** where the rush still scores 98 points overall: [`../tournaments/single_map/`](../tournaments/single_map/)
- 🎯 **Generalisation probes** (analogous structure, different question): [`../generalization_probes/`](../generalization_probes/)
