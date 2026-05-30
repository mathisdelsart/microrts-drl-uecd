# Architecture ablation

Systematic comparison of 7 neural network architectures trained **from
scratch for 100 M steps** on `basesWorkers16x16A`, identical
hyperparameters (baseline configuration, no advanced features), **3 seeds
each = 21 runs**. The architectures progressively refine the convolutional
backbone (`GridNet` → `IMPALA-CNN` → `U-Net`), then introduce the entity
Transformer and CBAM attention to build up the final thesis architecture
`unet_entity_cbam_deep` (referred to as `UECD` in the dissertation and the
CoG paper).

## Contents

| Path | What |
|---|---|
| [`agent/`](agent/) | The 21 trained runs (7 architectures × 3 seeds). Each `agent/<arch>_s<N>/` carries the *minimal* tier needed to **load the model and inspect its training**: `agent.pt` (inference state-dict), `config.json` (launch hyperparameters), `eval_results.csv` (in-training eval at multiple steps), `train.log` (end-to-end textual log). No resume checkpoint, no TensorBoard events — these are ablation runs and no one resumes from them. |
| [`eval/`](eval/) | The **formal post-training evaluation** of every run against the 5 base-pool bots (`RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari`), **1 000 games per matchup**. [`eval/results.csv`](eval/results.csv) is the aggregate (105 rows: arch × seed × opponent → P0/P1 win rate and episode length). [`eval/s{1,2,3}/`](eval/) hold the cleaned per-run stdout dumps (`<arch>_vs_<opponent>.txt`, 35 files per seed). |

## Headline results

Mean pool WR averaged over 3 seeds × 5 base-pool bots
(`RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari`),
**1 000 games per matchup** (500 P0 + 500 P1):

| Architecture                              | Mean WR | Δ vs `GridNet`  | Role |
|---|---:|---:|---|
| `gridnet`                                 | 64.8 %  | —               | Baseline. Encoder-decoder, fixed 16×16 map size. |
| `impala`                                  | 68.7 %  | +3.9 pp         | IMPALA residual encoder; decouples from map size. |
| `impala_entity`                           | 74.9 %  | +10.1 pp        | Parallel entity Transformer over unit tokens. |
| `unet`                                    | 77.0 %  | +12.2 pp        | U-Net skip connections + SE channel attention. |
| `unet_entity`                             | 82.7 %  | +17.9 pp        | Combines U-Net + entity Transformer. |
| `unet_entity_cbam`                        | 79.2 %  | +14.4 pp        | Replaces SE attention with CBAM (channel + spatial). |
| `unet_entity_cbam_deep` *(UECD)*          | **86.2 %** | **+21.4 pp** | Deeper CBAM stack — the final thesis architecture. |

The progressive refinement of the backbone (`GridNet` → `IMPALA-CNN` →
`U-Net`) lifts pool WR from 64.8 % to 77.0 %. Adding the entity
Transformer on top of `IMPALA` adds another 6.2 pp; combining U-Net with
the entity head adds 5.7 pp on top. CBAM on its own is noisier than SE
(79.2 % vs 82.7 %) but stacking it deeper recovers and overshoots
(86.2 %), validating the `UECD` choice. Full numerical data with
per-seed and per-opponent breakdown lives in
[`eval/results.csv`](eval/results.csv).

## Layout

```
data/ablation/arch/
├── README.md
├── eval/
│   ├── results.csv          # 105 rows: arch, seed, opponent, p0/p1 WR + avg_len, total_wr
│   ├── s1/                  # 35 .txt = 7 archs × 5 opps for seed 1
│   ├── s2/                  # 35 .txt for seed 2
│   └── s3/                  # 35 .txt for seed 3
└── agent/                   # 21 trained runs, minimal tier
    └── <arch>_s<seed>/
        ├── agent.pt         # inference-ready policy state-dict
        ├── config.json      # every CLI / hyperparameter the run was launched with
        ├── eval_results.csv # in-training eval (multiple steps, 10 games each)
        └── train.log        # end-to-end textual training log
```

`config.json` still uses the upstream architecture identifier
(`impala_unet` for the `unet` runs, `impala_unet_entity_cbam_v2` for the
`unet_entity_cbam_deep` runs); the shortened directory names match the
dissertation's display labels (`U-Net`, `U-Net-Entity-CBAM-Deep`).

## Final evaluation

`eval/s<seed>/<arch>_vs_<opponent>.txt` — 105 stdout dumps of the
`evaluate` CLI (one per architecture × seed × opponent), 1 000 games each.
The aggregate `RESULTS` block at the bottom of every file is what
[`eval/results.csv`](eval/results.csv) summarises:

| Column        | Meaning |
|---|---|
| `arch`        | one of the 7 architecture labels above |
| `seed`        | 1, 2, or 3 |
| `opponent`    | `RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari` |
| `games`       | always 1 000 (500 P0 + 500 P1) |
| `p0_wr`       | win rate as P0 (over 500 games) |
| `p0_avg_len`  | average game length (frames) as P0 |
| `p1_wr`       | win rate as P1 |
| `p1_avg_len`  | average game length as P1 |
| `total_wr`    | overall win rate (1 000 games) |

Bot names in raw dumps normalised to the canonical labels (no `PO`
prefix). The agent label inside each dump was the on-cluster run name at
evaluation time; cleaned to `<arch>_s<N>` for naming consistency with
the directory.

## Reproducing one architecture-seed combination

```bash
python -m microrts_agent train \
    --architecture <arch>  \
    --seed <N> \
    --total-timesteps 100000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --exp-name arch_ablation_<arch>_s<N>
```

For evaluation against the 5 base-pool bots:

```bash
for opp in RandomBiasedAI WorkerRush LightRush CoacAI Mayari; do
    python -m microrts_agent evaluate \
        --agent data/ablation/arch/agent/<arch>_s<N> \
        --opponent $opp \
        --map maps/open_competition/basesWorkers16x16A.xml \
        --num-games 500 \
        --positions both
done
```

Generator SLURM (parallel array job over the 7 architectures):
[`experiments/ablation/arch/arch_ablation_100M.slurm`](../../../experiments/ablation/arch/arch_ablation_100M.slurm).

## See also

- 🏗 **The final architecture** (UECD trained for 350M with two-phase
  fine-tuning): [`../../agents/UECD-SingleMap-Best/`](../../agents/UECD-SingleMap-Best/)
- 📊 **Tournament context** where these architectures compete against
  the wider field: [`../../tournaments/`](../../tournaments/)
- 📈 **Dissertation chapter** on architectures (Chapter 8 of
  `dissertation/dissertation.pdf`).
