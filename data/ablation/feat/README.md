# Feature ablation

Systematic comparison of **22 individual feature additions** on top of a
baseline `unet_entity_cbam_deep` agent, plus the baseline itself,
trained from scratch for 50 M steps on `basesWorkers16x16A`, **3 seeds
each = 64 trained runs** with two known holes documented below. The
ablation isolates the marginal contribution of every architectural,
representational, or training feature considered by the thesis before
they get composed in the top-5 selection used by
`UECD-SingleMap-TopFeats`.

### Known holes

- **`triple_heads`** — only seed 1 trained to completion on the cluster
  (seeds 2 and 3 jobs died). Eval was nonetheless run for all three
  seeds (presumably from intermediate cluster checkpoints that are no
  longer recoverable), so the feature has **1 agent dir + 15 eval rows**.
- **`buildtime_rewards`** — all three seeds trained, but the formal
  1 000-game eval was never launched. The feature has **3 agent dirs +
  0 eval rows** and is therefore absent from the headline table below.

## Contents

| Path | What |
|---|---|
| [`agent/`](agent/) | The 64 trained runs (20 features × 3 seeds + `buildtime_rewards_s{1,2,3}` + `triple_heads_s1`). Each `agent/<feature>_s<N>/` carries the *minimal* tier: `agent.pt` (inference state-dict), `config.json` (launch hyperparameters), `train.log` (end-to-end textual log). No `eval_results.csv` (the feat-ablation training script logs in-training eval only to stdout), no resume checkpoint, no TensorBoard events. |
| [`eval/`](eval/) | The **formal post-training evaluation** of every evaluated run against the 5 base-pool bots (`RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari`), **1 000 games per matchup**. [`eval/results.csv`](eval/results.csv) is the aggregate (315 rows: 21 features with eval coverage × 3 seeds × 5 opponents). [`eval/s{1,2,3}/`](eval/) hold the per-seed cleaned stdout dumps (`<feature>_vs_<opponent>.txt`, 105 files per seed). `buildtime_rewards` is **absent** from `eval/`. |

## Headline results

Mean pool WR averaged over 3 seeds × 5 base-pool bots, 1 000 games per
matchup, **sorted by mean WR descending**. The baseline configuration is
the unmarked reference point; the Δ column shows the per-feature
deviation when added on top.

| Feature | Mean WR | Δ vs baseline |
|---|---:|---:|
| `extended_obs`         | 84.3 % | **+19.3 pp** |
| `filtmask_resobs`      | 80.6 % | +15.5 pp |
| `priori_samp` *(MCW)*  | 78.4 % | +13.3 pp |
| `opponent_modeling`    | 77.8 % | +12.7 pp |
| `triple_heads`         | 72.7 % |  +7.6 pp |
| `pae`                  | 71.3 % |  +6.2 pp |
| `hierarchical_mask`    | 71.1 % |  +6.0 pp |
| `adaptive`             | 69.6 % |  +4.6 pp |
| `popart`               | 68.5 % |  +3.5 pp |
| `aux_unit_count`       | 67.6 % |  +2.5 pp |
| `bots_selfplay_pfsp`   | 66.9 % |  +1.8 pp |
| `gelu`                 | 66.3 % |  +1.2 pp |
| `framestack4`          | 65.3 % |  +0.3 pp |
| `autoregressive_hmask` | 65.1 % |  +0.1 pp |
| **`baseline`**         | **65.1 %** | — |
| `autoregressive`       | 63.7 % |  −1.3 pp |
| `aux_spatial`          | 63.3 % |  −1.8 pp |
| `spp_critic`           | 63.2 % |  −1.8 pp |
| `aux_contrastive`      | 61.6 % |  −3.4 pp |
| `augment_symmetry`     | 57.5 % |  −7.6 pp |
| `hl_gauss`             | 39.5 % | **−25.6 pp** |

The top-band (≥ +10 pp lift) — `extended_obs`, `filtmask_resobs`,
`priori_samp` (MCW), `opponent_modeling` — is exactly the set carried
forward into `UECD-SingleMap-TopFeats` (the *top-5* configuration
validated at 100 M steps in the architecture ablation chapter). The
bottom-band tells a clearer story: `hl_gauss` (HL-Gauss value
classification, −25.6 pp) catastrophically destabilises training at
this budget, and `augment_symmetry` (symmetry-augmented rollouts,
−7.6 pp) hurts — neither survives into the published recipe.

Full per-seed and per-opponent breakdown lives in
[`eval/results.csv`](eval/results.csv).

## Reproducing one feature-seed combination

```bash
python -m microrts_agent train \
    --architecture unet_entity_cbam_deep \
    --<feature>             # e.g. --extended-obs True
    --seed <N> \
    --total-timesteps 50000000 \
    --map maps/open_competition/basesWorkers16x16A.xml \
    --exp-name feats_<feature>_s<N>
```

For evaluation against the 5 base-pool bots (1 000 games per matchup):

```bash
for opp in RandomBiasedAI WorkerRush LightRush CoacAI Mayari; do
    python -m microrts_agent evaluate \
        --agent data/ablation/feat/agent/<feature>_s<N> \
        --opponent $opp \
        --map maps/open_competition/basesWorkers16x16A.xml \
        --num-games 500 \
        --positions both
done
```

Generator SLURM (parallel array job over the features):
[`experiments/ablation/feat/feats_ablation_50M.slurm`](../../../experiments/ablation/feat/feats_ablation_50M.slurm).

## See also

- 🏗 **Architecture ablation** (validates the backbone choice before this
  feature ablation builds on it): [`../arch/`](../arch/)
- 🤖 **The top-5 selection** trained 2× longer: [`../../agents/UECD-SingleMap-TopFeats/`](../../agents/UECD-SingleMap-TopFeats/)
- 🤖 **The all-features selection** for comparison: [`../../agents/UECD-SingleMap-AllFeats/`](../../agents/UECD-SingleMap-AllFeats/)
- 📈 **Dissertation chapter** on the training system (Chapter 9 of
  `dissertation/dissertation.pdf`).
