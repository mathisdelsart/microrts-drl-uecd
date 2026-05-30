# `data/` — curated artefacts shipped with the thesis

Everything under `data/` is hand-picked output kept in version control because
it backs the dissertation, the CoG short paper, or the supplementary site. The
code never writes here: the runner / evaluator / BC generator all write to
`outputs/` (git-ignored, fully volatile), and a curated subset is then copied
into this tree.

## Contents

| Subdirectory | What it holds |
|---|---|
| [`recordings/`](recordings/) | 36 showcase MP4 game clips of UECD-Best vs the field. These are the videos shown on the supplementary website. |
| [`tournaments/`](tournaments/) | Both canonical thesis tournaments (`single_map/` + `multi_map/`): tournament CSV, parsed JSON, chunked CSVs from the SLURM runner, and the full PDF visualisation tree (basic + game-theoretic metrics, global + per-map). |
| [`generalization_probes/`](generalization_probes/) | Generalisation probes — UECD-Best (trained on `basesWorkers16x16A`) tested on two unseen maps against 8 opponents (100 games each). CSV summary + markdown table + cleaned per-probe stdout logs. |
| [`bc_training/`](bc_training/) | The behaviour-cloning teacher dataset used to warm-start the BC+PPO pipeline: 6 NPZ chunks (~65 MB), `RAISocketAI` playing 100 games against each of {`RAISocketAI`, `CoacAI`, `Mayari`} on `basesWorkers16x16A`. |
| [`agents/`](agents/)           | All trained agents (5 single-map + 2 multi-map + GridNet baseline) in *medium* form: inference model + resume checkpoint + config + final eval + training log. ~685 MB total; `UECD-SingleMap-Best` carries an additional `lineage/` subdir reconstructing its full training history from step 0. |
| [`rush_collapse/`](rush_collapse/) | Pre/post-collapse evaluation of `UECD-SingleMap-Rushed` (checkpoints at 150M and 300M, 1 000 games each against `CoacAI`, `Mayari`, `ObiBotKenobi`, `TMA`) — the data behind the dissertation's *rush collapse* / *out-of-distribution cost* sections. |
| [`bc_baseline/`](bc_baseline/) | Win-rate evaluation of `UECD-BC` against the 5 base-pool bots (`RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari`) — the **78 % baseline** of the dissertation's BC+VF→PPO vs from-scratch figure. |
| [`ablation/arch/`](ablation/arch/) | Architecture ablation — 7 architectures × 3 seeds = 21 runs trained from scratch for 100 M steps on `basesWorkers16x16A`. Each run shipped in minimal tier (agent + config + eval + log); 105 cleaned 1 000-game evaluation dumps under `raw/`; aggregate `results.csv`. |

## Links

- 🌐 **Supplementary site:** <https://mathisdelsart.github.io/microrts-drl-uecd-website/>
- 📦 **Site source:** <https://github.com/mathisdelsart/microrts-drl-uecd-website>
- 📄 **Dissertation PDF:** [`../dissertation/dissertation.pdf`](../dissertation/dissertation.pdf)
- 📝 **CoG 2026 short paper (under review):** [`../cog-2026-paper/`](../cog-2026-paper/)
