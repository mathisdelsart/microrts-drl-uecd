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
| [`probes/`](probes/) | Generalisation probes — UECD-Best (trained on `basesWorkers16x16A`) tested on two unseen maps against 8 opponents (100 games each). CSV summary + markdown table + cleaned per-probe stdout logs. |
| [`bc_training/`](bc_training/) | The behaviour-cloning teacher dataset used to warm-start the BC+PPO pipeline: 6 NPZ chunks (~65 MB), `RAISocketAI` playing 100 games against each of {`RAISocketAI`, `CoacAI`, `Mayari`} on `basesWorkers16x16A`. |

## Links

- 🌐 **Supplementary site:** <https://mathisdelsart.github.io/microrts-drl-uecd-website/>
- 📦 **Site source:** <https://github.com/mathisdelsart/microrts-drl-uecd-website>
- 📄 **Dissertation PDF:** [`../dissertation/dissertation.pdf`](../dissertation/dissertation.pdf)
- 📝 **CoG 2026 short paper (under review):** [`../cog-2026-paper/`](../cog-2026-paper/)
