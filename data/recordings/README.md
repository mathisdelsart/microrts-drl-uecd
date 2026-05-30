# Recordings — UECD-Best (single-map agent)

Showcase game recordings of `UECD-Best`, the best agent trained on a single
map (16x16 `basesWorkers`), played against the full competition field plus
several of our own ablation agents. 36 MP4 clips on the training map —
18 opponents × P0/P1 mirrored viewpoints.

These are the same clips embedded in the supplementary website
([https://mathisdelsart.github.io/microrts-drl-uecd-website/](https://mathisdelsart.github.io/microrts-drl-uecd-website/),
source repo: [`mathisdelsart/microrts-drl-uecd-website`](https://github.com/mathisdelsart/microrts-drl-uecd-website))
— the site pulls them directly from this folder so that a public reader can
watch a match without cloning anything.

## Layout

One directory per opponent:

```
data/recordings/
└── UECD-Best_vs_<opponent>/
    ├── basesWorkers16x16A_P0_game01.mp4   # P0 viewpoint
    └── basesWorkers16x16A_P1_game01.mp4   # P1 viewpoint
```

`P0` and `P1` are mirrored renderings of the same game from each player's
point of view.

## Opponents covered

| Category | Bots / agents |
|----------|--------------|
| Competition winners | `RAISocketAI`, `CoacAI`, `Mayari`, `Tiamat`, `StrategyTactics`, `MixedBot`, `TMA`, `Droplet`, `ObiBotKenobi`, `UtsImass` |
| Scripted baselines | `WorkerRush`, `LightRush`, `RandomBiasedAI`, `NaiveMCTS` |
| Our agents (ablations) | `GridNet`, `UECD-AllFeats`, `UECD-TopFeats`, `UECD-Rushed` |

## Reproducing a recording

The full set lives under `outputs/recordings/` (git-ignored) and is produced
by the `evaluate` CLI with `--record-video`:

```bash
python -m microrts_agent evaluate \
    --agent outputs/runs/single_map/UECD-Best \
    --opponent RAISocketAI \
    --map basesWorkers16x16A.xml \
    --num-games 1 \
    --record-video
```

The on-disk run directory still uses its training-time identifier
(`UECD-Best`); only the public name in plots and shipped recordings was
updated to `UECD-Best`. See `python -m microrts_agent evaluate --help` for
all flags.

## More

- 🌐 **Interactive viewer / extended archive:** <https://mathisdelsart.github.io/microrts-drl-uecd-website/>
- 📦 **Site source:** <https://github.com/mathisdelsart/microrts-drl-uecd-website>
- 📊 **Numerical tournament results behind these matches:** [`../tournaments/`](../tournaments/)
