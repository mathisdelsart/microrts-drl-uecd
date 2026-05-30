# Recordings — BestRL-350M (single-map agent)

Showcase game recordings of `BestRL-350M`, the best agent trained on a single
map (16x16 `basesWorkers`), played against the full competition field plus
several of our own ablation agents. 36 MP4 clips on the training map —
18 opponents × P0/P1 mirrored viewpoints.

## Layout

One directory per opponent:

```
recordings/
└── BestRL-350M_vs_<opponent>/
    ├── basesWorkers16x16A_P0_game01.mp4   # P0 viewpoint
    └── basesWorkers16x16A_P1_game01.mp4   # P1 viewpoint
```

`P0` and `P1` are mirrored renderings of the same game from each player's
point of view.

## Opponents covered

| Category | Bots / agents |
|----------|--------------|
| Competition winners | `RAISocketAI`, `CoacAI`, `Mayari`, `Tiamat`, `StrategyTactics`, `MixedBot`, `TMA`, `Droplet`, `ObiBotKenobi`, `UtsImass` |
| Scripted baselines | `POWorkerRush`, `POLightRush`, `RandomBiasedAI`, `NaiveMCTS` |
| Our agents (ablations) | `GridNet-300M`, `AllFeatsRL-100M`, `TopFeatsRL-100M`, `PhasedRL-300M` |

## Reproducing a recording

The full set lives under `outputs/recordings/` (git-ignored) and is produced
by the `evaluate` CLI with `--record-video`:

```bash
python -m microrts_agent evaluate \
    --agent outputs/runs/BestRL-350M \
    --opponent RAISocketAI \
    --map basesWorkers16x16A.xml \
    --num-games 1 \
    --record-video
```

See `python -m microrts_agent evaluate --help` for all flags.

## More recordings

The full tournament archive (additional opponents, more games per match,
larger maps) is on the [supplementary
site](https://mathisdelsart.github.io/microrts-drl-uecd-website/).
