# Behaviour-cloning teacher dataset

Training data for the behaviour-cloning warm-start used by the **BC + PPO**
training pipeline of the thesis. The agent is trained in supervised fashion
to imitate `RAISocketAI` (the 2023/2024 IEEE-CoG MicroRTS competition winner),
before being fine-tuned with PPO and an optional KL teacher penalty.

## What's in here

Six NPZ chunks, one pair per opponent — `RAISocketAI` playing **100 games**
against each of:

| Opponent | Files | Transitions | Total size |
|---|---|---:|---:|
| `RAISocketAI` (self-play) | `bc_chunk_RAISocketAI_{0,1}.npz` | 161 065 | 34 MB |
| `CoacAI`                  | `bc_chunk_CoacAI_{0,1}.npz`      |  85 826 | 18 MB |
| `Mayari`                  | `bc_chunk_Mayari_{0,1}.npz`      |  66 503 | 14 MB |
| **Total**                 | 6 files                          | **313 394** | **65 MB** |

All games are on `basesWorkers16x16A.xml` (the canonical 16×16 thesis map).
Each transition records the demonstrator's view: a 16×16×29 observation,
a per-cell action of shape (256, 7) (gridnet action encoding), and the
sparse reward at that step.

## Chunk format

Each `.npz` file contains three arrays produced by
`microrts_agent/bc/generate.py`:

| Key | Shape | Dtype | Meaning |
|---|---|---|---|
| `obs`     | (N, 16, 16, 29) | float32 | Per-cell observation stack at step *t* |
| `actions` | (N, 256, 7)     | int     | Gridnet action emitted by the demonstrator at step *t* |
| `rewards` | (N,)            | float32 | Sparse reward observed at step *t* |

`N` varies per chunk (the generator flushes to disk every N games to avoid
OOM, so a 100-game run for an opponent may split into two chunks of unequal
size).

## How it is consumed

`microrts_agent/bc/train.py` globs every `bc_chunk_*.npz` it is pointed at
and merges them on the fly. To re-train from the shipped data:

```bash
python -m microrts_agent bc train \
    --data data/bc_training/bc_chunk_*.npz \
    --epochs 20 \
    --batch-size 256 \
    --output outputs/runs/bc_warmstart
```

Then chain into PPO with the trained checkpoint as the warm-start (with or
without a KL teacher penalty). See the dissertation chapter on the training
system for the BC + PPO pipeline rationale and ablation results.

## How it was generated

Three back-to-back invocations of the BC generator, one per opponent:

```bash
for opp in RAISocketAI CoacAI Mayari; do
    python -m microrts_agent bc generate \
        --bot RAISocketAI \
        --opponents $opp \
        --games-per-opponent 100 \
        --map maps/open_competition/basesWorkers16x16A.xml
done
```

The full pipeline (generation + BC training + PPO fine-tune + evaluation)
is in `experiments/bc/bc_pipeline_v2.slurm`.
