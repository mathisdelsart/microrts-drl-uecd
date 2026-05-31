# `microrts_agent/bench/`

Decision-time benchmarks. Used to compare the wall-clock cost of one
policy step across architectures, and to compare the practical
head-to-head playing strength of two agents under matched time
budgets.

## Files

| File | Role |
|---|---|
| [`__main__.py`](__main__.py) | Sub-subcommand dispatcher: `microrts-agent bench {inference\|head2head}`. |
| [`inference.py`](inference.py) | `microrts-agent bench inference`: measures per-step inference latency (forward + sample) for one agent over a configurable horizon. Reports mean/p50/p95/p99. |
| [`head_to_head.py`](head_to_head.py) | `microrts-agent bench head2head`: plays two agents against each other for N games and reports win rates + draws + average game length. Both sides share the same time-budget configuration. |

## Inference benchmark

```bash
microrts-agent bench inference \
    --agent data/agents/UECD-SingleMap-Best \
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --steps 1000 \
    --device cpu
```

Reports rows like:

```
Loaded   UECD-SingleMap-Best from data/agents/UECD-SingleMap-Best (...)
1000 steps, mean= 12.4 ms, p50= 11.8 ms, p95= 18.3 ms, p99= 24.1 ms
```

Use `--device cuda` on a GPU node to time the GPU path. The vec env is
single-env so the timings reflect per-step latency, not throughput.

## Head-to-head benchmark

```bash
microrts-agent bench head2head \
    --agent-a data/agents/UECD-SingleMap-Best \
    --agent-b data/agents/UECD-SingleMap-TopFeats \
    --maps maps/open_competition/basesWorkers16x16A.xml \
    --nb_games 50
```

Plays each side both P0 and P1 (so totals are `2 * nb_games`). Useful
to A/B two ablation variants under matched conditions.

## See also

- The cluster-wide round-robin lives in [`../tournament/`](../tournament/);
  use that when comparing more than two agents or across maps.
- Architecture parameter counts: `microrts-agent analysis params`
  (implementation in [`../analysis/params.py`](../analysis/params.py)).
