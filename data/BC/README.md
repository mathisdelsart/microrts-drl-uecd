# Behaviour-cloning artefacts (BC)

Everything BC-related in one place: the teacher dataset used to warm-start
[`UECD-BC`](../agents/UECD-BC/) and [`UECD-BC-PPO`](../agents/UECD-BC-PPO/),
and the 78% pool-win-rate evaluation of `UECD-BC` that anchors the
dissertation's BC+VF→PPO vs from-scratch figure.

## Subtrees

| Path | What |
|---|---|
| [`training/`](training/) | The supervised BC teacher dataset: 6 NPZ chunks (~65 MB), `RAISocketAI` playing 100 games each against `RAISocketAI`, `CoacAI`, `Mayari` on `basesWorkers16x16A`. 313 394 transitions. Consumed by `microrts-agent bc train`. |
| [`baseline/`](baseline/) | Win-rate evaluation of the BC-only model against the 5 base-pool bots (`RandomBiasedAI`, `WorkerRush`, `LightRush`, `CoacAI`, `Mayari`), 20 games per opponent. The **78%** pool mean is the horizontal reference line in the BC+VF→PPO vs from-scratch figure. |

The BC pipeline at a glance:

1. [`training/`](training/) NPZ chunks &rarr; supervised loss in `microrts-agent bc train`
   &rarr; [`UECD-BC`](../agents/UECD-BC/) (BC-only model).
2. [`UECD-BC`](../agents/UECD-BC/) evaluated against the base pool &rarr;
   [`baseline/`](baseline/) (78% pool win rate, the horizontal line).
3. [`UECD-BC`](../agents/UECD-BC/) warm-start + 100M PPO steps &rarr;
   [`UECD-BC-PPO`](../agents/UECD-BC-PPO/) (the climbing curve that hits ~96%).

The dissertation figure that pulls all three together is
[`dissertation/figs/figs-python/bc_vs_scratch_overall.py`](../../dissertation/figs/figs-python/bc_vs_scratch_overall.py).
