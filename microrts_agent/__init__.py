"""MicroRTS deep-RL agent (UCLouvain master's thesis).

Subpackages
-----------
architectures/   Neural network policies (GridNet, IMPALA, U-Net / entity / CBAM) + features/ + factory
envs/            Vectorized MicroRTS environments (base, rl, padded, bot) + factory
wrappers/        VecEnv wrappers (StatsRecorder, FrameStack, symmetry, ...) + factory
training/        PPO core, scheduling, self-play, in-training eval, logging
tournament/      Round-robin tournament engine + result visualisation (viz/)
registries/      Registries for Java bots (ai.py) and maps (maps.py)

Modules
-------
obs_adapter.py   Per-model observation adapter for agent-vs-agent evaluation
paths.py         Canonical output paths (RUNS_DIR, RECORDINGS_DIR, ...)

Entry points are the top-level scripts, run as `python -m microrts_agent <name>`
(e.g. train, evaluate, tournament).
"""
