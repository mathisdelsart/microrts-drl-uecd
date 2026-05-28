"""
Library modules for microrts_agent — import only, not entry points.

Packages
--------
architectures/   Neural network architectures (GridNetAgent, IMPALAAgent, U-Net / entity / CBAM variants)
envs/            Vectorized MicroRTS environments (base, rl, padded, bot)
mappings/        Registries for Java bots (ai.py) and maps (maps.py)
training/        Training utilities (PPO core, scheduling, in-training eval)
tournament/      Tournament infrastructure (config, env pool, game loops, CSV I/O, servers)
wrappers/        Gym wrappers (StatsRecorder, FrameStack, ...)

Factories
---------
arch_factory.py     Unified agent creation: create_agent(), load_agent_from_config()
env_factory.py      Unified env creation: make_agent_env(), make_bot_env()
wrapper_factory.py  Unified wrapper application: apply_env_wrappers()

Modules
-------
paths.py         Canonical output paths (RUNS_DIR, RECORDINGS_DIR, ...)
"""
