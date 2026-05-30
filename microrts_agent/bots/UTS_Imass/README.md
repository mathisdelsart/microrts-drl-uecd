# UTS_Imass

IEEE-CoG MicroRTS competition entrant (Tsinghua University team, 2019). A
hybrid AI that combines online planning with a Python inference server
running a learned policy.

## Layout

| Path | Purpose |
|---|---|
| `Remote_MicroRTS_Java/` | Java side — the in-engine bot that ships in the tournament JAR; opens a TCP connection to the Python server on `localhost:9823` for inference. |
| `UTS_Imass_2019_Server/` | Python side — the 2019 inference server (`UTS_Imass_Server.py`) and its supporting modules; includes the native BL_JPS pathfinding extension. |
| `MicroRTS_Modifications/` | Patches applied on top of the upstream MicroRTS engine to expose the data the bot needs at decision time. |
| `training_data/` | Per-map data folders consumed by the bot via `preGameAnalysis(gs, budget, folder)`. |
| `build.sh` | Compiles `Remote_MicroRTS_Java/` against the vendored `microrts.jar`. |
| `start_server.sh` | Launches the Python inference server on port 9823. |

## Python 3.6 requirement

The native `BL_JPS` pathfinding module shipped with `UTS_Imass_2019_Server/`
is compiled against the CPython 3.6 ABI and cannot be loaded under any
newer interpreter. `start_server.sh` searches for a 3.6 interpreter in
this order:

1. A dedicated env at `<repo-root>/uts_imass_env/bin/python` (created by
   `bash setup/cluster.sh` via micromamba, the only path that works on a
   cluster with no system Python 3.6).
2. `pyenv` installations (`pyenv install 3.6.15 && pyenv local 3.6.15`).
3. A system-wide `python3.6` on `PATH`.

If none of the three are found the server refuses to start and the bot
is unavailable in tournaments. Every other feature of the project
remains functional.

## Tournament use

Tournaments that include `UTS_Imass` auto-start the Python server before
the first match (`microrts_agent.tournament.uts_imass_server`) and stop
it on exit. To launch the server manually for testing:

```bash
bash microrts_agent/bots/UTS_Imass/start_server.sh
```

## Upstream

[Tsinghua University, IEEE-CoG MicroRTS competition 2019 entry](https://github.com/santiontanon/microrts/wiki/2019-MicroRTS-AI-Competition-Entries).
The contents under `UTS_Imass_2019_Server/` and `Remote_MicroRTS_Java/`
are vendored as-is from the competition submission; only the path levels
in `start_server.sh` have been adjusted to the in-tree location.
