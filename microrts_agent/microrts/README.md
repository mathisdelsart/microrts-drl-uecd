# `microrts_agent/microrts/`

The vendored MicroRTS Java engine + our JNI bridge. Everything the
Python side talks to on the Java side lives here.

## Layout

| Path | What it is |
|---|---|
| [`microrts.jar`](microrts.jar) | The MicroRTS engine itself (vendored). Built by upstream; we don't compile it. |
| [`lib/`](lib/) | Vendored compiled JARs that the engine and bridge depend on at runtime (logger, GSON, ...) and our compiled `bridge.jar`. |
| [`maps/`](maps/) | All MicroRTS map XMLs (open + closed competition + assorted misc). |
| [`src/`](src/) | **Our** JNI bridge source (Java). See [`src/README.md`](src/README.md) for the full per-field contract (observation / action / mask / reward). |
| [`build_bridge.sh`](build_bridge.sh) | Compiles `src/` against `microrts.jar` and writes `lib/bridge.jar`. Always run after editing `src/`. |

## The bridge

Python <-> Java is mediated by a small set of classes in `src/`:

- `JNIGridnetVecClient`: per-env JNI client; mirrors the
  `MicroRTSGridnetVecEnv` Python contract.
- `JNIClient`: shared singleton, drives `reset`/`step`/`getMasks`/`close`.
- Reward functions: each piece of shaped reward (kills, productions,
  resource harvesting, ...) is a separate `RewardFunctionInterface`
  implementation.

Build:

```bash
bash microrts_agent/microrts/build_bridge.sh
```

The committed `lib/bridge.jar` is **Java 17 bytecode**. The build
script enforces `javac --release 17`; a JDK >= 17 must be on PATH.

## Maps

The two canonical map families are:

- **Open competition** (training maps): under `maps/open_competition/`,
  16x16 sizes typically (e.g. `basesWorkers16x16A.xml`).
- **Closed competition** (held-out): under `maps/closed_competition/`,
  larger sizes including 24x24 and 32x32.

The registry [`../registries/maps.py`](../registries/maps.py) exposes
these as `COMPETITION_OPEN_MAPS` / `COMPETITION_CLOSED_MAPS`.

The classpath used by Python (in
[`../envs/base_vec_env.py`](../envs/base_vec_env.py)) is `microrts.jar`
+ everything under `lib/` + every vendored bot JAR under
[`../bots/<Bot>/`](../bots/).

Detailed reference for every field the bridge exchanges with Python
(observation channels, action heads, mask layout, reward signals) is in
[`src/README.md`](src/README.md).
