# Java↔Python bridge

Hand-written JNI layer that lets a PyTorch agent in Python drive the Java
MicroRTS engine. The Java sources in this directory are the **source of
truth** for `microrts_agent/microrts/lib/bridge.jar`; the compiled JAR is
committed to keep clones fast, but `bash microrts_agent/microrts/build_bridge.sh`
rebuilds it from `src/` against the vendored `microrts.jar` (Java 17).

## Module layout

```
src/
├── ai/jni/
│   ├── JNIInterface.java          # contract every JNI agent must implement
│   ├── JNIAI.java                 # stateless RL-side proxy (Python actions -> PlayerAction)
│   ├── Response.java              # one tick of (obs, mask, rewards, dones, terminalReward, terminalDone)
│   └── Responses.java             # batched response across N envs
├── ai/reward/                     # 10 RewardFunctionInterface subclasses (one signal each)
│   ├── WinDrawLossRewardFunction.java
│   ├── MilitaryScoreRewardFunction.java
│   ├── ResourceGatherRewardFunction.java
│   ├── AttackRewardFunction.java
│   ├── ProduceBaseRewardFunction.java
│   ├── ProduceBarracksRewardFunction.java
│   ├── ProduceWorkerRewardFunction.java
│   ├── ProduceLightUnitRewardFunction.java
│   ├── ProduceHeavyUnitRewardFunction.java
│   └── ProduceRangedUnitRewardFunction.java
├── ai/wrapper/
│   └── GameStateWrapper.java      # raw obs extraction + action-mask computation
└── tests/                         # the actual env clients consumed by Python
    ├── JNIGridnetClient.java          # 1 RL agent vs 1 Java bot
    ├── JNIGridnetClientSelfPlay.java  # 2 RL agents (self-play)
    ├── JNIBotClient.java              # bot vs bot (used by the tournament runner)
    ├── JNIGridnetVecClient.java       # vectorised batch over N envs
    └── FilteredMaskClient.java        # destination-aware action masking variant
```

The package names (`ai.jni`, `ai.reward`, `ai.wrapper`, `tests`) are kept
on purpose: they match the conventions of the upstream MicroRTS code so
the bridge sources can sit alongside vendored Java classes at runtime.

## Observation contract

`JNIAI.getObservation()` returns `int[13][H][W]`: 13 raw feature planes per
cell, defined in `GameStateWrapper.numVectorObservationFeatureMaps`:

| Plane | Meaning | Range |
|---|---|---|
| 0  | unit type id           | 0–6 |
| 1  | unit hit points        | 0–127 (clamped) |
| 2  | unit resources carried | 0–127 (clamped) |
| 3  | owner                  | 0 (none), 1 (P0), 2 (P1) |
| 4–8 | action type one-hot (none, move, harvest, return, produce) |
| 9–12| action direction one-hot (up, right, down, left) |
| 13 | ETA to action completion (offset by −128) |
| terrain | included once per cell (resource / wall / ...) |

Python's encoders turn this into the 29-channel "gridnet" stack and the
73-channel "extended" stack consumed by the policies under
`microrts_agent/architectures/`. The encoding logic is entirely Python-side
, Java emits the raw planes only.

## Action contract

Every tick Python sends `int[numUnits][8]` to `JNIAI.getAction()`. Each row
encodes one unit's action:

| Column | Field         | Semantics |
|---|---|---|
| 0 | `cellIdx`      | flattened cell index `y * W + x` (selects which unit) |
| 1 | `actionType`   | NONE / MOVE / HARVEST / RETURN / PRODUCE / ATTACK_LOCATION |
| 2 | `moveDir`      | direction for MOVE (0–3) |
| 3 | `harvestDir`   | direction for HARVEST |
| 4 | `returnDir`    | direction for RETURN |
| 5 | `produceDir`   | direction for PRODUCE |
| 6 | `produceType`  | RESOURCE / BASE / BARRACKS / WORKER / LIGHT / HEAVY / RANGED |
| 7 | `attackTarget` | offset into the attack grid (size `(2*range+1)^2`) |

`PlayerAction.fromVectorAction()` maps each row to a `UnitAction`;
`PlayerAction.fillWithNones()` then assigns `TYPE_NONE` to every idle unit
not in `actions[]`, because the engine requires *every* unit to have an
action each tick.

## Action mask contract

`JNIGridnetClient.getMasks(player)` returns
`int[H][W][23 + numUnitTypes + (2*maxAttackRange+1)^2]`. The leading 23
slots cover `actionType` (6) + `moveDir` (4) + `harvestDir` (4) + `returnDir` (4)
+ `produceDir` (4) + 1 reserved, the middle band covers `produceType`,
and the trailing band covers attack-target offsets. Python applies the mask
before softmaxing the policy logits so illegal moves are never sampled.

## Reward contract

The RL client receives a `double[N]` reward vector per tick, one entry per
reward function passed to the constructor (`RewardFunctionInterface[] rfs`).
Each function implements `computeReward(maxplayer, minplayer, TraceEntry te,
GameState afterGs)` and writes one scalar; Python combines them with the
weights set in the training config.

Default thesis stack (see `microrts_agent/envs/`):
`WinDrawLoss, ResourceGather, ProduceWorker, ProduceBase, ProduceBarracks,
Attack, ProduceLightUnit, ProduceHeavyUnit, ProduceRangedUnit,
MilitaryScore`.

## Lifecycle

Per env:

```
reset(player)
  └─ Response { obs, mask, reward=0, done=false, terminalReward=0, terminalDone=false }

repeat
    gameStep(actions, player)
      └─ Response { obs, mask, reward, done, terminalReward, terminalDone }
    getMasks(player)
      └─ int[H][W][maskDim]

close()
```

`JNIGridnetVecClient` wraps `numSelfplayEnvs` `JNIGridnetClientSelfPlay`
clients plus `numBotEnvs` `JNIGridnetClient` clients and exposes the same
shape vectorised over the batch. **Auto-reset is built in**: when an env is
done or hits `maxSteps`, it resets at the next `step()` and the terminal
reward/done are preserved in `terminalReward[i]` / `terminalDone[i]` so
PyTorch can do bootstrapping correctly.

## Threading model

Single-threaded per env. The engine tick (Python action → both AIs decide
→ engine advances → reward functions evaluate) is synchronous on the JVM
thread. Vectorisation happens at the *batch* level: `JNIGridnetVecClient`
holds N independent client instances and Python iterates over them in a
loop (the inner loop releases the GIL during JPype JNI calls, so the
vectorised step actually overlaps Java work across envs in practice).

If you change the bridge while a Python process is running, the JVM is
already up: restart the Python process to pick up the new
`bridge.jar`. The `setup/*.sh` scripts always rebuild the bridge before
activating the env so a fresh terminal session is always consistent.

## Where Python touches this

- `microrts_agent/envs/`           : instantiates `JNIGridnetVecClient`, owns the lifecycle.
- `microrts_agent/obs_adapter.py`  : encodes the 13-plane raw obs into the 29/73-channel stack.
- `microrts_agent/registries/ai.py`: maps user-facing bot names ("CoacAI", "Mayari", …) to
                                       the Java class to pass as `ai2`.
- `microrts_agent/wrappers/`       : frame-stack, symmetry, action-mask filtering on top of
                                       the raw `Response`.

If you add a new reward function: drop a `.java` in `ai/reward/`, rebuild
the bridge, and add it to the `rfs` list constructed in `envs/`.
