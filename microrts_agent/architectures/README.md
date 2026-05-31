# `microrts_agent/architectures/`

Neural-network policies + their building blocks. Every architecture used
in the thesis lives here, and every shipped agent under
[`../../data/agents/`](../../data/agents/) and the ablations under
[`../../data/ablation/arch/`](../../data/ablation/arch/) is reproducible
from this directory + the run's `config.json`.

## Files

| File | What it is |
|---|---|
| [`base_actor_critic.py`](base_actor_critic.py) | Base class. Implements the shared forward / `get_action` / `get_value` plumbing, the dual/triple value-head logic, optional HL-Gauss heads, and the autoregressive action sampler. |
| [`gridnet.py`](gridnet.py) | Baseline policy: GridNet-style ConvNet + flat action head. |
| [`impala.py`](impala.py) | IMPALA residual-tower variant. |
| [`impala_entity.py`](impala_entity.py) | IMPALA tower + entity-transformer mixer. |
| [`unet.py`](unet.py) | U-Net policy (encoder + decoder + skip connections). |
| [`unet_entity.py`](unet_entity.py) | U-Net + entity transformer at the bottleneck. |
| [`unet_entity_cbam.py`](unet_entity_cbam.py) | U-Net + entity transformer + CBAM blocks (channel + spatial attention). |
| [`unet_entity_cbam_deep.py`](unet_entity_cbam_deep.py) | Final UECD architecture: deeper U-Net + CBAM + entity transformer. Used by every shipped UECD-* agent. |
| [`factory.py`](factory.py) | `ARCHITECTURE_REGISTRY`, `create_agent(cfg)`, `_build_agent_kwargs(cfg)`, `load_agent_from_config(config_path)`. |
| [`features/`](features/) | Optional submodules pluggable into any base: autoregressive head, auxiliary heads, HL-Gauss, PopArt. |

## Registry

```python
from microrts_agent.architectures.factory import ARCHITECTURE_REGISTRY

list(ARCHITECTURE_REGISTRY)
# ['gridnet', 'impala', 'impala_entity', 'unet', 'unet_entity',
#  'unet_entity_cbam', 'unet_entity_cbam_deep']
```

The registry is the only entry point: `train.py`, `evaluate.py` and the
ablation drivers all go through `create_agent(cfg)` /
`load_agent_from_config(path)`.

## Loading a shipped agent

```python
from microrts_agent.architectures.factory import load_agent_from_config

agent = load_agent_from_config("data/agents/UECD-SingleMap-Best/config.json")
```

`load_agent_from_config` reads the run's `config.json`, dispatches to the
right architecture class, restores the optional features
(`autoregressive`, `hl_gauss`, `popart`, `aux_tasks`, `dual_value_heads`,
`triple_value_heads`, ...), and loads `agent.pt` from the same directory.
