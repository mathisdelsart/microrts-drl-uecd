"""
Unified architecture creation for MicroRTS PPO agents.

All agent instantiation must go through this factory:
  - ARCHITECTURE_REGISTRY    — name -> class mapping
  - create_agent()           — from CLI args (training)
  - load_agent_from_config() — from saved config.json (eval, resume)

The actual architecture classes live in lib/architectures/ and should
not be imported directly by external code.
"""

import json
import os
import sys

import torch

from microrts_agent.lib.architectures.gridnet import GridNetAgent
from microrts_agent.lib.architectures.impala import IMPALAAgent
from microrts_agent.lib.architectures.impala_entity import IMPALAEntityAgent
from microrts_agent.lib.architectures.unet import IMPALAUNetAgent
from microrts_agent.lib.architectures.unet_entity import IMPALAUNetEntityAgent
from microrts_agent.lib.architectures.unet_entity_cbam import IMPALAUNetEntityCBAMAgent
from microrts_agent.lib.architectures.unet_entity_cbam_deep import IMPALAUNetEntityCBAMV2Agent

ARCHITECTURE_REGISTRY = {
    "gridnet": GridNetAgent,
    "impala": IMPALAAgent,
    "impala_entity": IMPALAEntityAgent,
    "unet": IMPALAUNetAgent,
    "unet_entity": IMPALAUNetEntityAgent,
    "unet_entity_cbam": IMPALAUNetEntityCBAMAgent,
    "unet_entity_cbam_deep": IMPALAUNetEntityCBAMV2Agent,
}


def _build_agent_kwargs(cfg):
    """Extract architecture kwargs from a config dict.
    Works with both argparse Namespace (via vars()) and saved config.json dicts.
    Returns (AgentClass, constructor_kwargs)."""
    AgentClass = ARCHITECTURE_REGISTRY[cfg["architecture"]]

    kwargs = {
        "obs_channels": cfg["obs_channels"],
        "action_nvec": cfg["action_nvec"],
        "dual_value_heads": cfg.get("dual_value_heads", False),
        "aux_tasks": cfg.get("aux_tasks", None),
    }

    # Architecture-specific
    if cfg.get("arch_channels") is not None:
        kwargs["channels"] = cfg["arch_channels"]
    if cfg.get("constant_channels"):
        kwargs["constant_channels"] = True
    if cfg.get("gelu") or cfg.get("use_gelu"):
        kwargs["use_gelu"] = True
    if cfg.get("spp_critic") or cfg.get("use_spp"):
        kwargs["use_spp"] = True

    # Ownership channel indices for entity extractor
    if cfg["architecture"] in (
        "impala_entity",
        "unet_entity",
        "unet_entity_cbam",
        "unet_entity_cbam_deep",
    ):
        if cfg.get("extended_obs"):
            p0_ch, p1_ch = 4, 5
        else:
            p0_ch, p1_ch = 11, 12
        frame_stack = cfg.get("frame_stack", 0)
        if frame_stack > 0:
            base_c = cfg["obs_channels"] // frame_stack
            offset = (frame_stack - 1) * base_c
            p0_ch += offset
            p1_ch += offset
        kwargs["p0_channel"] = p0_ch
        kwargs["p1_channel"] = p1_ch

    # Optional features (passed via **kwargs to _finalize)
    if cfg.get("triple_value_heads"):
        kwargs["triple_value_heads"] = True
    if cfg.get("hl_gauss"):
        kwargs["hl_gauss"] = True
        kwargs["hl_gauss_bins"] = cfg.get("hl_gauss_bins", 255)
    if cfg.get("autoregressive"):
        kwargs["autoregressive"] = True
        kwargs["ar_embed_dim"] = cfg.get("ar_embed_dim", 8)
    if cfg.get("hierarchical_mask"):
        kwargs["hierarchical_mask"] = True
    if cfg.get("popart"):
        kwargs["popart"] = True

    return AgentClass, kwargs


def create_agent(args, obs_channels, action_nvec, device):
    """Instantiate agent from CLI args + optional pretrained weights.
    Returns the agent on `device`.
    """
    cfg = vars(args)
    cfg["obs_channels"] = obs_channels
    cfg["action_nvec"] = action_nvec

    AgentClass, kwargs = _build_agent_kwargs(cfg)
    agent = AgentClass(**kwargs).to(device)

    if args.load_model:
        if not os.path.exists(args.load_model):
            print(f"ERROR: --load-model file not found: {args.load_model}")
            sys.exit(1)
        print(f"Loading pretrained weights from {args.load_model}")
        agent.load_state_dict(torch.load(args.load_model, map_location=device, weights_only=False))

    return agent


def load_agent_from_config(run_dir, device="cpu", checkpoint_name="agent.pt"):
    """Load an agent from a run directory using its config.json.
    Returns (agent, config) tuple — agent is in eval mode.
    """
    config_path = os.path.join(run_dir, "config.json")
    if not os.path.exists(config_path):
        raise FileNotFoundError(
            f"Config not found: {config_path}\nIs '{run_dir}' a valid run directory?"
        )

    with open(config_path) as f:
        try:
            config = json.load(f)
        except json.JSONDecodeError as e:
            raise ValueError(f"Corrupted config.json in {run_dir}: {e}") from e

    required_fields = ["architecture", "obs_channels", "action_nvec"]
    missing = [f for f in required_fields if f not in config]
    if missing:
        raise ValueError(f"Config in {run_dir} is missing required fields: {missing}")

    if config["architecture"] not in ARCHITECTURE_REGISTRY:
        raise ValueError(
            f"Unknown architecture '{config['architecture']}' in {run_dir}. "
            f"Available: {list(ARCHITECTURE_REGISTRY.keys())}"
        )

    AgentClass, kwargs = _build_agent_kwargs(config)
    agent = AgentClass(**kwargs)

    ckpt_path = os.path.join(run_dir, checkpoint_name)
    if not os.path.exists(ckpt_path):
        raise FileNotFoundError(
            f"Checkpoint not found: {ckpt_path}\nAvailable files: {os.listdir(run_dir)}"
        )

    state_dict = torch.load(ckpt_path, map_location=device, weights_only=False)
    # Full checkpoints store weights under "model"; weights-only files are the state_dict directly
    if "model" in state_dict and isinstance(state_dict["model"], dict):
        state_dict = state_dict["model"]
    missing, unexpected = agent.load_state_dict(state_dict, strict=False)
    if missing:
        import warnings

        warnings.warn(
            f"load_agent_from_config: {len(missing)} missing keys "
            f"(randomly initialized): {missing[:5]}{'...' if len(missing) > 5 else ''}",
            stacklevel=2,
        )
    if unexpected:
        import warnings

        warnings.warn(
            f"load_agent_from_config: {len(unexpected)} unexpected keys "
            f"(ignored): {unexpected[:5]}{'...' if len(unexpected) > 5 else ''}",
            stacklevel=2,
        )
    agent.to(device)
    agent.eval()
    return agent, config
