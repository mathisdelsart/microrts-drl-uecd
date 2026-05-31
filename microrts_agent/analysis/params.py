"""Count parameters added by each feature in the ablation study.

For each feature, instantiate UNet-Entity-CBAM-Deep with that feature enabled
(all others disabled) and report param count + delta vs baseline.

Usage:
    microrts-agent analysis params
"""

from microrts_agent.architectures.factory import _build_agent_kwargs

ACTION_NVEC = [6, 4, 4, 4, 4, 7, 49]
BASE_OBS_CHANNELS = 27  # default (non-extended, no frame stack)
EXT_OBS_CHANNELS = 73  # extended obs (RAI-style)


def baseline_cfg(obs_channels=BASE_OBS_CHANNELS):
    """Minimal baseline config: UNet-Entity-CBAM-Deep, no optional features."""
    return {
        "architecture": "unet_entity_cbam_deep",
        "arch_channels": 48,
        "obs_channels": obs_channels,
        "action_nvec": ACTION_NVEC,
        "dual_value_heads": False,
        "triple_value_heads": False,
        "aux_tasks": None,
        "extended_obs": False,
        "frame_stack": 0,
        "reserved_obs": False,
        "filtered_masks": False,
        "gelu": False,
        "spp_critic": False,
        "autoregressive": False,
        "ar_embed_dim": 8,
        "hl_gauss": False,
        "hl_gauss_bins": 255,
        "popart": False,
    }


def count_params(cfg):
    """Instantiate agent from cfg and return total parameter count."""
    AgentClass, kwargs = _build_agent_kwargs(cfg)
    agent = AgentClass(**kwargs)
    return sum(p.nelement() for p in agent.parameters())


def make(feature_name, **overrides):
    """Return (feature_name, cfg) with baseline + overrides applied."""
    cfg = baseline_cfg()
    # If extended_obs is enabled, switch to the RAI 73-channel obs
    if overrides.get("extended_obs"):
        cfg["obs_channels"] = EXT_OBS_CHANNELS
    # If frame_stack, multiply obs channels
    fs = overrides.get("frame_stack", 0)
    if fs and fs > 0:
        cfg["obs_channels"] = cfg["obs_channels"] * fs
    # Reserved obs wrapper adds +1 binary channel after frame stack
    if overrides.get("reserved_obs"):
        cfg["obs_channels"] = cfg["obs_channels"] + 1
    cfg.update(overrides)
    return feature_name, cfg


def main():
    # Baseline (no features)
    baseline_params = count_params(baseline_cfg())

    # Every feature from the ablation table
    configs = [
        make("Baseline (UNet-Entity-CBAM-Deep)"),
        make("Extended Obs (73ch)", extended_obs=True),
        make("Filt. Masks + Res. Obs", filtered_masks=True, reserved_obs=True),
        make("Prioritized Sampling"),  # no model change
        make("Opponent Modeling", aux_tasks=["opponent_modeling"]),
        make("PAE (keep=95%)"),  # no model change
        make("PopArt", popart=True),
        make("Aux Unit Count", aux_tasks=["unit_count"]),
        make("Bots + Self-Play + PFSP"),  # no model change
        make("GELU", gelu=True),
        make("Frame Stack (4)", frame_stack=4),
        make("Autoregressive", autoregressive=True, ar_embed_dim=8),
        make("Aux Spatial", aux_tasks=["spatial"]),
        make("SPP Critic", spp_critic=True),
        make("Aux Contrastive", aux_tasks=["contrastive"]),
        make("Build-Time Rewards"),  # no model change
        make("Augment Symmetry"),  # no model change
        make("Triple Value Heads", triple_value_heads=True, dual_value_heads=True),
        make("Adaptive Opponents"),  # no model change
        make("HL-Gauss", hl_gauss=True, hl_gauss_bins=255),
    ]

    rows = []
    for name, cfg in configs:
        try:
            p = count_params(cfg)
            delta = p - baseline_params
            rows.append((name, p, delta))
        except Exception as e:
            rows.append((name, None, str(e)))

    # Print table
    print(f"\n{'Feature':<40} {'Params':>12} {'Δ vs baseline':>18}")
    print("-" * 72)
    for name, p, delta in rows:
        if p is None:
            print(f"{name:<40} {'ERROR':>12}  {delta}")
        else:
            d_str = f"{delta:+,}" if delta != 0 else "—"
            print(f"{name:<40} {p:>12,} {d_str:>18}")


if __name__ == "__main__":
    main()
