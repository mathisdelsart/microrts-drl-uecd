"""Opponent selection and configuration for bot training environments."""


def _parse_opponent_list(spec, num_bot_envs):
    """Parse 'BotA:3,BotB:2' into (labels, summary_string). Validates counts and names."""
    from microrts_agent.registries.ai import AI_MAPPING

    # Split comma-separated entries and strip whitespace
    entries = [e.strip() for e in spec.split(",") if e.strip()]
    labels = []
    parts = []
    for entry in entries:
        if ":" not in entry:
            raise ValueError(f"Bad --opponent-list entry '{entry}': expected 'BotName:count'")
        name, count_str = entry.rsplit(":", 1)
        name = name.strip()

        # Validate bot name against the AI registry
        if name not in AI_MAPPING:
            raise ValueError(
                f"Unknown opponent '{name}' in --opponent-list. Valid: {sorted(AI_MAPPING.keys())}"
            )
        try:
            count = int(count_str.strip())
        except ValueError:
            raise ValueError(
                f"Bad count '{count_str}' for opponent '{name}' in --opponent-list"
            ) from None
        if count < 1:
            raise ValueError(f"Count for '{name}' must be >= 1, got {count}")
        labels.extend([name] * count)
        parts.append(f"{count} {name}")

    # Total env count must match --num-bot-envs exactly
    total = len(labels)
    if total != num_bot_envs:
        raise ValueError(
            f"--opponent-list sums to {total} but --num-bot-envs is {num_bot_envs}. "
            f"They must match."
        )
    summary = "CUSTOM (" + " + ".join(parts) + ")"
    return labels, summary


def build_opponent_config(args):
    """Build per-env opponent type lists from CLI args. Returns (labels, summary_string)."""
    # --opponent-list overrides everything
    if args.opponent_list:
        return _parse_opponent_list(args.opponent_list, args.num_bot_envs)

    if args.diverse_opponents:
        n = args.num_bot_envs
        rpt = args.rush_per_type

        # Baseline mode uses full-obs rush bots; normal mode uses partial-obs
        if args.baseline:
            rush_worker = "WorkerRushAI"
            rush_light = "LightRushAI"
        else:
            rush_worker = "WorkerRush"
            rush_light = "LightRush"

        # 3 rush types (random, worker, light), each up to rush_per_type envs
        # Remaining envs go to hard bots (CoacAI + Mayari)
        n_rush = min(rpt, n // 3) if n >= 3 else 0
        n_remaining = n - 3 * n_rush

        # Baseline only uses CoacAI; normal splits between CoacAI and Mayari
        if args.baseline:
            n_coac = n_remaining
            n_mayari = 0
        else:
            n_coac = n_remaining // 2
            n_mayari = n_remaining - n_coac

        labels = (
            ["CoacAI"] * n_coac
            + (["Mayari"] * n_mayari if n_mayari > 0 else [])
            + ["RandomBiasedAI"] * n_rush
            + [rush_worker] * n_rush
            + [rush_light] * n_rush
        )

        # Build human-readable summary
        parts = []
        if n_coac > 0:
            parts.append(f"{n_coac} CoacAI")
        if n_mayari > 0:
            parts.append(f"{n_mayari} Mayari")
        if n_rush > 0:
            parts.append(f"{n_rush} random + {n_rush} {rush_worker} + {n_rush} {rush_light}")
        mode = "BASELINE" if args.baseline else "DIVERSE"
        summary = f"{mode} ({' + '.join(parts)})"
    else:
        # Uniform: all envs run the same opponent
        labels = [args.opponent] * args.num_bot_envs
        summary = f"{args.num_bot_envs}x {args.opponent}"
    return labels, summary
