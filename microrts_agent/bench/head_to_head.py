"""
Head-to-head per-tick decision time benchmark.

Plays RL agent (P0) vs Java bot (P1) on the same game sequence, timing
both sides on matched game states for a fair apples-to-apples comparison.

    python -m microrts_agent bench head2head --agent data/agents/UECD-SingleMap-Best \\
                                              --opponent RAISocketAI \\
                                              --games 5
"""

import argparse
import csv
import os
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import contextlib

import matplotlib.pyplot as plt
import numpy as np
import torch

from microrts_agent.architectures.factory import load_agent_from_config
from microrts_agent.envs.base_vec_env import suppress_java_output
from microrts_agent.envs.factory import JVM_ARGS, make_agent_env
from microrts_agent.registries.ai import AI_MAPPING
from microrts_agent.wrappers.factory import apply_env_wrappers


def benchmark_h2h(
    agent_path, opponent_name, num_games, map_path, max_steps, time_budget, device, deterministic
):
    """Play RL agent (P0) vs Java bot (P1) and time both sides on the same ticks.
    Returns (rl_tick_times, bot_tick_times, metadata).
    """
    agent, config = load_agent_from_config(agent_path, device=str(device))
    n_params = sum(p.nelement() for p in agent.parameters())
    arch = config["architecture"]
    rl_name = os.path.basename(agent_path.rstrip("/"))

    print(f"  RL agent: {rl_name}  |  Arch: {arch}  |  Params: {n_params:,}  |  Device: {device}")
    print(f"  Opponent: {opponent_name} (Java bot, budget {time_budget} ms)")

    is_padded = config["multi_map"]
    env = make_agent_env(
        num_bot_envs=1,
        num_selfplay_envs=0,
        ai2s=[AI_MAPPING[opponent_name]],
        map_paths=[map_path],
        player=0,
        partial_obs=config["partial_obs"],
        max_steps=max_steps,
        reward_weight=np.array(config["reward_weight"]),
        jvm_args=JVM_ARGS,
        padded=is_padded,
        max_height=config["max_map_size"] if is_padded else 0,
        max_width=config["max_map_size"] if is_padded else 0,
        alternate_players=False,
        filtered_masks=config["filtered_masks"],
        extended_obs=config["extended_obs"],
    )
    env = apply_env_wrappers(
        env,
        gamma=config["gamma"],
        frame_stack=config["frame_stack"],
        reserved_obs=config["reserved_obs"],
    )
    suppress_java_output()

    # Configure the Java bot's time budget
    try:
        from microrts_agent.tournament.game_loops import _configure_ai_with_budget

        client = env.vec_client.clients[0]
        client.ai2 = _configure_ai_with_budget(client.ai2, time_budget)
    except Exception:
        # Simpler fallback: just set time budget directly if available
        with contextlib.suppress(Exception):
            env.vec_client.clients[0].ai2.setTimeBudget(time_budget)

    # Warmup
    obs = torch.as_tensor(env.reset()).to(device)
    for _ in range(50):
        masks = torch.as_tensor(env.get_action_mask()).to(device)
        with torch.no_grad():
            out = agent.forward(obs, invalid_action_masks=masks)
        obs_np, _, dones, _ = env.step(out["action"].cpu().numpy().reshape(1, -1))
        obs = torch.as_tensor(obs_np).to(device)
        if dones[0]:
            obs = torch.as_tensor(env.reset()).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    rl_times = []
    bot_times = []
    games_done = 0
    obs = torch.as_tensor(env.reset()).to(device)
    client = env.vec_client.clients[0]

    while games_done < num_games:
        masks = torch.as_tensor(env.get_action_mask()).to(device)

        # --- Time RL forward pass ---
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if deterministic:
            action = agent.predict_batch(obs, masks)
        else:
            with torch.no_grad():
                action = agent.forward(obs, invalid_action_masks=masks)["action"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        rl_tick = time.perf_counter() - t0
        rl_times.append(rl_tick)

        # --- Time bot via JNI counter delta around env.step() ---
        # NOTE: ai2TotalTimeNs is reset to 0 on game auto-reset, so the
        # last tick of each game produces a negative delta. Skip those.
        ns_before = int(client.ai2TotalTimeNs)
        obs_np, _, dones, _ = env.step(action.cpu().numpy().reshape(1, -1))
        ns_after = int(client.ai2TotalTimeNs)
        delta_ns = ns_after - ns_before
        if delta_ns >= 0:
            bot_times.append(delta_ns / 1e9)
        # else: counter was reset (end of game) -- skip this tick
        #       (we lose one bot tick per game, out of ~1000)

        obs = torch.as_tensor(obs_np).to(device)

        if dones[0]:
            games_done += 1
            np.mean(rl_times[-len(rl_times) :]) * 1000 if rl_times else 0
            # Compute last-game stats
            print(
                f"    Game {games_done}/{num_games}: "
                f"RL mean={np.mean(rl_times) * 1000:.3f} ms, "
                f"Bot mean={np.mean(bot_times) * 1000:.3f} ms"
            )

    with contextlib.suppress(Exception):
        env.vec_client.close()

    return (
        rl_times,
        bot_times,
        {
            "rl_name": rl_name,
            "arch": arch,
            "params": n_params,
            "bot_name": opponent_name,
        },
    )


def save_csv(rl_times, bot_times, meta, csv_path):
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    with open(csv_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "agent",
                "type",
                "architecture",
                "params",
                "num_ticks",
                "mean_ms",
                "median_ms",
                "p95_ms",
                "p99_ms",
                "std_ms",
            ],
        )
        writer.writeheader()
        for label, times, arch, params, typ in [
            (meta["rl_name"], rl_times, meta["arch"], meta["params"], "rl"),
            (meta["bot_name"], bot_times, "java", 0, "bot"),
        ]:
            t_ms = np.array(times) * 1000
            writer.writerow(
                {
                    "agent": label,
                    "type": typ,
                    "architecture": arch,
                    "params": params,
                    "num_ticks": len(times),
                    "mean_ms": f"{np.mean(t_ms):.4f}",
                    "median_ms": f"{np.median(t_ms):.4f}",
                    "p95_ms": f"{np.percentile(t_ms, 95):.4f}",
                    "p99_ms": f"{np.percentile(t_ms, 99):.4f}",
                    "std_ms": f"{np.std(t_ms):.4f}",
                }
            )
    print(f"\n  CSV saved: {csv_path}")


def generate_plot(rl_times, bot_times, meta, output_path):
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))
    fig.patch.set_facecolor("white")

    rl_ms = np.array(rl_times) * 1000
    bot_ms = np.array(bot_times) * 1000

    labels = [
        f"{meta['rl_name']}\n({meta['params'] / 1e6:.2f}M)",
        f"{meta['bot_name']}\n(Java bot)",
    ]
    data = [rl_ms, bot_ms]
    colors = ["#1f77b4", "#ff7f0e"]

    # Boxplot
    ax1.set_facecolor("white")
    bp = ax1.boxplot(
        data,
        tick_labels=labels,
        patch_artist=True,
        showmeans=True,
        meanline=True,
        widths=0.5,
        medianprops={"color": "#1f4e79", "linewidth": 2.2},
        meanprops={"color": "#1f4e79", "linewidth": 1.5, "linestyle": "--"},
        flierprops={
            "marker": "o",
            "markersize": 2.5,
            "markerfacecolor": "#555555",
            "markeredgecolor": "none",
            "alpha": 0.3,
        },
    )
    for i, patch in enumerate(bp["boxes"]):
        patch.set_facecolor(colors[i])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.3)
    for w in bp["whiskers"]:
        w.set(linewidth=1.3, linestyle="--", alpha=0.6)
    for c in bp["caps"]:
        c.set(linewidth=1.3)

    ax1.axhline(
        100, color="#CC4444", linestyle="--", linewidth=1.5, alpha=0.6, label="100 ms budget"
    )
    ax1.set_ylabel("Decision Time (ms)", fontsize=13, fontweight="bold")
    ax1.set_title(
        "Per-Tick Decision Time (H2H: same game sequence)", fontsize=14, fontweight="bold", pad=8
    )
    ax1.grid(False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)

    # CDF
    ax2.set_facecolor("white")
    for i, (lbl, t_ms) in enumerate(zip(labels, data)):
        sorted_t = np.sort(t_ms)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        ax2.plot(sorted_t, cdf, color=colors[i], linewidth=2, label=lbl.split("\n")[0], alpha=0.85)
    ax2.axvline(
        100, color="#CC4444", linestyle="--", linewidth=1.5, alpha=0.6, label="100 ms budget"
    )
    ax2.set_xlabel("Decision Time (ms)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("CDF", fontsize=13, fontweight="bold")
    ax2.set_title("Decision Time CDF", fontsize=14, fontweight="bold", pad=8)
    ax2.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax2.set_axisbelow(True)
    ax2.legend(loc="lower right", fontsize=10, framealpha=0.9, edgecolor="#cccccc")
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  Plot saved: {output_path}")


def parse_args():
    parser = argparse.ArgumentParser(description="H2H per-tick decision time benchmark")
    parser.add_argument("--agent", type=str, required=True, help="RL agent run directory")
    parser.add_argument(
        "--opponent", type=str, required=True, help="Java bot name (from AI_MAPPING)"
    )
    parser.add_argument("--games", type=int, default=5)
    parser.add_argument(
        "--map", type=str, default=None, help="map path (default: from agent config)"
    )
    parser.add_argument("--max-steps", type=int, default=4000)
    parser.add_argument("--time-budget", type=int, default=100)
    parser.add_argument("--deterministic", action="store_true")
    parser.add_argument("-o", "--output", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Resolve map
    agent_config_path = Path(args.agent) / "config.json"
    if args.map:
        map_path = args.map
    else:
        import json

        with open(agent_config_path) as f:
            cfg = json.load(f)
        map_path = cfg.get("map", "maps/open_competition/basesWorkers16x16A.xml")

    # Output
    if args.output is None:
        tag = f"{os.path.basename(args.agent.rstrip('/'))}_vs_{args.opponent}"
        output_dir = f"outputs/inference_bench/h2h_{tag}"
    else:
        output_dir = args.output
    os.makedirs(output_dir, exist_ok=True)

    print(f"\n{'=' * 60}")
    print("  H2H Decision Time Benchmark")
    print(f"{'=' * 60}")
    print(f"  Agent:      {args.agent}")
    print(f"  Opponent:   {args.opponent}")
    print(f"  Games:      {args.games}")
    print(f"  Map:        {map_path}")
    print(f"  Device:     {device}")
    print(f"  Bot budget: {args.time_budget} ms")
    print(f"{'=' * 60}\n")

    rl_times, bot_times, meta = benchmark_h2h(
        args.agent,
        args.opponent,
        args.games,
        map_path,
        args.max_steps,
        args.time_budget,
        device,
        args.deterministic,
    )

    # Summary
    rl_ms = np.array(rl_times) * 1000
    bot_ms = np.array(bot_times) * 1000
    print(f"\n{'=' * 70}")
    print("  SUMMARY (H2H, same game sequence)")
    print(f"{'=' * 70}")
    print(f"  {'Agent':25s} {'Ticks':>6s} {'Mean ms':>8s} {'Med ms':>8s} {'P95 ms':>8s}")
    print(f"  {'-' * 70}")
    print(
        f"  {meta['rl_name']:25s} {len(rl_times):>6d} "
        f"{np.mean(rl_ms):>8.3f} {np.median(rl_ms):>8.3f} {np.percentile(rl_ms, 95):>8.3f}"
    )
    print(
        f"  {meta['bot_name']:25s} {len(bot_times):>6d} "
        f"{np.mean(bot_ms):>8.3f} {np.median(bot_ms):>8.3f} {np.percentile(bot_ms, 95):>8.3f}"
    )
    print(f"{'=' * 70}\n")

    save_csv(rl_times, bot_times, meta, os.path.join(output_dir, "h2h_inference_time.csv"))
    generate_plot(rl_times, bot_times, meta, os.path.join(output_dir, "h2h_inference_time.pdf"))


if __name__ == "__main__":
    main()
