"""
Per-tick decision time benchmark (self-play).

    python -m microrts_agent bench inference --agents data/agents/GridNet-SingleMap CoacAI
    python -m microrts_agent bench inference --agents data/agents/GridNet-SingleMap --device cuda --deterministic
"""

import argparse
import csv
import os
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import contextlib

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import torch

from microrts_agent.architectures.factory import load_agent_from_config
from microrts_agent.envs.base_vec_env import suppress_java_output
from microrts_agent.envs.factory import JVM_ARGS, make_agent_env, make_bot_env
from microrts_agent.obs_adapter import is_agent_dir
from microrts_agent.paths import BENCHMARKS_DIR
from microrts_agent.registries.ai import AI_MAPPING
from microrts_agent.wrappers.factory import apply_env_wrappers

# ── Java bot helpers ─────────────────────────────────────────────────────


def _configure_ai(ai, time_budget):
    """Set budgets and wrap with ContinuingAI (matches tournament behavior)."""
    import jpype

    AIWithBudget = jpype.JClass("ai.core.AIWithComputationBudget")
    InterruptibleAI = jpype.JClass("ai.core.InterruptibleAI")
    ContinuingAI = jpype.JClass("ai.core.ContinuingAI")

    if isinstance(ai, AIWithBudget):
        ai.setTimeBudget(time_budget)
        ai.setIterationsBudget(-1)
    if isinstance(ai, InterruptibleAI):
        ai = ContinuingAI(ai)

    return ai


# ── RL agent benchmark ───────────────────────────────────────────────────


def benchmark_rl(agent_path, num_games, map_override, max_steps, device, deterministic):
    """Self-play: RL agent vs itself. Times one forward pass per tick.
    Returns (tick_times_seconds, metadata_dict).
    """
    agent, config = load_agent_from_config(agent_path, device=str(device))
    n_params = sum(p.nelement() for p in agent.parameters())
    arch = config["architecture"]
    name = os.path.basename(agent_path.rstrip("/"))
    map_path = map_override or config["map"]

    print(f"  Architecture: {arch}  |  Params: {n_params:,}  |  Device: {device}")

    # 2 self-play envs (P0 + P1), no bots
    is_padded = config["multi_map"]
    env = make_agent_env(
        num_bot_envs=0,
        num_selfplay_envs=2,
        ai2s=[],
        map_paths=[map_path] * 2,
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

    # Warmup: 100 untimed steps to stabilize JIT, caches, allocations
    obs = torch.as_tensor(env.reset()).to(device)
    for _ in range(100):
        masks = torch.as_tensor(env.get_action_mask()).to(device)
        with torch.no_grad():
            out = agent.forward(obs, invalid_action_masks=masks)
        obs_np, _, dones, _ = env.step(out["action"].cpu().numpy().reshape(2, -1))
        obs = torch.as_tensor(obs_np).to(device)
        if dones[0]:
            obs = torch.as_tensor(env.reset()).to(device)
    if device.type == "cuda":
        torch.cuda.synchronize()

    # Play games, timing P0 forward pass only (batch=1)
    all_tick_times = []
    games_done = 0
    game_ticks = []
    obs = torch.as_tensor(env.reset()).to(device)

    while games_done < num_games:
        masks = torch.as_tensor(env.get_action_mask()).to(device)

        # Time P0 forward pass
        if device.type == "cuda":
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        if deterministic:
            a_p0 = agent.predict_batch(obs[0:1], masks[0:1])
        else:
            with torch.no_grad():
                a_p0 = agent.forward(obs[0:1], invalid_action_masks=masks[0:1])["action"]
        if device.type == "cuda":
            torch.cuda.synchronize()
        game_ticks.append(time.perf_counter() - t0)

        # P1 forward pass (same model, not timed)
        with torch.no_grad():
            if deterministic:
                a_p1 = agent.predict_batch(obs[1:2], masks[1:2])
            else:
                a_p1 = agent.forward(obs[1:2], invalid_action_masks=masks[1:2])["action"]

        # Step both envs
        all_actions = torch.zeros(2, *a_p0.shape[1:], dtype=a_p0.dtype, device=device)
        all_actions[0] = a_p0
        all_actions[1] = a_p1
        obs_np, _, dones, _ = env.step(all_actions.cpu().numpy().reshape(2, -1))
        obs = torch.as_tensor(obs_np).to(device)

        if dones[0]:
            games_done += 1
            avg_ms = np.mean(game_ticks) * 1000
            print(
                f"    Game {games_done}/{num_games}: "
                f"{len(game_ticks):>5d} ticks, avg={avg_ms:.3f} ms/tick"
            )
            all_tick_times.extend(game_ticks)
            game_ticks = []

    with contextlib.suppress(Exception):
        env.vec_client.close()

    return all_tick_times, {"name": name, "arch": arch, "params": n_params, "type": "rl"}


# ── Java bot benchmark ───────────────────────────────────────────────────


def benchmark_bot(bot_name, num_games, map_path, max_steps, time_budget):
    """Self-play: Java bot vs itself. Per-tick timing from JNI nanosecond counters.
    Returns (tick_times_seconds, metadata_dict).
    """
    from jpype.types import JInt

    print(f"  Type: Java bot  |  Budget: {time_budget} ms")

    env = make_bot_env(
        ai1s=[AI_MAPPING[bot_name]],
        ai2s=[AI_MAPPING[bot_name]],
        map_paths=[map_path],
        partial_obs=False,
        max_steps=max_steps,
        jvm_args=JVM_ARGS,
    )
    suppress_java_output()

    all_tick_times = []

    for game_idx in range(num_games):
        # Fresh AI instances per game
        client = env.vec_client.botClients[0]
        client.ai1 = AI_MAPPING[bot_name](env.real_utt)
        client.ai2 = AI_MAPPING[bot_name](env.real_utt)
        client.timeBudgetMs = time_budget
        client.reset(JInt(0))
        client.ai1 = _configure_ai(client.ai1, time_budget)
        client.ai2 = _configure_ai(client.ai2, time_budget)

        game_ticks = []
        steps = 0

        while True:
            # Read JNI nanosecond counter before/after game step
            ns_before = int(client.ai1TotalTimeNs)
            try:
                response = client.gameStep(JInt(0))
            except Exception as e:
                print(f"    [ERROR] gameStep failed at step {steps}: {e}")
                break
            ns_after = int(client.ai1TotalTimeNs)
            game_ticks.append((ns_after - ns_before) / 1e9)

            steps += 1
            if bool(list(response.done)[0]) or steps >= max_steps:
                break

        all_tick_times.extend(game_ticks)
        avg_ms = np.mean(game_ticks) * 1000 if game_ticks else 0
        print(
            f"    Game {game_idx + 1}/{num_games}: "
            f"{len(game_ticks):>5d} ticks, avg={avg_ms:.3f} ms/tick"
        )

    with contextlib.suppress(Exception):
        env.vec_client.close()

    return all_tick_times, {"name": bot_name, "arch": "java", "params": 0, "type": "bot"}


# ── Output ───────────────────────────────────────────────────────────────


def save_csv(results, csv_path):
    """Save per-agent timing summary (mean, median, p95, p99, std)."""
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
        for r in results:
            t_ms = np.array(r["times"]) * 1000
            writer.writerow(
                {
                    "agent": r["name"],
                    "type": r["type"],
                    "architecture": r["arch"],
                    "params": r["params"],
                    "num_ticks": len(r["times"]),
                    "mean_ms": f"{np.mean(t_ms):.4f}",
                    "median_ms": f"{np.median(t_ms):.4f}",
                    "p95_ms": f"{np.percentile(t_ms, 95):.4f}",
                    "p99_ms": f"{np.percentile(t_ms, 99):.4f}",
                    "std_ms": f"{np.std(t_ms):.4f}",
                }
            )
    print(f"\n  CSV saved: {csv_path}")


def generate_plot(results, output_path):
    """Boxplot (left) + CDF (right) comparison PDF."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

    # Scale colors to any number of agents
    cmap = plt.cm.tab20 if len(results) > 10 else plt.cm.tab10
    colors = [cmap(i / max(len(results) - 1, 1)) for i in range(len(results))]
    fig.patch.set_facecolor("white")

    # Sort by median ascending
    entries = []
    for r in results:
        t_ms = np.array(r["times"]) * 1000
        label = (
            f"{r['name']}\n({r['params'] / 1e6:.2f}M)"
            if r["params"] > 0
            else f"{r['name']}\n(Java bot)"
        )
        entries.append((label, r["name"], t_ms, float(np.median(t_ms))))
    entries.sort(key=lambda x: x[3])

    # ── Boxplot ──
    ax1.set_facecolor("white")
    bp = ax1.boxplot(
        [e[2] for e in entries],
        tick_labels=[e[0] for e in entries],
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
        patch.set_facecolor(colors[i % len(colors)])
        patch.set_alpha(0.75)
        patch.set_edgecolor("black")
        patch.set_linewidth(1.3)
    for w in bp["whiskers"]:
        w.set(linewidth=1.3, linestyle="--", alpha=0.6)
    for c in bp["caps"]:
        c.set(linewidth=1.3)

    # 100ms budget line (horizontal, soft red)
    ax1.axhline(
        100, color="#CC4444", linestyle="--", linewidth=1.5, alpha=0.6, label="100 ms budget"
    )

    ax1.set_ylabel("Decision Time (ms)", fontsize=13, fontweight="bold")
    ax1.set_title("Per-Tick Decision Time", fontsize=16, fontweight="bold", pad=8)
    ax1.grid(False)
    ax1.yaxis.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax1.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax1.spines[spine].set_visible(False)
    ax1.legend(
        handles=[
            mlines.Line2D([], [], color="#1f4e79", linewidth=2.2, label="Median"),
            mlines.Line2D([], [], color="#1f4e79", linewidth=1.5, linestyle="--", label="Mean"),
            mlines.Line2D(
                [],
                [],
                color="#CC4444",
                linewidth=1.5,
                linestyle="--",
                alpha=0.6,
                label="100 ms budget",
            ),
        ],
        loc="upper right",
        fontsize=9,
        framealpha=0.9,
        edgecolor="#cccccc",
    )

    # ── CDF ──
    ax2.set_facecolor("white")
    for i, (_, sname, t_ms, _) in enumerate(entries):
        sorted_t = np.sort(t_ms)
        cdf = np.arange(1, len(sorted_t) + 1) / len(sorted_t)
        # Stop plotting once CDF reaches 1.0 (trim flat tail)
        last_idx = len(cdf)  # default: all points
        for j in range(len(cdf) - 1, 0, -1):
            if cdf[j] < 1.0:
                last_idx = j + 2  # include one point at 1.0
                break
        ax2.plot(
            sorted_t[:last_idx],
            cdf[:last_idx],
            color=colors[i % len(colors)],
            linewidth=2,
            label=sname,
            alpha=0.85,
        )

    # 100ms budget line (vertical, soft red)
    ax2.axvline(
        100, color="#CC4444", linestyle="--", linewidth=1.5, alpha=0.6, label="100 ms budget"
    )

    ax2.set_xlabel("Decision Time (ms)", fontsize=13, fontweight="bold")
    ax2.set_ylabel("CDF", fontsize=13, fontweight="bold")
    ax2.set_title("Cumulative Distribution", fontsize=16, fontweight="bold", pad=8)
    ax2.legend(fontsize=9, framealpha=0.9, edgecolor="#cccccc")
    ax2.grid(False)
    ax2.yaxis.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax2.xaxis.grid(True, linestyle="--", alpha=0.25, color="lightgray")
    ax2.set_axisbelow(True)
    for spine in ["top", "right"]:
        ax2.spines[spine].set_visible(False)
    ax2.set_ylim(0, 1.02)

    plt.tight_layout()
    plt.savefig(output_path, bbox_inches="tight", format="pdf", dpi=300)
    plt.close(fig)
    print(f"  Plot saved: {output_path}")


# ── Printing ─────────────────────────────────────────────────────────────


def print_banner(args, rl_agents, bot_agents, device):
    """Print benchmark configuration banner."""
    action_mode = "deterministic" if args.deterministic else "stochastic"
    map_label = args.map or "(from config)"
    print(f"\n{'=' * 60}")
    print("  Decision Time Benchmark (self-play)")
    print(f"{'=' * 60}")
    print(f"  Agents:     {len(rl_agents) + len(bot_agents)}")
    for a in rl_agents:
        print(f"              - {os.path.basename(a.rstrip('/'))} (RL)")
    for b in bot_agents:
        print(f"              - {b} (Java bot)")
    print(f"  Games:      {args.games}")
    print(f"  Map:        {map_label}")
    if rl_agents:
        print(f"  Device:     {device}")
        print(f"  RL actions: {action_mode}")
    if bot_agents:
        print(f"  Bot budget: {args.time_budget} ms")
    print(f"{'=' * 60}")


def print_summary(results):
    """Print final summary table sorted by median time."""
    print(f"\n{'=' * 70}")
    print("  SUMMARY")
    print(f"{'=' * 70}")
    print(
        f"  {'Agent':25s} {'Type':6s} {'Arch':12s} "
        f"{'Mean ms':>8s} {'Med ms':>8s} {'P95 ms':>8s} {'Params':>10s}"
    )
    print(f"  {'-' * 75}")
    for r in sorted(results, key=lambda x: np.median(np.array(x["times"]) * 1000)):
        t_ms = np.array(r["times"]) * 1000
        params_str = f"{r['params']:>10,}" if r["params"] > 0 else "       N/A"
        print(
            f"  {r['name']:25s} {r['type']:6s} {r['arch']:12s} "
            f"{np.mean(t_ms):>8.3f} {np.median(t_ms):>8.3f} "
            f"{np.percentile(t_ms, 95):>8.3f} {params_str}"
        )
    print(f"{'=' * 70}")


def print_agent_result(tick_times, label):
    """Print summary line after benchmarking one agent."""
    if len(tick_times) == 0:
        print("  Total: 0 ticks  |  no timing data recorded")
        return
    t_ms = np.array(tick_times) * 1000
    print(
        f"  Total: {len(tick_times)} ticks  |  "
        f"mean={np.mean(t_ms):.3f} ms  |  "
        f"median={np.median(t_ms):.3f} ms  |  "
        f"p95={np.percentile(t_ms, 95):.3f} ms"
    )


# ── CLI + main ───────────────────────────────────────────────────────────


def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark per-tick decision time (self-play)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--agents", nargs="+", required=True, help="RL run dirs and/or Java bot names"
    )
    parser.add_argument("--games", type=int, default=5, help="games per agent (default: 5)")
    parser.add_argument(
        "--map", type=str, default=None, help="map path (default: from agent config)"
    )
    parser.add_argument(
        "--max-steps", type=int, default=3000, help="max steps per game (default: 3000)"
    )
    parser.add_argument(
        "--time-budget", type=int, default=100, help="bot time budget ms (default: 100)"
    )
    parser.add_argument(
        "--deterministic", action="store_true", help="greedy argmax for RL (instead of stochastic)"
    )
    parser.add_argument(
        "-o",
        "--output",
        type=str,
        default=None,
        help="output directory (default: outputs/inference_bench/<agents>_<map>/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Classify agents into RL vs Java bot
    rl_agents, bot_agents = [], []
    for a in args.agents:
        if is_agent_dir(a):
            rl_agents.append(a)
        elif a in AI_MAPPING:
            bot_agents.append(a)
        else:
            print(f"ERROR: '{a}' is neither an RL run dir nor a known bot.")
            print(f"  Known bots: {sorted(AI_MAPPING.keys())}")
            sys.exit(1)

    # Build output folder name: agents_mapname (e.g. "overfit_1map_s1_CoacAI_basesWorkers16x16A")
    agent_names = [os.path.basename(a.rstrip("/")) for a in rl_agents] + bot_agents
    map_name = Path(args.map).stem if args.map else "default_map"
    folder_name = "_".join(agent_names) + f"_{map_name}"
    output_dir = Path(args.output) if args.output else BENCHMARKS_DIR / folder_name
    os.makedirs(output_dir, exist_ok=True)

    print_banner(args, rl_agents, bot_agents, device)

    # Benchmark each agent
    results = []

    for agent_path in rl_agents:
        name = os.path.basename(agent_path.rstrip("/"))
        print(f"\n  --- {name} (RL) ---")
        tick_times, meta = benchmark_rl(
            agent_path, args.games, args.map, args.max_steps, device, args.deterministic
        )
        results.append({**meta, "times": tick_times})
        print_agent_result(tick_times, name)

    for bot_name in bot_agents:
        print(f"\n  --- {bot_name} (Java bot) ---")
        map_path = args.map or "maps/open_competition/basesWorkers16x16A.xml"
        tick_times, meta = benchmark_bot(
            bot_name, args.games, map_path, args.max_steps, args.time_budget
        )
        results.append({**meta, "times": tick_times})
        print_agent_result(tick_times, bot_name)

    # Output
    print_summary(results)
    save_csv(results, output_dir / "inference_time.csv")
    generate_plot(results, output_dir / "inference_time.pdf")

    # Force exit (JVM daemon threads prevent clean shutdown)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    main()
