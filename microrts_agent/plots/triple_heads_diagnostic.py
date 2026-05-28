"""Triple value heads diagnostic plot.

Visualizes whether the three value heads (shaped, sparse, cost) behave as
expected during training. Compares feats_triple_heads_s2 against feats_baseline_s2
(same seed, no triple heads) to isolate the effect.

Usage:
    python microrts_agent/plots/triple_heads_diagnostic.py
"""

import os
import sys

import matplotlib.pyplot as plt
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from analyze_metrics import load_tensorboard

ROOT = os.path.join(os.path.dirname(__file__), "..", "..")
TRIPLE_DIR = os.path.join(ROOT, "outputs/runs/feat_ablation/feats_triple_heads_s2")
BASELINE_DIR = os.path.join(ROOT, "outputs/runs/feat_ablation/feats_baseline_s2")
FIGURES_DIR = os.path.join(ROOT, "figures")
os.makedirs(FIGURES_DIR, exist_ok=True)
OUT_PDF = os.path.join(FIGURES_DIR, "triple_heads_diagnostic_s2.pdf")


def ema(values, alpha=0.95):
    if len(values) == 0:
        return values
    out = np.empty_like(values, dtype=float)
    out[0] = values[0]
    for i in range(1, len(values)):
        out[i] = alpha * out[i - 1] + (1 - alpha) * values[i]
    return out


def plot_tag(ax, data, tag, label, color, smooth=0.9, log=False):
    if tag not in data:
        return False
    steps, values = data[tag]
    ax.plot(steps / 1e6, values, color=color, alpha=0.15, linewidth=0.8)
    ax.plot(steps / 1e6, ema(values, smooth), color=color, label=label, linewidth=1.8)
    if log:
        ax.set_yscale("log")
    return True


def main():
    print(f"Loading triple: {TRIPLE_DIR}")
    triple = load_tensorboard(TRIPLE_DIR)
    print(f"Loading baseline: {BASELINE_DIR}")
    baseline = load_tensorboard(BASELINE_DIR)

    fig, axes = plt.subplots(3, 2, figsize=(13, 12))
    fig.suptitle(
        "Triple Value Heads — Diagnostic (feats_triple_heads_s2 vs feats_baseline_s2)",
        fontsize=13,
        fontweight="bold",
    )

    # Panel 1: v_loss per head (log scale)
    ax = axes[0, 0]
    plot_tag(ax, triple, "losses/value_loss", "shaped", "#1f77b4", log=True)
    plot_tag(ax, triple, "losses/value_loss_sparse", "sparse (win-loss)", "#ff7f0e", log=True)
    plot_tag(ax, triple, "losses/value_loss_cost", "cost", "#2ca02c", log=True)
    ax.set_title("Value loss per head (log)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("v_loss")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 2: Explained variance (single, global)
    ax = axes[0, 1]
    plot_tag(ax, triple, "losses/explained_variance", "triple", "#1f77b4")
    plot_tag(ax, baseline, "losses/explained_variance", "baseline", "#888888")
    ax.set_title("Explained variance (global)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("EV")
    ax.axhline(0, color="k", linewidth=0.5, linestyle="--")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 3: Advantage weights schedule
    ax = axes[1, 0]
    plot_tag(ax, triple, "schedule/adv_w_shaped", "shaped", "#1f77b4", smooth=0)
    plot_tag(ax, triple, "schedule/adv_w_sparse", "sparse", "#ff7f0e", smooth=0)
    plot_tag(ax, triple, "schedule/adv_w_cost", "cost", "#2ca02c", smooth=0)
    ax.set_title("Advantage mixing weights (schedule)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("adv_w")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 4: Effective VF coef schedule
    ax = axes[1, 1]
    plot_tag(ax, triple, "schedule/eff_vf_coef", "shaped", "#1f77b4", smooth=0)
    plot_tag(ax, triple, "schedule/eff_vf_sparse", "sparse", "#ff7f0e", smooth=0)
    plot_tag(ax, triple, "schedule/eff_vf_cost", "cost", "#2ca02c", smooth=0)
    ax.set_title("Value loss coefficients (schedule)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("vf_coef")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 5: Reward weight schedule
    ax = axes[2, 0]
    plot_tag(ax, triple, "schedule/reward_weight_winloss", "winloss", "#d62728", smooth=0)
    plot_tag(ax, triple, "schedule/reward_weight_shaped_sum", "shaped (sum)", "#9467bd", smooth=0)
    ax.set_title("Reward component weights (schedule)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("weight")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # Panel 6: Comparison shaped head v_loss triple vs baseline
    ax = axes[2, 1]
    plot_tag(ax, triple, "losses/value_loss", "triple (shaped head)", "#1f77b4", log=True)
    plot_tag(ax, baseline, "losses/value_loss", "baseline", "#888888", log=True)
    ax.set_title("Shaped head v_loss: triple vs baseline")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("v_loss")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.97])
    plt.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PDF}")

    # Quick numeric sanity check
    print("\n=== Final values (last point) ===")
    for label, data in [("triple", triple), ("baseline", baseline)]:
        for tag in [
            "losses/value_loss",
            "losses/value_loss_sparse",
            "losses/value_loss_cost",
            "losses/explained_variance",
        ]:
            if tag in data:
                s, v = data[tag]
                print(f"  [{label}] {tag:40s} = {v[-1]:.4f}  @ step {s[-1]:,}")

    available_triple = sorted(
        [t for t in triple if "loss" in t or "schedule" in t or "charts" in t]
    )
    print(f"\n=== Triple run tags (loss/schedule/charts): {len(available_triple)} ===")
    for t in available_triple[:40]:
        print(f"  {t}")


if __name__ == "__main__":
    main()
