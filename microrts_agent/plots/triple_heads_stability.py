"""Triple value heads — STABILITY diagnostic.

Focused on answering: "can I trust the sparse/cost heads to carry the signal
when the shaped reward fades out in a phased schedule?"

Key tests:
1. Cost head: is v_loss_cost stable over time? Rolling std should be low but
   non-zero (zero = dead prediction, high = diverging).
2. Sparse head: same test.
3. Reward components (MilitaryScore, WinDrawLoss, etc.): what's the *scale* of
   each reward signal? If sparse/cost rewards have tiny magnitude compared to
   shaped, their heads may be learning trivially and will fail to carry signal
   in a phased run.
4. Shaped head trajectory: does it plateau cleanly or drift?
5. Loss ratio sparse/shaped and cost/shaped: shows relative contribution.

Usage:
    python microrts_agent/plots/triple_heads_stability.py
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
OUT_PDF = os.path.join(FIGURES_DIR, "triple_heads_stability_s2.pdf")


def rolling_std(values, window=50):
    """Rolling std over a window. Returns array of same length."""
    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = np.std(values[lo : i + 1]) if i > 0 else 0
    return out


def rolling_mean(values, window=50):
    out = np.zeros_like(values, dtype=float)
    for i in range(len(values)):
        lo = max(0, i - window + 1)
        out[i] = np.mean(values[lo : i + 1])
    return out


def get(data, tag):
    if tag not in data:
        return None, None
    s, v = data[tag]
    return s / 1e6, v


def main():
    print(f"Loading triple: {TRIPLE_DIR}")
    triple = load_tensorboard(TRIPLE_DIR)
    print(f"Loading baseline: {BASELINE_DIR}")
    baseline = load_tensorboard(BASELINE_DIR)

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle(
        "Triple Value Heads — STABILITY TEST (feats_triple_heads_s2)\n"
        "Can sparse/cost heads carry signal when shaped fades? Check for flat-dead vs diverging.",
        fontsize=12,
        fontweight="bold",
    )

    # ── Panel 1: Cost head stability (loss + rolling std band) ──────────────
    ax = axes[0, 0]
    s, v = get(triple, "losses/value_loss_cost")
    if v is not None:
        mean = rolling_mean(v, 50)
        std = rolling_std(v, 50)
        ax.plot(s, v, color="#2ca02c", alpha=0.2, linewidth=0.7, label="raw")
        ax.plot(s, mean, color="#2ca02c", linewidth=2, label="rolling mean (50)")
        ax.fill_between(s, mean - std, mean + std, color="#2ca02c", alpha=0.2, label="±1σ")
        final_std = std[-100:].mean() if len(std) > 100 else std.mean()
        ax.text(
            0.02,
            0.95,
            f"final mean={mean[-1]:.5f}\nfinal std={final_std:.5f}\nratio σ/μ={final_std / max(mean[-1], 1e-9):.2%}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    ax.set_title("COST head: v_loss_cost (stability test)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("v_loss_cost")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 2: Sparse head stability ──────────────────────────────────────
    ax = axes[0, 1]
    s, v = get(triple, "losses/value_loss_sparse")
    if v is not None:
        mean = rolling_mean(v, 50)
        std = rolling_std(v, 50)
        ax.plot(s, v, color="#ff7f0e", alpha=0.2, linewidth=0.7, label="raw")
        ax.plot(s, mean, color="#ff7f0e", linewidth=2, label="rolling mean (50)")
        ax.fill_between(s, mean - std, mean + std, color="#ff7f0e", alpha=0.2, label="±1σ")
        final_std = std[-100:].mean() if len(std) > 100 else std.mean()
        ax.text(
            0.02,
            0.95,
            f"final mean={mean[-1]:.5f}\nfinal std={final_std:.5f}\nratio σ/μ={final_std / max(mean[-1], 1e-9):.2%}",
            transform=ax.transAxes,
            fontsize=9,
            va="top",
            bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.8},
        )
    ax.set_title("SPARSE (win-loss) head: v_loss_sparse (stability test)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("v_loss_sparse")
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 3: Shaped head trajectory vs baseline ─────────────────────────
    ax = axes[1, 0]
    s, v = get(triple, "losses/value_loss")
    if v is not None:
        ax.plot(s, v, color="#1f77b4", alpha=0.2, linewidth=0.7)
        ax.plot(s, rolling_mean(v, 50), color="#1f77b4", linewidth=2, label="triple (shaped)")
    s, v = get(baseline, "losses/value_loss")
    if v is not None:
        ax.plot(s, v, color="#888888", alpha=0.2, linewidth=0.7)
        ax.plot(s, rolling_mean(v, 50), color="#888888", linewidth=2, label="baseline (single)")
    ax.set_title("SHAPED head: triple vs baseline (does triple degrade shaped?)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("v_loss_shaped")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 4: Loss ratios (contribution of sparse/cost vs shaped) ────────
    ax = axes[1, 1]
    s_sh, v_sh = get(triple, "losses/value_loss")
    s_sp, v_sp = get(triple, "losses/value_loss_sparse")
    s_co, v_co = get(triple, "losses/value_loss_cost")
    if v_sh is not None and v_sp is not None:
        # interpolate to shared steps
        n = min(len(v_sh), len(v_sp), len(v_co))
        ratio_sp = rolling_mean(v_sp[:n], 50) / np.maximum(rolling_mean(v_sh[:n], 50), 1e-9)
        ratio_co = rolling_mean(v_co[:n], 50) / np.maximum(rolling_mean(v_sh[:n], 50), 1e-9)
        ax.plot(s_sh[:n], ratio_sp, color="#ff7f0e", linewidth=2, label="sparse / shaped")
        ax.plot(s_sh[:n], ratio_co, color="#2ca02c", linewidth=2, label="cost / shaped")
        ax.axhline(1.0, color="k", linewidth=0.5, linestyle="--", alpha=0.5)
    ax.set_title("Loss ratio: sparse/shaped and cost/shaped")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("ratio")
    ax.set_yscale("log")
    ax.legend(loc="best", fontsize=9)
    ax.grid(alpha=0.3)

    # ── Panel 5: Reward components (episodic returns) ───────────────────────
    ax = axes[2, 0]
    reward_tags = [
        ("charts/episodic_return/WinDrawLossRewardFunction", "WinDrawLoss (sparse)", "#ff7f0e"),
        ("charts/episodic_return/MilitaryScoreRewardFunction", "MilitaryScore", "#d62728"),
        ("charts/episodic_return/AttackRewardFunction", "Attack", "#9467bd"),
        ("charts/episodic_return/ProduceWorkerRewardFunction", "ProduceWorker", "#1f77b4"),
        ("charts/episodic_return/ResourceGatherRewardFunction", "ResourceGather", "#8c564b"),
    ]
    for tag, lbl, col in reward_tags:
        s, v = get(triple, tag)
        if v is not None:
            ax.plot(s, rolling_mean(v, 20), color=col, linewidth=1.6, label=lbl)
    ax.set_title("Reward components scale (episodic returns)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("return")
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.3)

    # ── Panel 6: Advantage weights + VF coef schedules (actual effective) ───
    ax = axes[2, 1]
    for tag, lbl, col, ls in [
        ("schedule/adv_w_shaped", "adv_w shaped", "#1f77b4", "-"),
        ("schedule/adv_w_sparse", "adv_w sparse", "#ff7f0e", "-"),
        ("schedule/adv_w_cost", "adv_w cost", "#2ca02c", "-"),
        ("schedule/eff_vf_coef", "vf_coef shaped", "#1f77b4", "--"),
        ("schedule/eff_vf_sparse", "vf_coef sparse", "#ff7f0e", "--"),
        ("schedule/eff_vf_cost", "vf_coef cost", "#2ca02c", "--"),
    ]:
        s, v = get(triple, tag)
        if v is not None:
            ax.plot(s, v, color=col, linestyle=ls, linewidth=1.5, label=lbl)
    ax.set_title("Effective schedules: adv_w (solid) + vf_coef (dashed)")
    ax.set_xlabel("Steps (M)")
    ax.set_ylabel("weight")
    ax.legend(loc="best", fontsize=8, ncol=2)
    ax.grid(alpha=0.3)

    plt.tight_layout(rect=[0, 0, 1, 0.95])
    plt.savefig(OUT_PDF, dpi=150, bbox_inches="tight")
    print(f"\nSaved: {OUT_PDF}")

    # ── Diagnostic summary ──────────────────────────────────────────────────
    print("\n" + "=" * 65)
    print("  TRIPLE HEADS STABILITY DIAGNOSTIC")
    print("=" * 65)
    for tag, name in [
        ("losses/value_loss", "shaped"),
        ("losses/value_loss_sparse", "sparse"),
        ("losses/value_loss_cost", "cost"),
    ]:
        if tag in triple:
            _, v = triple[tag]
            tail = v[-200:] if len(v) > 200 else v
            m = tail.mean()
            sd = tail.std()
            ratio = sd / max(m, 1e-9)
            # first-quarter mean vs last-quarter mean = drift
            q1 = v[: len(v) // 4].mean()
            q4 = v[-len(v) // 4 :].mean()
            drift = (q4 - q1) / max(q1, 1e-9)
            verdict = "OK"
            if m < 1e-4 and ratio < 0.01:
                verdict = "⚠ SUSPECT COLLAPSE (flat+dead)"
            elif ratio > 1.0:
                verdict = "⚠ UNSTABLE (high variance)"
            elif abs(drift) > 0.5:
                verdict = f"⚠ DRIFT ({drift:+.1%})"
            print(
                f"  {name:8s}: μ={m:.5f}  σ={sd:.5f}  σ/μ={ratio:.1%}  drift={drift:+.1%}  → {verdict}"
            )

    print("\n  Reward component scales (final 25% mean):")
    for tag, name in [
        ("charts/episodic_return/WinDrawLossRewardFunction", "WinDrawLoss"),
        ("charts/episodic_return/MilitaryScoreRewardFunction", "MilitaryScore"),
        ("charts/episodic_return/AttackRewardFunction", "Attack"),
        ("charts/episodic_return/ProduceWorkerRewardFunction", "ProduceWorker"),
        ("charts/episodic_return/ResourceGatherRewardFunction", "ResourceGather"),
    ]:
        if tag in triple:
            _, v = triple[tag]
            q4 = v[-len(v) // 4 :].mean()
            print(f"    {name:18s} = {q4:+.4f}")


if __name__ == "__main__":
    main()
