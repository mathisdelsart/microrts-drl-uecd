#!/usr/bin/env python3
"""Recolor white figure/logo backgrounds to the metropolis canvas (#FAFAFA).

Why: the metropolis slide background is #FAFAFA, while dissertation figures and
the cropped logos have a pure-white (#FFFFFF) background. On a slide the white
shows as a brighter rectangle. This renders each chart/diagram (and the two
logos) at high DPI and maps near-white pixels (>=248) to #FAFAFA so they blend
into the slide. Photos are excluded (their bright areas must stay).

Run from anywhere:  python3 defense/recolor_figures.py
Outputs:            defense/figures/<name>_bg.png
If the slide background color changes, update BG and re-run.
"""

import os
import subprocess

import numpy as np
from PIL import Image

BG = (250, 250, 250)  # #FAFAFA, the metropolis canvas
DPI = 400
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "defense", "figures")

_CH = "dissertation/chapters"
SEARCH = [
    os.path.join(ROOT, _CH, d, "figures")
    for d in (
        "02_microrts_decision_problem",
        "03_reinforcement_learning",
        "04_classical_approaches_rts",
        "05_deep_rl_approaches_rts",
        "06_evaluation_framework",
        "07_env_stack",
        "08_architectures",
        "09_training_system",
        "10_results",
        "13_future_work",
        "appendix",
    )
] + [
    os.path.join(ROOT, _CH, "10_results", "figures", "multi-maps"),
    OUT,
]

# Charts/diagrams (white background): recolor. Photos are intentionally absent.
PDF_FIGS = [
    "bridge_architecture",
    "ppo_loop",
    "ppo_cycle",
    "arch_cbam_deep",
    "arch_uecd_simplified",
    "bc_vs_scratch_overall",
    "cbam_resblock_simplified",
    "drl_timeline",
    "entity_transformer_simplified",
    "final_standings",
    "generalization_probes",
    "h2h_matrix",
    "metrics_comparison",
    "phased_schedule",
    "RL-MDP",
    "rush_collapse",
    "rush_fragility",
    "tournament_architecture",
    "training_pipeline",
    "winrates_per_map",
    "paradigm_progression",
    "unit_counters",
    "obs_encoding_viz",
    "action_masking_viz",
    "padded_env",
    "mcw_weight",
    "best_vs_stier_100M",
    "copeland_scores",
    "alpha_rank_sweep",
    "nash_scores",
    "robustness_score",
    "h2h_matrix_global",
]
PNG_FIGS = ["actor_critic_loop", "logo_uclouvain", "logo_epl", "nature-alphastar-model"]


def find(name, ext):
    for d in SEARCH:
        p = os.path.join(d, name + ext)
        if os.path.exists(p):
            return p
    return None


def recolor(src, dst):
    im = Image.open(src).convert("RGB")
    a = np.asarray(im).astype(np.int16)
    m = (a[:, :, 0] >= 248) & (a[:, :, 1] >= 248) & (a[:, :, 2] >= 248)
    a[m] = BG
    Image.fromarray(a.astype("uint8")).save(dst)
    return im.size, int(m.sum())


def main():
    for name in PDF_FIGS:
        src = find(name, ".pdf")
        if not src:
            print("MISSING", name)
            continue
        tmp = f"/tmp/_rc_{name}"
        subprocess.run(["pdftoppm", "-png", "-r", str(DPI), "-singlefile", src, tmp], check=True)
        sz, n = recolor(tmp + ".png", os.path.join(OUT, name + "_bg.png"))
        os.remove(tmp + ".png")
        print(f"{name:30s} {sz} recolored={n}")
    for name in PNG_FIGS:
        src = find(name, ".png") or find(name, ".pdf")
        if src and src.endswith(".pdf"):
            tmp = f"/tmp/_rc_{name}"
            subprocess.run(["pdftoppm", "-png", "-r", "600", "-singlefile", src, tmp], check=True)
            src = tmp + ".png"
        sz, n = recolor(src, os.path.join(OUT, name + "_bg.png"))
        print(f"{name:30s} {sz} recolored={n}")


if __name__ == "__main__":
    main()
