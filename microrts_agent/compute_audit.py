"""Compute audit: wall-clock GPU-hours and total steps per training run.

Reads TensorBoard event files to get the first/last event wall_time, then
prints elapsed hours, total steps reached, and average SPS.

Usage:
    python microrts_agent/compute_audit.py outputs/runs/rai_style_v3_300M_s1
    python microrts_agent/compute_audit.py outputs/runs/*
    python microrts_agent/compute_audit.py --all
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from glob import glob

from tensorboard.backend.event_processing.event_accumulator import EventAccumulator  # type: ignore

from lib.paths import RUNS_DIR


def audit_run(run_dir: str) -> dict | None:
    if not os.path.isdir(run_dir):
        return None
    tfevents = [f for f in os.listdir(run_dir) if f.startswith("events.out.tfevents")]
    if not tfevents:
        return None

    ea = EventAccumulator(run_dir, size_guidance={"scalars": 0})
    ea.Reload()
    tags = ea.Tags().get("scalars", [])
    if not tags:
        return None

    first_wt, last_wt, last_step = float("inf"), 0.0, 0
    for tag in tags:
        events = ea.Scalars(tag)
        if not events:
            continue
        first_wt = min(first_wt, events[0].wall_time)
        if events[-1].wall_time > last_wt:
            last_wt = events[-1].wall_time
            last_step = events[-1].step

    elapsed_s = last_wt - first_wt
    hours = elapsed_s / 3600.0

    cfg_path = os.path.join(run_dir, "config.json")
    cfg = {}
    if os.path.exists(cfg_path):
        with open(cfg_path) as f:
            cfg = json.load(f)
    target_steps = cfg.get("total_timesteps")
    sps = last_step / elapsed_s if elapsed_s > 0 else 0.0

    return {
        "name": os.path.basename(run_dir.rstrip("/")),
        "hours": hours,
        "steps": last_step,
        "target_steps": target_steps,
        "sps": sps,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("run_dirs", nargs="*")
    p.add_argument("--all", action="store_true", help=f"Audit every run in {RUNS_DIR}")
    args = p.parse_args()

    run_dirs = sorted(glob(os.path.join(RUNS_DIR, "*"))) if args.all else args.run_dirs
    if not run_dirs:
        p.error("Provide run dirs or --all")

    results = []
    for rd in run_dirs:
        if not os.path.isdir(rd):
            continue
        # Walk one level down if the dir contains sub-runs (no tfevents)
        has_events = any(f.startswith("events.out.tfevents") for f in os.listdir(rd))
        if has_events:
            r = audit_run(rd)
            if r:
                results.append(r)
        else:
            for sub in sorted(os.listdir(rd)):
                sub_path = os.path.join(rd, sub)
                if os.path.isdir(sub_path):
                    r = audit_run(sub_path)
                    if r:
                        results.append(r)

    if not results:
        print("No runs with TB events found.", file=sys.stderr)
        sys.exit(1)

    name_w = max(len(r["name"]) for r in results)
    print(f"{'Run':<{name_w}}  {'Hours':>8}  {'Steps':>14}  {'Target':>14}  {'SPS':>8}  {'%':>5}")
    print("-" * (name_w + 60))
    total_hours = total_steps = 0
    for r in results:
        target = f"{r['target_steps']:,}" if r["target_steps"] else "-"
        pct = f"{100 * r['steps'] / r['target_steps']:.0f}" if r["target_steps"] else "-"
        print(
            f"{r['name']:<{name_w}}  {r['hours']:>8.1f}  {r['steps']:>14,}  {target:>14}  {r['sps']:>8.0f}  {pct:>5}"
        )
        total_hours += r["hours"]
        total_steps += r["steps"]
    print("-" * (name_w + 60))
    print(f"{'TOTAL':<{name_w}}  {total_hours:>8.1f}  {total_steps:>14,}")
    print(f"\n= {total_hours / 24:.1f} GPU-days (1 GPU per run)")


if __name__ == "__main__":
    main()
