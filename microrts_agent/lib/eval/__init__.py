"""Evaluation utilities shared across evaluate.py, inference_bench.py, and tournaments."""

import os


def is_agent_dir(path):
    """True if path contains config.json (= RL run directory)."""
    return os.path.isdir(path) and os.path.exists(os.path.join(path, "config.json"))
