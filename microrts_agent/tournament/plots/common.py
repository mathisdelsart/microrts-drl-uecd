"""
Shared utilities for tournament plot modules.

- Matplotlib / Seaborn configuration (applied on import)
- calculate_rankings() used by multiple plots
"""

import re

import matplotlib.pyplot as plt
import seaborn as sns

# Matplotlib / Seaborn defaults
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 300
plt.rcParams["savefig.dpi"] = 300
plt.rcParams["font.size"] = 11
plt.rcParams["savefig.format"] = "pdf"
plt.rcParams["pdf.fonttype"] = 42
plt.rcParams["figure.max_open_warning"] = 0


def calculate_rankings(games, ais) -> list[tuple]:
    """
    Calculate final rankings with Win/Tie/Loss records.

    Returns list of (ai_name, score, wins, ties, losses) sorted by score desc.
    """
    wins = dict.fromkeys(ais, 0)
    ties = dict.fromkeys(ais, 0)
    losses = dict.fromkeys(ais, 0)

    for game in games:
        ai1, ai2 = game.ai1_name, game.ai2_name
        if game.winner == -1:
            ties[ai1] += 1
            ties[ai2] += 1
        elif game.winner == 0:
            wins[ai1] += 1
            losses[ai2] += 1
        elif game.winner == 1:
            losses[ai1] += 1
            wins[ai2] += 1

    scores = {ai: wins[ai] * 1 + ties[ai] * 0.5 for ai in ais}

    return sorted(
        [(ai, scores[ai], wins[ai], ties[ai], losses[ai]) for ai in ais],
        key=lambda x: (-x[1], -x[2], x[3]),
    )


def clean_name(ai_name: str) -> str:
    """Strip parenthesised suffixes from AI names and apply thesis display labels.

    Historic CSVs (parsed_tournament.json files produced before the registry
    rename) still reference ``POWorkerRush`` / ``POLightRush``; map them to
    the current names for visual consistency. The agent renames cover the two
    flagship runs as they appeared in the multi-map tournament, so they read
    "UECD-SingleMap" and "UECD-MultiMap" in the rendered plots even though the
    underlying run directories on disk keep their training-time identifiers.
    """
    name = re.sub(r"\s*\(.*?\)", "", ai_name)
    rename = {
        "POWorkerRush": "WorkerRush",
        "POLightRush": "LightRush",
        "BestRL-350M": "UECD-SingleMap",
        "multimap_small_200M_s1": "UECD-MultiMap",
    }
    return rename.get(name, name)
