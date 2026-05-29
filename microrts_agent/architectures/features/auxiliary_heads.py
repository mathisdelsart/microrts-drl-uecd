"""Optional aux heads attached to the shared encoder (architecture side).

Each head takes the encoder hidden (B, C, H', W') and produces a prediction.
Loss computation lives in microrts_agent/training/auxiliary.py.
Activated via --aux-spatial / --aux-contrastive / --aux-unit-count / --aux-opponent-modeling.
"""

import torch.nn as nn

from ..base_actor_critic import layer_init

# Registry: task_name -> attribute_name
AUX_HEAD_REGISTRY = {
    "spatial": "aux_spatial",
    "contrastive": "aux_contrastive",
    "unit_count": "aux_unit_count",
    "opponent_modeling": "aux_opponent_modeling",
}


def _build_spatial(hidden_channels):
    """Predict per-cell player ownership from encoder hidden.
    (B, C, H', W') -> (B, 2, H', W') logits for [P0, P1]."""
    return nn.Sequential(
        layer_init(nn.Conv2d(hidden_channels, hidden_channels // 2, 1)),
        nn.ReLU(),
        layer_init(nn.Conv2d(hidden_channels // 2, 2, 1), std=0.01),
    )


def _build_contrastive(hidden_channels, proj_dim=128):
    """Projection head for temporal contrastive learning (InfoNCE).
    (B, C, H', W') -> (B, proj_dim) embeddings."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        layer_init(nn.Linear(hidden_channels, proj_dim)),
        nn.ReLU(),
        layer_init(nn.Linear(proj_dim, proj_dim), std=0.01),
    )


def _build_unit_count(hidden_channels):
    """Predict unit counts per player from encoder hidden.
    (B, C, H', W') -> (B, 12) = 6 unit types × 2 players.
    Order: [P0_base, P0_barracks, ..., P0_ranged, P1_base, ..., P1_ranged]."""
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        layer_init(nn.Linear(hidden_channels, 64)),
        nn.ReLU(),
        layer_init(nn.Linear(64, 6 * 2), std=0.01),  # 6 = n_unit_types and 2 = n_players
    )


def _build_opponent_modeling(hidden_channels):
    """Predict per-cell enemy action type from encoder hidden.
    (B, C, H', W') -> (B, 6, H', W') logits for 6 action categories:
    [none, move, harvest, return, produce, attack]."""
    return nn.Sequential(
        layer_init(nn.Conv2d(hidden_channels, hidden_channels // 2, 3, padding=1)),
        nn.ReLU(),
        layer_init(nn.Conv2d(hidden_channels // 2, 6, 1), std=0.01),
    )


def build_aux_heads(model, hidden_channels, aux_tasks):
    """Attach auxiliary heads to a model based on task list."""
    _BUILDERS = {
        "spatial": _build_spatial,
        "contrastive": _build_contrastive,
        "unit_count": _build_unit_count,
        "opponent_modeling": _build_opponent_modeling,
    }
    model.aux_tasks = [t for t in (aux_tasks or []) if t in AUX_HEAD_REGISTRY]
    for task in model.aux_tasks:
        setattr(model, AUX_HEAD_REGISTRY[task], _BUILDERS[task](hidden_channels))
