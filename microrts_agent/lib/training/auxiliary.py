"""Auxiliary task loss computation (spatial, unit_count, contrastive, opponent_modeling)."""

import torch
import torch.nn.functional as F


def _focal_cross_entropy(logits, targets, mask, gamma=2.0):
    """Focal loss for masked per-cell classification.

    Downweights easy examples (e.g. idle units) and upweights hard ones
    (e.g. attack, produce). Only computed on cells where mask > 0.

    Args:
        logits:  (B, C, H, W) raw class logits
        targets: (B, H, W) long class indices
        mask:    (B, H, W) float, 1 where loss should be computed
        gamma:   focal exponent (0 = standard CE, 2 = strong focus on hard examples)

    Returns scalar loss averaged over masked cells.
    """
    # Per-cell cross-entropy (no reduction)
    ce = F.cross_entropy(logits, targets, reduction="none")  # (B, H, W)

    # Focal weighting: (1 - p_t)^gamma
    with torch.no_grad():
        p = F.softmax(logits, dim=1)  # (B, C, H, W)
        # Gather the probability of the true class
        p_t = p.gather(1, targets.unsqueeze(1)).squeeze(1)  # (B, H, W)
        focal_weight = (1.0 - p_t) ** gamma

    loss = focal_weight * ce * mask
    n_cells = mask.sum().clamp(min=1.0)
    return loss.sum() / n_cells


def compute_aux_losses(agent, hidden, aux_targets, active_tasks):
    """Compute all active auxiliary losses from the shared encoder hidden.

    Args:
        agent:        the neural network (has aux heads as attributes)
        hidden:       (B, C, H', W') encoder output shared by all heads
        aux_targets:  dict of pre-computed target tensors
        active_tasks: list of active task names (e.g. ['spatial', 'unit_count'])

    Returns:
        dict of {task_name: scalar_loss}
    """
    losses = {}

    # Spatial: predict per-cell ownership for P0 and P1
    # pred (B, 2, H', W') upsampled to (B, 2, H, W) vs target (B, 2, H, W)
    if "spatial" in active_tasks:
        pred = agent.aux_spatial(hidden)
        target = aux_targets["spatial_target"]
        if pred.shape[2:] != target.shape[2:]:
            pred = F.interpolate(pred, size=target.shape[2:], mode="bilinear", align_corners=False)
        losses["aux_spatial"] = F.binary_cross_entropy_with_logits(pred, target)

    # Unit count: predict how many of each unit type each player has
    # pred (B, 12) vs target (B, 12) = [P0_base, ..., P0_ranged, P1_base, ..., P1_ranged]
    if "unit_count" in active_tasks:
        pred = agent.aux_unit_count(hidden)
        target = aux_targets["unit_count_target"]
        losses["aux_unit_count"] = F.mse_loss(pred, target)

    # Opponent modeling: predict enemy unit action types per cell
    # pred (B, 6, H', W') upsampled to (B, 6, H, W) vs target (B, H, W) masked to enemy cells
    if "opponent_modeling" in active_tasks:
        pred = agent.aux_opponent_modeling(hidden)
        target = aux_targets["opponent_action_target"]
        mask = aux_targets["opponent_enemy_mask"]
        if pred.shape[2:] != target.shape[1:]:
            pred = F.interpolate(pred, size=target.shape[1:], mode="bilinear", align_corners=False)
        losses["aux_opponent_modeling"] = _focal_cross_entropy(pred, target, mask, gamma=2.0)

    # Contrastive (InfoNCE): embeddings of consecutive steps should be similar
    # logits (B, B) similarity matrix — diagonal = true pairs, rest = negatives
    if "contrastive" in active_tasks:
        z_anchor = agent.aux_contrastive(hidden)
        z_positive = aux_targets["contrastive_positive"]
        z_a = F.normalize(z_anchor, dim=1)
        z_p = F.normalize(z_positive, dim=1)
        logits = z_a @ z_p.T / 0.1  # temperature sharpens the distribution
        labels = torch.arange(z_a.shape[0], device=z_a.device)
        losses["aux_contrastive"] = F.cross_entropy(logits, labels)

    return losses
