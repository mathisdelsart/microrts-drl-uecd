"""PPO algorithm core: GAE computation and clipped policy gradient update."""

import numpy as np
import torch
import torch.nn as nn


def compute_gae(rewards, values, dones, next_value, next_done, gamma, gae_lambda, num_steps):
    """Generalized Advantage Estimation (Schulman et al. 2016).

    Supports both single-head and multi-head modes:
      - Single-head: all inputs (T, N), gamma/lambda scalars → outputs (T, N)
      - Multi-head:  all inputs (T, N, H), gamma/lambda arrays of len H → outputs (T, N, H)

    next_value: (1, N) or (1, N, H) bootstrap value; next_done: (N,) terminal flag.
    Returns (advantages, returns), both same shape as rewards.
    """
    advantages = torch.zeros_like(rewards)
    lastgaelam = 0

    # Convert gamma/lambda to tensors for broadcasting if they are arrays
    if isinstance(gamma, (list, tuple)):
        gamma = torch.tensor(gamma, dtype=rewards.dtype, device=rewards.device)
    if isinstance(gae_lambda, (list, tuple)):
        gae_lambda = torch.tensor(gae_lambda, dtype=rewards.dtype, device=rewards.device)

    for t in reversed(range(num_steps)):
        is_last_step = t == num_steps - 1

        next_value_t = next_value if is_last_step else values[t + 1]
        next_non_terminal = 1.0 - (next_done if is_last_step else dones[t + 1])

        # For multi-head: next_non_terminal is (N,), needs unsqueeze to (N, 1)
        if rewards.dim() == 3 and next_non_terminal.dim() == 1:
            next_non_terminal = next_non_terminal.unsqueeze(-1)

        # TD residual: r_t + gamma * V(s_{t+1}) - V(s_t)
        delta = rewards[t] + gamma * next_value_t * next_non_terminal - values[t]

        # GAE: A_t = delta_t + (gamma * lambda) * A_{t+1}
        lastgaelam = delta + gamma * gae_lambda * next_non_terminal * lastgaelam

        advantages[t] = lastgaelam

    # returns = advantages + V(s)
    returns = advantages + values
    return advantages, returns


def compute_multi_head_gae(heads, dones, next_done, num_steps, advantage_weights):
    """Multi-head GAE: unified computation on stacked (T, N, H) tensors.

    Following RAISocketAI's approach:
      1. Stack rewards/values/bootstrap into (T, N, H) tensors
      2. Single GAE pass with per-head gamma/lambda arrays
      3. Return per-head advantages (T, N, H): blending and normalization
         happen later in ppo_update so that each head can be normalized
         independently before being combined via advantage_weights.

    Args:
        heads: list of (rewards, values, last_v, gamma, gae_lambda) per head.
               Order convention: [shaped, sparse, cost, ...].
        dones: (T, N) terminal flags (shared across heads).
        next_done: (N,) terminal flag for bootstrap step.
        num_steps: rollout length T.
        advantage_weights: iterable of weights, one per head (should sum to 1.0).
            Kept in the signature for API compatibility and logging; no blend
            is performed here.

    Returns:
        (advantages_per_head, *returns_per_head)
        advantages_per_head: (T, N, H) tensor, one advantage sequence per head.
        returns_per_head: tuple of (T, N) tensors, one per head.
    """
    num_heads = len(heads)
    assert num_heads == len(advantage_weights), (
        f"Got {num_heads} heads but {len(advantage_weights)} weights"
    )

    # Stack into (T, N, H) tensors
    stacked_rewards = torch.stack([h[0] for h in heads], dim=-1)  # (T, N, H)
    stacked_values = torch.stack([h[1] for h in heads], dim=-1)  # (T, N, H)
    stacked_last_v = torch.stack([h[2] for h in heads], dim=-1)  # (1, N, H)
    gammas = [h[3] for h in heads]  # list of scalars
    gae_lambdas = [h[4] for h in heads]  # list of scalars

    # Single unified GAE pass on the 3D tensor
    adv_all, ret_all = compute_gae(
        stacked_rewards,
        stacked_values,
        dones,
        stacked_last_v,
        next_done,
        gammas,
        gae_lambdas,
        num_steps,
    )
    # adv_all: (T, N, H), ret_all: (T, N, H)

    # Split returns back into per-head tensors; advantages stay per-head so
    # ppo_update can normalize each column independently before blending.
    returns_per_head = tuple(ret_all[..., i] for i in range(num_heads))

    return (adv_all, *returns_per_head)


def _clipped_value_loss(new_values, old_values, returns, clip_coef, clip_vloss):
    """Optionally clipped value function MSE loss. Shared across all value heads."""
    new_values = new_values.view(-1)

    # Standard MSE loss
    value_error = new_values - returns
    loss_unclipped = value_error.pow(2)

    if not clip_vloss:
        return 0.5 * loss_unclipped.mean()

    # Clipped value prediction
    value_delta = new_values - old_values
    value_delta_clipped = torch.clamp(value_delta, -clip_coef, clip_coef)
    values_clipped = old_values + value_delta_clipped

    # Clipped MSE loss
    clipped_error = values_clipped - returns
    loss_clipped = clipped_error.pow(2)

    # Take worst-case loss (conservative update)
    loss = torch.max(loss_unclipped, loss_clipped)

    return 0.5 * loss.mean()


def _value_loss_for_head(
    new_values,
    old_values,
    returns,
    logits,
    clip_coef,
    clip_vloss,
    hl_gauss,
    hl_gauss_bins,
    popart_module=None,
):
    """Compute value loss for a single head (shaped, sparse, or cost).

    Dispatches to HL-Gauss, PopArt-normalized clipped MSE, or standard clipped MSE.
    """
    if hl_gauss:
        from microrts_agent.architectures.features.hl_gauss import hl_gauss_loss

        return hl_gauss_loss(logits.view(-1, hl_gauss_bins), returns, hl_gauss_bins)

    if popart_module is not None:
        return _clipped_value_loss(
            popart_module.normalize(new_values),
            popart_module.normalize(old_values),
            popart_module.normalize(returns),
            clip_coef,
            clip_vloss,
        )

    return _clipped_value_loss(new_values, old_values, returns, clip_coef, clip_vloss)


def ppo_update(
    agent,
    optimizer,
    b_obs,
    b_actions,
    b_logprobs,
    b_advantages,
    b_returns,
    b_values,
    b_invalid_action_masks,
    batch_size,
    minibatch_size,
    update_epochs,
    clip_coef,
    vf_coef,
    current_ent_coef,
    max_grad_norm,
    norm_adv,
    clip_vloss,
    dual_value_heads,
    b_values_sparse,
    b_returns_sparse,
    vf_coef_sparse,
    triple_value_heads,
    b_values_cost,
    b_returns_cost,
    vf_coef_cost,
    b_importance_weights,
    aux_tasks,
    aux_coefs,
    b_aux_targets,
    hl_gauss,
    hl_gauss_bins,
    popart,
    bc_teacher=None,
    bc_kl_coef=0.0,
    advantage_weights=None,
):
    """Run PPO update epochs on a collected rollout.

    All b_* tensors have shape (batch_size, ...) where batch_size = T*N.

    With multiple value heads, b_advantages has shape (batch_size, H) and
    advantage_weights is the (H,) blend vector. Each head is normalized
    independently before being linearly combined via advantage_weights,
    matching RAISocketAI's normalize_advantage=True,
    normalize_advantages_after_scaling=False default.

    Returns dict with loss metrics: v_loss, pg_loss, entropy_loss, approx_kl,
    and optionally v_loss_sparse, v_loss_cost, aux_*.
    """
    multi_head_adv = b_advantages.dim() > 1
    if multi_head_adv:
        assert advantage_weights is not None, (
            "Multi-head advantages require advantage_weights to blend them."
        )
        adv_weights_tensor = torch.as_tensor(
            advantage_weights, dtype=b_advantages.dtype, device=b_advantages.device
        )
    # Use aux path whenever we need hidden (aux tasks)
    need_aux_path = bool(aux_tasks)

    # PopArt: update running stats and adjust critic weights before epochs
    if popart:
        agent.popart_shaped.update_and_adjust(b_returns, agent._popart_linear_shaped)
        if dual_value_heads:
            agent.popart_sparse.update_and_adjust(b_returns_sparse, agent._popart_linear_sparse)
        if triple_value_heads:
            agent.popart_cost.update_and_adjust(b_returns_cost, agent._popart_linear_cost)

    # Indices for minibatch sampling (shuffled each epoch)
    inds = np.arange(batch_size)

    clip_fracs = []  # fraction of transitions where |ρ-1| > ε (monitoring)
    grad_norms = []  # gradient norms after clipping (monitoring)

    # Track aux losses across minibatches for metrics
    aux_loss_accum = {}
    aux_loss_count = 0

    for _ in range(update_epochs):
        np.random.shuffle(inds)
        # Drop trailing partial minibatch when it would have <2 samples:
        # std(dim=0) with N=1 returns NaN under Bessel correction, which
        # poisons advantages -> gradients -> model weights.
        mb_schedule = [
            inds[s : s + minibatch_size]
            for s in range(0, batch_size, minibatch_size)
            if min(s + minibatch_size, batch_size) - s >= 2
        ]

        for mb_inds in mb_schedule:
            # ---- Forward pass ----
            out = agent.forward(
                b_obs[mb_inds],
                invalid_action_masks=b_invalid_action_masks[mb_inds],
                action=b_actions.long()[mb_inds],
                return_hidden=need_aux_path,
            )
            newlogprob = out["logprob"]
            entropy = out["entropy"]
            new_values = out["v_shaped"]
            new_values_sparse = out.get("v_sparse")
            new_values_cost = out.get("v_cost")
            logits_shaped = out.get("logits_shaped")
            logits_sparse = out.get("logits_sparse")
            logits_cost = out.get("logits_cost")
            hidden = out.get("hidden")

            # ---- Advantage normalization (per minibatch) ----
            mb_advantages = b_advantages[mb_inds]  # (MB,) or (MB, H)
            if multi_head_adv:
                # Normalize EACH head independently along the minibatch, then
                # blend via advantage_weights. This is RAISocketAI's default
                # (normalize_advantage=True, normalize_advantages_after_scaling=False)
                # and equalizes the contribution of heads with very different
                # reward magnitudes (shaped vs sparse vs cost).
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean(dim=0, keepdim=True)) / (
                        mb_advantages.std(dim=0, keepdim=True) + 1e-8
                    )
                mb_advantages = mb_advantages @ adv_weights_tensor  # (MB,)
            else:
                if norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (
                        mb_advantages.std() + 1e-8
                    )

            # Importance sampling ratio: pi_new(a|s) / pi_old(a|s)
            ratio = (newlogprob - b_logprobs[mb_inds]).exp()  # (MB,)
            approx_kl = (b_logprobs[mb_inds] - newlogprob).mean()

            with torch.no_grad():
                clip_fracs.append(((ratio - 1.0).abs() > clip_coef).float().mean().item())

            # PPO clipped surrogate objective
            pg_loss1 = -mb_advantages * ratio
            pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - clip_coef, 1 + clip_coef)
            pg_loss_per_sample = torch.max(pg_loss1, pg_loss2)  # pessimistic bound

            # Apply per-transition importance weights (prioritized sampling)
            if b_importance_weights is not None:
                pg_loss = (pg_loss_per_sample * b_importance_weights[mb_inds]).mean()
            else:
                pg_loss = pg_loss_per_sample.mean()

            entropy_loss = entropy.mean()

            # Value losses: one per active head
            popart_shaped = agent.popart_shaped if popart else None
            v_loss = _value_loss_for_head(
                new_values,
                b_values[mb_inds],
                b_returns[mb_inds],
                logits_shaped,
                clip_coef,
                clip_vloss,
                hl_gauss,
                hl_gauss_bins,
                popart_shaped,
            )

            loss = pg_loss - current_ent_coef * entropy_loss + vf_coef * v_loss

            if dual_value_heads:
                popart_sparse = agent.popart_sparse if popart else None
                v_loss_sparse = _value_loss_for_head(
                    new_values_sparse,
                    b_values_sparse[mb_inds],
                    b_returns_sparse[mb_inds],
                    logits_sparse,
                    clip_coef,
                    clip_vloss,
                    hl_gauss,
                    hl_gauss_bins,
                    popart_sparse,
                )
                loss = loss + vf_coef_sparse * v_loss_sparse
            else:
                v_loss_sparse = None

            if triple_value_heads and new_values_cost is not None:
                popart_cost = agent.popart_cost if popart else None
                v_loss_cost = _value_loss_for_head(
                    new_values_cost,
                    b_values_cost[mb_inds],
                    b_returns_cost[mb_inds],
                    logits_cost,
                    clip_coef,
                    clip_vloss,
                    hl_gauss,
                    hl_gauss_bins,
                    popart_cost,
                )
                loss = loss + vf_coef_cost * v_loss_cost
            else:
                v_loss_cost = None

            # Auxiliary losses
            if aux_tasks and b_aux_targets:
                from microrts_agent.training.auxiliary import compute_aux_losses

                mb_aux_targets = {
                    k: v[mb_inds] if v.dim() > 0 and v.shape[0] == batch_size else v
                    for k, v in b_aux_targets.items()
                }
                aux_losses = compute_aux_losses(agent, hidden, mb_aux_targets, aux_tasks)
                for name, aloss in aux_losses.items():
                    loss = loss + aux_coefs[name] * aloss
                    if name not in aux_loss_accum:
                        aux_loss_accum[name] = 0.0
                    aux_loss_accum[name] += aloss.item()
                aux_loss_count += 1

            # BC teacher penalty (single-sample importance-weight estimator).
            # Actions come from the agent's rollout, so this is
            #   E_{a ~ π_agent}[log π_teacher(a) − log π_agent(a)]
            # which is the negative of the (k1) Monte-Carlo estimate of
            # KL(π_agent ‖ π_teacher) on the agent's own samples: minimising
            # it pulls π_agent toward π_teacher on states the agent visits.
            # Note: this is NOT an unbiased estimate of the forward KL
            # (KL(π_teacher ‖ π_agent)); estimating that would require sampling
            # actions from π_teacher.
            bc_kl_loss = torch.tensor(0.0, device=b_obs.device)
            if bc_teacher is not None and bc_kl_coef > 0:
                with torch.no_grad():
                    teacher_out = bc_teacher.forward(
                        b_obs[mb_inds],
                        invalid_action_masks=b_invalid_action_masks[mb_inds],
                        action=b_actions.long()[mb_inds],
                    )
                    teacher_logprob = teacher_out["logprob"]
                bc_kl_loss = (teacher_logprob - newlogprob).mean()
                loss = loss + bc_kl_coef * bc_kl_loss

            # Backprop + gradient clipping + optimizer step
            optimizer.zero_grad()
            loss.backward()
            gn = nn.utils.clip_grad_norm_(agent.parameters(), max_grad_norm)
            grad_norms.append(gn.item() if hasattr(gn, "item") else float(gn))
            optimizer.step()

    # Explained variance: 1 - Var(returns - values) / Var(returns)
    # EV=1 means perfect predictions, EV=0 means no better than mean
    with torch.no_grad():
        y_pred = b_values.cpu().numpy()  # (batch_size,)
        y_true = b_returns.cpu().numpy()  # (batch_size,)
        var_y = np.var(y_true)
        explained_var = 1.0 - np.var(y_true - y_pred) / (var_y + 1e-8) if var_y > 1e-8 else 0.0

    metrics = {
        "v_loss": v_loss.item(),
        "pg_loss": pg_loss.item(),
        "entropy_loss": entropy_loss.item(),
        "approx_kl": approx_kl.item(),
        "clip_fraction": float(np.mean(clip_fracs)),
        "explained_variance": float(explained_var),
        "grad_norm": float(np.mean(grad_norms)),
        "value_shaped_mean": float(b_values.mean().item()),
        "value_shaped_std": float(b_values.std().item()),
    }

    if dual_value_heads and v_loss_sparse is not None:
        metrics["v_loss_sparse"] = v_loss_sparse.item()
    if dual_value_heads and b_values_sparse is not None:
        metrics["value_sparse_mean"] = float(b_values_sparse.mean().item())
        metrics["value_sparse_std"] = float(b_values_sparse.std().item())
    if triple_value_heads and v_loss_cost is not None:
        metrics["v_loss_cost"] = v_loss_cost.item()
    if triple_value_heads and b_values_cost is not None:
        metrics["value_cost_mean"] = float(b_values_cost.mean().item())
        metrics["value_cost_std"] = float(b_values_cost.std().item())

    if aux_tasks and aux_loss_count > 0:
        for name, total in aux_loss_accum.items():
            metrics[name] = total / aux_loss_count

    if bc_teacher is not None:
        metrics["bc_kl_loss"] = bc_kl_loss.item()

    return metrics
