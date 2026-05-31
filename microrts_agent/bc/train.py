"""
Behavior cloning + value fitting (BC+VF), following RAISocketAI's approach.

Trains both the actor (cross-entropy on bot actions) and the critic (MSE on
discounted returns) simultaneously, so the value head is ready for PPO fine-tuning.

    L_total = L_BC + c1 * L_VF

where L_BC is the per-unit cross-entropy loss (scaled by 1/num_active_units)
and L_VF is the standard value function MSE loss on reward-to-go.

Usage:
    microrts-agent bc train --data data/bc_training/bc_chunk_*.npz \
                                      --architecture unet_entity_cbam_deep --gelu --epochs 30
"""

import argparse
import glob
import json
import os
import time

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from microrts_agent.architectures.factory import ARCHITECTURE_REGISTRY, create_agent
from microrts_agent.paths import OUTPUTS_DIR

ACTION_NVEC = [6, 4, 4, 4, 4, 7, 49]  # MicroRTS action space per cell


def parse_args():
    p = argparse.ArgumentParser(description="BC+VF training for MicroRTS")
    p.add_argument(
        "--data", type=str, nargs="+", required=True, help=".npz file(s) from bc_generate.py"
    )
    p.add_argument(
        "--architecture",
        type=str,
        default="unet_entity_cbam_deep",
        choices=list(ARCHITECTURE_REGISTRY.keys()),
    )
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument(
        "--vf-coef", type=float, default=0.5, help="Value loss coefficient c1 (default: 0.5)"
    )
    p.add_argument(
        "--gamma", type=float, default=0.99, help="Discount factor for reward-to-go (default: 0.99)"
    )
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--gelu", action="store_true")
    p.add_argument("--output", type=str, default=None)
    p.add_argument("--val-split", type=float, default=0.1)
    return p.parse_args()


def load_data(file_paths):
    """Load and concatenate multiple .npz files."""
    all_obs, all_actions, all_rewards, all_dones = [], [], [], []
    any_missing_dones = False
    for path in file_paths:
        for f in sorted(glob.glob(path)):
            print(f"  Loading {f}")
            data = np.load(f)
            n = len(data["obs"])
            all_obs.append(data["obs"])
            all_actions.append(data["actions"])
            if "rewards" in data:
                all_rewards.append(data["rewards"])
            else:
                # Backward compat: no rewards in old files, fill with zeros
                all_rewards.append(np.zeros(n, dtype=np.float32))
            if "dones" in data:
                all_dones.append(data["dones"].astype(bool))
            else:
                # Backward compat: legacy chunks predate dones; signal so the caller
                # can fall back to the reward-magnitude heuristic.
                any_missing_dones = True
                all_dones.append(np.zeros(n, dtype=bool))
            print(f"    {n} samples")
    obs = np.concatenate(all_obs, axis=0)
    actions = np.concatenate(all_actions, axis=0)
    rewards = np.concatenate(all_rewards, axis=0)
    dones = None if any_missing_dones else np.concatenate(all_dones, axis=0)
    print(f"  Total: {len(obs)} samples")
    if dones is None:
        print(
            "  WARNING: at least one chunk has no 'dones' key: falling back to "
            "reward-magnitude heuristic for episode boundaries."
        )
    return obs, actions, rewards, dones


def compute_returns(rewards, gamma=0.99, dones=None):
    """Compute discounted reward-to-go for each timestep.

    `rewards` is a flat array of weighted scalar rewards (w^T @ r_t).
    `dones` (optional) is a parallel bool array; True marks the last step
    of an episode. When provided, episode boundaries are taken from `dones`
    and the function is robust to any reward weighting.

    Backward-compat fallback (dones=None): detect boundaries by reward
    magnitude: works only when the win/loss component dominates
    (e.g. weight ≥ ~5).
    """
    returns = np.zeros_like(rewards)
    running_return = 0.0
    for t in reversed(range(len(rewards))):
        # Legacy heuristic fallback (dones=None) assumes |win/loss reward| > 5
        is_boundary = bool(dones[t]) if dones is not None else abs(rewards[t]) > 5.0
        running_return = rewards[t] if is_boundary else rewards[t] + gamma * running_return
        returns[t] = running_return
    return returns


def compute_bc_vf_loss(agent, obs, actions, returns, action_nvec, vf_coef, device):
    """Compute combined BC + VF loss.

    L = L_BC + vf_coef * L_VF

    L_BC: cross-entropy on active cells, scaled by 1/num_active_units (per sample)
    L_VF: MSE between critic output and discounted returns
    """
    B = obs.shape[0]
    H, W = obs.shape[1], obs.shape[2]
    mapsize = H * W

    # Forward: encoder → actor logits + critic value
    hidden = agent.encoder(obs)
    logits = agent.actor(hidden)  # (B, H, W, 78)
    v_shaped = agent._shaped_value(hidden)  # (B, 1)

    # ── Actor loss (BC) ──
    grid_logits = logits.reshape(B * mapsize, -1)
    split_logits = torch.split(grid_logits, list(action_nvec), dim=1)

    target = actions.reshape(B * mapsize, 7)
    has_action = target.sum(dim=1) > 0
    num_active = has_action.sum()

    if num_active == 0:
        bc_loss = torch.tensor(0.0, device=device)
    else:
        bc_loss = torch.tensor(0.0, device=device)
        for i, (logit_i, nvec_i) in enumerate(zip(split_logits, action_nvec, strict=False)):
            target_i = target[has_action, i].long().clamp(0, nvec_i - 1)
            logit_i = logit_i[has_action]
            bc_loss += nn.functional.cross_entropy(logit_i, target_i, reduction="sum")
        # Scale by 1/num_active_units (RAISocketAI's trick: normalize by unit count)
        bc_loss = bc_loss / (num_active * len(action_nvec))

    # ── Critic loss (VF) ──
    vf_loss = 0.5 * ((v_shaped.squeeze(-1) - returns) ** 2).mean()

    total_loss = bc_loss + vf_coef * vf_loss

    return total_loss, bc_loss.item(), vf_loss.item()


def main():
    args = parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Load data
    print("Loading data...")
    obs, actions, rewards, dones = load_data(args.data)

    # Compute discounted returns for value head training
    print(f"Computing discounted returns (gamma={args.gamma})...")
    returns = compute_returns(rewards, gamma=args.gamma, dones=dones)
    print(f"  Returns range: [{returns.min():.3f}, {returns.max():.3f}]")

    # Infer dimensions
    C = obs.shape[3]
    action_nvec = ACTION_NVEC

    # Train/val split
    n = len(obs)
    n_val = int(n * args.val_split)
    n_train = n - n_val
    perm = np.random.permutation(n)
    train_idx, val_idx = perm[:n_train], perm[n_train:]

    # Keep data on CPU, send batches to GPU on the fly (avoids VRAM OOM)
    train_obs = torch.from_numpy(obs[train_idx]).float()
    train_act = torch.from_numpy(actions[train_idx]).long()
    train_ret = torch.from_numpy(returns[train_idx]).float()
    val_obs = torch.from_numpy(obs[val_idx]).float()
    val_act = torch.from_numpy(actions[val_idx]).long()
    val_ret = torch.from_numpy(returns[val_idx]).float()
    # Free numpy arrays
    del obs, actions, returns

    print(f"Train: {n_train} samples | Val: {n_val} samples")

    # Create agent
    fake_args = argparse.Namespace(
        architecture=args.architecture,
        arch_channels=None,
        gelu=args.gelu,
        spp_critic=False,
        hl_gauss=False,
        hl_gauss_bins=255,
        popart=False,
        autoregressive=False,
        ar_embed_dim=8,
        aux_tasks=None,
        dual_value_heads=False,
        triple_value_heads=False,
        load_model=None,
    )

    agent = create_agent(fake_args, C, action_nvec, device)
    agent.to(device)
    agent.train()

    n_params = sum(p.numel() for p in agent.parameters())
    print(f"Architecture: {args.architecture} ({n_params:,} params)")

    # Optimizer
    optimizer = optim.Adam(agent.parameters(), lr=args.lr)

    # Output directory
    if args.output is None:
        args.output = os.path.join(OUTPUTS_DIR, "runs", f"bc_{args.architecture}_s{args.seed}")
    os.makedirs(args.output, exist_ok=True)

    # Save config.json
    config = {
        "architecture": args.architecture,
        "obs_channels": C,
        "action_nvec": list(action_nvec),
        "gelu": args.gelu,
        "spp_critic": False,
        "hl_gauss": False,
        "hl_gauss_bins": 255,
        "popart": False,
        "autoregressive": False,
        "ar_embed_dim": 8,
        "dual_value_heads": False,
        "triple_value_heads": False,
        "extended_obs": False,
        "frame_stack": 0,
        "filtered_masks": False,
        "reserved_obs": False,
        "multi_map": False,
        "seed": args.seed,
        "map": "maps/open_competition/basesWorkers16x16A.xml",
        "reward_weight": [10.0, 1.0, 1.0, 0.2, 0.2, 1.0, 4.0, 4.0, 4.0, 0.0],
        "gamma": args.gamma,
        "partial_obs": False,
        "max_steps": 4000,
        "alternate_players": True,
    }
    with open(os.path.join(args.output, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    print(f"Output: {args.output}")
    print(
        f"Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr} | VF coef: {args.vf_coef}"
    )
    print("=" * 60)

    # Training loop
    best_val_loss = float("inf")
    patience = 5
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        t0 = time.time()
        agent.train()

        perm = torch.randperm(n_train)
        epoch_bc, epoch_vf, n_batches = 0.0, 0.0, 0

        for start in range(0, n_train, args.batch_size):
            end = min(start + args.batch_size, n_train)
            idx = perm[start:end]

            loss, bc_l, vf_l = compute_bc_vf_loss(
                agent,
                train_obs[idx].to(device),
                train_act[idx].to(device),
                train_ret[idx].to(device),
                action_nvec,
                args.vf_coef,
                device,
            )

            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(agent.parameters(), 0.5)
            optimizer.step()

            epoch_bc += bc_l
            epoch_vf += vf_l
            n_batches += 1

        avg_bc = epoch_bc / max(n_batches, 1)
        avg_vf = epoch_vf / max(n_batches, 1)

        # Validation
        agent.eval()
        with torch.no_grad():
            val_bc, val_vf, val_batches = 0.0, 0.0, 0
            for start in range(0, n_val, args.batch_size):
                end = min(start + args.batch_size, n_val)
                _, bc_l, vf_l = compute_bc_vf_loss(
                    agent,
                    val_obs[start:end].to(device),
                    val_act[start:end].to(device),
                    val_ret[start:end].to(device),
                    action_nvec,
                    args.vf_coef,
                    device,
                )
                val_bc += bc_l
                val_vf += vf_l
                val_batches += 1
            avg_val_bc = val_bc / max(val_batches, 1)
            avg_val_vf = val_vf / max(val_batches, 1)
            avg_val_total = avg_val_bc + args.vf_coef * avg_val_vf

        elapsed = time.time() - t0

        # Save best
        improved = ""
        if avg_val_total < best_val_loss:
            best_val_loss = avg_val_total
            torch.save(agent.state_dict(), os.path.join(args.output, "agent.pt"))
            improved = " *best*"
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        print(
            f"Epoch {epoch:3d}/{args.epochs} | "
            f"BC: {avg_bc:.4f}/{avg_val_bc:.4f} | "
            f"VF: {avg_vf:.4f}/{avg_val_vf:.4f}{improved} | "
            f"{elapsed:.1f}s"
        )

        if epochs_without_improvement >= patience:
            print(f"\nEarly stopping: no improvement for {patience} epochs.")
            break

    # Save final
    torch.save(agent.state_dict(), os.path.join(args.output, "agent_final.pt"))
    print(f"\nBest val loss: {best_val_loss:.4f}")
    print(f"Best model:    {args.output}/agent.pt")
    print(f"Final model:   {args.output}/agent_final.pt")
    print("\nTo fine-tune with RL:")
    print(
        f"  microrts-agent train --load-model {args.output}/agent.pt --architecture {args.architecture} ..."
    )


if __name__ == "__main__":
    main()
