"""
Base class and shared utilities for all grid-based actor-critic architectures.

Utils functions:
  - layer_init     — orthogonal weight initialization (PPO standard)
  - get_activation — switch GELU/ReLU via --gelu flag

Utils classes:
  - CategoricalMasked   — categorical distribution with invalid action masking
  - Transpose           — nn.Module wrapper for permute() in nn.Sequential
  - SpatialPyramidPooling — multi-scale pooling for map-size-independent critic

Base class:
  - GridActorCriticBase — abstract base with action sampling, value heads, predict
"""

import numpy as np
import torch
import torch.nn as nn
from torch.distributions.categorical import Categorical

# PPO orthogonal-init gain for hidden layers (Andrychowicz 2020)
_SQRT2 = np.sqrt(2)


def layer_init(layer, std=_SQRT2, bias_const=0.0):
    """Orthogonal weight init (PPO standard, Andrychowicz 2020).
    std=sqrt(2) for hidden layers,
    std=0.01 for actor output (near-uniform exploration at start),
    std=1 for critic output. Returns the layer for inline use: layer_init(nn.Linear(...))"""
    torch.nn.init.orthogonal_(layer.weight, std)
    if layer.bias is not None:
        torch.nn.init.constant_(layer.bias, bias_const)
    return layer


def get_activation(use_gelu=False):
    """Switch between GELU (smoother gradients) and ReLU (faster) via --gelu flag."""
    return nn.GELU() if use_gelu else nn.ReLU()


class CategoricalMasked(Categorical):
    """Categorical distribution that forces invalid actions to probability ~0.
    Used for sampling + log_prob + entropy. For greedy argmax, use torch.where directly."""

    def __init__(self, logits, masks, mask_value):
        # Valid actions keep their logits; invalid ones get -1e8 -> softmax ≈ 0
        logits = torch.where(masks.bool(), logits, mask_value)
        super().__init__(logits=logits, validate_args=False)


class Transpose(nn.Module):
    """nn.Module wrapper around permute() so it can be used in nn.Sequential.
    Converts between gym (B,H,W,C) and PyTorch (B,C,H,W) conventions."""

    def __init__(self, permutation):
        super().__init__()
        self.permutation = permutation

    def forward(self, x):
        return x.permute(self.permutation)


class SpatialPyramidPooling(nn.Module):
    """Multi-scale spatial pooling for scale-invariant feature aggregation (He 2014).

    Pools spatial features at multiple resolutions and concatenates them,
    producing a fixed-size output regardless of input spatial dimensions.
    This allows a single critic to handle any map size without retraining.

    Input:  (B, C, H, W)  — any spatial size
    Output: (B, C * num_bins)  — fixed size

    Default levels (1, 2, 4) -> 1 + 4 + 16 = 21 bins per channel.
    """

    def __init__(self, levels=(1, 2, 4)):
        super().__init__()
        self.levels = levels
        self.pools = nn.ModuleList([nn.AdaptiveAvgPool2d(level) for level in levels])

    @property
    def num_bins(self):
        """Total number of spatial bins across all levels. Used to size the Linear layer after SPP."""
        return sum(level * level for level in self.levels)

    def forward(self, x):
        pooled = [
            pool(x).flatten(2) for pool in self.pools
        ]  # flatten spatial dims only (keep B, C)
        return torch.cat(pooled, dim=2).flatten(1)  # (B, C * num_bins)


class GridActorCriticBase(nn.Module):
    """Base class for grid-based actor-critic architectures.

    Subclass __init__ pattern:
        1. super().__init__()
        2. self._init_common(action_nvec)
        3. Build self.encoder, self.actor, self.critic_shaped
        4. self._finalize(dual_value_heads, hidden_channels, aux_tasks, ...)
    """

    def _init_common(self, action_nvec):
        """Initialize action space attributes. Call before building actor."""
        self.action_nvec = action_nvec
        self.num_action_planes = sum(action_nvec)

    def _finalize(
        self,
        hidden_channels,
        dual_value_heads=False,
        aux_tasks=None,
        extra_critic_builder=None,
        triple_value_heads=False,
        hl_gauss=False,
        hl_gauss_bins=255,
        autoregressive=False,
        ar_embed_dim=8,
        hierarchical_mask=False,
        popart=False,
    ):
        """Finalize model: value heads, mask buffer, aux tasks.
        Call at the end of __init__ after encoder, actor, and critic are built.
        """
        # Triple value heads requires dual (shaped + sparse + cost)
        if triple_value_heads and not dual_value_heads:
            dual_value_heads = True
        self.dual_value_heads = dual_value_heads
        self.triple_value_heads = triple_value_heads
        if dual_value_heads:
            self.critic_sparse = extra_critic_builder()
        if triple_value_heads:
            self.critic_cost = extra_critic_builder()

        # HL-Gauss: replace scalar critic outputs (Linear->1) with categorical bins (Linear->N)
        self.hl_gauss = hl_gauss
        self.hl_gauss_bins = hl_gauss_bins
        if hl_gauss:
            from microrts_agent.lib.architectures.features.hl_gauss import convert_critic_to_bins

            convert_critic_to_bins(self.critic_shaped, hl_gauss_bins)
            if dual_value_heads:
                convert_critic_to_bins(self.critic_sparse, hl_gauss_bins)
            if triple_value_heads:
                convert_critic_to_bins(self.critic_cost, hl_gauss_bins)

        # Auto-regressive sub-action head: each sub-action conditioned on previous ones.
        # Needs rich per-cell features BEFORE the actor's final projection to 78 logits.
        # All remaining architectures (GridNet/IMPALA/UNet) have upsampling in the actor,
        # so we create a separate upsample path that preserves channel dim.
        self.use_autoregressive = autoregressive
        if autoregressive:
            modules = list(self.actor.children())
            self._actor_transpose = modules[-1]  # Transpose (B,C,H,W) -> (B,H,W,C)
            final_conv = modules[-2]  # last ConvTranspose -> 78 channels
            pre_conv_dim = final_conv.in_channels  # channel dim before projection

            # Build upsample path: same spatial transform as final_conv but keeps pre_conv_dim channels
            body = nn.Sequential(*modules[:-2]) if len(modules) > 2 else nn.Identity()
            self._ar_feature_net = nn.Sequential(
                body,
                layer_init(
                    nn.ConvTranspose2d(
                        pre_conv_dim,
                        pre_conv_dim,
                        kernel_size=final_conv.kernel_size,
                        stride=final_conv.stride,
                        padding=final_conv.padding,
                        output_padding=final_conv.output_padding,
                    )
                ),
                nn.ReLU(),
            )

            from .features.autoregressive import AutoRegressiveHead

            self.ar_head = AutoRegressiveHead(
                spatial_dim=pre_conv_dim,
                action_nvec=self.action_nvec,
                embed_dim=ar_embed_dim,
            )

        # Hierarchical sub-action masking: zero out log_prob/entropy of sub-actions
        # whose parent action_type does not make them active (RAISocketAI-style).
        # Parent action_type per sub-action index (with -1 meaning "always active"):
        #   idx 0: action_type       → always active
        #   idx 1: move_dir          → only when action_type == 1 (MOVE)
        #   idx 2: harvest_dir       → only when action_type == 2 (HARVEST)
        #   idx 3: return_dir        → only when action_type == 3 (RETURN)
        #   idx 4: produce_dir       → only when action_type == 4 (PRODUCE)
        #   idx 5: produce_type      → only when action_type == 4 (PRODUCE)
        #   idx 6: attack_target     → only when action_type == 5 (ATTACK)
        self.use_hierarchical_mask = hierarchical_mask
        if hierarchical_mask:
            self.register_buffer(
                "_hmask_parent",
                torch.tensor([-1, 1, 2, 3, 4, 4, 5], dtype=torch.long),
            )

        # PopArt: adaptive value normalization. Tracks running mean/std per head
        # and adjusts the output layer weights to preserve denormalized values.
        self.use_popart = popart
        if popart:
            from microrts_agent.lib.architectures.features.popart import (
                PopArtNormalizer,
                get_output_layer,
            )

            self.popart_shaped = PopArtNormalizer()
            self._popart_linear_shaped = get_output_layer(self.critic_shaped)
            if dual_value_heads:
                self.popart_sparse = PopArtNormalizer()
                self._popart_linear_sparse = get_output_layer(self.critic_sparse)
            if triple_value_heads:
                self.popart_cost = PopArtNormalizer()
                self._popart_linear_cost = get_output_layer(self.critic_cost)

        # mask_value as registered buffer so it follows .to(device) automatically
        self.register_buffer("mask_value", torch.tensor(-1e8))
        from .features.auxiliary_heads import build_aux_heads

        build_aux_heads(self, hidden_channels, aux_tasks)

    # ------------------------------------------------------------------
    # Value head helpers
    # ------------------------------------------------------------------

    def _head_value(self, head, hidden, popart_norm=None, use_tanh=False):
        """Generic value extraction: handles HL-Gauss / PopArt / raw_output_critic transparently.
        HL-Gauss: bins -> scalar via expected value. PopArt: denormalize. Tanh: for sparse (bounded)."""
        raw_output_critic = head(hidden)
        if self.hl_gauss:
            from microrts_agent.lib.architectures.features.hl_gauss import hl_gauss_value

            val = hl_gauss_value(raw_output_critic, self.hl_gauss_bins).unsqueeze(-1)
            return torch.tanh(val) if use_tanh else val
        if self.use_popart and popart_norm is not None:
            return popart_norm.denormalize(raw_output_critic)
        return torch.tanh(raw_output_critic) if use_tanh else raw_output_critic

    def _shaped_value(self, hidden):
        """Shaped value head (main reward signal)."""
        return self._head_value(self.critic_shaped, hidden, getattr(self, "popart_shaped", None))

    def _sparse_value(self, hidden):
        """Sparse value head (win/loss, bounded [-1,1] via tanh)."""
        return self._head_value(
            self.critic_sparse, hidden, getattr(self, "popart_sparse", None), use_tanh=True
        )

    def _cost_value(self, hidden):
        """Cost value head (MilitaryScore, normalized [-1,1] per step, no tanh needed)."""
        return self._head_value(self.critic_cost, hidden, getattr(self, "popart_cost", None))

    # ------------------------------------------------------------------
    # Action sampling
    # ------------------------------------------------------------------

    def _sample_action(self, hidden, action, invalid_action_masks):
        """Logits -> masked categoricals -> action + logprob + entropy.
        action=None: sample new actions (rollout collection).
        action=tensor (B, mapsize, 7): evaluate existing actions (PPO update).
        """
        mapsize = invalid_action_masks.shape[-2]  # derive from masks (B, mapsize, 78)
        invalid_action_masks = invalid_action_masks.reshape(
            -1, invalid_action_masks.shape[-1]
        )  # (B * mapsize, 78)
        split_masks = torch.split(
            invalid_action_masks, self.action_nvec, dim=1
        )  # 7 tensors: [(B * mapsize, 6), ...]

        if self.use_autoregressive:
            ar_features = self._get_ar_features(hidden)  # (B * mapsize, D)
            if action is None:
                action = self._ar_act(ar_features, split_masks, greedy=False)  # (7, B * mapsize)
                flat_actions = action.T  # (B * mapsize, 7)
            else:
                action = action.view(-1, action.shape[-1]).T  # (7, B*mapsize)
                flat_actions = action.T  # (B * mapsize, 7)
            split_logits = self.ar_head.forward_all(
                ar_features, flat_actions
            )  # 7 tensors: [(B * mapsize, 6), ...]
        else:
            logits = self.actor(hidden)  # (B, H, W, 78)
            grid_logits = logits.reshape(-1, self.num_action_planes)  # (B * mapsize, 78)
            split_logits = torch.split(
                grid_logits, self.action_nvec, dim=1
            )  # 7 tensors: [(B * mapsize, 6), ...]

        # Build one masked distribution per sub-action
        multi_categoricals = [
            CategoricalMasked(logits=sl, masks=sm, mask_value=self.mask_value)
            for sl, sm in zip(split_logits, split_masks)  # 7 distributions
        ]

        if not self.use_autoregressive:
            if action is None:
                action = torch.stack(
                    [cat.sample() for cat in multi_categoricals]
                )  # (7, B * mapsize)
            else:
                action = action.view(-1, action.shape[-1]).T  # (7, B * mapsize)

        # Compute log_prob and entropy per sub-action
        logprob = torch.stack(
            [cat.log_prob(a) for a, cat in zip(action, multi_categoricals)]
        )  # (7, B * mapsize)
        entropy = torch.stack([cat.entropy() for cat in multi_categoricals])  # (7, B * mapsize)

        # Hierarchical sub-action mask (RAISocketAI-style): zero out contributions
        # from sub-actions whose parent action_type doesn't activate them.
        # action[0] is the action_type for each (B*mapsize) cell.
        if self.use_hierarchical_mask:
            at = action[0]  # (B*mapsize,)
            parent = self._hmask_parent  # (7,)
            # mask[k, b] = 1 iff parent[k] < 0 (always active) OR at[b] == parent[k]
            hmask = (parent.unsqueeze(1) < 0) | (at.unsqueeze(0) == parent.unsqueeze(1))
            hmask = hmask.to(logprob.dtype)  # (7, B*mapsize)
            logprob = logprob * hmask
            entropy = entropy * hmask

        logprob = logprob.T.view(-1, mapsize, len(self.action_nvec))  # (B, mapsize, 7)
        entropy = entropy.T.view(-1, mapsize, len(self.action_nvec))  # (B, mapsize, 7)
        action = action.T.view(-1, mapsize, len(self.action_nvec))  # (B, mapsize, 7)

        # Sum over 7 sub-actions and mapsize cells -> one scalar per batch element
        return action, logprob.sum(1).sum(1), entropy.sum(1).sum(1), invalid_action_masks

    def _ar_act(self, spatial_features, split_masks, greedy=False):
        """Auto-regressive action selection. greedy=True: argmax, False: sample.
        spatial_features: (N, D) where N = B*mapsize, D = pre-projection channel dim.
        split_masks: 7 tensors [(N, 6), (N, 4), ..., (N, 49)].
        Each sub-action k receives [spatial_features, embed(a_0), ..., embed(a_{k-1})]."""
        results = []  # will hold 7 tensors of shape (N,)
        embedded = []  # will hold up to 6 embedding tensors of shape (N, embed_dim)

        for k in range(len(self.action_nvec)):
            # Build input: spatial features + embeddings of all previous sub-actions
            if k == 0:
                logits_k = self.ar_head.heads[k](spatial_features)  # (N, D) → (N, nvec[0])
            else:
                inp = torch.cat([spatial_features] + embedded, dim=1)  # (N, D + k * embed_dim)
                logits_k = self.ar_head.heads[k](inp)  # (N, nvec[k])

            masked = torch.where(split_masks[k].bool(), logits_k, self.mask_value)
            if greedy:
                a_k = masked.argmax(dim=1)  # (N,) best valid action
            else:
                a_k = Categorical(
                    logits=masked, validate_args=False
                ).sample()  # (N,) sampled action
            results.append(a_k)

            # Embed chosen action for conditioning next sub-action (skip last, nothing follows)
            if k < len(self.action_nvec) - 1:
                embedded.append(self.ar_head.embeddings[k](a_k))  # (N,) -> (N, embed_dim)

        # Stack: greedy → (N, 7) for predict_batch; sample -> (7, N) for _sample_action compat
        return torch.stack(results, dim=1) if greedy else torch.stack(results, dim=0)

    def _get_ar_features(self, hidden):
        """Upsample encoder hidden to full-res per-cell features for AR head.
        Returns (B*mapsize, D) where D = pre-projection channel dim."""
        features = self._ar_feature_net(hidden)
        features = self._actor_transpose(features)  # (B, C, H, W) -> (B, H, W, C)
        return features.reshape(-1, features.shape[-1])  # (B * mapsize, C)

    # ------------------------------------------------------------------
    # Forward methods
    # ------------------------------------------------------------------

    def forward(self, x, invalid_action_masks, action=None, return_hidden=False):
        """Unified forward: encode + sample/evaluate + all active value heads.
        Returns dict with keys:
            action, logprob, entropy, masks, v_shaped,
            v_sparse (if dual), v_cost (if triple),
            logits_shaped/logits_sparse/logits_cost (if hl_gauss),
            hidden (if return_hidden)
        """
        hidden = self.encoder(x)
        action, logprob, entropy, masks = self._sample_action(hidden, action, invalid_action_masks)

        # Compute critic logits once, reuse for both value extraction and HL-Gauss loss
        if self.hl_gauss:
            logits_shaped = self.critic_shaped(hidden)
            from microrts_agent.lib.architectures.features.hl_gauss import hl_gauss_value

            v_shaped = hl_gauss_value(logits_shaped, self.hl_gauss_bins).unsqueeze(-1)
        else:
            v_shaped = self._shaped_value(hidden)

        result = {
            "action": action,
            "logprob": logprob,
            "entropy": entropy,
            "masks": masks,
            "v_shaped": v_shaped,
        }

        if self.dual_value_heads:
            if self.hl_gauss:
                logits_sparse = self.critic_sparse(hidden)
                v_sparse_raw = hl_gauss_value(logits_sparse, self.hl_gauss_bins).unsqueeze(-1)
                result["v_sparse"] = torch.tanh(v_sparse_raw)
            else:
                result["v_sparse"] = self._sparse_value(hidden)
        if self.triple_value_heads:
            if self.hl_gauss:
                logits_cost = self.critic_cost(hidden)
                result["v_cost"] = hl_gauss_value(logits_cost, self.hl_gauss_bins).unsqueeze(-1)
            else:
                result["v_cost"] = self._cost_value(hidden)
        if self.hl_gauss:
            result["logits_shaped"] = logits_shaped
            if self.dual_value_heads:
                result["logits_sparse"] = logits_sparse
            if self.triple_value_heads:
                result["logits_cost"] = logits_cost
        if return_hidden:
            result["hidden"] = hidden

        return result

    def get_values(self, x):
        """Bootstrap GAE: returns only value(s), no action sampling.
        Returns dict with v_shaped, v_sparse (if dual), v_cost (if triple).
        """
        hidden = self.encoder(x)
        result = {"v_shaped": self._shaped_value(hidden)}
        if self.dual_value_heads:
            result["v_sparse"] = self._sparse_value(hidden)
        if self.triple_value_heads:
            result["v_cost"] = self._cost_value(hidden)
        return result

    @torch.no_grad()
    def predict_batch(self, obs, masks):
        """Batched deterministic action selection (greedy argmax). Used for evaluation.
        Input:  obs (B, H, W, C)  |  masks (B, mapsize, sum(action_nvec))
        Output: actions (B, mapsize, len(action_nvec)) int tensor
        """
        B = obs.shape[0]
        hidden = self.encoder(obs)
        mask_flat = masks.reshape(B, -1, self.num_action_planes)  # (B, mapsize, 78)

        if self.use_autoregressive:
            ar_features = self._get_ar_features(hidden)  # (B * mapsize, D)
            flat_masks = mask_flat.reshape(-1, self.num_action_planes)  # (B * mapsize, 78)
            split_masks = torch.split(flat_masks, self.action_nvec, dim=1)  # 7 tensors
            actions = self._ar_act(ar_features, split_masks, greedy=True)  # (B * mapsize, 7)
            return actions.reshape(B, -1, len(self.action_nvec))  # (B, mapsize, 7)

        # Standard path: independent greedy argmax per sub-action
        logits = self.actor(hidden)  # (B, H, W, 78)
        grid_logits = logits.reshape(B, -1, self.num_action_planes)  # (B, mapsize, 78)
        split_logits = torch.split(
            grid_logits, self.action_nvec, dim=2
        )  # 7 tensors: [(B, mapsize, 6), ...]
        split_masks = torch.split(
            mask_flat, self.action_nvec, dim=2
        )  # 7 masks:   [(B, mapsize, 6), ...]

        actions = []
        for sl, sm in zip(split_logits, split_masks):
            masked = torch.where(sm.bool(), sl, self.mask_value)  # invalid → -1e8
            actions.append(masked.argmax(dim=2))  # (B, mapsize) best valid action

        return torch.stack(actions, dim=2)  # (B, mapsize, 7)
