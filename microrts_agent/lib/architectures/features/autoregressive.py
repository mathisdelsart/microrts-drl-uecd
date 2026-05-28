"""Sequential sub-action sampling: each sub-action conditioned on previous choices.
head k input = [spatial_features, embed(a_0), ..., embed(a_{k-1})].
Activated via --autoregressive. Sampling loop in base_actor_critic._ar_act().
"""

import torch
import torch.nn as nn

from ..base_actor_critic import layer_init


class AutoRegressiveHead(nn.Module):
    def __init__(self, spatial_dim, action_nvec, embed_dim=8):
        super().__init__()
        self.action_nvec = action_nvec
        self.embed_dim = embed_dim
        self.num_sub = len(action_nvec)

        # action index -> learned vector (8d), fed as input to next head
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(n, embed_dim)
                for n in action_nvec  # e.g. Embedding(6, 8) for action_type
            ]
        )

        # head k: Linear(spatial_dim + k*embed_dim, n_k) -> logits for sub-action k
        self.heads = nn.ModuleList(
            [
                layer_init(nn.Linear(spatial_dim + k * embed_dim, n), std=0.01)
                for k, n in enumerate(action_nvec)
            ]
        )

    def forward_all(self, spatial_features, actions):
        """Recompute all logits given known actions (PPO update, for log_prob/entropy).
        spatial_features: (N, D), actions: (N, 7) -> returns list of 7× (N, n_k) logits.
        """
        all_logits = [self.heads[0](spatial_features)]  # head 0: spatial only -> (N, 6)

        embedded = []
        for k in range(1, self.num_sub):
            embedded.append(self.embeddings[k - 1](actions[:, k - 1]))  # (N, embed_dim)
            inp = torch.cat([spatial_features] + embedded, dim=1)  # (N, D + k*embed_dim)
            all_logits.append(self.heads[k](inp))  # (N, n_k)

        return all_logits
