"""PopArt (van Hasselt 2016): adaptive value normalization with weight-preserving updates.
Critic predicts in ~N(0,1) space, denormalized for GAE. Last linear layer adjusted
when running stats change so output stays continuous. Activated via --popart.
"""

import torch
import torch.nn as nn


def get_output_layer(module):
    """Find the last nn.Linear layer in a module (typically nn.Sequential)."""
    last = None
    for m in module.modules():
        if isinstance(m, nn.Linear):
            last = m
    if last is None:
        raise ValueError(f"No nn.Linear found in {type(module).__name__}")
    return last


class PopArtNormalizer(nn.Module):
    """Running normalization with output-preserving weight adjustment.

    Uses exponential moving average to track target statistics.

    Args:
        beta: EMA decay rate for statistics update (default: 3e-4)
    """

    def __init__(self, beta=3e-4):
        super().__init__()
        self.beta = beta
        self.register_buffer("mu", torch.zeros(1))
        self.register_buffer("sigma", torch.ones(1))
        self.register_buffer("initialized", torch.zeros(1, dtype=torch.bool))

    def normalize(self, x):
        """Normalize targets to ~N(0,1) space."""
        return (x - self.mu) / self.sigma.clamp(min=1e-6)

    def denormalize(self, x):
        """Convert normalized predictions back to real value space."""
        return x * self.sigma + self.mu

    @torch.no_grad()
    def update_and_adjust(self, targets, linear_layer):
        """Update running stats from a batch of targets and adjust linear layer.

        The weight adjustment ensures that:
            new_output * new_sigma + new_mu == old_output * old_sigma + old_mu
        so the denormalized value function is continuous across updates.

        Args:
            targets: (N,) batch of real-valued returns
            linear_layer: the last nn.Linear of the critic to adjust
        """
        old_mu = self.mu.clone()
        old_sigma = self.sigma.clone()

        batch_mean = targets.mean()
        batch_var = targets.var()

        if not self.initialized.item():
            # First call: initialize directly from batch
            self.mu.copy_(batch_mean)
            self.sigma.copy_(batch_var.clamp(min=1e-6).sqrt())
            self.initialized.fill_(True)
        else:
            # EMA update
            new_mu = (1 - self.beta) * self.mu + self.beta * batch_mean
            new_var = (1 - self.beta) * (self.sigma**2) + self.beta * batch_var
            self.mu.copy_(new_mu)
            self.sigma.copy_(new_var.clamp(min=1e-6).sqrt())

        # Adjust last linear layer to preserve denormalized outputs
        # W_new = W_old * old_sigma / new_sigma
        # b_new = (b_old * old_sigma + old_mu - new_mu) / new_sigma
        ratio = old_sigma / self.sigma.clamp(min=1e-6)
        linear_layer.weight.data.mul_(ratio)
        linear_layer.bias.data.mul_(ratio)
        linear_layer.bias.data.add_((old_mu - self.mu) / self.sigma.clamp(min=1e-6))
