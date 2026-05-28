"""HL-Gauss (Stop Regressing, 2024): value prediction via classification over 255 bins.
Critic outputs bin logits instead of scalar. Target = soft Gaussian in symlog space.
More stable than MSE regression for PPO. Activated via --hl-gauss.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

# Default HL-Gauss parameters
HL_GAUSS_DEFAULT_BINS = 255
HL_GAUSS_MIN = -10.0  # after symlog
HL_GAUSS_MAX = 10.0  # after symlog
HL_GAUSS_SIGMA = 0.75  # Gaussian smoothing width


def symlog(x):
    """Symmetric logarithmic compression (DreamerV3).
    symlog(x) = sign(x) * ln(|x| + 1)
    Maps large values to manageable range while preserving sign and small values.
    """
    return torch.sign(x) * torch.log1p(x.abs())


def symexp(x):
    """Inverse of symlog.
    symexp(x) = sign(x) * (exp(|x|) - 1)
    """
    return torch.sign(x) * (torch.exp(x.abs()) - 1)


def _build_bin_centers(num_bins, v_min=HL_GAUSS_MIN, v_max=HL_GAUSS_MAX):
    """Return (num_bins,) tensor of evenly spaced bin centers in [v_min, v_max]."""
    return torch.linspace(v_min, v_max, num_bins)


def hl_gauss_target(target, num_bins, v_min=HL_GAUSS_MIN, v_max=HL_GAUSS_MAX, sigma=HL_GAUSS_SIGMA):
    """Create soft Gaussian target distribution over bins.

    Args:
        target: (*,) tensor of scalar targets (raw, NOT symlog-compressed)
        num_bins: number of discrete bins
        v_min, v_max: bin range (in symlog space)
        sigma: Gaussian smoothing width

    Returns:
        (*,num_bins) soft target distribution (sums to ~1)
    """
    # Compress target to symlog space
    target_compressed = symlog(target)
    # Clamp to bin range
    target_compressed = target_compressed.clamp(v_min, v_max)

    # Bin centers
    centers = _build_bin_centers(num_bins, v_min, v_max).to(target.device)

    # Gaussian kernel: exp(-0.5 * ((center - target) / sigma)^2)
    # target: (*,1), centers: (num_bins,)
    diff = centers - target_compressed.unsqueeze(-1)  # (*, num_bins)
    log_probs = -0.5 * (diff / sigma) ** 2
    # Normalize to valid distribution
    probs = F.softmax(log_probs, dim=-1)
    return probs


def hl_gauss_loss(
    logits, target, num_bins, v_min=HL_GAUSS_MIN, v_max=HL_GAUSS_MAX, sigma=HL_GAUSS_SIGMA
):
    """Cross-entropy loss between predicted bin logits and HL-Gauss soft target.

    Args:
        logits: (B, num_bins) raw critic output
        target: (B,) scalar return targets (raw, NOT symlog-compressed)
        num_bins: number of bins
        v_min, v_max: bin range (in symlog space)
        sigma: Gaussian smoothing width

    Returns:
        scalar loss (mean over batch)
    """
    soft_target = hl_gauss_target(target, num_bins, v_min, v_max, sigma)
    # Cross-entropy with soft targets: -sum(target * log_softmax(logits))
    log_probs = F.log_softmax(logits, dim=-1)
    loss = -(soft_target * log_probs).sum(dim=-1)
    return loss.mean()


def hl_gauss_value(logits, num_bins, v_min=HL_GAUSS_MIN, v_max=HL_GAUSS_MAX):
    """Extract scalar value from bin logits.
    Computes expected value in symlog space, then applies symexp.

    Args:
        logits: (*, num_bins) raw critic output

    Returns:
        (*,) scalar values in original (uncompressed) space
    """
    centers = _build_bin_centers(num_bins, v_min, v_max).to(logits.device)
    probs = F.softmax(logits, dim=-1)
    # Expected value in symlog space
    expected_symlog = (probs * centers).sum(dim=-1)
    # Decompress back to original scale
    return symexp(expected_symlog)


def convert_critic_to_bins(module, num_bins):
    """Replace the last Linear(..., 1) layer in a nn.Sequential with Linear(..., num_bins).
    Walks the sequential to find the last Linear layer, verifies its out_features=1,
    and replaces it with a new Linear(..., num_bins) with proper initialization.

    Args:
        module: nn.Sequential critic head
        num_bins: number of output bins

    Returns:
        The modified module (in-place modification)
    """
    # Find the last Linear layer
    last_linear_idx = None
    last_linear = None
    if isinstance(module, nn.Sequential):
        for i, layer in enumerate(module):
            if isinstance(layer, nn.Linear):
                last_linear_idx = i
                last_linear = layer

    if last_linear is None or last_linear.out_features != 1:
        raise ValueError(
            f"Cannot apply HL-Gauss: expected last Linear layer with out_features=1, "
            f"got {last_linear}"
        )

    # Replace with wider output
    new_linear = nn.Linear(last_linear.in_features, num_bins)
    # Initialize with small std (like layer_init with std=1 but scaled for bins)
    nn.init.orthogonal_(new_linear.weight, gain=0.01)
    nn.init.constant_(new_linear.bias, 0.0)

    module[last_linear_idx] = new_linear
    return module
