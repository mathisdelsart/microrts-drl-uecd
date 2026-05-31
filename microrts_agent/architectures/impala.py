"""Registry: "impala": IMPALA-CNN with 3-stage ResBlock encoder (~1M params).
Encoder: 3x [Conv+MaxPool+ResBlockx2]: spatial H->H/8, channels C->32->64->128.
Decoder: 3x ConvTranspose: spatial H/8->H, channels 128->64->32->78.
Critic: AdaptiveAvgPool -> MLP (any map size). Supports all optional features via **kwargs.
Exports reusable builders: build_impala_encoder, build_impala_actor, build_adaptive_critic.
"""

import torch.nn as nn

from .base_actor_critic import (
    GridActorCriticBase,
    SpatialPyramidPooling,
    Transpose,
    get_activation,
    layer_init,
)


class ResBlock(nn.Module):
    """Pre-activation residual block (IMPALA-style).
    act -> Conv3x3 -> act -> Conv3x3 + skip
    """

    def __init__(self, channels, use_gelu=False):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.act = get_activation(use_gelu)

    def forward(self, x):
        out = self.act(x)
        out = self.conv1(out)
        out = self.act(out)
        out = self.conv2(out)
        return out + x


def build_impala_encoder(obs_channels, out_channels=128, use_gelu=False):
    """Build 3-stage IMPALA CNN encoder.
    Each stage: Conv(c_in -> c_out) -> MaxPool(/2) -> ResBlock x 2.
    For 16x16 input: output is (B, out_channels, 2, 2).
    """
    return nn.Sequential(
        Transpose((0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
        layer_init(nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1)),  # (B, 32, H, W)
        nn.MaxPool2d(3, stride=2, padding=1),  # (B, 32, H/2, W/2)
        ResBlock(32, use_gelu),  # (B, 32, H/2, W/2)
        ResBlock(32, use_gelu),  # (B, 32, H/2, W/2)
        layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),  # (B, 64, H/2, W/2)
        nn.MaxPool2d(3, stride=2, padding=1),  # (B, 64, H/4, W/4)
        ResBlock(64, use_gelu),  # (B, 64, H/4, W/4)
        ResBlock(64, use_gelu),  # (B, 64, H/4, W/4)
        layer_init(nn.Conv2d(64, out_channels, kernel_size=3, padding=1)),  # (B, 128, H/4, W/4)
        nn.MaxPool2d(3, stride=2, padding=1),  # (B, 128, H/8, W/8)
        ResBlock(out_channels, use_gelu),  # (B, 128, H/8, W/8)
        ResBlock(out_channels, use_gelu),  # (B, 128, H/8, W/8)
        get_activation(use_gelu),  # final activation
    )


def build_impala_actor(num_action_planes, in_channels=128, use_gelu=False):
    """Build 3-stage ConvTranspose decoder (mirror of IMPALA encoder).
    For 16x16 maps: (B, 128, 2, 2) -> (B, 78, 16, 16) -> (B, 16, 16, 78)."""
    return nn.Sequential(
        layer_init(
            nn.ConvTranspose2d(in_channels, 64, 3, stride=2, padding=1, output_padding=1)
        ),  # (B, 64, H/4, W/4)
        get_activation(use_gelu),  # activation
        layer_init(
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
        ),  # (B, 32, H/2, W/2)
        get_activation(use_gelu),  # activation
        layer_init(
            nn.ConvTranspose2d(32, num_action_planes, 3, stride=2, padding=1, output_padding=1)
        ),  # (B, 78, H, W)
        Transpose((0, 2, 3, 1)),  # (B, H, W, 78)
    )


def build_adaptive_critic(in_channels=128, hidden=256, use_gelu=False, use_spp=False):
    """Build map-size-independent critic: AdaptiveAvgPool or SPP -> MLP.
    Input: (B, C, H', W') any spatial size. Output: (B, 1)."""
    if use_spp:
        spp = SpatialPyramidPooling(levels=(1, 2, 4))
        return nn.Sequential(
            spp,  # (B, C, H', W') -> (B, C*21)
            layer_init(nn.Linear(in_channels * spp.num_bins, hidden)),  # (B, 256)
            get_activation(use_gelu),  # activation
            layer_init(nn.Linear(hidden, 1), std=1),  # (B, 1)
        )
    return nn.Sequential(
        nn.AdaptiveAvgPool2d(1),  # (B, C, H', W') -> (B, C, 1, 1)
        nn.Flatten(),  # (B, C)
        layer_init(nn.Linear(in_channels, hidden)),  # (B, 256)
        get_activation(use_gelu),  # activation
        layer_init(nn.Linear(hidden, 1), std=1),  # (B, 1)
    )


class IMPALAAgent(GridActorCriticBase):
    """IMPALA-CNN encoder with ConvTranspose decoder.
    3 stages: Conv -> MaxPool -> ResBlock x2, channels 32 -> 64 -> 128.
    Critic uses AdaptiveAvgPool (map-size independent).
    """

    def __init__(self, obs_channels, action_nvec, use_gelu=False, use_spp=False, **kwargs):
        super().__init__()
        self._init_common(action_nvec)

        self.encoder = build_impala_encoder(obs_channels, use_gelu=use_gelu)
        self.actor = build_impala_actor(self.num_action_planes, use_gelu=use_gelu)
        self.critic_shaped = build_adaptive_critic(use_gelu=use_gelu, use_spp=use_spp)

        self._finalize(
            hidden_channels=128,
            extra_critic_builder=lambda: build_adaptive_critic(use_gelu=use_gelu, use_spp=use_spp),
            **kwargs,
        )
