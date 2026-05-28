"""Registry: "unet" — U-shaped encoder-decoder with SE attention (~2.5M at C=48).
Encoder: spatial H->H/2->H/4, channels C->2C->4C. Decoder: H/4->H/2->H with U-Net skip-add.
SE-ResBlocks (channel attention) at every stage, learnable Conv stride-2 downsampling.
Actor: direct Conv 3x3 at full-res. Channels configurable via --arch-channels.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .base_actor_critic import (
    GridActorCriticBase,
    SpatialPyramidPooling,
    Transpose,
    get_activation,
    layer_init,
)


class SEResBlock(nn.Module):
    """Residual block with Squeeze-and-Excitation channel attention.
    conv -> act -> conv -> SE(pool -> fc↓ -> act -> fc↑ -> sigmoid -> scale) -> act(out + skip)"""

    def __init__(self, channels, se_ratio=16, use_gelu=False):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        se_hidden = max(channels // se_ratio, 1)
        self.se_fc1 = nn.Linear(channels, se_hidden, bias=False)
        self.se_fc2 = nn.Linear(se_hidden, channels, bias=False)
        self.act = get_activation(use_gelu)

    def forward(self, x):
        out = self.act(self.conv1(x))  # (B, C, H, W)
        out = self.conv2(out)  # (B, C, H, W)
        se = F.adaptive_avg_pool2d(out, 1).flatten(1)  # (B, C) global squeeze
        se = torch.sigmoid(self.se_fc2(self.act(self.se_fc1(se))))  # (B, C) channel weights
        return self.act(out * se.unsqueeze(-1).unsqueeze(-1) + x)  # scale + skip + act


class UNetEncoder(nn.Module):
    """3-stage encoder: full-res -> /2 -> /4 bottleneck. Returns (bottleneck, skip1, skip2)."""

    def __init__(
        self, obs_channels, ch1, ch2, ch3, blocks_per_stage=2, se_ratio=16, use_gelu=False
    ):
        super().__init__()
        self.transpose_in = Transpose((0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
        self.input_proj = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, ch1, 3, padding=1)), get_activation(use_gelu)
        )
        self.stage1 = nn.Sequential(
            *[SEResBlock(ch1, se_ratio, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.down1 = nn.Sequential(
            layer_init(nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)), get_activation(use_gelu)
        )
        self.stage2 = nn.Sequential(
            *[SEResBlock(ch2, se_ratio, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.down2 = nn.Sequential(
            layer_init(nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)), get_activation(use_gelu)
        )
        self.bottleneck = nn.Sequential(
            *[SEResBlock(ch3, se_ratio, use_gelu) for _ in range(blocks_per_stage)]
        )

    def forward(self, x):
        x = self.input_proj(self.transpose_in(x))  # (B, ch1, H, W)
        s1 = self.stage1(x)  # (B, ch1, H, W)      — skip1
        s2 = self.stage2(self.down1(s1))  # (B, ch2, H/2, W/2)  — skip2
        bot = self.bottleneck(self.down2(s2))  # (B, ch3, H/4, W/4)
        return bot, s1, s2


class UNetDecoder(nn.Module):
    """2-stage decoder: /4 -> /2 -> full-res with U-Net skip-add."""

    def __init__(self, ch1, ch2, ch3, blocks_per_stage=2, se_ratio=16, use_gelu=False):
        super().__init__()
        self.up1 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch3, ch2, 2, stride=2)), get_activation(use_gelu)
        )
        self.fuse1 = nn.Sequential(
            *[SEResBlock(ch2, se_ratio, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.up2 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch2, ch1, 2, stride=2)), get_activation(use_gelu)
        )
        self.fuse2 = nn.Sequential(
            *[SEResBlock(ch1, se_ratio, use_gelu) for _ in range(blocks_per_stage)]
        )

    def forward(self, bot, skip1, skip2):
        up1 = self.up1(bot)
        up1 = up1[..., : skip2.shape[2], : skip2.shape[3]]
        x = self.fuse1(up1 + skip2)  # (B, ch2, H/2, W/2)
        up2 = self.up2(x)
        up2 = up2[..., : skip1.shape[2], : skip1.shape[3]]
        x = self.fuse2(up2 + skip1)  # (B, ch1, H, W)
        return x


def build_unet_critic(ch1, use_gelu=False, use_spp=False):
    """Critic: full-res -> 2x stride-2 conv -> pool -> MLP -> scalar.
    Input: (B, ch1, H, W). Output: (B, 1)."""
    c = 64
    if use_spp:
        spp = SpatialPyramidPooling(levels=(1, 2, 4))
        return nn.Sequential(
            layer_init(nn.Conv2d(ch1, c, 3, stride=2, padding=1)),
            get_activation(use_gelu),  # (B, 64, H/2, W/2)
            layer_init(nn.Conv2d(c, c, 3, stride=2, padding=1)),
            get_activation(use_gelu),  # (B, 64, H/4, W/4)
            spp,  # (B, 64*21)
            layer_init(nn.Linear(c * spp.num_bins, c)),
            get_activation(use_gelu),  # (B, 64)
            layer_init(nn.Linear(c, 1), std=1),  # (B, 1)
        )
    return nn.Sequential(
        layer_init(nn.Conv2d(ch1, c, 3, stride=2, padding=1)),
        get_activation(use_gelu),  # (B, 64, H/2, W/2)
        layer_init(nn.Conv2d(c, c, 3, stride=2, padding=1)),
        get_activation(use_gelu),  # (B, 64, H/4, W/4)
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),  # (B, 64)
        layer_init(nn.Linear(c, c)),
        get_activation(use_gelu),  # (B, 64)
        layer_init(nn.Linear(c, 1), std=1),  # (B, 1)
    )


class _UNetEncoderDecoder(nn.Module):
    def __init__(self, enc, dec):
        super().__init__()
        self.enc = enc
        self.dec = dec

    def forward(self, x):
        bot, s1, s2 = self.enc(x)
        return self.dec(bot, s1, s2)  # (B, ch1, H, W)


class IMPALAUNetAgent(GridActorCriticBase):
    def __init__(
        self,
        obs_channels,
        action_nvec,
        channels=48,
        se_ratio=16,
        use_gelu=False,
        use_spp=False,
        **kwargs,
    ):
        super().__init__()
        self._init_common(action_nvec)
        ch1, ch2, ch3 = channels, channels * 2, channels * 4

        self.encoder = _UNetEncoderDecoder(
            UNetEncoder(obs_channels, ch1, ch2, ch3, se_ratio=se_ratio, use_gelu=use_gelu),
            UNetDecoder(ch1, ch2, ch3, se_ratio=se_ratio, use_gelu=use_gelu),
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Conv2d(ch1, self.num_action_planes, 3, padding=1), std=0.01
            ),  # (B, 78, H, W)
            Transpose((0, 2, 3, 1)),  # (B, H, W, 78)
        )
        self.critic_shaped = build_unet_critic(ch1, use_gelu, use_spp)

        self._finalize(
            hidden_channels=ch1,
            extra_critic_builder=lambda: build_unet_critic(ch1, use_gelu, use_spp),
            **kwargs,
        )
