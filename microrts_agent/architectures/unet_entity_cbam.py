"""Registry: "unet_entity_cbam": UNet + Entity Transformer + CBAM (~3.0M at C=48).
Same as unet_entity but replaces SE-only attention with CBAM (channel + spatial).

CBAM (Woo et al. 2018): Convolutional Block Attention Module.
  - Channel attention: avg_pool + max_pool -> shared MLP -> sigmoid (what features matter)
  - Spatial attention: channel-wise avg+max -> 7x7 conv -> sigmoid (where to attend)
  - SE only answers "what channels matter". CBAM also answers "where on the map to focus".

This is directly relevant for RTS: identifying combat fronts, resource locations, and
enemy bases requires explicit spatial attention that SE blocks cannot provide.

Flow (same as unet_entity, with CBAM instead of SE):
  1. CBAM-UNet encoder: obs (B,H,W,C) -> (bot H/4, skip1 H, skip2 H/2)
  2. Entity extractor: detect units from obs + skip1 features -> (B, max_ent, d_model)
  3. Transformer: enrich entity representations via self-attention
  4. Dual scatter-add: inject entity deltas into bot (H/4) and skip2 (H/2)
  5. CBAM-UNet decoder: (enriched bot + skip1 + enriched skip2) -> (B, ch1, H, W)
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
from .impala_entity import EntityExtractor, scatter_entities_to_grid

# ── CBAM modules ─────────────────────────────────────────────────────────


class ChannelAttention(nn.Module):
    """Channel attention: dual-pool (avg + max) -> shared MLP -> sigmoid.
    Richer than SE which only uses avg pool.

    Input:  (B, C, H, W)
    Output: (B, C, H, W) : each channel rescaled by its importance weight.
    """

    def __init__(self, channels, reduction=16):
        super().__init__()
        hidden = max(channels // reduction, 1)
        self.fc1 = nn.Linear(channels, hidden, bias=False)  # C -> C/r
        self.fc2 = nn.Linear(hidden, channels, bias=False)  # C/r -> C

    def forward(self, x):
        # x: (B, C, H, W)
        avg = F.adaptive_avg_pool2d(x, 1).flatten(1)  # (B, C): average response per channel
        mx = F.adaptive_max_pool2d(x, 1).flatten(1)  # (B, C): peak response per channel
        # Shared MLP applied to both, then summed
        att = torch.sigmoid(
            self.fc2(F.relu(self.fc1(avg))) + self.fc2(F.relu(self.fc1(mx)))
        )  # (B, C): importance weight per channel
        return x * att.unsqueeze(-1).unsqueeze(-1)  # (B, C, H, W): broadcast multiply


class SpatialAttention(nn.Module):
    """Spatial attention: channel-wise avg+max -> concat -> 7x7 conv -> sigmoid.
    Learns 'where on the map to focus'.

    Input:  (B, C, H, W)
    Output: (B, C, H, W) : each spatial position rescaled by its importance.
    """

    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(
            2, 1, kernel_size, padding=padding, bias=False
        )  # (B,2,H,W) -> (B,1,H,W)

    def forward(self, x):
        # x: (B, C, H, W)
        avg = x.mean(dim=1, keepdim=True)  # (B, 1, H, W): mean across channels
        mx = x.max(dim=1, keepdim=True)[0]  # (B, 1, H, W): max across channels
        cat = torch.cat([avg, mx], dim=1)  # (B, 2, H, W): concat statistics
        att = torch.sigmoid(self.conv(cat))  # (B, 1, H, W): spatial saliency map
        return x * att  # (B, C, H, W): broadcast multiply


class CBAMResBlock(nn.Module):
    """Residual block with CBAM (channel + spatial attention).

    Flow: x -> Conv3x3 -> Act -> Conv3x3
              -> ChannelAtt (reweight channels)
              -> SpatialAtt (reweight positions)
              -> Act(out + x)  [residual connection]

    Input:  (B, C, H, W)
    Output: (B, C, H, W) : same shape, spatial resolution unchanged.
    """

    def __init__(self, channels, reduction=16, use_gelu=False):
        super().__init__()
        self.conv1 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.conv2 = layer_init(nn.Conv2d(channels, channels, 3, padding=1))
        self.channel_att = ChannelAttention(channels, reduction)
        self.spatial_att = SpatialAttention(kernel_size=7)
        self.act = get_activation(use_gelu)

    def forward(self, x):
        # x: (B, C, H, W)
        out = self.act(self.conv1(x))  # (B, C, H, W)
        out = self.conv2(out)  # (B, C, H, W)
        out = self.channel_att(out)  # (B, C, H, W): channel-reweighted
        out = self.spatial_att(out)  # (B, C, H, W): spatially-reweighted
        return self.act(out + x)  # (B, C, H, W): residual + activation


# ── UNet Encoder/Decoder with CBAM ──────────────────────────────────────


class CBAMUNetEncoder(nn.Module):
    """3-stage encoder with CBAM: full-res -> H/2 -> H/4 bottleneck.

    Flow:
      obs (B,H,W,C) -> transpose -> (B,C,H,W)
      -> input_proj Conv3x3+Act -> (B, ch1, H, W)
      -> stage1: 2x CBAMRes     -> (B, ch1, H, W)         = skip1
      -> down1: Conv3x3 stride2  -> (B, ch2, H/2, W/2)
      -> stage2: 2x CBAMRes     -> (B, ch2, H/2, W/2)     = skip2
      -> down2: Conv3x3 stride2  -> (B, ch3, H/4, W/4)
      -> bottleneck: 2x CBAMRes -> (B, ch3, H/4, W/4)     = bot

    Returns: (bot, skip1, skip2)
    """

    def __init__(
        self, obs_channels, ch1, ch2, ch3, blocks_per_stage=2, reduction=16, use_gelu=False
    ):
        super().__init__()
        self.transpose_in = Transpose((0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
        self.input_proj = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, ch1, 3, padding=1)),  # C -> ch1
            get_activation(use_gelu),
        )
        self.stage1 = nn.Sequential(
            *[CBAMResBlock(ch1, reduction, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.down1 = nn.Sequential(
            layer_init(nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)),  # ch1 -> ch2, H -> H/2
            get_activation(use_gelu),
        )
        self.stage2 = nn.Sequential(
            *[CBAMResBlock(ch2, reduction, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.down2 = nn.Sequential(
            layer_init(nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)),  # ch2 -> ch3, H/2 -> H/4
            get_activation(use_gelu),
        )
        self.bottleneck = nn.Sequential(
            *[CBAMResBlock(ch3, reduction, use_gelu) for _ in range(blocks_per_stage)]
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.input_proj(self.transpose_in(x))  # (B, ch1, H, W)
        s1 = self.stage1(x)  # (B, ch1, H, W)      : skip1
        s2 = self.stage2(self.down1(s1))  # (B, ch2, H/2, W/2)  : skip2
        bot = self.bottleneck(self.down2(s2))  # (B, ch3, H/4, W/4)  : bottleneck
        return bot, s1, s2


class CBAMUNetDecoder(nn.Module):
    """2-stage decoder with CBAM: H/4 -> H/2 -> full-res with skip-add.

    Flow:
      bot (B, ch3, H/4, W/4)
      -> up1: ConvTranspose2d stride2 -> (B, ch2, H/2, W/2)
      -> + skip2                      -> (B, ch2, H/2, W/2)   [element-wise add]
      -> fuse1: 3x CBAMRes            -> (B, ch2, H/2, W/2)
      -> up2: ConvTranspose2d stride2 -> (B, ch1, H, W)
      -> + skip1                      -> (B, ch1, H, W)       [element-wise add]
      -> fuse2: 2x CBAMRes            -> (B, ch1, H, W)

    Returns: (B, ch1, H, W)
    """

    def __init__(self, ch1, ch2, ch3, blocks_per_stage=2, reduction=16, use_gelu=False):
        super().__init__()
        self.up1 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch3, ch2, 2, stride=2)),  # ch3 -> ch2, H/4 -> H/2
            get_activation(use_gelu),
        )
        self.fuse1 = nn.Sequential(
            *[CBAMResBlock(ch2, reduction, use_gelu) for _ in range(blocks_per_stage)]
        )
        self.up2 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch2, ch1, 2, stride=2)),  # ch2 -> ch1, H/2 -> H
            get_activation(use_gelu),
        )
        self.fuse2 = nn.Sequential(
            *[CBAMResBlock(ch1, reduction, use_gelu) for _ in range(blocks_per_stage)]
        )

    def forward(self, bot, skip1, skip2):
        # bot: (B, ch3, H/4, W/4), skip1: (B, ch1, H, W), skip2: (B, ch2, H/2, W/2)
        up1 = self.up1(bot)
        up1 = up1[..., : skip2.shape[2], : skip2.shape[3]]
        x = self.fuse1(up1 + skip2)  # (B, ch2, H/2, W/2)
        up2 = self.up2(x)
        up2 = up2[..., : skip1.shape[2], : skip1.shape[3]]
        x = self.fuse2(up2 + skip1)  # (B, ch1, H, W)
        return x


# ── Critic (reuse structure from unet) ───────────────────────────────


def build_cbam_unet_critic(ch1, use_gelu=False, use_spp=False):
    """Critic: Conv stride-2 x2 -> pool -> MLP -> scalar.
    Same structure as unet critic. Map-size independent via AdaptiveAvgPool or SPP."""
    c = 64
    if use_spp:
        spp = SpatialPyramidPooling(levels=(1, 2, 4))
        return nn.Sequential(
            layer_init(nn.Conv2d(ch1, c, 3, stride=2, padding=1)),
            get_activation(use_gelu),  # ch1 -> 64, /2
            layer_init(nn.Conv2d(c, c, 3, stride=2, padding=1)),
            get_activation(use_gelu),  # 64 -> 64, /4
            spp,  # -> (B, 64*21)
            layer_init(nn.Linear(c * spp.num_bins, c)),
            get_activation(use_gelu),  # -> (B, 64)
            layer_init(nn.Linear(c, 1), std=1),  # -> (B, 1)
        )
    return nn.Sequential(
        layer_init(nn.Conv2d(ch1, c, 3, stride=2, padding=1)),
        get_activation(use_gelu),  # ch1 -> 64, /2
        layer_init(nn.Conv2d(c, c, 3, stride=2, padding=1)),
        get_activation(use_gelu),  # 64 -> 64, /4
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),  # -> (B, 64)
        layer_init(nn.Linear(c, c)),
        get_activation(use_gelu),  # -> (B, 64)
        layer_init(nn.Linear(c, 1), std=1),  # -> (B, 1)
    )


# ── Full encoder-decoder with Entity Transformer ────────────────────────


class _CBAMUNetEntityEncoderDecoder(nn.Module):
    """CBAM-UNet encoder-decoder with Entity Transformer and dual scatter.

    Flow:
      1. Encoder: obs (B,H,W,C) -> (bot, s1, s2)
      2. Entity:  obs + s1 -> extract -> Transformer -> enriched entities (B, max_ent, d_model)
      3. Scatter:
         - proj_bot: (B, max_ent, d_model) -> (B, max_ent, ch3) -> scatter-add to bot (H/4)
         - proj_s2:  (B, max_ent, d_model) -> (B, max_ent, ch2) -> scatter-add to s2  (H/2)
      4. Decoder: (enriched bot, s1, enriched s2) -> (B, ch1, H, W)
    """

    def __init__(
        self,
        obs_channels,
        ch1,
        ch2,
        ch3,
        d_model=128,
        nhead=4,
        num_layers=2,
        d_ff=256,
        max_entities=128,
        reduction=16,
        use_gelu=False,
        p0_channel=11,
        p1_channel=12,
    ):
        super().__init__()
        # UNet with CBAM blocks
        self.enc = CBAMUNetEncoder(
            obs_channels, ch1, ch2, ch3, reduction=reduction, use_gelu=use_gelu
        )
        self.dec = CBAMUNetDecoder(ch1, ch2, ch3, reduction=reduction, use_gelu=use_gelu)

        # Entity Transformer: uses skip1 (ch1 channels, full-res) as feature source
        self.entity_extractor = EntityExtractor(
            obs_channels,
            ch1,
            d_model,
            max_entities,
            ds_factor=1,
            use_gelu=use_gelu,
            p0_channel=p0_channel,
            p1_channel=p1_channel,
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True, dropout=0.0),
            num_layers=num_layers,
        )  # (B, max_ent, d_model) -> (B, max_ent, d_model)

        # Dual scatter projections: d_model -> target channel count
        self.scatter_proj_bot = nn.Sequential(
            layer_init(nn.Linear(d_model, ch3)),  # d_model -> ch3 (bottleneck channels)
            get_activation(use_gelu),
        )
        self.scatter_proj_s2 = nn.Sequential(
            layer_init(nn.Linear(d_model, ch2)),  # d_model -> ch2 (skip2 channels)
            get_activation(use_gelu),
        )

    def forward(self, x):
        # x: (B, H, W, C)
        bot, s1, s2 = self.enc(x)
        # bot: (B, ch3, H/4, W/4), s1: (B, ch1, H, W), s2: (B, ch2, H/2, W/2)

        # Entity path: extract from raw obs + full-res features (s1)
        ent_feat, ent_mask, positions = self.entity_extractor(x, s1)  # (B, max_ent, d_model)
        ent_enriched = self.transformer(
            ent_feat, src_key_padding_mask=ent_mask
        )  # (B, max_ent, d_model)

        # Dual scatter: inject entity info at two spatial scales
        # 1) Bottleneck (H/4): enriches global planning features
        deltas_bot = self.scatter_proj_bot(ent_enriched)  # (B, max_ent, ch3)
        ds_bot = x.shape[1] // bot.shape[2]  # H / (H/4) = 4
        bot = scatter_entities_to_grid(deltas_bot, ent_mask, positions, bot, ds_bot)

        # 2) Skip2 (H/2): enriches mid-resolution spatial features
        deltas_s2 = self.scatter_proj_s2(ent_enriched)  # (B, max_ent, ch2)
        ds_s2 = x.shape[1] // s2.shape[2]  # H / (H/2) = 2
        s2 = scatter_entities_to_grid(deltas_s2, ent_mask, positions, s2, ds_s2)

        # Decode with enriched bottleneck + skip1 + enriched skip2
        return self.dec(bot, s1, s2)  # (B, ch1, H, W)


# ── Agent ────────────────────────────────────────────────────────────────


class IMPALAUNetEntityCBAMAgent(GridActorCriticBase):
    """Full agent: CBAM-UNet encoder-decoder + Entity Transformer + actor/critic heads.

    Default channel config (C=48): ch1=48, ch2=96, ch3=192.
    constant_channels=True: ch1=ch2=ch3=C (used for some ablations).
    """

    def __init__(
        self,
        obs_channels,
        action_nvec,
        channels=48,
        reduction=16,
        use_gelu=False,
        use_spp=False,
        constant_channels=False,
        p0_channel=11,
        p1_channel=12,
        **kwargs,
    ):
        super().__init__()
        self._init_common(action_nvec)
        if constant_channels:
            ch1, ch2, ch3 = channels, channels, channels
        else:
            ch1, ch2, ch3 = channels, channels * 2, channels * 4

        self.encoder = _CBAMUNetEntityEncoderDecoder(
            obs_channels,
            ch1,
            ch2,
            ch3,
            reduction=reduction,
            use_gelu=use_gelu,
            p0_channel=p0_channel,
            p1_channel=p1_channel,
        )
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(ch1, self.num_action_planes, 3, padding=1), std=0.01),  # ch1 -> 78
            Transpose((0, 2, 3, 1)),  # (B, 78, H, W) -> (B, H, W, 78)
        )
        self.critic_shaped = build_cbam_unet_critic(ch1, use_gelu, use_spp)

        self._finalize(
            hidden_channels=ch1,
            extra_critic_builder=lambda: build_cbam_unet_critic(ch1, use_gelu, use_spp),
            **kwargs,
        )
