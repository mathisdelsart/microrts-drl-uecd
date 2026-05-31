"""Registry: "unet_entity_cbam_deep": UNet-Entity-CBAM Deep (~4.7M at C=48).

Changes vs unet_entity_cbam:
  - Deeper: 2+3+4+3+2 = 14 CBAM-ResBlocks (matching RAISocketAI's depth)
  - Spatial self-attention at bottleneck (H/4 * W/4 tokens, very cheap)
  - Default C=48 (same as other UNet variants)

Flow:
  1. Encoder: obs (B,H,W,C) -> input_proj -> stage1 (2x CBAM, H) -> down1
     -> stage2 (3x CBAM, H/2) -> down2 -> bottleneck (4x CBAM, H/4)
     -> SpatialSelfAttention (H/4) -> (bot, s1, s2)
  2. Entity: obs + s1 -> extract -> Transformer 2L/4H/d128 -> enriched entities
  3. Scatter: proj -> scatter-add to bot (ch3) and s2 (ch2)
  4. Decoder: up1 + skip2 -> fuse1 (3x CBAM, H/2) -> up2 + skip1
     -> fuse2 (2x CBAM, H) -> (B, ch1, H, W)
  5. Actor: Conv3x3 -> 78 logits per cell
  6. Critic: Conv stride-2 x2 -> AdaptiveAvgPool/SPP -> MLP -> scalar
"""

import torch.nn as nn

from .base_actor_critic import GridActorCriticBase, Transpose, get_activation, layer_init
from .impala_entity import EntityExtractor, scatter_entities_to_grid
from .unet_entity_cbam import CBAMResBlock, build_cbam_unet_critic

# ── Spatial Self-Attention at bottleneck ─────────────────────────────────


class SpatialSelfAttention(nn.Module):
    """Multi-head self-attention over spatial positions at the bottleneck.

    Each (h, w) position is treated as a token. For a 16x16 map downsampled 4x,
    this gives 4x4 = 16 tokens: O(256) attention, negligible compute.

    Uses pre-norm (LayerNorm before attention) and residual connection.

    Input:  (B, C, H, W)
    Output: (B, C, H, W) : same shape, enriched with global spatial context.
    """

    def __init__(self, channels, nhead=4):
        super().__init__()
        self.nhead = nhead
        self.head_dim = channels // nhead
        assert channels % nhead == 0, f"channels ({channels}) must be divisible by nhead ({nhead})"
        self.scale = self.head_dim**-0.5

        self.norm = nn.LayerNorm(channels)
        self.qkv = layer_init(nn.Linear(channels, 3 * channels))  # -> Q, K, V
        self.proj = layer_init(nn.Linear(channels, channels))  # output projection

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        N = H * W  # number of spatial tokens

        # Reshape spatial dims to sequence: (B, C, H, W) -> (B, N, C)
        tokens = x.flatten(2).transpose(1, 2)  # (B, N, C)
        tokens_normed = self.norm(tokens)  # (B, N, C): pre-norm

        # QKV projection: (B, N, C) -> (B, N, 3*C) -> (3, B, nhead, N, head_dim)
        qkv = self.qkv(tokens_normed).reshape(B, N, 3, self.nhead, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, B, nhead, N, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Scaled dot-product attention
        att = (q @ k.transpose(-2, -1)) * self.scale  # (B, nhead, N, N)
        att = att.softmax(dim=-1)  # (B, nhead, N, N)
        out = (att @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)

        # Project + residual
        out = self.proj(out)  # (B, N, C)
        tokens = tokens + out  # (B, N, C): residual connection

        # Reshape back: (B, N, C) -> (B, C, H, W)
        return tokens.transpose(1, 2).reshape(B, C, H, W)


# ── UNet Encoder with variable depth + bottleneck attention ──────────────


class CBAMUNetEncoderV2(nn.Module):
    """3-stage encoder with CBAM + spatial self-attention at bottleneck.

    blocks_config: tuple of (stage1, stage2, bottleneck) block counts.
    Default (2, 3, 4): more blocks at bottleneck where spatial dims are smallest.

    Flow:
      obs (B,H,W,C) -> transpose -> (B,C,H,W)
      -> input_proj Conv3x3+Act              -> (B, ch1, H, W)
      -> stage1: b1 x CBAMRes               -> (B, ch1, H, W)         = skip1
      -> down1: Conv3x3 stride2 + Act       -> (B, ch2, H/2, W/2)
      -> stage2: b2 x CBAMRes               -> (B, ch2, H/2, W/2)     = skip2
      -> down2: Conv3x3 stride2 + Act       -> (B, ch3, H/4, W/4)
      -> bottleneck: b3 x CBAMRes           -> (B, ch3, H/4, W/4)
      -> SpatialSelfAttention (4 heads)      -> (B, ch3, H/4, W/4)     = bot

    Returns: (bot, skip1, skip2)
    """

    def __init__(
        self,
        obs_channels,
        ch1,
        ch2,
        ch3,
        blocks_config=(2, 3, 4),
        reduction=16,
        use_gelu=False,
        bottleneck_attention=True,
        sa_nhead=4,
    ):
        super().__init__()
        b1, b2, b3 = blocks_config

        self.transpose_in = Transpose((0, 3, 1, 2))  # (B,H,W,C) -> (B,C,H,W)
        self.input_proj = nn.Sequential(
            layer_init(nn.Conv2d(obs_channels, ch1, 3, padding=1)),  # C -> ch1
            get_activation(use_gelu),
        )
        self.stage1 = nn.Sequential(*[CBAMResBlock(ch1, reduction, use_gelu) for _ in range(b1)])
        self.down1 = nn.Sequential(
            layer_init(nn.Conv2d(ch1, ch2, 3, stride=2, padding=1)),  # ch1 -> ch2, H -> H/2
            get_activation(use_gelu),
        )
        self.stage2 = nn.Sequential(*[CBAMResBlock(ch2, reduction, use_gelu) for _ in range(b2)])
        self.down2 = nn.Sequential(
            layer_init(nn.Conv2d(ch2, ch3, 3, stride=2, padding=1)),  # ch2 -> ch3, H/2 -> H/4
            get_activation(use_gelu),
        )
        self.bottleneck = nn.Sequential(
            *[CBAMResBlock(ch3, reduction, use_gelu) for _ in range(b3)]
        )

        # Spatial self-attention after bottleneck CBAM blocks
        self.bottleneck_attention = (
            SpatialSelfAttention(ch3, nhead=sa_nhead)  # (B, ch3, H/4, W/4) -> same
            if bottleneck_attention
            else nn.Identity()
        )

    def forward(self, x):
        # x: (B, H, W, C)
        x = self.input_proj(self.transpose_in(x))  # (B, ch1, H, W)
        s1 = self.stage1(x)  # (B, ch1, H, W)      : skip1
        s2 = self.stage2(self.down1(s1))  # (B, ch2, H/2, W/2)  : skip2
        bot = self.bottleneck(self.down2(s2))  # (B, ch3, H/4, W/4)
        bot = self.bottleneck_attention(bot)  # (B, ch3, H/4, W/4)  : + global context
        return bot, s1, s2


# ── UNet Decoder with variable depth ────────────────────────────────────


class CBAMUNetDecoderV2(nn.Module):
    """2-stage decoder with CBAM and variable depth per stage.

    blocks_config: tuple of (fuse1, fuse2) block counts. Default (3, 2).

    Flow:
      bot (B, ch3, H/4, W/4)
      -> up1: ConvTranspose stride2 + Act    -> (B, ch2, H/2, W/2)
      -> + skip2                              -> (B, ch2, H/2, W/2)   [element-wise add]
      -> fuse1: b1 x CBAMRes                -> (B, ch2, H/2, W/2)
      -> up2: ConvTranspose stride2 + Act    -> (B, ch1, H, W)
      -> + skip1                              -> (B, ch1, H, W)       [element-wise add]
      -> fuse2: b2 x CBAMRes                -> (B, ch1, H, W)

    Returns: (B, ch1, H, W)
    """

    def __init__(self, ch1, ch2, ch3, blocks_config=(3, 2), reduction=16, use_gelu=False):
        super().__init__()
        b1, b2 = blocks_config

        self.up1 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch3, ch2, 2, stride=2)),  # ch3 -> ch2, H/4 -> H/2
            get_activation(use_gelu),
        )
        self.fuse1 = nn.Sequential(*[CBAMResBlock(ch2, reduction, use_gelu) for _ in range(b1)])
        self.up2 = nn.Sequential(
            layer_init(nn.ConvTranspose2d(ch2, ch1, 2, stride=2)),  # ch2 -> ch1, H/2 -> H
            get_activation(use_gelu),
        )
        self.fuse2 = nn.Sequential(*[CBAMResBlock(ch1, reduction, use_gelu) for _ in range(b2)])

    def forward(self, bot, skip1, skip2):
        # bot: (B, ch3, H/4, W/4), skip1: (B, ch1, H, W), skip2: (B, ch2, H/2, W/2)
        up1 = self.up1(bot)  # may overshoot skip2 by 1 on non-square maps
        up1 = up1[..., : skip2.shape[2], : skip2.shape[3]]  # crop to match skip2
        x = self.fuse1(up1 + skip2)  # (B, ch2, H/2, W/2)
        up2 = self.up2(x)  # may overshoot skip1 by 1
        up2 = up2[..., : skip1.shape[2], : skip1.shape[3]]  # crop to match skip1
        x = self.fuse2(up2 + skip1)  # (B, ch1, H, W)
        return x


# ── Full encoder-decoder with Entity Transformer ────────────────────────


class _CBAMUNetEntityEncoderDecoderV2(nn.Module):
    """CBAM-UNet V2 encoder-decoder with Entity Transformer and dual scatter.

    Flow:
      1. Encoder: obs -> (bot, s1, s2) with SpatialSelfAttention at bottleneck
      2. Entity:  obs + s1 -> EntityExtractor -> Transformer -> (B, max_ent, d_model)
      3. Scatter:
         - scatter_proj_bot: d_model -> ch3, scatter-add to bot at unit positions (H/4)
         - scatter_proj_s2:  d_model -> ch2, scatter-add to s2 at unit positions  (H/2)
      4. Decoder: (enriched bot, s1, enriched s2) -> (B, ch1, H, W)
    """

    def __init__(
        self,
        obs_channels,
        ch1,
        ch2,
        ch3,
        enc_blocks=(2, 3, 4),
        dec_blocks=(3, 2),
        d_model=128,
        nhead=4,
        num_layers=2,
        d_ff=256,
        max_entities=128,
        reduction=16,
        use_gelu=False,
        bottleneck_attention=True,
        sa_nhead=4,
        p0_channel=11,
        p1_channel=12,
    ):
        super().__init__()
        self.enc = CBAMUNetEncoderV2(
            obs_channels,
            ch1,
            ch2,
            ch3,
            blocks_config=enc_blocks,
            reduction=reduction,
            use_gelu=use_gelu,
            bottleneck_attention=bottleneck_attention,
            sa_nhead=sa_nhead,
        )
        self.dec = CBAMUNetDecoderV2(
            ch1, ch2, ch3, blocks_config=dec_blocks, reduction=reduction, use_gelu=use_gelu
        )

        # Entity Transformer: detects units, builds per-unit representations
        self.entity_extractor = EntityExtractor(
            obs_channels,
            ch1,
            d_model,
            max_entities,  # obs_C + ch1 + 2 -> d_model
            ds_factor=1,
            use_gelu=use_gelu,
            p0_channel=p0_channel,
            p1_channel=p1_channel,
        )
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, d_ff, batch_first=True, dropout=0.0),
            num_layers=num_layers,
        )  # (B, max_ent, 128) -> (B, max_ent, 128)

        # Dual scatter projections: d_model -> target resolution channels
        self.scatter_proj_bot = nn.Sequential(
            layer_init(nn.Linear(d_model, ch3)),  # 128 -> ch3 (bottleneck)
            get_activation(use_gelu),
        )
        self.scatter_proj_s2 = nn.Sequential(
            layer_init(nn.Linear(d_model, ch2)),  # 128 -> ch2 (mid-res)
            get_activation(use_gelu),
        )

    def forward(self, x):
        # x: (B, H, W, C)
        bot, s1, s2 = self.enc(x)
        # bot: (B, ch3, H/4, W/4), s1: (B, ch1, H, W), s2: (B, ch2, H/2, W/2)

        # Entity path: extract units from raw obs + full-res encoder features (s1)
        ent_feat, ent_mask, positions = self.entity_extractor(x, s1)  # (B, max_ent, d_model)
        ent_enriched = self.transformer(
            ent_feat, src_key_padding_mask=ent_mask
        )  # (B, max_ent, d_model)

        # Dual scatter: inject entity info at two spatial scales
        # 1) Bottleneck (H/4): enriches compressed global features
        deltas_bot = self.scatter_proj_bot(ent_enriched)  # (B, max_ent, ch3)
        ds_bot = x.shape[1] // bot.shape[2]  # H / (H/4) = 4
        bot = scatter_entities_to_grid(deltas_bot, ent_mask, positions, bot, ds_bot)

        # 2) Skip2 (H/2): enriches mid-resolution spatial features
        deltas_s2 = self.scatter_proj_s2(ent_enriched)  # (B, max_ent, ch2)
        ds_s2 = x.shape[1] // s2.shape[2]  # H / (H/2) = 2
        s2 = scatter_entities_to_grid(deltas_s2, ent_mask, positions, s2, ds_s2)

        # Decode with enriched tensors
        return self.dec(bot, s1, s2)  # (B, ch1, H, W)


# ── Agent ────────────────────────────────────────────────────────────────


class IMPALAUNetEntityCBAMV2Agent(GridActorCriticBase):
    """Full agent: CBAM-UNet-Deep encoder-decoder + Entity Transformer + actor/critic.

    Default channel config (C=48): ch1=48, ch2=96, ch3=192.
    Encoder blocks: (2, 3, 4) = 9 CBAM-ResBlocks.
    Decoder blocks: (3, 2) = 5 CBAM-ResBlocks. Total: 14 blocks.
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

        self.encoder = _CBAMUNetEntityEncoderDecoderV2(
            obs_channels,
            ch1,
            ch2,
            ch3,
            enc_blocks=(2, 3, 4),
            dec_blocks=(3, 2),
            reduction=reduction,
            use_gelu=use_gelu,
            bottleneck_attention=True,
            sa_nhead=4,
            p0_channel=p0_channel,
            p1_channel=p1_channel,
        )
        self.actor = nn.Sequential(
            layer_init(
                nn.Conv2d(ch1, self.num_action_planes, 3, padding=1), std=0.01
            ),  # ch1 -> 78 logits
            Transpose((0, 2, 3, 1)),  # (B, 78, H, W) -> (B, H, W, 78)
        )
        self.critic_shaped = build_cbam_unet_critic(ch1, use_gelu, use_spp)

        self._finalize(
            hidden_channels=ch1,
            extra_critic_builder=lambda: build_cbam_unet_critic(ch1, use_gelu, use_spp),
            **kwargs,
        )
