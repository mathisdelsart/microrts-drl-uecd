"""Registry: "unet_entity" — UNet + Entity Transformer (~2.9M at C=48).
Combines the spatial preservation of UNet (skip connections, SE-ResBlocks)
with the entity-relational reasoning of the Entity Transformer.

Flow:
  1. UNet encoder: obs -> (bottleneck H/4, skip1 H, skip2 H/2)
  2. Entity extractor: detect units from obs, gather features from skip1 (full-res)
  3. Transformer: enrich entity representations via self-attention
  4. Dual scatter-add: inject entity deltas into both bottleneck (H/4) AND skip2 (H/2)
  5. UNet decoder: (enriched bottleneck + skip1 + enriched skip2) -> full-res features

Key design choices vs IMPALA-Entity:
  - Reuses skip1 features instead of a separate full-res CNN (saves ~150K params)
  - Dual scatter at two scales: bottleneck for global planning, skip2 for spatial precision
  - UNet skip connections preserve fine spatial detail through the decoder
"""

import torch.nn as nn

from .base_actor_critic import GridActorCriticBase, Transpose, get_activation, layer_init
from .impala_entity import EntityExtractor, scatter_entities_to_grid
from .unet import UNetDecoder, UNetEncoder, build_unet_critic


class _UNetEntityEncoderDecoder(nn.Module):
    """UNet encoder-decoder with Entity Transformer and dual scatter.
    Output: (B, ch1, H, W) — same as pure UNet encoder-decoder."""

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
        se_ratio=16,
        use_gelu=False,
        p0_channel=11,
        p1_channel=12,
    ):
        super().__init__()
        # UNet encoder & decoder
        self.enc = UNetEncoder(obs_channels, ch1, ch2, ch3, se_ratio=se_ratio, use_gelu=use_gelu)
        self.dec = UNetDecoder(ch1, ch2, ch3, se_ratio=se_ratio, use_gelu=use_gelu)

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
        )

        # Dual scatter projections: one per target scale
        self.scatter_proj_bot = nn.Sequential(
            layer_init(nn.Linear(d_model, ch3)),  # -> bottleneck channels (192)
            get_activation(use_gelu),
        )
        self.scatter_proj_s2 = nn.Sequential(
            layer_init(nn.Linear(d_model, ch2)),  # -> skip2 channels (96)
            get_activation(use_gelu),
        )

    def forward(self, x):
        # x: (B, H, W, C)
        bot, s1, s2 = self.enc(x)
        # bot: (B, ch3, H/4, W/4), s1: (B, ch1, H, W), s2: (B, ch2, H/2, W/2)

        # Entity path: extract from raw obs + full-res UNet features (s1)
        ent_feat, ent_mask, positions = self.entity_extractor(x, s1)
        ent_enriched = self.transformer(ent_feat, src_key_padding_mask=ent_mask)

        # Dual scatter: inject entity info at two scales
        # 1) Bottleneck (H/4): global planning, deep processing through decoder
        deltas_bot = self.scatter_proj_bot(ent_enriched)
        ds_bot = x.shape[1] // bot.shape[2]  # H / (H/4) = 4
        bot = scatter_entities_to_grid(deltas_bot, ent_mask, positions, bot, ds_bot)

        # 2) Skip2 (H/2): finer spatial precision, enters decoder at mid-resolution
        deltas_s2 = self.scatter_proj_s2(ent_enriched)
        ds_s2 = x.shape[1] // s2.shape[2]  # H / (H/2) = 2
        s2 = scatter_entities_to_grid(deltas_s2, ent_mask, positions, s2, ds_s2)

        # Decode with enriched bottleneck + skip1 + enriched skip2
        return self.dec(bot, s1, s2)  # (B, ch1, H, W)


class IMPALAUNetEntityAgent(GridActorCriticBase):
    def __init__(
        self,
        obs_channels,
        action_nvec,
        channels=48,
        se_ratio=16,
        use_gelu=False,
        use_spp=False,
        p0_channel=11,
        p1_channel=12,
        **kwargs,
    ):
        super().__init__()
        self._init_common(action_nvec)
        ch1, ch2, ch3 = channels, channels * 2, channels * 4

        self.encoder = _UNetEntityEncoderDecoder(
            obs_channels,
            ch1,
            ch2,
            ch3,
            se_ratio=se_ratio,
            use_gelu=use_gelu,
            p0_channel=p0_channel,
            p1_channel=p1_channel,
        )
        self.actor = nn.Sequential(
            layer_init(nn.Conv2d(ch1, self.num_action_planes, 3, padding=1), std=0.01),
            Transpose((0, 2, 3, 1)),
        )
        self.critic_shaped = build_unet_critic(ch1, use_gelu, use_spp)

        self._finalize(
            hidden_channels=ch1,
            extra_critic_builder=lambda: build_unet_critic(ch1, use_gelu, use_spp),
            **kwargs,
        )
