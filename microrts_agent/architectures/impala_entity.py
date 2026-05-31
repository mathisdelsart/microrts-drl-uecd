"""Registry: "impala_entity": IMPALA + Entity Transformer (~1.35M params).
Inherits IMPALAAgent, replaces encoder with IMPALAEntityEncoder.
Parallel paths: IMPALA (spatial H->H/8, ch C->128) + full-res CNN (ch C->64, no spatial loss).
Units detected from obs -> Transformer (2L, 4H) -> scatter-add to IMPALA H/8 bottleneck.
"""

import torch
import torch.nn as nn

from .base_actor_critic import Transpose, get_activation, layer_init
from .impala import IMPALAAgent, build_impala_encoder


class EntityExtractor(nn.Module):
    """Detect units from ownership channels, build per-entity feature vectors.
    Returns padded (B, max_entities, d_model) with attention mask for Transformer."""

    def __init__(
        self,
        obs_channels,
        feature_channels,
        d_model,
        max_entities=128,
        ds_factor=1,
        use_gelu=False,
        p0_channel=11,
        p1_channel=12,
    ):
        super().__init__()
        self.max_entities = max_entities
        self.ds_factor = ds_factor
        self.p0_channel = p0_channel
        self.p1_channel = p1_channel
        # Per-entity: [raw_obs(C) + spatial_features(F) + normalized_position(2)] -> d_model
        self.projection = nn.Sequential(
            layer_init(
                nn.Linear(obs_channels + feature_channels + 2, d_model)
            ),  # (*, C+F+2) -> (*, d_model)
            get_activation(use_gelu),
        )

    def forward(self, raw_obs, feature_map):
        """raw_obs: (B, H, W, C). feature_map: (B, F, H', W').
        Returns: (entity_features, padding_mask, positions)."""
        B, H, W, C = raw_obs.shape
        feat_ch = feature_map.shape[1]
        device = raw_obs.device

        # Detect units: ownership channels for P0 and P1
        unit_flat = (
            (raw_obs[:, :, :, self.p0_channel] + raw_obs[:, :, :, self.p1_channel]) > 0.5
        ).view(B, -1)  # (B, H * W)
        all_pos = torch.nonzero(unit_flat, as_tuple=False)  # (N_total, 2)

        # Pre-allocate padded outputs
        feat_dim = C + feat_ch + 2
        all_features = torch.zeros(B, self.max_entities, feat_dim, device=device)
        padding_mask = torch.ones(B, self.max_entities, dtype=torch.bool, device=device)
        positions = torch.zeros(B, self.max_entities, 2, dtype=torch.long, device=device)

        if all_pos.shape[0] == 0:
            return self.projection(all_features), padding_mask, positions

        # Compute per-batch entity index and filter to max_entities
        batch_idx, flat_pos = all_pos[:, 0], all_pos[:, 1]
        rows, cols = flat_pos // W, flat_pos % W
        counts = unit_flat.sum(dim=1)  # (B,)
        cum_counts = torch.zeros(B, dtype=torch.long, device=device)
        cum_counts[1:] = counts[:-1].cumsum(0)
        entity_idx = torch.arange(all_pos.shape[0], device=device) - cum_counts[batch_idx]
        valid = entity_idx < self.max_entities
        batch_idx, entity_idx = batch_idx[valid], entity_idx[valid]
        rows, cols = rows[valid], cols[valid]

        # Gather features at unit positions
        raw_feats = raw_obs[batch_idx, rows, cols, :]  # (N_valid, C)
        ds_rows = (rows // self.ds_factor).clamp(max=feature_map.shape[2] - 1)
        ds_cols = (cols // self.ds_factor).clamp(max=feature_map.shape[3] - 1)
        spatial_feats = feature_map[batch_idx, :, ds_rows, ds_cols]  # (N_valid, F)
        norm_pos = torch.stack([rows.float() / H, cols.float() / W], dim=1)  # (N_valid, 2)

        # Scatter into padded tensors
        all_features[batch_idx, entity_idx] = torch.cat([raw_feats, spatial_feats, norm_pos], dim=1)
        padding_mask[batch_idx, entity_idx] = False
        positions[batch_idx, entity_idx, 0] = rows
        positions[batch_idx, entity_idx, 1] = cols

        return self.projection(all_features), padding_mask, positions  # (B, max_ent, d_model)


def scatter_entities_to_grid(deltas, ent_mask, positions, target_grid, ds_factor=1):
    """Add enriched entity deltas back to spatial grid at their positions.
    deltas: (B, max_ent, C). target_grid: (B, C, H_t, W_t). Returns same shape."""
    deltas = deltas * (~ent_mask).unsqueeze(-1).float()  # zero out padding
    B, C, H_t, W_t = target_grid.shape
    non_pad = torch.nonzero(~ent_mask, as_tuple=False)  # (N_real, 2)
    if non_pad.shape[0] == 0:
        return target_grid

    b_idx, e_idx = non_pad[:, 0], non_pad[:, 1]
    rows_t = (positions[b_idx, e_idx, 0] // ds_factor).clamp(max=H_t - 1)
    cols_t = (positions[b_idx, e_idx, 1] // ds_factor).clamp(max=W_t - 1)

    enriched = target_grid.permute(0, 2, 3, 1).reshape(-1, C).clone()  # (B * H_t * W_t, C)
    flat_idx = b_idx * H_t * W_t + rows_t * W_t + cols_t
    enriched.index_add_(0, flat_idx, deltas[b_idx, e_idx, :])
    return enriched.reshape(B, H_t, W_t, C).permute(0, 3, 1, 2)  # (B, C, H_t, W_t)


class IMPALAEntityEncoder(nn.Module):
    """IMPALA CNN (8x compress) + full-res Entity Transformer merged via scatter-add.
    Output: (B, 128, H/8, W/8): same shape as pure IMPALA encoder."""

    def __init__(
        self,
        obs_channels,
        d_model=128,
        nhead=4,
        num_layers=2,
        d_ff=256,
        max_entities=128,
        use_gelu=False,
        p0_channel=11,
        p1_channel=12,
    ):
        super().__init__()
        fullres_ch = 64

        # Full-res CNN: lightweight feature extraction for entity quality
        self.fullres_cnn = nn.Sequential(
            Transpose((0, 3, 1, 2)),  # (B,H,W,C) -> (B,C,H,W)
            layer_init(nn.Conv2d(obs_channels, fullres_ch, 3, padding=1)),  # (B, 64, H, W)
            get_activation(use_gelu),
            layer_init(nn.Conv2d(fullres_ch, fullres_ch, 3, padding=1)),  # (B, 64, H, W)
            get_activation(use_gelu),
        )

        # Standard IMPALA encoder (runs in parallel)
        self.cnn = build_impala_encoder(obs_channels, use_gelu=use_gelu)  # -> (B, 128, H/8, W/8)

        # Entity Transformer
        self.entity_extractor = EntityExtractor(
            obs_channels,
            fullres_ch,
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
        self.scatter_proj = nn.Sequential(
            layer_init(nn.Linear(d_model, 128)),  # (B, max_ent, 128)
            get_activation(use_gelu),
        )

    def forward(self, x):
        fullres = self.fullres_cnn(x)  # (B, 64, H, W)
        cnn_out = self.cnn(x)  # (B, 128, H/8, W/8)

        # Entity path: extract -> transform -> scatter back to IMPALA grid
        ent_feat, ent_mask, positions = self.entity_extractor(x, fullres)  # (B, max_ent, 128)
        ent_enriched = self.transformer(
            ent_feat, src_key_padding_mask=ent_mask
        )  # (B, max_ent, 128)
        deltas = self.scatter_proj(ent_enriched)  # (B, max_ent, 128)

        ds_factor = x.shape[1] // cnn_out.shape[2]  # H / (H/8) = 8
        return scatter_entities_to_grid(deltas, ent_mask, positions, cnn_out, ds_factor)


class IMPALAEntityAgent(IMPALAAgent):
    def __init__(
        self,
        obs_channels,
        action_nvec,
        use_gelu=False,
        use_spp=False,
        p0_channel=11,
        p1_channel=12,
        **kwargs,
    ):
        super().__init__(obs_channels, action_nvec, use_gelu=use_gelu, use_spp=use_spp, **kwargs)
        self.encoder = IMPALAEntityEncoder(
            obs_channels, use_gelu=use_gelu, p0_channel=p0_channel, p1_channel=p1_channel
        )
