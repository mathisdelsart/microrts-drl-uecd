"""Registry: "gridnet": Baseline encoder-decoder from Huang et al. 2021 (~838k params).
Encoder: 4x Conv+MaxPool: spatial 16x16->1x1, channels 29->32->64->128->256.
Decoder: 4x ConvTranspose: spatial 1x1->16x16, channels 256->128->64->32->78.
Critic: Flatten(256) -> MLP. Fixed to 16x16 maps. No optional features.
"""

import torch.nn as nn

from .base_actor_critic import GridActorCriticBase, Transpose, layer_init


class GridNetAgent(GridActorCriticBase):
    """Encoder-decoder GridNet from Huang et al. 2021 (best GridNet config).
    Baseline only: no dual/triple heads, no aux tasks, no optional features."""

    def __init__(self, obs_channels, action_nvec, **kwargs):
        super().__init__()
        self._init_common(action_nvec)

        self.encoder = nn.Sequential(
            Transpose((0, 3, 1, 2)),  # (B, H, W, C) -> (B, C, H, W)
            layer_init(nn.Conv2d(obs_channels, 32, kernel_size=3, padding=1)),  # (B, 32, 16, 16)
            nn.MaxPool2d(3, stride=2, padding=1),  # (B, 32, 8, 8)
            nn.ReLU(),
            layer_init(nn.Conv2d(32, 64, kernel_size=3, padding=1)),  # (B, 64, 8, 8)
            nn.MaxPool2d(3, stride=2, padding=1),  # (B, 64, 4, 4)
            nn.ReLU(),
            layer_init(nn.Conv2d(64, 128, kernel_size=3, padding=1)),  # (B, 128, 4, 4)
            nn.MaxPool2d(3, stride=2, padding=1),  # (B, 128, 2, 2)
            nn.ReLU(),
            layer_init(nn.Conv2d(128, 256, kernel_size=3, padding=1)),  # (B, 256, 2, 2)
            nn.MaxPool2d(3, stride=2, padding=1),  # (B, 256, 1, 1)
        )

        self.actor = nn.Sequential(
            layer_init(
                nn.ConvTranspose2d(256, 128, 3, stride=2, padding=1, output_padding=1)
            ),  # (B, 128, 2, 2)
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1)
            ),  # (B, 64, 4, 4)
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1)
            ),  # (B, 32, 8, 8)
            nn.ReLU(),
            layer_init(
                nn.ConvTranspose2d(
                    32, self.num_action_planes, 3, stride=2, padding=1, output_padding=1
                )
            ),  # (B, 78, 16, 16)
            Transpose((0, 2, 3, 1)),  # (B, 16, 16, 78)
        )

        self.critic_shaped = nn.Sequential(
            nn.Flatten(),  # (B, 256)
            layer_init(nn.Linear(256, 128)),  # (B, 128)
            nn.ReLU(),
            layer_init(nn.Linear(128, 1), std=1),  # (B, 1)
        )

        self._finalize(hidden_channels=256)
