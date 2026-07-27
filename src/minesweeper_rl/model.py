from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from minesweeper_rl.features import ACTION_CHANNELS, BOARD_CHANNELS, GLOBAL_FEATURES


class ResidualBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        groups = 8 if channels % 8 == 0 else 1
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(groups, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(groups, channels)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        residual = x
        x = F.silu(self.norm1(self.conv1(x)))
        x = self.norm2(self.conv2(x))
        return F.silu(x + residual)


class MinesweeperNet(nn.Module):
    """Fully convolutional actor-critic network for visible Minesweeper state."""

    def __init__(
        self,
        in_channels: int = BOARD_CHANNELS,
        global_features: int = GLOBAL_FEATURES,
        hidden_channels: int = 64,
        residual_blocks: int = 3,
    ) -> None:
        super().__init__()
        groups = 8 if hidden_channels % 8 == 0 else 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
        ]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(residual_blocks))
        self.backbone = nn.Sequential(*layers)
        self.policy_global = nn.Sequential(
            nn.Linear(global_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.policy_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, ACTION_CHANNELS, kernel_size=1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels + global_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, board: torch.Tensor, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        features = self.backbone(board)
        policy_context = self.policy_global(global_features).view(global_features.shape[0], -1, 1, 1)
        policy_logits = self.policy_head(features + policy_context)
        pooled = features.mean(dim=(2, 3))
        value_input = torch.cat([pooled, global_features], dim=1)
        value = self.value_head(value_input).squeeze(-1)
        return policy_logits, value
