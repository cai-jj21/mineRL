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
        global_policy_context: bool = False,
        long_range_context: bool = False,
    ) -> None:
        super().__init__()
        self.global_policy_context = global_policy_context
        self.long_range_context = long_range_context
        groups = 8 if hidden_channels % 8 == 0 else 1
        layers: list[nn.Module] = [
            nn.Conv2d(in_channels, hidden_channels, kernel_size=3, padding=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
        ]
        layers.extend(ResidualBlock(hidden_channels) for _ in range(residual_blocks))
        self.backbone = nn.Sequential(*layers)
        self.long_range = nn.Sequential(
            nn.Conv2d(
                hidden_channels,
                hidden_channels,
                kernel_size=(15, 31),
                padding=(7, 15),
                groups=hidden_channels,
            ),
            nn.Conv2d(hidden_channels, hidden_channels, kernel_size=1),
            nn.GroupNorm(groups, hidden_channels),
            nn.SiLU(),
        )
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
        self.risk_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )
        # The risk estimate needs board-wide state such as remaining mines and
        # covered-cell ratio, especially when no forced-safe move exists.
        self.risk_global = nn.Sequential(
            nn.Linear(global_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, hidden_channels),
        )
        self.counterfactual_head = nn.Sequential(
            nn.Conv2d(hidden_channels, hidden_channels // 2, kernel_size=1),
            nn.SiLU(),
            nn.Conv2d(hidden_channels // 2, 1, kernel_size=1),
        )
        self.value_head = nn.Sequential(
            nn.Linear(hidden_channels + global_features, hidden_channels),
            nn.SiLU(),
            nn.Linear(hidden_channels, 1),
        )

    def forward(self, board: torch.Tensor, global_features: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        policy_logits, value, _risk_logits, _counterfactual_values = self.forward_with_aux(
            board,
            global_features,
        )
        return policy_logits, value

    def forward_with_risk(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        policy_logits, value, risk_logits, _counterfactual_values = self.forward_with_aux(
            board,
            global_features,
        )
        return policy_logits, value, risk_logits

    def forward_with_aux(
        self,
        board: torch.Tensor,
        global_features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        features = self._policy_features(board)
        policy_context = self.policy_global(global_features).view(global_features.shape[0], -1, 1, 1)
        policy_features = features + policy_context
        policy_logits = self.policy_head(policy_features)
        risk_context = self.risk_global(global_features).view(global_features.shape[0], -1, 1, 1)
        risk_logits = self.risk_head(features + risk_context)
        counterfactual_values = self.counterfactual_head(policy_features).squeeze(1)
        pooled = features.mean(dim=(2, 3))
        value_input = torch.cat([pooled, global_features], dim=1)
        value = self.value_head(value_input).squeeze(-1)
        return policy_logits, value, risk_logits, counterfactual_values

    def _policy_features(self, board: torch.Tensor) -> torch.Tensor:
        features = self.backbone(board)
        if self.long_range_context:
            features = features + 0.5 * self.long_range(features)
        if not self.global_policy_context:
            return features

        mean_context = features.mean(dim=(2, 3), keepdim=True)
        max_context = features.amax(dim=(2, 3), keepdim=True)
        return features + 0.5 * mean_context + 0.25 * max_context
