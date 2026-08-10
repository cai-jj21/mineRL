from __future__ import annotations

import torch

from minesweeper_rl.features import ACTION_CHANNELS, BOARD_CHANNELS, GLOBAL_FEATURES
from minesweeper_rl.model import MinesweeperNet


def test_model_forward_shapes() -> None:
    model = MinesweeperNet()
    board = torch.zeros(2, BOARD_CHANNELS, 16, 30)
    global_features = torch.zeros(2, GLOBAL_FEATURES)
    logits, values = model(board, global_features)

    assert logits.shape == (2, ACTION_CHANNELS, 16, 30)
    assert values.shape == (2,)

    logits, values, risk_logits = model.forward_with_risk(board, global_features)
    assert logits.shape == (2, ACTION_CHANNELS, 16, 30)
    assert values.shape == (2,)
    assert risk_logits.shape == (2, 1, 16, 30)
