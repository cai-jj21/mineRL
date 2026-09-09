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

    logits, values, risk_logits, counterfactual = model.forward_with_aux(board, global_features)
    assert logits.shape == (2, ACTION_CHANNELS, 16, 30)
    assert values.shape == (2,)
    assert risk_logits.shape == (2, 1, 16, 30)
    assert counterfactual.shape == (2, 16, 30)


def test_risk_head_uses_global_context() -> None:
    model = MinesweeperNet()
    board = torch.zeros(1, BOARD_CHANNELS, 16, 30)
    low_remaining = torch.zeros(1, GLOBAL_FEATURES)
    high_remaining = low_remaining.clone()
    high_remaining[:, 4] = 1.0

    with torch.no_grad():
        low_risk = model.forward_with_risk(board, low_remaining)[2]
        high_risk = model.forward_with_risk(board, high_remaining)[2]

    assert not torch.equal(low_risk, high_risk)
