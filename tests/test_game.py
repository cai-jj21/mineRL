from __future__ import annotations

from minesweeper_rl.types import Action, ActionType
from minesweeper_rl.game import MinesweeperGame


def test_first_click_is_safe_and_reveals_zero_area() -> None:
    game = MinesweeperGame(rows=16, cols=30, mines=99, safe_radius=1, seed=123)
    _, reward, done, info = game.open_cell(8, 15)

    assert info["valid"] is True
    assert info["hit_mine"] is False
    assert game.revealed[8, 15]
    assert game.adjacent[8, 15] == 0
    assert game.revealed.sum() > 1
    assert reward > 0
    assert not done


def test_flag_and_unflag_change_visible_state() -> None:
    game = MinesweeperGame(rows=3, cols=3, mines=1, safe_radius=0, seed=7)
    game.open_cell(1, 1)
    game.step(Action(ActionType.FLAG, 0, 0))
    assert game.flagged[0, 0]
    game.step(Action(ActionType.UNFLAG, 0, 0))
    assert not game.flagged[0, 0]
