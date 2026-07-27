from __future__ import annotations

import numpy as np

from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.solver import Constraint, MinesweeperSolver


def test_solver_marks_single_hidden_neighbor_as_mine() -> None:
    game = MinesweeperGame(rows=1, cols=3, mines=1, safe_radius=0, seed=0)
    game.mines_placed = True
    game.mines = np.array([[False, True, False]], dtype=bool)
    game.adjacent = np.array([[1, 0, 1]], dtype=np.int8)
    game.revealed = np.array([[True, False, True]], dtype=bool)
    game.flagged = np.zeros((1, 3), dtype=bool)

    snapshot = MinesweeperSolver().analyze(game)
    assert snapshot.mine_mask[0, 1]
    assert snapshot.risk_map[0, 1] == 1.0


def test_solver_marks_hidden_neighbor_safe_after_flag() -> None:
    game = MinesweeperGame(rows=1, cols=3, mines=1, safe_radius=0, seed=0)
    game.mines_placed = True
    game.mines = np.array([[True, False, False]], dtype=bool)
    game.adjacent = np.array([[0, 1, 0]], dtype=np.int8)
    game.revealed = np.array([[False, True, False]], dtype=bool)
    game.flagged = np.array([[True, False, False]], dtype=bool)

    snapshot = MinesweeperSolver().analyze(game)
    assert snapshot.safe_mask[0, 2]
    assert snapshot.risk_map[0, 2] == 0.0


def test_area_enumeration_compresses_symmetric_cells() -> None:
    solver = MinesweeperSolver(exact_limit=1)
    enumeration = solver._enumerate_component_counts(
        cells=[0, 1, 2, 3],
        constraints=[Constraint(cells=(0, 1, 2, 3), required=2)],
    )

    assert enumeration is not None
    assert enumeration.mine_totals == {2: 6}
    for cell in (0, 1, 2, 3):
        assert enumeration.cell_mine_totals[cell] == {2: 3}
