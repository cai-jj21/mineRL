from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
from PIL import Image


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "windows_minesweeper_agent.py"
SPEC = importlib.util.spec_from_file_location("windows_minesweeper_agent_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
windows_agent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = windows_agent
SPEC.loader.exec_module(windows_agent)


def test_restore_revealed_cells_keeps_prior_open_cells_visible() -> None:
    prev_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    prev_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    prev_revealed[0, 2] = True
    prev_adjacent[0, 2] = 0
    previous = windows_agent.ScreenBoard(
        revealed=prev_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=prev_adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    current = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    restored = windows_agent.restore_revealed_cells(current, previous)

    assert restored.revealed[0, 2]
    assert restored.adjacent[0, 2] == 0
    assert restored.read_restores == 1


def test_classify_cell_fast_prefers_center_over_blue_border() -> None:
    open_zero = np.full((24, 24, 3), [166, 212, 247], dtype=np.uint8)
    open_zero[6:18, 6:18] = [138, 143, 147]
    cell = windows_agent.classify_cell_fast(open_zero)

    assert cell["kind"] == "revealed"
    assert cell["number"] == 0


def test_classify_cell_fast_keeps_blue_hidden_cells_hidden() -> None:
    hidden = np.full((24, 24, 3), [166, 212, 247], dtype=np.uint8)
    cell = windows_agent.classify_cell_fast(hidden)

    assert cell["kind"] == "hidden"


def test_read_board_from_array_reuses_unchanged_cells() -> None:
    cell_w = 24
    cell_h = 24
    board_array = np.full((windows_agent.ROWS * cell_h, windows_agent.COLS * cell_w, 3), [166, 212, 247], dtype=np.uint8)
    prev_array = board_array.copy()
    board_array[0:cell_h, 2 * cell_w : 3 * cell_w] = [190, 190, 190]
    prev_board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(
            x_lines=[i * cell_w for i in range(windows_agent.COLS + 1)],
            y_lines=[i * cell_h for i in range(windows_agent.ROWS + 1)],
        ),
        screenshot=Image.fromarray(prev_array, mode="RGB"),
    )

    board = windows_agent.WindowsMinesweeper._read_board_from_array(
        object(),
        board_array,
        prev_board.grid,
        Image.fromarray(board_array, mode="RGB"),
        0,
        prev_board,
    )

    assert board.revealed[0, 2]
    assert board.adjacent[0, 2] == 0
    assert not board.revealed[0, 3]
