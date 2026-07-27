from __future__ import annotations

import numpy as np

from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.types import Action, ActionType


BOARD_CHANNELS = 20
COORDINATE_CHANNELS = 4
COORDINATE_CHANNEL_START = BOARD_CHANNELS - COORDINATE_CHANNELS
GLOBAL_FEATURES = 6
ACTION_KINDS: tuple[ActionType, ...] = (ActionType.OPEN, ActionType.FLAG, ActionType.UNFLAG, ActionType.CHORD)
ACTION_CHANNELS = len(ACTION_KINDS)
_ACTION_KIND_TO_CHANNEL = {kind: index for index, kind in enumerate(ACTION_KINDS)}


def encode_state(game: MinesweeperGame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Encode only visible game state plus a legal action mask."""

    hidden = game.hidden_mask().astype(np.float32)
    flagged = game.flagged.astype(np.float32)

    channels: list[np.ndarray] = [hidden, flagged]
    for clue in range(9):
        channels.append(((game.revealed) & (game.adjacent == clue)).astype(np.float32))

    flag_neighbors = _neighbor_count(game, game.flagged)
    hidden_neighbors = _neighbor_count(game, game.hidden_mask())
    revealed_neighbors = _neighbor_count(game, game.revealed)
    frontier = (game.hidden_mask() & (revealed_neighbors > 0)).astype(np.float32)
    chord_ready = _chord_ready_mask(game, flag_neighbors, hidden_neighbors).astype(np.float32)

    channels.append(frontier)
    channels.append((flag_neighbors / 8.0).astype(np.float32))
    channels.append((hidden_neighbors / 8.0).astype(np.float32))
    channels.append((revealed_neighbors / 8.0).astype(np.float32))
    channels.append(chord_ready)
    channels.extend(coordinate_channels(game.rows, game.cols))

    board = np.stack(channels, axis=0).astype(np.float32)
    if board.shape[0] != BOARD_CHANNELS:
        raise RuntimeError(f"expected {BOARD_CHANNELS} channels, got {board.shape[0]}")

    revealed_count = int(game.revealed.sum())
    hidden_count = int(game.hidden_mask().sum())
    covered_count = int((~game.revealed).sum())
    global_features = np.array(
        [
            game.step_count / max(1, game.total_cells),
            revealed_count / max(1, game.total_safe_cells),
            int(game.flagged.sum()) / max(1, game.mine_count),
            hidden_count / max(1, game.total_cells),
            game.remaining_mines_estimate() / max(1, game.mine_count),
            covered_count / max(1, game.total_cells),
        ],
        dtype=np.float32,
    )

    action_mask = legal_action_mask(game)
    return board, global_features, action_mask


def legal_action_mask(game: MinesweeperGame) -> np.ndarray:
    mask = np.zeros((ACTION_CHANNELS, game.rows, game.cols), dtype=bool)
    if game.done:
        return mask

    mask[action_channel(ActionType.OPEN)] = game.legal_open_mask()
    if game.mines_placed:
        mask[action_channel(ActionType.FLAG)] = game.hidden_mask()
        mask[action_channel(ActionType.UNFLAG)] = game.flagged
        mask[action_channel(ActionType.CHORD)] = _chord_ready_mask(game)
    return mask


def action_channel(kind: ActionType) -> int:
    return _ACTION_KIND_TO_CHANNEL[kind]


def action_to_index(action: Action, rows: int, cols: int) -> int:
    if not (0 <= action.row < rows and 0 <= action.col < cols):
        raise ValueError(f"action cell out of bounds: {action}")
    return action_channel(action.kind) * rows * cols + action.row * cols + action.col


def decode_action_index(index: int, rows: int, cols: int) -> Action:
    cells = rows * cols
    kind_index, cell_index = divmod(int(index), cells)
    if kind_index < 0 or kind_index >= ACTION_CHANNELS:
        raise ValueError(f"action index out of bounds: {index}")
    row, col = divmod(cell_index, cols)
    return Action(ACTION_KINDS[kind_index], row, col)


def _neighbor_count(game: MinesweeperGame, mask: np.ndarray) -> np.ndarray:
    padded = np.pad(mask.astype(np.float32), 1, mode="constant")
    counts = np.zeros((game.rows, game.cols), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            counts += padded[dr : dr + game.rows, dc : dc + game.cols]
    return counts


def coordinate_channels(rows_count: int, cols_count: int) -> list[np.ndarray]:
    rows = np.linspace(-1.0, 1.0, rows_count, dtype=np.float32)[:, None]
    cols = np.linspace(-1.0, 1.0, cols_count, dtype=np.float32)[None, :]
    row_grid = np.repeat(rows, cols_count, axis=1)
    col_grid = np.repeat(cols, rows_count, axis=0)

    row_edge = np.minimum(np.arange(rows_count), np.arange(rows_count)[::-1])[:, None]
    col_edge = np.minimum(np.arange(cols_count), np.arange(cols_count)[::-1])[None, :]
    edge_distance = np.minimum(row_edge, col_edge).astype(np.float32)
    edge_distance /= max(1.0, float(edge_distance.max()))

    center_row = (rows_count - 1) / 2.0
    center_col = (cols_count - 1) / 2.0
    row_scale = max(1.0, center_row)
    col_scale = max(1.0, center_col)
    center_distance = np.sqrt(
        ((np.arange(rows_count)[:, None] - center_row) / row_scale) ** 2
        + ((np.arange(cols_count)[None, :] - center_col) / col_scale) ** 2
    )
    center_prior = (1.0 - np.clip(center_distance / np.sqrt(2.0), 0.0, 1.0)).astype(np.float32)

    return [
        row_grid.astype(np.float32),
        col_grid.astype(np.float32),
        edge_distance.astype(np.float32),
        center_prior,
    ]


def _chord_ready_mask(
    game: MinesweeperGame,
    flag_neighbors: np.ndarray | None = None,
    hidden_neighbors: np.ndarray | None = None,
) -> np.ndarray:
    if flag_neighbors is None:
        flag_neighbors = _neighbor_count(game, game.flagged)
    if hidden_neighbors is None:
        hidden_neighbors = _neighbor_count(game, game.hidden_mask())
    return (
        game.revealed
        & (game.adjacent > 0)
        & (hidden_neighbors > 0)
        & (flag_neighbors.astype(np.int16) == game.adjacent.astype(np.int16))
    )
