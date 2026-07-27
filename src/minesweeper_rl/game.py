from __future__ import annotations

from collections import deque
from typing import Iterable

import numpy as np

from minesweeper_rl.types import Action, ActionType, GameConfig


class MinesweeperGame:
    """Classic Minesweeper board with delayed mine placement for first-click safety."""

    def __init__(
        self,
        rows: int = 16,
        cols: int = 30,
        mines: int = 99,
        safe_radius: int = 1,
        seed: int | None = None,
    ) -> None:
        if mines >= rows * cols:
            raise ValueError("mine count must be smaller than board size")
        self.config = GameConfig(rows=rows, cols=cols, mines=mines, safe_radius=safe_radius)
        self.rng = np.random.default_rng(seed)
        self.seed = seed

        self.open_reward_per_cell = 0.01
        self.flag_correct_reward = 0.0
        self.flag_wrong_penalty = 0.03
        self.invalid_action_penalty = 0.02
        self.step_penalty = 0.0005
        self.win_reward = 5.0
        self.loss_penalty = 5.0

        self.reset(seed=seed)

    @property
    def rows(self) -> int:
        return self.config.rows

    @property
    def cols(self) -> int:
        return self.config.cols

    @property
    def mine_count(self) -> int:
        return self.config.mines

    @property
    def total_cells(self) -> int:
        return self.config.cells

    @property
    def total_safe_cells(self) -> int:
        return self.config.safe_cells

    def reset(self, seed: int | None = None) -> dict[str, np.ndarray | bool | int]:
        if seed is not None:
            self.seed = seed
            self.rng = np.random.default_rng(seed)

        shape = (self.rows, self.cols)
        self.mines = np.zeros(shape, dtype=bool)
        self.adjacent = np.zeros(shape, dtype=np.int8)
        self.revealed = np.zeros(shape, dtype=bool)
        self.flagged = np.zeros(shape, dtype=bool)

        self.mines_placed = False
        self.done = False
        self.won = False
        self.lost = False
        self.step_count = 0
        return self.observation()

    def neighbors(self, row: int, col: int) -> Iterable[tuple[int, int]]:
        for dr in (-1, 0, 1):
            for dc in (-1, 0, 1):
                if dr == 0 and dc == 0:
                    continue
                nr, nc = row + dr, col + dc
                if 0 <= nr < self.rows and 0 <= nc < self.cols:
                    yield nr, nc

    def hidden_mask(self) -> np.ndarray:
        return ~(self.revealed | self.flagged)

    def legal_open_mask(self) -> np.ndarray:
        return self.hidden_mask()

    def remaining_mines_estimate(self) -> int:
        return max(0, self.mine_count - int(self.flagged.sum()))

    def observation(self) -> dict[str, np.ndarray | bool | int]:
        numbers = np.full((self.rows, self.cols), -1, dtype=np.int8)
        numbers[self.revealed] = self.adjacent[self.revealed]
        return {
            "revealed": self.revealed.copy(),
            "flagged": self.flagged.copy(),
            "numbers": numbers,
            "mines_placed": self.mines_placed,
            "done": self.done,
            "won": self.won,
            "lost": self.lost,
            "step_count": self.step_count,
            "remaining_mines_estimate": self.remaining_mines_estimate(),
        }

    def step(self, action: Action) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        if action.kind == ActionType.OPEN:
            return self.open_cell(action.row, action.col)
        if action.kind == ActionType.FLAG:
            return self.flag_cell(action.row, action.col)
        if action.kind == ActionType.UNFLAG:
            return self.unflag_cell(action.row, action.col)
        if action.kind == ActionType.CHORD:
            return self.chord_cell(action.row, action.col)
        raise ValueError(f"unsupported action: {action.kind}")

    def open_cell(self, row: int, col: int) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        if self.done:
            return self.observation(), 0.0, True, {"valid": False, "reason": "game_done"}
        if not self._in_bounds(row, col):
            return self._invalid("out_of_bounds")

        self.step_count += 1
        if not self.mines_placed:
            self._place_mines(row, col)

        if self.flagged[row, col]:
            return self._invalid("flagged_cell")
        if self.revealed[row, col]:
            return self._invalid("already_revealed")

        if self.mines[row, col]:
            self.revealed[row, col] = True
            self.done = True
            self.lost = True
            reward = -self.loss_penalty - self.step_penalty
            return self.observation(), reward, True, {"valid": True, "hit_mine": True, "new_cells": 0}

        newly_revealed = self._reveal_from([(row, col)])
        reward = self.open_reward_per_cell * len(newly_revealed) - self.step_penalty
        if self._check_win():
            reward += self.win_reward

        return self.observation(), reward, self.done, {
            "valid": True,
            "hit_mine": False,
            "new_cells": len(newly_revealed),
        }

    def flag_cell(self, row: int, col: int) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        if self.done:
            return self.observation(), 0.0, True, {"valid": False, "reason": "game_done"}
        if not self._in_bounds(row, col):
            return self._invalid("out_of_bounds")
        if not self.mines_placed:
            return self._invalid("before_first_open")
        if self.revealed[row, col]:
            return self._invalid("revealed_cell")
        if self.flagged[row, col]:
            return self._invalid("already_flagged")

        self.step_count += 1
        self.flagged[row, col] = True
        is_mine = bool(self.mines[row, col])
        reward = self.flag_correct_reward if is_mine else -self.flag_wrong_penalty

        reward -= self.step_penalty
        return self.observation(), reward, self.done, {
            "valid": True,
            "placed": True,
            "correct": is_mine,
        }

    def unflag_cell(self, row: int, col: int) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        if self.done:
            return self.observation(), 0.0, True, {"valid": False, "reason": "game_done"}
        if not self._in_bounds(row, col):
            return self._invalid("out_of_bounds")
        if not self.mines_placed:
            return self._invalid("before_first_open")
        if self.revealed[row, col]:
            return self._invalid("revealed_cell")
        if not self.flagged[row, col]:
            return self._invalid("not_flagged")

        self.step_count += 1
        self.flagged[row, col] = False
        is_mine = bool(self.mines[row, col])
        reward = -self.flag_correct_reward if is_mine else self.flag_wrong_penalty

        reward -= self.step_penalty
        return self.observation(), reward, self.done, {
            "valid": True,
            "removed": True,
            "correct": is_mine,
        }

    def chord_cell(self, row: int, col: int) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        if self.done:
            return self.observation(), 0.0, True, {"valid": False, "reason": "game_done"}
        if not self._in_bounds(row, col):
            return self._invalid("out_of_bounds")
        if not self.revealed[row, col]:
            return self._invalid("not_revealed")

        self.step_count += 1
        clue = int(self.adjacent[row, col])
        flagged_neighbors = [(r, c) for r, c in self.neighbors(row, col) if self.flagged[r, c]]
        if clue == 0 or len(flagged_neighbors) != clue:
            return self._invalid("chord_condition_not_met")

        targets = [(r, c) for r, c in self.neighbors(row, col) if not self.revealed[r, c] and not self.flagged[r, c]]
        hit_mine = any(self.mines[r, c] for r, c in targets)
        if hit_mine:
            for r, c in targets:
                if self.mines[r, c]:
                    self.revealed[r, c] = True
            self.done = True
            self.lost = True
            reward = -self.loss_penalty - self.step_penalty
            return self.observation(), reward, True, {"valid": True, "hit_mine": True, "new_cells": 0}

        newly_revealed = self._reveal_from(targets)
        reward = self.open_reward_per_cell * len(newly_revealed) - self.step_penalty
        if self._check_win():
            reward += self.win_reward
        return self.observation(), reward, self.done, {
            "valid": True,
            "hit_mine": False,
            "new_cells": len(newly_revealed),
        }

    def _invalid(self, reason: str) -> tuple[dict[str, np.ndarray | bool | int], float, bool, dict[str, object]]:
        reward = -self.invalid_action_penalty - self.step_penalty
        return self.observation(), reward, self.done, {"valid": False, "reason": reason}

    def _place_mines(self, first_row: int, first_col: int) -> None:
        allowed = np.ones((self.rows, self.cols), dtype=bool)
        radius = max(0, self.config.safe_radius)
        r0, r1 = max(0, first_row - radius), min(self.rows, first_row + radius + 1)
        c0, c1 = max(0, first_col - radius), min(self.cols, first_col + radius + 1)
        allowed[r0:r1, c0:c1] = False

        candidates = np.flatnonzero(allowed.ravel())
        if len(candidates) < self.mine_count:
            raise ValueError("safe radius leaves too few cells for mines")

        mine_indices = self.rng.choice(candidates, size=self.mine_count, replace=False)
        self.mines.fill(False)
        self.mines.ravel()[mine_indices] = True
        self._recompute_adjacent()
        self.mines_placed = True

    def _recompute_adjacent(self) -> None:
        self.adjacent.fill(0)
        for row, col in zip(*np.where(self.mines)):
            for nr, nc in self.neighbors(int(row), int(col)):
                self.adjacent[nr, nc] += 1

    def _reveal_from(self, starts: list[tuple[int, int]]) -> list[tuple[int, int]]:
        queue: deque[tuple[int, int]] = deque(starts)
        newly_revealed: list[tuple[int, int]] = []
        while queue:
            row, col = queue.popleft()
            if not self._in_bounds(row, col):
                continue
            if self.revealed[row, col] or self.flagged[row, col] or self.mines[row, col]:
                continue

            self.revealed[row, col] = True
            newly_revealed.append((row, col))
            if self.adjacent[row, col] == 0:
                for nr, nc in self.neighbors(row, col):
                    if not self.revealed[nr, nc] and not self.flagged[nr, nc] and not self.mines[nr, nc]:
                        queue.append((nr, nc))

        return newly_revealed

    def _check_win(self) -> bool:
        if self.lost:
            return False
        if int(self.revealed.sum()) >= self.total_safe_cells:
            self.done = True
            self.won = True
            return True
        return False

    def _in_bounds(self, row: int, col: int) -> bool:
        return 0 <= row < self.rows and 0 <= col < self.cols
