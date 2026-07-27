from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np


class ActionType(str, Enum):
    OPEN = "open"
    FLAG = "flag"
    UNFLAG = "unflag"
    CHORD = "chord"


@dataclass(frozen=True)
class Action:
    kind: ActionType
    row: int
    col: int


@dataclass(frozen=True)
class GameConfig:
    rows: int = 16
    cols: int = 30
    mines: int = 99
    safe_radius: int = 1

    @property
    def cells(self) -> int:
        return self.rows * self.cols

    @property
    def safe_cells(self) -> int:
        return self.cells - self.mines


@dataclass
class SolverSnapshot:
    hidden_mask: np.ndarray
    frontier_mask: np.ndarray
    safe_mask: np.ndarray
    mine_mask: np.ndarray
    risk_map: np.ndarray
    frontier_degree_map: np.ndarray
    best_guess: tuple[int, int] | None
    best_guess_index: int | None
    best_guess_risk: float | None

    @property
    def has_forced_moves(self) -> bool:
        return bool(self.safe_mask.any() or self.mine_mask.any())


@dataclass
class EpisodeTransition:
    board: np.ndarray
    global_features: np.ndarray
    action_mask: np.ndarray
    action_index: int
    expert_action_index: int | None
    reward: float
    done: bool
    expert_action_mask: np.ndarray | None = None
    mine_mask: np.ndarray | None = None
    risk_map: np.ndarray | None = None


@dataclass
class EpisodeSummary:
    won: bool
    lost: bool
    reward: float
    game_steps: int
    guess_steps: int
    forced_steps: int
    revealed_safe_cells: int
    flags: int
    seed: int | None

    def as_dict(self) -> dict[str, Any]:
        return {
            "won": self.won,
            "lost": self.lost,
            "reward": self.reward,
            "game_steps": self.game_steps,
            "guess_steps": self.guess_steps,
            "forced_steps": self.forced_steps,
            "revealed_safe_cells": self.revealed_safe_cells,
            "flags": self.flags,
            "seed": self.seed,
        }
