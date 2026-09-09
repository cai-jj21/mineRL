from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm import tqdm

from minesweeper_rl.features import (
    COORDINATE_CHANNEL_START,
    action_channel,
    action_to_index,
    coordinate_channels,
    decode_action_index,
    encode_state,
)
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.model import MinesweeperNet
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.types import Action, ActionType, EpisodeSummary, EpisodeTransition, GameConfig, SolverSnapshot


MODEL_MODES = {"rl", "policy", "hybrid"}
VALID_MODES = ("rl", "policy", "solver", "hybrid")
DECISION_ACTION_MODES = ("open", "full")
INFERENCE_ENSEMBLES = ("logits", "probs")


@dataclass
class TrainingConfig:
    rows: int = 16
    cols: int = 30
    mines: int = 99
    safe_radius: int = 1
    exact_limit: int = 24
    seed: int = 0
    episodes: int = 1000
    gamma: float = 0.995
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    solver_imitation_coef: float = 0.2
    counterfactual_coef: float = 0.0
    counterfactual_margin: float = 0.1
    counterfactual_temperature: float = 0.08
    counterfactual_policy_coef: float = 0.0
    counterfactual_policy_temperature: float = 0.25
    counterfactual_policy_topk: int = 16
    counterfactual_policy_min_gap: float = 0.0
    counterfactual_policy_full_action: bool = False
    counterfactual_only_guess: bool = False
    counterfactual_value_head_coef: float = 0.0
    counterfactual_value_head_clip: float = 4.0
    counterfactual_value_rank_coef: float = 0.0
    counterfactual_value_target_mode: str = "raw"
    mine_aux_coef: float = 0.0
    risk_supervision_coef: float = 0.0
    risk_temperature: float = 0.08
    counterfactual_risk_head_coef: float = 0.0
    counterfactual_risk_temperature: float = 1.0
    counterfactual_risk_loss_mode: str = "bce"
    guess_survival_coef: float = 0.0
    guess_survival_mine_weight: float = 4.0
    guess_survival_topk: int = 32
    guess_survival_margin: float = 0.05
    behavior_mine_demotion_coef: float = 0.0
    behavior_mine_demotion_topk: int = 4
    behavior_mine_demotion_margin: float = 0.1
    grad_clip: float = 1.0
    hidden_channels: int = 64
    residual_blocks: int = 3
    global_policy_context: bool = False
    long_range_context: bool = False
    eval_every: int = 100
    eval_games: int = 50
    pretrain_episodes: int = 0
    pretrain_imitation_coef: float = 1.0
    pretrain_epochs_per_episode: int = 1
    pretrain_batch_size: int = 256
    expert_replay_size: int = 8192
    dagger_replay_size: int = 8192
    dagger_batch_size: int = 256
    dagger_updates_per_episode: int = 1
    guess_supervision_topk: int = 5
    guess_imitation_weight: float = 1.0
    exploration_temperature: float = 1.0
    exploration_topk: int = 0
    inference_augment_flips: bool = False
    inference_ensemble: str = "logits"
    augment_flips: bool = True
    decision_actions: str = "open"
    risk_weight: float = 0.0
    risk_head_coef: float = 0.0
    risk_head_weight: float = 0.0
    max_steps: int = 2000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.decision_actions not in DECISION_ACTION_MODES:
            raise ValueError(f"unknown decision_actions {self.decision_actions!r}; expected one of {DECISION_ACTION_MODES}")
        if self.exploration_temperature <= 0.0:
            raise ValueError("exploration_temperature must be positive")
        if self.exploration_topk < 0:
            raise ValueError("exploration_topk must be non-negative")
        if self.inference_ensemble not in INFERENCE_ENSEMBLES:
            raise ValueError(f"unknown inference_ensemble {self.inference_ensemble!r}; expected one of {INFERENCE_ENSEMBLES}")
        if self.counterfactual_policy_temperature <= 0.0:
            raise ValueError("counterfactual_policy_temperature must be positive")
        if self.counterfactual_policy_topk < 0:
            raise ValueError("counterfactual_policy_topk must be non-negative")
        if self.counterfactual_policy_min_gap < 0.0:
            raise ValueError("counterfactual_policy_min_gap must be non-negative")
        if self.counterfactual_value_head_coef < 0.0:
            raise ValueError("counterfactual_value_head_coef must be non-negative")
        if self.counterfactual_value_head_clip <= 0.0:
            raise ValueError("counterfactual_value_head_clip must be positive")
        if self.counterfactual_value_rank_coef < 0.0:
            raise ValueError("counterfactual_value_rank_coef must be non-negative")
        if self.counterfactual_value_target_mode not in {"raw", "minmax", "zscore", "rank"}:
            raise ValueError(
                "counterfactual_value_target_mode must be one of "
                "{'raw', 'minmax', 'zscore', 'rank'}"
            )
        if self.counterfactual_risk_loss_mode not in {"bce", "listwise"}:
            raise ValueError(
                "counterfactual_risk_loss_mode must be one of {'bce', 'listwise'}"
            )
        if self.guess_survival_coef < 0.0:
            raise ValueError("guess_survival_coef must be non-negative")
        if self.guess_survival_mine_weight <= 0.0:
            raise ValueError("guess_survival_mine_weight must be positive")
        if self.guess_survival_topk < 0:
            raise ValueError("guess_survival_topk must be non-negative")
        if self.guess_survival_margin < 0.0:
            raise ValueError("guess_survival_margin must be non-negative")
        if self.behavior_mine_demotion_coef < 0.0:
            raise ValueError("behavior_mine_demotion_coef must be non-negative")
        if self.behavior_mine_demotion_topk < 0:
            raise ValueError("behavior_mine_demotion_topk must be non-negative")
        if self.behavior_mine_demotion_margin < 0.0:
            raise ValueError("behavior_mine_demotion_margin must be non-negative")

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(rows=self.rows, cols=self.cols, mines=self.mines, safe_radius=self.safe_radius)


def make_game(config: TrainingConfig | GameConfig, seed: int | None = None) -> MinesweeperGame:
    return MinesweeperGame(
        rows=config.rows,
        cols=config.cols,
        mines=config.mines,
        safe_radius=config.safe_radius,
        seed=seed,
    )


def center_first_open(game: MinesweeperGame) -> tuple[float, bool]:
    if game.mines_placed or game.done:
        return 0.0, game.done
    row, col = game.rows // 2, game.cols // 2
    _, reward, done, _ = game.open_cell(row, col)
    return reward, done


def transition_sample_weight(transition: EpisodeTransition) -> float:
    progress = 0.0
    if transition.global_features.size > 1:
        progress = float(np.clip(transition.global_features[1], 0.0, 1.0))

    weight = 1.0 + 1.1 * progress
    if progress >= 0.85:
        weight += 0.25
    if transition.expert_is_guess:
        weight += 0.6
        if progress >= 0.75:
            weight += 0.15
    if transition.done:
        weight += 0.5
    if transition.reward < 0.0:
        weight += min(1.0, -float(transition.reward))
    if transition.risk_map is not None:
        weight += 0.25
    if _transition_behavior_mine(transition):
        weight += 1.0
    if transition.extreme_score > 0.0:
        weight += min(2.0, 0.5 * float(transition.extreme_score))
    regret = _transition_counterfactual_regret(transition)
    if regret is not None:
        weight += min(2.0, 0.6 * regret)
    rows = int(transition.board.shape[1]) if transition.board.ndim >= 3 else 0
    cols = int(transition.board.shape[2]) if transition.board.ndim >= 3 else 0
    cells = rows * cols
    if cells > 0:
        kind_index, cell_index = divmod(int(transition.action_index), cells)
        if kind_index == action_channel(ActionType.OPEN):
            row, col = divmod(cell_index, cols)
            is_edge = row in {0, rows - 1} or col in {0, cols - 1}
            is_corner = row in {0, rows - 1} and col in {0, cols - 1}
            if is_edge:
                weight += 0.25
            if is_corner:
                weight += 0.25
    return float(max(weight, 0.1))


def _transition_behavior_mine(transition: EpisodeTransition) -> bool:
    if transition.mine_mask is None or transition.action_mask.ndim != 3:
        return False
    rows = int(transition.mine_mask.shape[0])
    cols = int(transition.mine_mask.shape[1])
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != action_channel(ActionType.OPEN):
        return False
    row, col = divmod(cell_index, cols)
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    return bool(np.asarray(transition.mine_mask, dtype=bool)[row, col])


def _transition_counterfactual_regret(transition: EpisodeTransition) -> float | None:
    values = transition.counterfactual_open_values
    if values is None:
        return None
    value_array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(value_array)
    if value_array.ndim != 2 or not bool(finite.any()):
        return None

    rows, cols = value_array.shape
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != action_channel(ActionType.OPEN):
        return None
    row, col = divmod(cell_index, cols)
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    behavior_value = float(value_array[row, col])
    if not np.isfinite(behavior_value):
        return None
    return float(max(0.0, float(np.max(value_array[finite])) - behavior_value))


def _normalize_counterfactual_values(values: np.ndarray, mode: str) -> np.ndarray:
    """Normalize finite per-state labels while preserving their ordering."""

    values = np.asarray(values, dtype=np.float32).copy()
    if mode == "raw":
        return values
    if mode not in {"minmax", "zscore", "rank"}:
        raise ValueError(f"unknown counterfactual value target mode {mode!r}")

    finite = np.isfinite(values)
    if not bool(finite.any()):
        return values
    selected = values[finite]
    if mode == "minmax":
        low = float(selected.min())
        high = float(selected.max())
        scale = high - low
        values[finite] = 0.0 if scale <= 1e-6 else (selected - low) / scale
    elif mode == "zscore":
        mean_value = float(selected.mean())
        scale = float(selected.std())
        values[finite] = (selected - mean_value) / max(scale, 1e-3)
    else:
        order = np.argsort(np.argsort(selected, kind="stable"), kind="stable")
        denominator = max(1, int(selected.size) - 1)
        values[finite] = order.astype(np.float32) / float(denominator)
    return values


def transition_extreme_profile(
    transition: EpisodeTransition,
    *,
    mines: int = 99,
    safe_left_threshold: int = 100,
) -> dict[str, object]:
    """Classify replay states that deserve targeted endgame/guess training."""

    rows = int(transition.board.shape[-2]) if transition.board.ndim >= 3 else 0
    cols = int(transition.board.shape[-1]) if transition.board.ndim >= 3 else 0
    if rows <= 0 or cols <= 0:
        return {
            "extreme": False,
            "family": "invalid",
            "score": 0.0,
            "tail": False,
            "guess": False,
            "edge": False,
            "corner": False,
            "high_risk": False,
            "safe_left": None,
        }

    progress = 0.0
    if transition.global_features.size > 1:
        progress = float(np.clip(transition.global_features[1], 0.0, 1.0))
    total_safe = max(1, rows * cols - int(mines))
    safe_left = max(0, int(round((1.0 - progress) * total_safe)))
    tail = safe_left <= int(safe_left_threshold)

    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    row, col = divmod(cell_index, cols)
    is_open = kind_index == action_channel(ActionType.OPEN)
    edge = bool(is_open and (row in {0, rows - 1} or col in {0, cols - 1}))
    corner = bool(is_open and row in {0, rows - 1} and col in {0, cols - 1})
    guess = bool(is_open and transition.expert_is_guess)

    risk_value: float | None = None
    if is_open and transition.risk_map is not None:
        risk_map = np.asarray(transition.risk_map)
        if risk_map.ndim == 2 and risk_map.shape == (rows, cols):
            risk_value = float(risk_map[row, col])
    high_risk = bool(risk_value is not None and np.isfinite(risk_value) and risk_value >= 0.33)

    score = 0.0
    if tail:
        score += 1.0
    if guess:
        score += 1.25
    if edge:
        score += 0.5
    if corner:
        score += 0.75
    if high_risk:
        score += 0.75

    if corner and guess and tail:
        family = "corner_guess_tail"
    elif edge and guess and tail:
        family = "edge_guess_tail"
    elif guess and tail:
        family = "guess_tail"
    elif corner and guess:
        family = "corner_guess"
    elif edge and guess:
        family = "edge_guess"
    elif guess:
        family = "guess"
    elif corner and tail:
        family = "corner_tail"
    elif edge and tail:
        family = "edge_tail"
    elif tail:
        family = "tail"
    elif high_risk:
        family = "high_risk"
    else:
        family = "ordinary"

    return {
        "extreme": bool(score > 0.0),
        "family": family,
        "score": float(score),
        "tail": tail,
        "guess": guess,
        "edge": edge,
        "corner": corner,
        "high_risk": high_risk,
        "safe_left": safe_left,
        "risk": risk_value,
    }


def resolve_forced_moves(
    game: MinesweeperGame,
    solver: MinesweeperSolver,
    max_rounds: int = 256,
) -> tuple[SolverSnapshot, float, int]:
    total_reward = 0.0
    forced_steps = 0
    snapshot = solver.analyze(game)

    for _ in range(max_rounds):
        if game.done or not snapshot.has_forced_moves:
            return snapshot, total_reward, forced_steps

        progressed = False

        for row, col in zip(*np.where(snapshot.mine_mask)):
            if game.done:
                break
            if not game.revealed[row, col] and not game.flagged[row, col]:
                _, reward, _, _ = game.flag_cell(int(row), int(col))
                total_reward += reward
                forced_steps += 1
                progressed = True

        for row, col in zip(*np.where(snapshot.safe_mask)):
            if game.done:
                break
            if not game.revealed[row, col] and not game.flagged[row, col]:
                _, reward, _, _ = game.open_cell(int(row), int(col))
                total_reward += reward
                forced_steps += 1
                progressed = True

        if not progressed:
            return snapshot, total_reward, forced_steps
        snapshot = solver.analyze(game)

    return snapshot, total_reward, forced_steps


class MinesweeperTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self.solver = MinesweeperSolver(exact_limit=config.exact_limit)
        self.model = MinesweeperNet(
            hidden_channels=config.hidden_channels,
            residual_blocks=config.residual_blocks,
            global_policy_context=config.global_policy_context,
            long_range_context=config.long_range_context,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.rng = np.random.default_rng(config.seed)

    def collect_episode(
        self,
        seed: int | None,
        deterministic: bool = False,
        mode: str = "rl",
        risk_weight: float | None = None,
        learn_from_solver: bool = False,
    ) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        if mode == "solver":
            return self._collect_solver_episode(seed=seed)
        if mode not in MODEL_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        return self._collect_model_episode(
            seed=seed,
            deterministic=deterministic,
            mode=mode,
            risk_weight=self.config.risk_weight if risk_weight is None else risk_weight,
            learn_from_solver=learn_from_solver,
        )

    def _collect_model_episode(
        self,
        seed: int | None,
        deterministic: bool,
        mode: str,
        risk_weight: float,
        learn_from_solver: bool,
    ) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward = 0.0
        agent_steps = 0
        transitions: list[EpisodeTransition] = []

        while not game.done and agent_steps < self.config.max_steps:
            board, global_features, action_mask = encode_state(game)
            action_mask = self._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            mine_mask = game.mines.copy() if learn_from_solver and game.mines_placed else None
            risk_map = None
            snapshot = None
            if learn_from_solver and game.mines_placed:
                snapshot = self.solver.analyze(game)
            if learn_from_solver and self.config.risk_supervision_coef > 0.0 and game.mines_placed:
                risk_map = snapshot.risk_map if snapshot is not None else None
            expert_action_mask = self._solver_expert_action_mask(game, action_mask, snapshot) if learn_from_solver else None
            expert_action_index = None
            if expert_action_mask is not None and expert_action_mask.any():
                expert_action_index = int(np.flatnonzero(expert_action_mask.reshape(-1))[0])
            action_index = self._select_action(
                board=board,
                global_features=global_features,
                action_mask=action_mask,
                game=game,
                deterministic=deterministic,
                mode=mode,
                risk_weight=risk_weight,
            )
            action = decode_action_index(action_index, game.rows, game.cols)
            _, reward, _, _ = game.step(action)
            total_reward += reward
            agent_steps += 1

            transitions.append(
                EpisodeTransition(
                    board=board,
                    global_features=global_features,
                    action_mask=action_mask,
                    action_index=action_index,
                    expert_action_index=expert_action_index,
                    reward=float(reward),
                    done=bool(game.done),
                    expert_action_mask=expert_action_mask,
                    mine_mask=mine_mask,
                    risk_map=risk_map,
                    expert_is_guess=bool(snapshot is not None and not snapshot.has_forced_moves),
                )
            )

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=agent_steps,
            forced_steps=0,
        )
        return transitions, summary

    def _collect_solver_episode(self, seed: int | None) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward, done = center_first_open(game)
        forced_steps = 0
        guess_steps = 0

        if not done:
            snapshot, reward, steps = resolve_forced_moves(game, self.solver)
            total_reward += reward
            forced_steps += steps
        else:
            snapshot = self.solver.analyze(game)

        while not game.done and guess_steps + forced_steps < self.config.max_steps:
            action_mask = game.legal_open_mask().astype(bool)
            if not action_mask.any():
                break

            action_index = self._select_solver_guess(snapshot, action_mask)
            row, col = divmod(action_index, game.cols)
            _, reward, done, _ = game.open_cell(row, col)
            total_reward += reward
            guess_steps += 1

            if not done:
                snapshot, forced_reward, steps = resolve_forced_moves(game, self.solver)
                total_reward += forced_reward
                forced_steps += steps
            else:
                snapshot = self.solver.analyze(game)

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=guess_steps,
            forced_steps=forced_steps,
        )
        return [], summary

    def collect_expert_episode(self, seed: int | None) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward = 0.0
        expert_steps = 0
        transitions: list[EpisodeTransition] = []

        while not game.done and expert_steps < self.config.max_steps:
            board, global_features, action_mask = encode_state(game)
            action_mask = self._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            if not game.mines_placed:
                expert_action_mask = self._solver_expert_action_mask(game, action_mask)
                if expert_action_mask is None or not expert_action_mask.any():
                    break
                action_index = int(np.flatnonzero(expert_action_mask.reshape(-1))[0])
                reward = self._apply_expert_action(
                    game,
                    transitions,
                    board,
                    global_features,
                    action_mask,
                    action_index,
                    expert_is_guess=False,
                )
                total_reward += reward
                expert_steps += 1
                continue

            snapshot = self.solver.analyze(game)
            forced_actions: list[int] = []
            mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
            for row, col in zip(*np.where(mine_cells)):
                forced_actions.append(action_to_index(Action(ActionType.FLAG, int(row), int(col)), game.rows, game.cols))

            safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
            for row, col in zip(*np.where(safe_cells)):
                forced_actions.append(action_to_index(Action(ActionType.OPEN, int(row), int(col)), game.rows, game.cols))

            for action_index in forced_actions:
                if expert_steps >= self.config.max_steps or game.done:
                    break
                current_board, current_global_features, current_action_mask = encode_state(game)
                current_action_mask = self._decision_action_mask(current_action_mask)
                current_expert_mask = _expert_mask_from_indices(current_action_mask, forced_actions)
                if not current_expert_mask.any():
                    continue
                if not current_action_mask.reshape(-1)[action_index]:
                    action_index = int(np.flatnonzero(current_expert_mask.reshape(-1))[0])
                reward = self._apply_expert_action(
                    game,
                    transitions,
                    current_board,
                    current_global_features,
                    current_action_mask,
                    action_index,
                    current_expert_mask,
                    snapshot.risk_map,
                    expert_is_guess=False,
                )
                total_reward += reward
                expert_steps += 1

            if forced_actions:
                continue

            action_index = self._solver_expert_action_from_snapshot(game, snapshot, action_mask)
            if action_index is None:
                break
            reward = self._apply_expert_action(
                game,
                transitions,
                board,
                global_features,
                action_mask,
                action_index,
                _single_action_mask(action_mask, action_index),
                snapshot.risk_map,
                expert_is_guess=not snapshot.has_forced_moves,
            )
            total_reward += reward
            expert_steps += 1

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=expert_steps,
            forced_steps=0,
        )
        return transitions, summary

    def _apply_expert_action(
        self,
        game: MinesweeperGame,
        transitions: list[EpisodeTransition],
        board: np.ndarray,
        global_features: np.ndarray,
        action_mask: np.ndarray,
        action_index: int,
        expert_action_mask: np.ndarray | None = None,
        risk_map: np.ndarray | None = None,
        expert_is_guess: bool = False,
    ) -> float:
        mine_mask = game.mines.copy() if game.mines_placed else None
        action = decode_action_index(action_index, game.rows, game.cols)
        _, reward, _, _ = game.step(action)
        transitions.append(
            EpisodeTransition(
                board=board,
                global_features=global_features,
                action_mask=action_mask,
                action_index=action_index,
                expert_action_index=action_index,
                reward=float(reward),
                done=bool(game.done),
                expert_action_mask=expert_action_mask,
                mine_mask=mine_mask,
                risk_map=risk_map,
                expert_is_guess=expert_is_guess,
            )
        )
        return float(reward)

    def _episode_summary(
        self,
        game: MinesweeperGame,
        seed: int | None,
        total_reward: float,
        guess_steps: int,
        forced_steps: int,
    ) -> EpisodeSummary:
        return EpisodeSummary(
            won=bool(game.won),
            lost=bool(game.lost),
            reward=float(total_reward),
            game_steps=int(game.step_count),
            guess_steps=guess_steps,
            forced_steps=forced_steps,
            revealed_safe_cells=int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum()),
            flags=int(game.flagged.sum()),
            seed=seed,
        )

    def train(self, save_path: str | Path | None = None) -> dict[str, float | int | bool | None]:
        save_path = Path(save_path) if save_path else None
        best_win_rate = -1.0
        best_saved = False
        recent: list[EpisodeSummary] = []
        expert_replay: list[EpisodeTransition] = []
        dagger_replay: list[EpisodeTransition] = []

        if self.config.pretrain_episodes > 0:
            pretrain = tqdm(
                range(1, self.config.pretrain_episodes + 1),
                desc="solver pretrain",
                unit="episode",
            )
            for episode in pretrain:
                seed = int(self.rng.integers(0, 2**31 - 1))
                transitions, summary = self.collect_expert_episode(seed=seed)
                if transitions:
                    if self.config.expert_replay_size > 0:
                        expert_replay.extend(transitions)
                        overflow = len(expert_replay) - self.config.expert_replay_size
                        if overflow > 0:
                            del expert_replay[:overflow]
                        train_source = expert_replay
                    else:
                        train_source = transitions
                    losses = self._zero_losses()
                    for _ in range(max(1, self.config.pretrain_epochs_per_episode)):
                        batch = self._sample_transitions(train_source, self.config.pretrain_batch_size)
                        losses = self.update_imitation(batch)
                else:
                    losses = self._zero_losses()
                pretrain.set_postfix(
                    win_rate=f"{float(summary.won):.2f}",
                    expert_steps=summary.guess_steps,
                    replay=len(expert_replay),
                    loss=f"{losses['loss']:.3f}",
                    solver=f"{losses['solver_imitation_loss']:.3f}",
                    mine=f"{losses['mine_aux_loss']:.3f}",
                    risk=f"{losses['risk_supervision_loss']:.3f}",
                )

        progress = tqdm(range(1, self.config.episodes + 1), desc="training", unit="episode")
        for episode in progress:
            seed = int(self.rng.integers(0, 2**31 - 1))
            collect_training_labels = (
                self.config.solver_imitation_coef > 0.0
                or self.config.dagger_updates_per_episode > 0
                or self.config.mine_aux_coef > 0.0
                or self.config.risk_supervision_coef > 0.0
            )
            transitions, summary = self.collect_episode(
                seed=seed,
                deterministic=False,
                mode="rl",
                learn_from_solver=collect_training_labels,
            )
            if transitions:
                losses = self.update(transitions)
                expert_labeled = [transition for transition in transitions if transition.expert_action_index is not None]
                if expert_labeled and self.config.dagger_updates_per_episode > 0:
                    if self.config.dagger_replay_size > 0:
                        dagger_replay.extend(expert_labeled)
                        overflow = len(dagger_replay) - self.config.dagger_replay_size
                        if overflow > 0:
                            del dagger_replay[:overflow]
                        train_source = dagger_replay
                    else:
                        train_source = expert_labeled

                    dagger_losses = self._zero_losses()
                    for _ in range(self.config.dagger_updates_per_episode):
                        batch = self._sample_transitions(train_source, self.config.dagger_batch_size)
                        dagger_losses = self.update_imitation(batch)
                    losses["dagger_loss"] = dagger_losses["loss"]
            else:
                losses = self._zero_losses()
            recent.append(summary)
            recent = recent[-100:]

            progress.set_postfix(
                win_rate=f"{mean(s.won for s in recent):.2f}",
                reward=f"{summary.reward:.2f}",
                agent_steps=summary.guess_steps,
                loss=f"{losses['loss']:.3f}",
                solver=f"{losses['solver_imitation_loss']:.3f}",
                dagger=f"{losses.get('dagger_loss', 0.0):.3f}",
                mine=f"{losses['mine_aux_loss']:.3f}",
                risk=f"{losses['risk_supervision_loss']:.3f}",
            )

            if self.config.eval_every > 0 and episode % self.config.eval_every == 0:
                metrics = evaluate_policy(
                    self,
                    games=self.config.eval_games,
                    seed=self.config.seed + episode * 997,
                    mode="rl",
                )
                if metrics["win_rate"] > best_win_rate:
                    best_win_rate = metrics["win_rate"]
                    if save_path is not None:
                        save_checkpoint(save_path, self, episode=episode, metrics=metrics)
                        best_saved = True

        final_metrics = evaluate_policy(
            self,
            games=max(1, self.config.eval_games),
            seed=self.config.seed + 100000,
            mode="rl",
        )
        if save_path is not None and (not best_saved or final_metrics["win_rate"] >= best_win_rate):
            save_checkpoint(save_path, self, episode=self.config.episodes, metrics=final_metrics)
        return final_metrics

    def update(self, transitions: list[EpisodeTransition], imitation_coef: float | None = None) -> dict[str, float]:
        transitions = self._augment_transitions(transitions)
        boards = torch.tensor(np.stack([t.board for t in transitions]), dtype=torch.float32, device=self.device)
        global_features = torch.tensor(
            np.stack([t.global_features for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=self.device)
        actions = torch.tensor([t.action_index for t in transitions], dtype=torch.long, device=self.device)
        returns = self._discounted_returns([t.reward for t in transitions]).to(self.device)

        logits, values, risk_logits, counterfactual_values = self.model.forward_with_aux(
            boards,
            global_features,
        )
        flat_logits = logits.view(logits.shape[0], -1)
        flat_masks = masks.view(masks.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~flat_masks, -1e9)
        dist = Categorical(logits=self._exploration_logits(masked_logits, actions))

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        solver_imitation_loss = self._expert_policy_loss(masked_logits, transitions, allow_action_fallback=False)
        counterfactual_loss = self._counterfactual_preference_loss(masked_logits, transitions)
        counterfactual_policy_loss = self._counterfactual_policy_loss(logits, masks, transitions)
        counterfactual_value_head_loss = self._counterfactual_value_head_loss(
            counterfactual_values,
            masks,
            transitions,
        )
        loss = policy_loss + self.config.value_coef * value_loss
        loss = loss + (self.config.solver_imitation_coef if imitation_coef is None else imitation_coef) * solver_imitation_loss
        loss = loss + self.config.counterfactual_coef * counterfactual_loss
        loss = loss + self.config.counterfactual_policy_coef * counterfactual_policy_loss
        loss = loss + self.config.counterfactual_value_head_coef * counterfactual_value_head_loss
        mine_aux_loss = self._mine_auxiliary_loss(logits, masks, transitions)
        risk_supervision_loss = self._risk_supervision_loss(logits, masks, transitions)
        risk_head_loss = self._risk_head_loss(risk_logits, masks, transitions)
        counterfactual_risk_head_loss = self._counterfactual_risk_head_loss(risk_logits, masks, transitions)
        guess_survival_loss = self._guess_survival_loss(logits, masks, transitions)
        behavior_mine_demotion_loss = self._behavior_mine_demotion_loss(logits, masks, transitions)
        loss = loss + self.config.mine_aux_coef * mine_aux_loss
        loss = loss + self.config.risk_supervision_coef * risk_supervision_loss
        loss = loss + self.config.risk_head_coef * risk_head_loss
        loss = loss + self.config.counterfactual_risk_head_coef * counterfactual_risk_head_loss
        loss = loss + self.config.guess_survival_coef * guess_survival_loss
        loss = loss + self.config.behavior_mine_demotion_coef * behavior_mine_demotion_loss
        loss = loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "solver_imitation_loss": float(solver_imitation_loss.detach().cpu()),
            "counterfactual_loss": float(counterfactual_loss.detach().cpu()),
            "counterfactual_policy_loss": float(counterfactual_policy_loss.detach().cpu()),
            "counterfactual_value_head_loss": float(counterfactual_value_head_loss.detach().cpu()),
            "mine_aux_loss": float(mine_aux_loss.detach().cpu()),
            "risk_supervision_loss": float(risk_supervision_loss.detach().cpu()),
            "risk_head_loss": float(risk_head_loss.detach().cpu()),
            "counterfactual_risk_head_loss": float(counterfactual_risk_head_loss.detach().cpu()),
            "guess_survival_loss": float(guess_survival_loss.detach().cpu()),
            "behavior_mine_demotion_loss": float(behavior_mine_demotion_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def update_imitation(self, transitions: list[EpisodeTransition]) -> dict[str, float]:
        transitions = self._augment_transitions(transitions)
        boards = torch.tensor(np.stack([t.board for t in transitions]), dtype=torch.float32, device=self.device)
        global_features = torch.tensor(
            np.stack([t.global_features for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=self.device)
        logits, values, risk_logits, counterfactual_values = self.model.forward_with_aux(
            boards,
            global_features,
        )
        flat_logits = logits.view(logits.shape[0], -1)
        flat_masks = masks.view(masks.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~flat_masks, -1e9)
        imitation_loss = self._expert_policy_loss(masked_logits, transitions, allow_action_fallback=True)
        counterfactual_loss = self._counterfactual_preference_loss(masked_logits, transitions)
        counterfactual_policy_loss = self._counterfactual_policy_loss(logits, masks, transitions)
        counterfactual_value_head_loss = self._counterfactual_value_head_loss(
            counterfactual_values,
            masks,
            transitions,
        )
        mine_aux_loss = self._mine_auxiliary_loss(logits, masks, transitions)
        risk_supervision_loss = self._risk_supervision_loss(logits, masks, transitions)
        risk_head_loss = self._risk_head_loss(risk_logits, masks, transitions)
        counterfactual_risk_head_loss = self._counterfactual_risk_head_loss(risk_logits, masks, transitions)
        guess_survival_loss = self._guess_survival_loss(logits, masks, transitions)
        behavior_mine_demotion_loss = self._behavior_mine_demotion_loss(logits, masks, transitions)
        loss = imitation_loss * self.config.pretrain_imitation_coef
        loss = loss + self.config.counterfactual_coef * counterfactual_loss
        loss = loss + self.config.counterfactual_policy_coef * counterfactual_policy_loss
        loss = loss + self.config.counterfactual_value_head_coef * counterfactual_value_head_loss
        loss = loss + self.config.mine_aux_coef * mine_aux_loss
        loss = loss + self.config.risk_supervision_coef * risk_supervision_loss
        loss = loss + self.config.risk_head_coef * risk_head_loss
        loss = loss + self.config.counterfactual_risk_head_coef * counterfactual_risk_head_loss
        loss = loss + self.config.guess_survival_coef * guess_survival_loss
        loss = loss + self.config.behavior_mine_demotion_coef * behavior_mine_demotion_loss
        entropy = Categorical(logits=masked_logits).entropy().mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "solver_imitation_loss": float(imitation_loss.detach().cpu()),
            "counterfactual_loss": float(counterfactual_loss.detach().cpu()),
            "counterfactual_policy_loss": float(counterfactual_policy_loss.detach().cpu()),
            "counterfactual_value_head_loss": float(counterfactual_value_head_loss.detach().cpu()),
            "mine_aux_loss": float(mine_aux_loss.detach().cpu()),
            "risk_supervision_loss": float(risk_supervision_loss.detach().cpu()),
            "risk_head_loss": float(risk_head_loss.detach().cpu()),
            "counterfactual_risk_head_loss": float(counterfactual_risk_head_loss.detach().cpu()),
            "guess_survival_loss": float(guess_survival_loss.detach().cpu()),
            "behavior_mine_demotion_loss": float(behavior_mine_demotion_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def _expert_policy_loss(
        self,
        masked_logits: torch.Tensor,
        transitions: list[EpisodeTransition],
        allow_action_fallback: bool,
    ) -> torch.Tensor:
        if self.config.counterfactual_only_guess:
            keep = np.asarray(
                [
                    not (
                        transition.expert_is_guess
                        and transition.counterfactual_open_values is not None
                    )
                    for transition in transitions
                ],
                dtype=bool,
            )
            if not bool(keep.any()):
                return masked_logits.new_tensor(0.0)
            transitions = [
                transition
                for transition, include in zip(transitions, keep.tolist(), strict=True)
                if include
            ]
            masked_logits = masked_logits[torch.from_numpy(keep).to(self.device)]

        expert_weights = torch.tensor(
            np.stack(
                [
                    _expert_action_weights(
                        transition,
                        risk_temperature=self.config.risk_temperature,
                        allow_action_fallback=allow_action_fallback,
                    )
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        ).view(masked_logits.shape[0], -1)
        valid_rows = expert_weights.sum(dim=1) > 0.0
        if not bool(valid_rows.any()):
            return masked_logits.new_tensor(0.0)

        targets = expert_weights[valid_rows]
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        log_probs = F.log_softmax(masked_logits[valid_rows], dim=1)
        row_losses = -(targets * log_probs).sum(dim=1)
        row_weights = torch.tensor(
            [
                self.config.guess_imitation_weight if transition.expert_is_guess else 1.0
                for transition, valid in zip(transitions, valid_rows.detach().cpu().tolist())
                if valid
            ],
            dtype=torch.float32,
            device=self.device,
        )
        return (row_losses * row_weights).sum() / row_weights.sum().clamp_min(1.0)

    def _zero_losses(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "solver_imitation_loss": 0.0,
            "counterfactual_loss": 0.0,
            "counterfactual_policy_loss": 0.0,
            "counterfactual_value_head_loss": 0.0,
            "dagger_loss": 0.0,
            "mine_aux_loss": 0.0,
            "risk_supervision_loss": 0.0,
            "risk_head_loss": 0.0,
            "counterfactual_risk_head_loss": 0.0,
            "guess_survival_loss": 0.0,
            "behavior_mine_demotion_loss": 0.0,
            "entropy": 0.0,
        }

    def _counterfactual_preference_loss(
        self,
        masked_logits: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        """Prefer offline action outcomes on hard states without using them at inference."""

        if self.config.counterfactual_coef <= 0.0:
            return masked_logits.new_tensor(0.0)

        losses: list[torch.Tensor] = []
        margin = float(self.config.counterfactual_margin)
        temperature = max(float(self.config.counterfactual_temperature), 1e-6)
        for row_index, transition in enumerate(transitions):
            if transition.counterfactual_open_values is not None:
                open_channel = action_channel(ActionType.OPEN)
                open_mask = transition.action_mask[open_channel]
                values = np.asarray(transition.counterfactual_open_values, dtype=np.float32)
                valid = open_mask & np.isfinite(values)
                if bool(valid.any()):
                    cells = self.config.rows * self.config.cols
                    behavior_index = int(transition.action_index)
                    behavior_kind = behavior_index // cells
                    if behavior_kind != open_channel:
                        continue
                    behavior_cell_index = behavior_index % cells
                    flat_valid = valid.reshape(-1)
                    flat_values = values.reshape(-1)
                    if not bool(flat_valid.any()):
                        continue

                    if flat_valid[behavior_cell_index]:
                        behavior_value = float(flat_values[behavior_cell_index])
                    else:
                        behavior_value = float(np.nanmin(flat_values[flat_valid]))

                    candidate_cells = np.flatnonzero(flat_valid & (flat_values > behavior_value + margin))
                    if candidate_cells.size == 0:
                        continue
                    if candidate_cells.size > 8:
                        order = np.argsort(flat_values[candidate_cells])[::-1][:8]
                        candidate_cells = candidate_cells[order]

                    open_offset = open_channel * cells
                    candidate_indices = torch.tensor(open_offset + candidate_cells, dtype=torch.long, device=self.device)
                    candidate_scores = masked_logits[row_index][candidate_indices]
                    behavior_score = masked_logits[row_index, behavior_index]
                    weights = torch.tensor(
                        np.clip(flat_values[candidate_cells] - behavior_value, 0.0, 8.0),
                        dtype=torch.float32,
                        device=self.device,
                    )
                    if float(weights.sum()) > 0.0:
                        weights = weights / weights.sum().clamp_min(1.0)
                        losses.append((F.softplus(behavior_score - candidate_scores + margin) * weights).sum())
                    else:
                        losses.append(F.softplus(behavior_score - candidate_scores + margin).mean())
                    continue

            expert_mask = _transition_expert_mask(transition, allow_action_fallback=True)
            expert_flat = expert_mask.reshape(-1) & transition.action_mask.reshape(-1)
            behavior_index = int(transition.action_index)
            if not expert_flat.any() or not transition.action_mask.reshape(-1)[behavior_index]:
                continue
            if expert_flat[behavior_index]:
                continue

            expert_scores = masked_logits[row_index][torch.from_numpy(expert_flat).to(self.device)]
            behavior_score = masked_logits[row_index, behavior_index]
            target_score = expert_scores.max()
            losses.append(F.softplus(behavior_score - target_score + margin))

        if not losses:
            return masked_logits.new_tensor(0.0)
        return torch.stack(losses).mean()

    def _counterfactual_policy_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        """Fit open-candidate probabilities to offline outcome labels on hard states."""

        if self.config.counterfactual_policy_coef <= 0.0:
            return logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        open_logits = logits[:, open_channel, :, :].reshape(logits.shape[0], -1)
        open_masks = masks[:, open_channel, :, :].reshape(masks.shape[0], -1)
        flat_logits = logits.reshape(logits.shape[0], -1)
        flat_masks = masks.reshape(masks.shape[0], -1)
        temperature = max(float(self.config.counterfactual_policy_temperature), 1e-6)
        topk = int(self.config.counterfactual_policy_topk)
        min_gap = float(self.config.counterfactual_policy_min_gap)
        losses: list[torch.Tensor] = []
        row_weights: list[float] = []

        for row_index, transition in enumerate(transitions):
            values = transition.counterfactual_open_values
            if values is None:
                continue

            flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
            flat_open_mask = transition.action_mask[open_channel].reshape(-1)
            valid = flat_open_mask & np.isfinite(flat_values)
            if int(valid.sum()) < 2:
                continue
            valid_indices = np.flatnonzero(valid)
            valid_values = flat_values[valid_indices]
            value_gap = float(np.max(valid_values) - np.min(valid_values))
            if value_gap < min_gap:
                continue

            if topk > 0 and valid_indices.size > topk:
                order = np.argsort(valid_values)[::-1][:topk]
                valid_indices = valid_indices[order]
                valid_values = flat_values[valid_indices]

            # Always retain the behavior OPEN cell in the comparison set.
            # Otherwise a bad behavior action can be truncated away before
            # training sees the negative label it needs to move away from.
            behavior_cell = -1
            behavior_index = int(transition.action_index)
            behavior_kind = behavior_index // (self.config.rows * self.config.cols)
            if behavior_kind == open_channel:
                behavior_cell = behavior_index % (self.config.rows * self.config.cols)
                if behavior_cell in np.flatnonzero(valid) and behavior_cell not in valid_indices:
                    valid_indices = np.concatenate(
                        [valid_indices, np.asarray([behavior_cell], dtype=np.int64)]
                    )
                    valid_values = np.concatenate(
                        [valid_values, np.asarray([flat_values[behavior_cell]], dtype=np.float32)]
                    )

            indices = torch.tensor(valid_indices, dtype=torch.long, device=self.device)
            legal = open_masks[row_index, indices]
            if not bool(legal.any()):
                continue
            indices = indices[legal]
            selected_values = valid_values[legal.detach().cpu().numpy()]
            if selected_values.size < 2:
                continue

            if self.config.counterfactual_policy_full_action:
                full_log_probs = F.log_softmax(
                    flat_logits[row_index].masked_fill(~flat_masks[row_index], -1e9),
                    dim=0,
                )
                open_offset = open_channel * (self.config.rows * self.config.cols)
                student_log_probs = full_log_probs[open_offset + indices]
            else:
                student_log_probs = F.log_softmax(open_logits[row_index, indices], dim=0)
            target_logits = torch.tensor(selected_values, dtype=torch.float32, device=self.device) / temperature
            target = F.softmax(target_logits, dim=0)
            policy_loss = -(target * student_log_probs).sum()

            # Explicitly push the behavior cell below every labelled cell
            # with a materially better offline outcome.
            behavior_positions = np.flatnonzero(
                indices.detach().cpu().numpy() == behavior_cell
            )
            if behavior_positions.size:
                behavior_position = int(behavior_positions[0])
                better = selected_values > selected_values[behavior_position] + min_gap
                if bool(better.any()):
                    behavior_score = open_logits[row_index, indices[behavior_position]]
                    better_scores = open_logits[
                        row_index,
                        indices[torch.from_numpy(better).to(self.device)],
                    ]
                    policy_loss = policy_loss + F.softplus(
                        behavior_score - better_scores + min_gap
                    ).mean()

            losses.append(policy_loss)
            row_weights.append(float(transition.source_quality) * (1.0 + min(1.0, float(transition.extreme_score))))

        if not losses:
            return logits.new_tensor(0.0)
        weights = torch.tensor(row_weights, dtype=torch.float32, device=self.device).clamp_min(0.05)
        stacked = torch.stack(losses)
        return (stacked * weights).sum() / weights.sum().clamp_min(1.0)

    def _counterfactual_value_head_loss(
        self,
        counterfactual_values: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        """Regress offline OPEN outcomes into a model-only spatial value head."""

        if self.config.counterfactual_value_head_coef <= 0.0:
            return counterfactual_values.new_tensor(0.0)

        targets = np.full(
            (len(transitions), self.config.rows, self.config.cols),
            np.nan,
            dtype=np.float32,
        )
        row_weights: list[float] = []
        for index, transition in enumerate(transitions):
            if transition.counterfactual_open_values is not None:
                targets[index] = _normalize_counterfactual_values(
                    np.asarray(transition.counterfactual_open_values, dtype=np.float32),
                    self.config.counterfactual_value_target_mode,
                )
            row_weights.append(
                float(transition.source_quality)
                * (1.0 + min(1.0, float(transition.extreme_score)))
            )

        target_tensor = torch.tensor(
            np.nan_to_num(
                np.clip(
                    targets,
                    -float(self.config.counterfactual_value_head_clip),
                    float(self.config.counterfactual_value_head_clip),
                ),
                nan=0.0,
            ),
            dtype=torch.float32,
            device=self.device,
        )
        valid = torch.tensor(
            np.isfinite(targets),
            dtype=torch.bool,
            device=self.device,
        )
        valid = valid & masks[:, action_channel(ActionType.OPEN), :, :]
        if not bool(valid.any()):
            return counterfactual_values.new_tensor(0.0)

        prediction = counterfactual_values
        element_loss = F.smooth_l1_loss(prediction, target_tensor, reduction="none")
        row_mask = valid.view(valid.shape[0], -1).any(dim=1)
        weights = torch.tensor(row_weights, dtype=torch.float32, device=self.device)
        weights = weights[row_mask].clamp_min(0.05)
        row_loss = (element_loss * valid).view(valid.shape[0], -1).sum(dim=1)
        row_count = valid.view(valid.shape[0], -1).sum(dim=1).clamp_min(1)
        row_loss = row_loss / row_count
        regression_loss = (row_loss[row_mask] * weights).sum() / weights.sum().clamp_min(1.0)
        if self.config.counterfactual_value_rank_coef <= 0.0:
            return regression_loss

        # The downstream use is a candidate ranking decision, so add a
        # pairwise objective over the labelled cells instead of relying only
        # on absolute-value regression.
        ranking_losses: list[torch.Tensor] = []
        full_weights = torch.tensor(row_weights, dtype=torch.float32, device=self.device).clamp_min(0.05)
        for index in torch.nonzero(row_mask, as_tuple=False).flatten().tolist():
            cells = torch.nonzero(valid[index].view(-1), as_tuple=False).flatten()
            if cells.numel() < 2:
                continue
            target_values = target_tensor[index].view(-1)[cells]
            prediction_values = prediction[index].view(-1)[cells]
            order = torch.argsort(target_values, descending=True)
            target_values = target_values[order]
            prediction_values = prediction_values[order]
            pair_limit = min(16, int(target_values.numel()))
            top = prediction_values[:pair_limit]
            bottom = prediction_values[-pair_limit:]
            target_gap = target_values[:pair_limit].unsqueeze(1) - target_values[-pair_limit:].unsqueeze(0)
            pair_mask = target_gap > 0.0
            if not bool(pair_mask.any()):
                continue
            score_gap = top.unsqueeze(1) - bottom.unsqueeze(0)
            pair_loss = F.softplus(-score_gap)[pair_mask]
            ranking_losses.append(pair_loss.mean() * full_weights[index])
        if not ranking_losses:
            return regression_loss
        ranking_loss = torch.stack(ranking_losses).sum() / full_weights[row_mask].sum().clamp_min(1.0)
        return regression_loss + float(self.config.counterfactual_value_rank_coef) * ranking_loss

    def _mine_auxiliary_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.mine_aux_coef <= 0.0:
            return logits.new_tensor(0.0)
        # A full mine mask is useful for post-game diagnostics, but it is not
        # observable during play. Prefer the solver's visible-information risk
        # map whenever one is available so training does not memorize hidden layouts.
        mine_targets = torch.tensor(
            np.stack(
                [
                    np.zeros((self.config.rows, self.config.cols), dtype=bool)
                    if transition.mine_mask is None or transition.risk_map is not None
                    else transition.mine_mask
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        known_targets = torch.tensor(
            [transition.mine_mask is not None and transition.risk_map is None for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        ).view(-1, 1, 1)
        open_channel = action_channel(ActionType.OPEN)
        valid_cells = known_targets & masks[:, open_channel, :, :]
        if not bool(valid_cells.any()):
            return logits.new_tensor(0.0)

        safe_targets = 1.0 - mine_targets
        open_logits = logits[:, open_channel, :, :]
        return F.binary_cross_entropy_with_logits(open_logits[valid_cells], safe_targets[valid_cells])

    def _risk_supervision_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.risk_supervision_coef <= 0.0:
            return logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        risk_maps = torch.tensor(
            np.stack(
                [
                    np.zeros((self.config.rows, self.config.cols), dtype=np.float32)
                    if transition.risk_map is None
                    else transition.risk_map.astype(np.float32)
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        risk_valid = torch.tensor(
            [transition.risk_map is not None for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        open_masks = masks[:, open_channel, :, :].view(masks.shape[0], -1)
        guess_mask = risk_valid & open_masks.any(dim=1)
        if not bool(guess_mask.any()):
            return logits.new_tensor(0.0)

        open_logits = logits[:, open_channel, :, :].view(logits.shape[0], -1)
        selected_logits = open_logits[guess_mask].masked_fill(~open_masks[guess_mask], -1e9)
        selected_risk = risk_maps[guess_mask].view(risk_maps[guess_mask].shape[0], -1).masked_fill(
            ~open_masks[guess_mask],
            float("inf"),
        )
        target = torch.softmax(-selected_risk / max(self.config.risk_temperature, 1e-6), dim=1)
        log_probs = F.log_softmax(selected_logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()

    def _risk_head_loss(
        self,
        risk_logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.risk_head_coef <= 0.0:
            return risk_logits.new_tensor(0.0)

        risk_targets = torch.tensor(
            np.stack(
                [
                    np.zeros((self.config.rows, self.config.cols), dtype=np.float32)
                    if transition.mine_mask is None and transition.risk_map is None
                    else (
                        transition.risk_map.astype(np.float32)
                        if transition.risk_map is not None
                        else transition.mine_mask.astype(np.float32)
                    )
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        target_valid = torch.tensor(
            [transition.risk_map is not None or transition.mine_mask is not None for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        ).view(-1, 1, 1)
        open_mask = masks[:, action_channel(ActionType.OPEN), :, :]
        valid = target_valid & open_mask
        if not bool(valid.any()):
            return risk_logits.new_tensor(0.0)

        return F.binary_cross_entropy_with_logits(risk_logits[:, 0][valid], risk_targets[valid])

    def _counterfactual_risk_head_loss(
        self,
        risk_logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.counterfactual_risk_head_coef <= 0.0:
            return risk_logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        target_rows: list[np.ndarray] = []
        valid_rows: list[np.ndarray] = []
        for transition in transitions:
            values = transition.counterfactual_open_values
            if values is None:
                target_rows.append(np.zeros((self.config.rows, self.config.cols), dtype=np.float32))
                valid_rows.append(np.zeros((self.config.rows, self.config.cols), dtype=bool))
                continue

            values = np.asarray(values, dtype=np.float32)
            open_mask = transition.action_mask[open_channel] & np.isfinite(values)
            if not bool(open_mask.any()):
                target_rows.append(np.zeros((self.config.rows, self.config.cols), dtype=np.float32))
                valid_rows.append(open_mask)
                continue

            finite_values = values[open_mask]
            best_value = float(np.max(finite_values))
            temperature = max(float(self.config.counterfactual_risk_temperature), 1e-6)
            # Better counterfactual outcomes should look safer to the risk head.
            normalized_badness = np.clip((best_value - values) / temperature, 0.0, 12.0)
            targets = 1.0 - np.exp(-normalized_badness)
            target_rows.append(targets.astype(np.float32))
            valid_rows.append(open_mask)

        valid = torch.tensor(np.stack(valid_rows), dtype=torch.bool, device=self.device) & masks[:, open_channel, :, :]
        if not bool(valid.any()):
            return risk_logits.new_tensor(0.0)
        targets = torch.tensor(np.stack(target_rows), dtype=torch.float32, device=self.device)
        if self.config.counterfactual_risk_loss_mode == "bce":
            return F.binary_cross_entropy_with_logits(risk_logits[:, 0][valid], targets[valid])

        # Rank candidates within each state so large ordinary safe regions do
        # not drown out the few decisions that determine whether a guess wins.
        temperature = max(float(self.config.counterfactual_risk_temperature), 1e-6)
        row_losses: list[torch.Tensor] = []
        for row_index, transition in enumerate(transitions):
            row_valid = valid[row_index].view(-1)
            if int(row_valid.sum()) < 2 or transition.counterfactual_open_values is None:
                continue
            values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
            valid_indices = np.flatnonzero(row_valid.detach().cpu().numpy())
            row_values = torch.tensor(values[valid_indices], dtype=risk_logits.dtype, device=self.device)
            row_risk = risk_logits[row_index, 0].reshape(-1)[row_valid]
            target = F.softmax((row_values - row_values.max()) / temperature, dim=0)
            log_quality = F.log_softmax(-row_risk / temperature, dim=0)
            row_losses.append(-(target * log_quality).sum())
        if not row_losses:
            return risk_logits.new_tensor(0.0)
        return torch.stack(row_losses).mean()

    def _guess_survival_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.guess_survival_coef <= 0.0:
            return logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        row_losses: list[torch.Tensor] = []
        row_weights: list[float] = []
        for row_index, transition in enumerate(transitions):
            if transition.mine_mask is None or not transition.expert_is_guess:
                continue

            mine_mask = np.asarray(transition.mine_mask, dtype=bool)
            if mine_mask.shape != (self.config.rows, self.config.cols):
                continue
            open_mask = np.asarray(transition.action_mask[open_channel], dtype=bool)
            if open_mask.shape != mine_mask.shape:
                continue

            safe_mask = torch.tensor(open_mask & ~mine_mask, dtype=torch.bool, device=self.device)
            mine_mask_t = torch.tensor(open_mask & mine_mask, dtype=torch.bool, device=self.device)
            if not bool(safe_mask.any()) and not bool(mine_mask_t.any()):
                continue

            open_logits = logits[row_index, open_channel, :, :]
            row_loss = open_logits.new_tensor(0.0)
            if bool(safe_mask.any()) and bool(mine_mask_t.any()):
                safe_logits = open_logits[safe_mask]
                mine_logits = open_logits[mine_mask_t]
                topk = int(self.config.guess_survival_topk)
                if topk > 0:
                    safe_logits = torch.topk(safe_logits, k=min(topk, int(safe_logits.numel()))).values
                    mine_logits = torch.topk(mine_logits, k=min(topk, int(mine_logits.numel()))).values
                margin = float(self.config.guess_survival_margin)
                row_loss = row_loss + self.config.guess_survival_mine_weight * F.softplus(
                    mine_logits[:, None] - safe_logits[None, :] + margin
                ).mean()
                safe_target = torch.ones_like(safe_logits)
                mine_target = torch.zeros_like(mine_logits)
                row_loss = row_loss + 0.05 * F.binary_cross_entropy_with_logits(safe_logits, safe_target)
                row_loss = row_loss + 0.05 * self.config.guess_survival_mine_weight * F.binary_cross_entropy_with_logits(
                    mine_logits,
                    mine_target,
                )
            elif bool(safe_mask.any()):
                safe_target = torch.ones_like(open_logits[safe_mask])
                row_loss = row_loss + 0.1 * F.binary_cross_entropy_with_logits(
                    open_logits[safe_mask],
                    safe_target,
                )
            elif bool(mine_mask_t.any()):
                mine_target = torch.zeros_like(open_logits[mine_mask_t])
                row_loss = row_loss + 0.1 * self.config.guess_survival_mine_weight * F.binary_cross_entropy_with_logits(
                    open_logits[mine_mask_t],
                    mine_target,
                )
            non_open_mask = np.asarray(transition.action_mask[1:], dtype=bool)
            if bool(non_open_mask.any()) and bool(safe_mask.any()):
                non_open_logits = logits[row_index, 1:, :, :].reshape(-1)
                legal_non_open = torch.tensor(non_open_mask.reshape(-1), dtype=torch.bool, device=self.device)
                if bool(legal_non_open.any()):
                    safe_score = open_logits[safe_mask].amax()
                    non_open_score = non_open_logits[legal_non_open].amax()
                    row_loss = row_loss + 0.5 * F.softplus(non_open_score - safe_score)

            row_losses.append(row_loss)
            row_weights.append(
                float(transition.source_quality)
                * (1.0 + min(1.0, float(transition.extreme_score)))
            )

        if not row_losses:
            return logits.new_tensor(0.0)

        weights = torch.tensor(row_weights, dtype=logits.dtype, device=self.device).clamp_min(0.05)
        return (torch.stack(row_losses) * weights).sum() / weights.sum().clamp_min(1.0)

    def _behavior_mine_demotion_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        """Demote only the OPEN action that the policy actually took on a mine.

        This keeps the correction narrow: the hidden mine layout is used only
        offline to identify a bad behavior action, while better alternatives
        come from labelled counterfactual OPEN values for the same visible
        state.
        """

        if self.config.behavior_mine_demotion_coef <= 0.0:
            return logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        cells = self.config.rows * self.config.cols
        margin = float(self.config.behavior_mine_demotion_margin)
        topk = int(self.config.behavior_mine_demotion_topk)
        row_losses: list[torch.Tensor] = []
        row_weights: list[float] = []
        open_logits = logits[:, open_channel, :, :].reshape(logits.shape[0], -1)
        open_masks = masks[:, open_channel, :, :].reshape(masks.shape[0], -1)

        for row_index, transition in enumerate(transitions):
            if transition.counterfactual_open_values is None or not _transition_behavior_mine(transition):
                continue

            behavior_index = int(transition.action_index)
            behavior_kind, behavior_cell = divmod(behavior_index, cells)
            if behavior_kind != open_channel or not (0 <= behavior_cell < cells):
                continue
            if not bool(open_masks[row_index, behavior_cell]):
                continue

            flat_values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
            finite = np.isfinite(flat_values)
            valid = np.asarray(transition.action_mask[open_channel], dtype=bool).reshape(-1) & finite
            if not bool(valid[behavior_cell]):
                continue

            behavior_value = float(flat_values[behavior_cell])
            better = valid & (flat_values > behavior_value + margin)
            if not bool(better.any()):
                continue

            better_cells = np.flatnonzero(better)
            better_values = flat_values[better_cells]
            if topk > 0 and better_cells.size > topk:
                order = np.argsort(better_values)[::-1][:topk]
                better_cells = better_cells[order]
                better_values = better_values[order]

            better_indices = torch.tensor(better_cells, dtype=torch.long, device=self.device)
            better_scores = open_logits[row_index, better_indices]
            behavior_score = open_logits[row_index, behavior_cell]
            value_gaps = torch.tensor(
                np.clip(better_values - behavior_value, 0.0, 8.0),
                dtype=logits.dtype,
                device=self.device,
            )
            weights = value_gaps / value_gaps.sum().clamp_min(1e-6)
            row_losses.append((F.softplus(behavior_score - better_scores + margin) * weights).sum())
            row_weights.append(
                float(transition.source_quality)
                * (1.0 + min(1.0, float(transition.extreme_score)))
                * (1.0 + min(1.0, max(0.0, behavior_value * -1.0)))
            )

        if not row_losses:
            return logits.new_tensor(0.0)
        weights = torch.tensor(row_weights, dtype=logits.dtype, device=self.device).clamp_min(0.05)
        return (torch.stack(row_losses) * weights).sum() / weights.sum().clamp_min(1.0)

    def _sample_transitions(
        self,
        transitions: list[EpisodeTransition],
        batch_size: int,
    ) -> list[EpisodeTransition]:
        if batch_size <= 0 or len(transitions) <= batch_size:
            return list(transitions)
        weights = np.asarray([transition_sample_weight(transition) for transition in transitions], dtype=np.float64)
        total = float(weights.sum())
        if not np.isfinite(total) or total <= 0.0:
            indices = self.rng.choice(len(transitions), size=batch_size, replace=False)
        else:
            indices = self.rng.choice(len(transitions), size=batch_size, replace=False, p=weights / total)
        return [transitions[int(index)] for index in indices]

    def _augment_transitions(self, transitions: list[EpisodeTransition]) -> list[EpisodeTransition]:
        if not self.config.augment_flips or not transitions:
            return transitions
        augmented: list[EpisodeTransition] = []
        for transition in transitions:
            transform = int(self.rng.integers(0, 4))
            flip_vertical = bool(transform & 1)
            flip_horizontal = bool(transform & 2)
            if not flip_vertical and not flip_horizontal:
                augmented.append(transition)
                continue
            augmented.append(_flip_transition(transition, self.config.rows, self.config.cols, flip_vertical, flip_horizontal))
        return augmented

    def _decision_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        if self.config.decision_actions == "full":
            return action_mask
        mask = np.zeros_like(action_mask, dtype=bool)
        mask[action_channel(ActionType.OPEN)] = action_mask[action_channel(ActionType.OPEN)]
        return mask

    def _select_action(
        self,
        board: np.ndarray,
        global_features: np.ndarray,
        action_mask: np.ndarray,
        game: MinesweeperGame,
        deterministic: bool,
        mode: str,
        risk_weight: float,
    ) -> int:
        with torch.no_grad():
            mask_t = torch.tensor(action_mask.reshape(1, -1), dtype=torch.bool, device=self.device)
            use_flip_ensemble = deterministic and self.config.inference_augment_flips
            use_probability_ensemble = use_flip_ensemble and self.config.inference_ensemble == "probs" and mode != "hybrid"
            if use_probability_ensemble:
                masked_scores = self._predict_policy_scores_batch(
                    boards=np.stack([board]),
                    global_features_batch=np.stack([global_features]),
                    action_masks=np.stack([action_mask]),
                    use_flip_ensemble=True,
                )
            else:
                logits, risk_logits = self._predict_policy_risk_logits_batch(
                    boards=np.stack([board]),
                    global_features_batch=np.stack([global_features]),
                    use_flip_ensemble=use_flip_ensemble,
                )
                logits = self._apply_model_risk_prior(logits, risk_logits)
                if mode == "hybrid":
                    logits = self._apply_solver_risk_prior(logits, game, risk_weight)
                flat_logits = logits.view(1, -1)
                masked_scores = flat_logits.masked_fill(~mask_t, -1e9)
            if deterministic:
                return int(masked_scores.argmax(dim=1).item())
            dist = Categorical(logits=self._exploration_logits(masked_scores))
            return int(dist.sample().item())

    def _predict_policy_scores_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        action_masks: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        flat_masks = torch.tensor(action_masks.reshape(boards.shape[0], -1), dtype=torch.bool, device=self.device)
        if use_flip_ensemble and self.config.inference_ensemble == "probs":
            policy_parts, risk_parts = self._predict_flip_policy_risk_parts(
                boards=boards,
                global_features_batch=global_features_batch,
            )
            probabilities = []
            for policy_part, risk_part in zip(policy_parts, risk_parts):
                adjusted = self._apply_model_risk_prior(policy_part, risk_part)
                probabilities.append(
                    F.softmax(adjusted.view(boards.shape[0], -1).masked_fill(~flat_masks, -1e9), dim=1)
                )
            return torch.stack(probabilities, dim=0).mean(dim=0).masked_fill(~flat_masks, -1.0)

        logits, risk_logits = self._predict_policy_risk_logits_batch(
            boards=boards,
            global_features_batch=global_features_batch,
            use_flip_ensemble=use_flip_ensemble,
        )
        adjusted = self._apply_model_risk_prior(logits, risk_logits)
        return adjusted.view(boards.shape[0], -1).masked_fill(~flat_masks, -1e9)

    def _predict_counterfactual_values_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        if not use_flip_ensemble:
            board_t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            global_t = torch.tensor(global_features_batch, dtype=torch.float32, device=self.device)
            _policy, _value, _risk, counterfactual = self.model.forward_with_aux(board_t, global_t)
            return counterfactual

        transforms = [(False, False), (True, False), (False, True), (True, True)]
        transformed_boards = np.concatenate(
            [
                np.stack(
                    [
                        _flip_board(board, self.config.rows, self.config.cols, flip_vertical, flip_horizontal)
                        for board in boards
                    ]
                )
                for flip_vertical, flip_horizontal in transforms
            ],
            axis=0,
        )
        transformed_globals = np.concatenate([global_features_batch for _ in transforms], axis=0)
        board_t = torch.tensor(transformed_boards, dtype=torch.float32, device=self.device)
        global_t = torch.tensor(transformed_globals, dtype=torch.float32, device=self.device)
        _policy, _value, _risk, counterfactual = self.model.forward_with_aux(board_t, global_t)
        values_by_transform = counterfactual.view(
            len(transforms),
            boards.shape[0],
            counterfactual.shape[-2],
            counterfactual.shape[-1],
        )
        values = [
            _unflip_policy_logits(values_by_transform[index], flip_vertical, flip_horizontal)
            for index, (flip_vertical, flip_horizontal) in enumerate(transforms)
        ]
        return torch.stack(values, dim=0).mean(dim=0)

    def _predict_policy_risk_logits_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        use_flip_ensemble: bool,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if not use_flip_ensemble:
            board_t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            global_t = torch.tensor(global_features_batch, dtype=torch.float32, device=self.device)
            policy_logits, _, risk_logits = self.model.forward_with_risk(board_t, global_t)
            return policy_logits, risk_logits

        policy_parts, risk_parts = self._predict_flip_policy_risk_parts(
            boards=boards,
            global_features_batch=global_features_batch,
        )
        return torch.stack(policy_parts, dim=0).mean(dim=0), torch.stack(risk_parts, dim=0).mean(dim=0)

    def _predict_flip_policy_risk_parts(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
    ) -> tuple[list[torch.Tensor], list[torch.Tensor]]:
        transforms = [(False, False), (True, False), (False, True), (True, True)]
        transformed_boards = np.concatenate(
            [
                np.stack(
                    [
                        _flip_board(board, self.config.rows, self.config.cols, flip_vertical, flip_horizontal)
                        for board in boards
                    ]
                )
                for flip_vertical, flip_horizontal in transforms
            ],
            axis=0,
        )
        transformed_globals = np.concatenate([global_features_batch for _ in transforms], axis=0)
        board_t = torch.tensor(transformed_boards, dtype=torch.float32, device=self.device)
        global_t = torch.tensor(transformed_globals, dtype=torch.float32, device=self.device)
        policy_logits, _, risk_logits = self.model.forward_with_risk(board_t, global_t)
        policy_by_transform = policy_logits.view(
            len(transforms),
            boards.shape[0],
            policy_logits.shape[1],
            policy_logits.shape[2],
            policy_logits.shape[3],
        )
        risk_by_transform = risk_logits.view(
            len(transforms),
            boards.shape[0],
            risk_logits.shape[1],
            risk_logits.shape[2],
            risk_logits.shape[3],
        )
        policy_parts = [
            _unflip_policy_logits(policy_by_transform[index], flip_vertical, flip_horizontal)
            for index, (flip_vertical, flip_horizontal) in enumerate(transforms)
        ]
        risk_parts = [
            _unflip_policy_logits(risk_by_transform[index], flip_vertical, flip_horizontal)
            for index, (flip_vertical, flip_horizontal) in enumerate(transforms)
        ]
        return policy_parts, risk_parts

    def _predict_flip_policy_logits_parts(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
    ) -> list[torch.Tensor]:
        policy_parts, _ = self._predict_flip_policy_risk_parts(
            boards=boards,
            global_features_batch=global_features_batch,
        )
        return policy_parts

    def _predict_policy_logits(
        self,
        board: np.ndarray,
        global_features: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        return self._predict_policy_logits_batch(
            boards=np.stack([board]),
            global_features_batch=np.stack([global_features]),
            use_flip_ensemble=use_flip_ensemble,
        )

    def _predict_policy_logits_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        if not use_flip_ensemble:
            board_t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            global_t = torch.tensor(global_features_batch, dtype=torch.float32, device=self.device)
            logits, _ = self.model(board_t, global_t)
            return logits

        transforms = [(False, False), (True, False), (False, True), (True, True)]
        logits_by_transform = self._predict_flip_policy_logits_parts(boards=boards, global_features_batch=global_features_batch)
        unflipped = [
            logits_by_transform[index]
            for index, _ in enumerate(transforms)
        ]
        return torch.stack(unflipped, dim=0).mean(dim=0)

    def _apply_model_risk_prior(
        self,
        logits: torch.Tensor,
        risk_logits: torch.Tensor,
    ) -> torch.Tensor:
        if self.config.risk_head_weight == 0.0:
            return logits

        risk = torch.sigmoid(risk_logits[:, 0, :, :])
        adjusted = logits.clone()
        adjusted[:, action_channel(ActionType.OPEN), :, :] -= self.config.risk_head_weight * risk
        adjusted[:, action_channel(ActionType.FLAG), :, :] += self.config.risk_head_weight * risk
        return adjusted

    def _exploration_logits(
        self,
        masked_logits: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = masked_logits
        topk = int(self.config.exploration_topk)
        if topk > 0 and topk < logits.shape[1]:
            _, top_indices = torch.topk(logits, k=topk, dim=1)
            top_mask = torch.zeros_like(logits, dtype=torch.bool)
            top_mask.scatter_(1, top_indices, True)
            if actions is not None:
                top_mask.scatter_(1, actions.view(-1, 1), True)
            logits = logits.masked_fill(~top_mask, -1e9)

        temperature = max(float(self.config.exploration_temperature), 1e-6)
        return logits / temperature

    def _apply_solver_risk_prior(
        self,
        logits: torch.Tensor,
        game: MinesweeperGame,
        risk_weight: float,
    ) -> torch.Tensor:
        if risk_weight == 0.0 or not game.mines_placed:
            return logits
        adjusted = logits.clone()
        snapshot = self.solver.analyze(game)
        risk = torch.tensor(snapshot.risk_map, dtype=torch.float32, device=self.device)
        adjusted[:, action_channel(ActionType.OPEN), :, :] -= risk_weight * risk
        adjusted[:, action_channel(ActionType.FLAG), :, :] += risk_weight * risk
        return adjusted

    def _select_solver_guess(self, snapshot: SolverSnapshot, action_mask: np.ndarray) -> int:
        if snapshot.best_guess_index is not None and action_mask.ravel()[snapshot.best_guess_index]:
            return int(snapshot.best_guess_index)
        return self._solver_guess_from_risk(snapshot, action_mask)

    def _solver_expert_action(self, game: MinesweeperGame, action_mask: np.ndarray) -> int | None:
        if not game.mines_placed:
            row, col = game.rows // 2, game.cols // 2
            action = Action(ActionType.OPEN, row, col)
            index = action_to_index(action, game.rows, game.cols)
            return index if action_mask.reshape(-1)[index] else None

        snapshot = self.solver.analyze(game)
        expert_mask = self._solver_expert_action_mask(game, action_mask, snapshot)
        if expert_mask is None or not expert_mask.any():
            return None
        for kind in (ActionType.UNFLAG, ActionType.FLAG, ActionType.OPEN, ActionType.CHORD):
            kind_mask = expert_mask[action_channel(kind)]
            if kind_mask.any():
                if kind == ActionType.OPEN:
                    flat_index = self._select_open_index_from_snapshot(snapshot, kind_mask)
                else:
                    flat_index = int(np.flatnonzero(kind_mask.reshape(-1))[0])
                return action_channel(kind) * game.rows * game.cols + flat_index
        return None

    def _solver_expert_action_mask(
        self,
        game: MinesweeperGame,
        action_mask: np.ndarray,
        snapshot: SolverSnapshot | None = None,
    ) -> np.ndarray | None:
        if not game.mines_placed:
            row, col = game.rows // 2, game.cols // 2
            action = Action(ActionType.OPEN, row, col)
            index = action_to_index(action, game.rows, game.cols)
            return _single_action_mask(action_mask, index) if action_mask.reshape(-1)[index] else None

        if snapshot is None:
            snapshot = self.solver.analyze(game)
        expert_mask = np.zeros_like(action_mask, dtype=bool)

        wrong_flags = game.flagged & ~game.mines & action_mask[action_channel(ActionType.UNFLAG)]
        if self.config.decision_actions == "full":
            expert_mask[action_channel(ActionType.UNFLAG)] = wrong_flags

            mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
            expert_mask[action_channel(ActionType.FLAG)] = mine_cells

        safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
        expert_mask[action_channel(ActionType.OPEN)] = safe_cells

        if expert_mask.any():
            return expert_mask

        open_mask = action_mask[action_channel(ActionType.OPEN)]
        if not open_mask.any():
            return None
        guess_mask = self._topk_guess_open_mask(snapshot, open_mask)
        if guess_mask is None:
            return None
        expert_mask[action_channel(ActionType.OPEN)] = guess_mask
        return expert_mask

    def _solver_expert_action_from_snapshot(
        self,
        game: MinesweeperGame,
        snapshot: SolverSnapshot,
        action_mask: np.ndarray,
    ) -> int | None:
        mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
        for row, col in zip(*np.where(mine_cells)):
            return action_to_index(Action(ActionType.FLAG, int(row), int(col)), game.rows, game.cols)

        safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
        if safe_cells.any():
            open_index = self._select_open_index_from_snapshot(snapshot, safe_cells)
            row, col = divmod(open_index, game.cols)
            return action_to_index(Action(ActionType.OPEN, row, col), game.rows, game.cols)

        open_mask = action_mask[action_channel(ActionType.OPEN)]
        if not open_mask.any():
            return None
        open_index = self._select_open_index_from_snapshot(snapshot, open_mask)
        row, col = divmod(open_index, game.cols)
        return action_to_index(Action(ActionType.OPEN, row, col), game.rows, game.cols)

    def _select_open_index_from_snapshot(self, snapshot: SolverSnapshot, open_mask: np.ndarray) -> int:
        safe_mask = snapshot.safe_mask & open_mask
        if safe_mask.any():
            frontier = snapshot.frontier_degree_map.copy()
            frontier[~safe_mask] = -np.inf
            return int(np.argmax(frontier))

        open_flat = open_mask.reshape(-1)
        if snapshot.best_guess_index is not None and open_flat[snapshot.best_guess_index]:
            return int(snapshot.best_guess_index)
        return self._solver_guess_from_risk(snapshot, open_mask)

    def _topk_guess_open_mask(self, snapshot: SolverSnapshot, open_mask: np.ndarray) -> np.ndarray | None:
        open_flat = open_mask.reshape(-1)
        candidate_indices = np.flatnonzero(open_flat)
        if candidate_indices.size == 0:
            return None

        risk_flat = snapshot.risk_map.reshape(-1)
        topk = max(1, int(self.config.guess_supervision_topk))
        k = min(topk, candidate_indices.size)
        ranked_candidates = candidate_indices[np.argpartition(risk_flat[candidate_indices], k - 1)[:k]]
        ranked_candidates = ranked_candidates[np.argsort(risk_flat[ranked_candidates], kind="stable")]
        if snapshot.best_guess_index is not None and open_flat[snapshot.best_guess_index]:
            ranked_candidates = np.unique(np.concatenate([ranked_candidates, np.array([snapshot.best_guess_index])]))

        guess_mask = np.zeros_like(open_mask, dtype=bool)
        guess_mask.reshape(-1)[ranked_candidates] = True
        return guess_mask

    def _solver_guess_from_risk(self, snapshot: SolverSnapshot, action_mask: np.ndarray) -> int:
        risk = snapshot.risk_map.copy()
        risk[~action_mask] = np.inf
        return int(np.argmin(risk))

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)


def evaluate_policy(
    trainer: MinesweeperTrainer,
    games: int = 100,
    seed: int = 0,
    mode: str = "rl",
    risk_weight: float = 0.0,
) -> dict[str, float | int | bool | None]:
    summaries: list[EpisodeSummary] = []
    for i in range(games):
        _, summary = trainer.collect_episode(
            seed=seed + i,
            deterministic=True,
            mode=mode,
            risk_weight=risk_weight,
        )
        summaries.append(summary)

    return _metrics_from_summaries(
        summaries=summaries,
        games=games,
        seed=seed,
        mode=mode,
        decision_actions=trainer.config.decision_actions,
        inference_augment_flips=trainer.config.inference_augment_flips,
        inference_ensemble=trainer.config.inference_ensemble,
    )


def evaluate_policy_batched(
    trainer: MinesweeperTrainer,
    games: int = 100,
    seed: int = 0,
    mode: str = "rl",
    risk_weight: float = 0.0,
    batch_size: int = 64,
) -> dict[str, float | int | bool | None]:
    if mode not in {"rl", "policy"} or risk_weight != 0.0:
        return evaluate_policy(trainer, games=games, seed=seed, mode=mode, risk_weight=risk_weight)

    summaries: list[EpisodeSummary] = []
    batch_size = max(1, int(batch_size))

    for start in range(0, games, batch_size):
        current_batch = min(batch_size, games - start)
        game_batch = [make_game(trainer.config, seed=seed + start + offset) for offset in range(current_batch)]
        rewards = [0.0 for _ in range(current_batch)]
        agent_steps = [0 for _ in range(current_batch)]
        active = set(range(current_batch))

        while active:
            ready: list[int] = []
            boards: list[np.ndarray] = []
            global_features: list[np.ndarray] = []
            action_masks: list[np.ndarray] = []

            for index in list(active):
                game = game_batch[index]
                if game.done or agent_steps[index] >= trainer.config.max_steps:
                    active.remove(index)
                    continue

                board, global_feature, action_mask = encode_state(game)
                action_mask = trainer._decision_action_mask(action_mask)
                if not action_mask.any():
                    active.remove(index)
                    continue

                ready.append(index)
                boards.append(board)
                global_features.append(global_feature)
                action_masks.append(action_mask)

            if not ready:
                continue

            scores = trainer._predict_policy_scores_batch(
                boards=np.stack(boards),
                global_features_batch=np.stack(global_features),
                action_masks=np.stack(action_masks),
                use_flip_ensemble=trainer.config.inference_augment_flips,
            )
            action_indices = scores.argmax(dim=1).detach().cpu().numpy()

            for local_index, game_index in enumerate(ready):
                game = game_batch[game_index]
                action = decode_action_index(int(action_indices[local_index]), trainer.config.rows, trainer.config.cols)
                _, reward, _, _ = game.step(action)
                rewards[game_index] += float(reward)
                agent_steps[game_index] += 1
                if game.done or agent_steps[game_index] >= trainer.config.max_steps:
                    active.discard(game_index)

        for offset, game in enumerate(game_batch):
            summaries.append(
                trainer._episode_summary(
                    game=game,
                    seed=seed + start + offset,
                    total_reward=rewards[offset],
                    guess_steps=agent_steps[offset],
                    forced_steps=0,
                )
            )

    return _metrics_from_summaries(
        summaries=summaries,
        games=games,
        seed=seed,
        mode=mode,
        decision_actions=trainer.config.decision_actions,
        inference_augment_flips=trainer.config.inference_augment_flips,
        inference_ensemble=trainer.config.inference_ensemble,
    )


def _metrics_from_summaries(
    summaries: list[EpisodeSummary],
    games: int,
    seed: int,
    mode: str,
    decision_actions: str,
    inference_augment_flips: bool,
    inference_ensemble: str,
) -> dict[str, float | int | bool | None]:
    wins = [summary.won for summary in summaries]
    longest_streak = 0
    current = 0
    current_start = 0
    longest_start: int | None = None
    longest_end: int | None = None
    for index, won in enumerate(wins):
        if won:
            if current == 0:
                current_start = index
            current += 1
            if current > longest_streak:
                longest_streak = current
                longest_start = current_start
                longest_end = index
        else:
            current = 0

    avg_agent_steps = float(mean(summary.guess_steps for summary in summaries) if summaries else 0.0)
    return {
        "mode": mode,
        "decision_actions": decision_actions,
        "inference_augment_flips": inference_augment_flips,
        "inference_ensemble": inference_ensemble,
        "solver_decision": mode in {"solver", "hybrid"},
        "games": float(games),
        "win_rate": float(mean(wins) if wins else 0.0),
        "longest_streak": float(longest_streak),
        "longest_streak_start_index": longest_start,
        "longest_streak_end_index": longest_end,
        "longest_streak_start_seed": None if longest_start is None else seed + longest_start,
        "longest_streak_end_seed": None if longest_end is None else seed + longest_end,
        "avg_reward": float(mean(summary.reward for summary in summaries) if summaries else 0.0),
        "avg_agent_steps": avg_agent_steps,
        "avg_guess_steps": avg_agent_steps,
        "avg_forced_steps": float(mean(summary.forced_steps for summary in summaries) if summaries else 0.0),
        "avg_revealed_safe_cells": float(mean(summary.revealed_safe_cells for summary in summaries) if summaries else 0.0),
    }


def save_checkpoint(
    path: str | Path,
    trainer: MinesweeperTrainer,
    episode: int,
    metrics: dict[str, float | int | bool | None],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "config": asdict(trainer.config),
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    metrics_path = path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"episode": episode, "metrics": metrics}, indent=2), encoding="utf-8")


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    config_overrides: dict[str, object] | None = None,
) -> MinesweeperTrainer:
    checkpoint = torch.load(path, map_location=device)
    known_fields = {field.name for field in fields(TrainingConfig)}
    config_values = {key: value for key, value in checkpoint["config"].items() if key in known_fields}
    if config_overrides:
        config_values.update({key: value for key, value in config_overrides.items() if key in known_fields})
    config = TrainingConfig(**config_values)
    config.device = device
    trainer = MinesweeperTrainer(config)
    partial_model_load = False
    try:
        trainer.model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        try:
            partial_model_load = _load_compatible_model_state(trainer.model, checkpoint["model"])
        except RuntimeError as fallback_exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the current action-space model. "
                "Train a new checkpoint with `python -m minesweeper_rl.cli train`."
            ) from fallback_exc
    if "optimizer" in checkpoint and not partial_model_load:
        try:
            trainer.optimizer.load_state_dict(checkpoint["optimizer"])
        except (ValueError, RuntimeError):
            pass
    return trainer


def _load_compatible_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    current_state = model.state_dict()
    loaded_state: dict[str, torch.Tensor] = {}
    partial = False

    for key, current_tensor in current_state.items():
        checkpoint_tensor = state_dict.get(key)
        if checkpoint_tensor is None:
            partial = True
            continue
        if checkpoint_tensor.shape == current_tensor.shape:
            loaded_state[key] = checkpoint_tensor
            continue
        if checkpoint_tensor.ndim != current_tensor.ndim:
            raise RuntimeError(
                f"incompatible tensor shape for {key}: checkpoint {tuple(checkpoint_tensor.shape)} vs current {tuple(current_tensor.shape)}"
            )

        merged = current_tensor.clone()
        slices = tuple(slice(0, min(checkpoint_dim, current_dim)) for checkpoint_dim, current_dim in zip(checkpoint_tensor.shape, current_tensor.shape))
        merged[slices] = checkpoint_tensor[slices]
        loaded_state[key] = merged
        partial = True

    model.load_state_dict(loaded_state, strict=False)
    return partial


def _single_action_mask(action_mask: np.ndarray, action_index: int) -> np.ndarray:
    expert_mask = np.zeros_like(action_mask, dtype=bool)
    flat = expert_mask.reshape(-1)
    if action_mask.reshape(-1)[action_index]:
        flat[action_index] = True
    return expert_mask


def _expert_mask_from_indices(action_mask: np.ndarray, action_indices: list[int]) -> np.ndarray:
    expert_mask = np.zeros_like(action_mask, dtype=bool)
    flat_mask = action_mask.reshape(-1)
    flat_expert = expert_mask.reshape(-1)
    for action_index in action_indices:
        if flat_mask[action_index]:
            flat_expert[action_index] = True
    return expert_mask


def _transition_expert_mask(transition: EpisodeTransition, allow_action_fallback: bool) -> np.ndarray:
    if transition.expert_action_mask is not None and transition.expert_action_mask.any():
        return transition.expert_action_mask & transition.action_mask
    if allow_action_fallback:
        action_index = transition.action_index if transition.expert_action_index is None else transition.expert_action_index
        return _single_action_mask(transition.action_mask, action_index)
    if transition.expert_action_index is None:
        return np.zeros_like(transition.action_mask, dtype=bool)
    return _single_action_mask(transition.action_mask, transition.expert_action_index)


def _expert_action_weights(
    transition: EpisodeTransition,
    risk_temperature: float,
    allow_action_fallback: bool,
) -> np.ndarray:
    """Prefer lower-risk OPEN actions among the solver's candidate actions."""
    expert_mask = _transition_expert_mask(transition, allow_action_fallback)
    weights = expert_mask.astype(np.float32)
    if transition.risk_map is None:
        return weights

    open_channel = action_channel(ActionType.OPEN)
    open_mask = expert_mask[open_channel]
    if not bool(open_mask.any()):
        return weights

    risk = transition.risk_map.astype(np.float32, copy=False)
    finite_risk = risk[np.isfinite(risk) & open_mask]
    if finite_risk.size == 0:
        return weights
    shifted_risk = risk - float(finite_risk.min())
    temperature = max(float(risk_temperature), 1e-6)
    open_weights = np.exp(-np.clip(shifted_risk / temperature, 0.0, 80.0)).astype(np.float32)
    weights[open_channel] = open_weights * open_mask.astype(np.float32)
    return weights


def _flip_transition(
    transition: EpisodeTransition,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> EpisodeTransition:
    return EpisodeTransition(
        board=_flip_board(transition.board, rows, cols, flip_vertical, flip_horizontal),
        global_features=transition.global_features.copy(),
        action_mask=_flip_spatial(transition.action_mask, flip_vertical, flip_horizontal),
        action_index=_flip_action_index(transition.action_index, rows, cols, flip_vertical, flip_horizontal),
        expert_action_index=None
        if transition.expert_action_index is None
        else _flip_action_index(transition.expert_action_index, rows, cols, flip_vertical, flip_horizontal),
        reward=transition.reward,
        done=transition.done,
        expert_action_mask=None
        if transition.expert_action_mask is None
        else _flip_spatial(transition.expert_action_mask, flip_vertical, flip_horizontal),
        mine_mask=None
        if transition.mine_mask is None
        else _flip_spatial(transition.mine_mask, flip_vertical, flip_horizontal),
        risk_map=None
        if transition.risk_map is None
        else _flip_spatial(transition.risk_map, flip_vertical, flip_horizontal),
        counterfactual_open_values=None
        if transition.counterfactual_open_values is None
        else _flip_spatial(transition.counterfactual_open_values, flip_vertical, flip_horizontal),
        expert_is_guess=transition.expert_is_guess,
        source_quality=transition.source_quality,
        extreme_score=transition.extreme_score,
        extreme_family=transition.extreme_family,
    )


def _flip_board(
    board: np.ndarray,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> np.ndarray:
    flipped = _flip_spatial(board, flip_vertical, flip_horizontal)
    coord_layers = coordinate_channels(rows, cols)
    coord_end = COORDINATE_CHANNEL_START + len(coord_layers)
    if flipped.shape[0] < coord_end:
        return flipped

    flipped = flipped.copy()
    for offset, channel in enumerate(coord_layers):
        flipped[COORDINATE_CHANNEL_START + offset] = channel
    return flipped


def _flip_spatial(array: np.ndarray, flip_vertical: bool, flip_horizontal: bool) -> np.ndarray:
    flipped = array
    if flip_vertical:
        flipped = np.flip(flipped, axis=-2)
    if flip_horizontal:
        flipped = np.flip(flipped, axis=-1)
    return np.ascontiguousarray(flipped)


def _unflip_policy_logits(
    logits: torch.Tensor,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> torch.Tensor:
    unflipped = logits
    if flip_vertical:
        unflipped = torch.flip(unflipped, dims=(-2,))
    if flip_horizontal:
        unflipped = torch.flip(unflipped, dims=(-1,))
    return unflipped


def _flip_action_index(
    action_index: int,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> int:
    action = decode_action_index(action_index, rows, cols)
    row = rows - 1 - action.row if flip_vertical else action.row
    col = cols - 1 - action.col if flip_horizontal else action.col
    return action_to_index(Action(action.kind, row, col), rows, cols)
