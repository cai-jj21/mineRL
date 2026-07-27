from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from minesweeper_rl.features import decode_action_index, encode_state
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.trainer import MinesweeperTrainer, make_game
from minesweeper_rl.types import Action, ActionType, SolverSnapshot


ReplayFrame = dict[str, Any]
ReplayTrace = dict[str, Any]


def iter_episode_frames(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str = "rl",
    risk_weight: float = 1.0,
    deterministic: bool = True,
) -> Iterator[ReplayFrame]:
    """Yield one serializable frame after every visible game action."""

    if mode == "solver":
        yield from _iter_solver_episode_frames(
            trainer=trainer,
            seed=seed,
            mode=mode,
            risk_weight=risk_weight,
        )
        return

    yield from _iter_model_episode_frames(
        trainer=trainer,
        seed=seed,
        mode=mode,
        risk_weight=risk_weight,
        deterministic=deterministic,
    )


def _iter_model_episode_frames(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str,
    risk_weight: float,
    deterministic: bool,
) -> Iterator[ReplayFrame]:
    game = make_game(trainer.config, seed=seed)
    cumulative_reward = 0.0
    agent_steps = 0
    frame_index = 0

    while not game.done and agent_steps < trainer.config.max_steps:
        board, global_features, action_mask = encode_state(game)
        if not action_mask.any():
            yield _make_frame(
                game=game,
                snapshot=None,
                frame_index=frame_index,
                event="stalled",
                action=None,
                decision={"type": "stalled", "mode": mode, "risk": None, "note": "no legal actions"},
                reward=0.0,
                cumulative_reward=cumulative_reward,
                forced_steps=0,
                guess_steps=agent_steps,
                info={"valid": False, "reason": "no_legal_actions"},
            )
            return

        action_index = trainer._select_action(
            board=board,
            global_features=global_features,
            action_mask=action_mask,
            game=game,
            deterministic=deterministic,
            mode=mode,
            risk_weight=risk_weight,
        )
        action = decode_action_index(action_index, game.rows, game.cols)
        _, reward, _, info = game.step(action)
        cumulative_reward += reward
        agent_steps += 1

        decision = _model_decision(action, action_index, mode, risk_weight, int(action_mask.sum()))
        yield _make_frame(
            game=game,
            snapshot=None,
            frame_index=frame_index,
            event=decision["type"],
            action=action,
            decision=decision,
            reward=reward,
            cumulative_reward=cumulative_reward,
            forced_steps=0,
            guess_steps=agent_steps,
            info=info,
        )
        frame_index += 1

    if not game.done:
        yield _make_frame(
            game=game,
            snapshot=None,
            frame_index=frame_index,
            event="max_steps",
            action=None,
            decision={"type": "max_steps", "mode": mode, "risk": None, "note": "episode step cap reached"},
            reward=0.0,
            cumulative_reward=cumulative_reward,
            forced_steps=0,
            guess_steps=agent_steps,
            info={"valid": False, "reason": "max_steps"},
        )


def _iter_solver_episode_frames(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str,
    risk_weight: float,
) -> Iterator[ReplayFrame]:
    from minesweeper_rl.trainer import center_first_open, resolve_forced_moves

    game = make_game(trainer.config, seed=seed)
    cumulative_reward = 0.0
    forced_steps = 0
    guess_steps = 0
    frame_index = 0

    first_action = Action(ActionType.OPEN, game.rows // 2, game.cols // 2)
    reward, _ = center_first_open(game)
    cumulative_reward += reward
    snapshot = trainer.solver.analyze(game)
    yield _make_frame(
        game=game,
        snapshot=snapshot,
        frame_index=frame_index,
        event="first_open",
        action=first_action,
        decision={
            "type": "first_open",
            "mode": mode,
            "risk": 0.0,
            "note": "center first click",
        },
        reward=reward,
        cumulative_reward=cumulative_reward,
        forced_steps=forced_steps,
        guess_steps=guess_steps,
        info={"valid": True, "hit_mine": False},
    )
    frame_index += 1

    while not game.done and guess_steps + forced_steps < trainer.config.max_steps:
        forced = _next_forced_action(game, snapshot)
        if forced is not None:
            action, decision = forced
            decision_snapshot = snapshot
            _, reward, _, info = game.step(action)
            cumulative_reward += reward
            forced_steps += 1
            snapshot = trainer.solver.analyze(game)
            yield _make_frame(
                game=game,
                snapshot=decision_snapshot,
                frame_index=frame_index,
                event=decision["type"],
                action=action,
                decision=decision,
                reward=reward,
                cumulative_reward=cumulative_reward,
                forced_steps=forced_steps,
                guess_steps=guess_steps,
                info=info,
            )
            frame_index += 1
            continue

        action_mask = game.legal_open_mask().astype(bool)
        if not action_mask.any():
            yield _make_frame(
                game=game,
                snapshot=snapshot,
                frame_index=frame_index,
                event="stalled",
                action=None,
                decision={"type": "stalled", "mode": mode, "risk": None, "note": "no legal open cells"},
                reward=0.0,
                cumulative_reward=cumulative_reward,
                forced_steps=forced_steps,
                guess_steps=guess_steps,
                info={"valid": False, "reason": "no_legal_open_cells"},
            )
            return

        action_index = trainer._select_solver_guess(snapshot, action_mask)
        row, col = divmod(action_index, game.cols)
        action = Action(ActionType.OPEN, row, col)
        decision_snapshot = snapshot
        decision = _guess_decision(snapshot, action_index, mode, risk_weight, game.cols)
        _, reward, _, info = game.open_cell(row, col)
        cumulative_reward += reward
        guess_steps += 1
        snapshot = trainer.solver.analyze(game)
        yield _make_frame(
            game=game,
            snapshot=decision_snapshot,
            frame_index=frame_index,
            event="agent_guess",
            action=action,
            decision=decision,
            reward=reward,
            cumulative_reward=cumulative_reward,
            forced_steps=forced_steps,
            guess_steps=guess_steps,
            info=info,
        )
        frame_index += 1


def run_episode_trace(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str = "rl",
    risk_weight: float = 1.0,
    deterministic: bool = True,
) -> ReplayTrace:
    frames = list(
        iter_episode_frames(
            trainer=trainer,
            seed=seed,
            mode=mode,
            risk_weight=risk_weight,
            deterministic=deterministic,
        )
    )
    return make_trace(
        trainer=trainer,
        seed=seed,
        mode=mode,
        risk_weight=risk_weight,
        frames=frames,
    )


def find_winning_trace(
    trainer: MinesweeperTrainer,
    start_seed: int,
    max_attempts: int = 500,
    mode: str = "rl",
    risk_weight: float = 1.0,
) -> ReplayTrace:
    last_trace: ReplayTrace | None = None
    for offset in range(max_attempts):
        seed = start_seed + offset
        trace = run_episode_trace(trainer, seed=seed, mode=mode, risk_weight=risk_weight)
        last_trace = trace
        if trace["summary"]["won"]:
            trace["found_by_search"] = {
                "start_seed": start_seed,
                "attempts": offset + 1,
            }
            return trace
    raise RuntimeError(
        f"no winning episode found in {max_attempts} attempts from seed {start_seed}; "
        f"last seed was {last_trace['seed'] if last_trace else None}"
    )


def make_trace(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str,
    risk_weight: float,
    frames: list[ReplayFrame],
) -> ReplayTrace:
    summary = frames[-1]["summary"] if frames else _empty_summary(seed)
    return {
        "version": 2,
        "seed": seed,
        "mode": mode,
        "risk_weight": risk_weight,
        "config": asdict(trainer.config.game_config),
        "summary": summary,
        "frames": frames,
    }


def save_trace(path: str | Path, trace: ReplayTrace) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")


def load_trace(path: str | Path) -> ReplayTrace:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _next_forced_action(
    game: MinesweeperGame,
    snapshot: SolverSnapshot,
) -> tuple[Action, dict[str, Any]] | None:
    for row, col in zip(*np.where(snapshot.mine_mask & game.hidden_mask())):
        return Action(ActionType.FLAG, int(row), int(col)), {
            "type": "forced_mine",
            "mode": "logic",
            "risk": 1.0,
            "note": "constraint says this cell is a mine",
        }

    for row, col in zip(*np.where(snapshot.safe_mask & game.hidden_mask())):
        return Action(ActionType.OPEN, int(row), int(col)), {
            "type": "forced_safe",
            "mode": "logic",
            "risk": 0.0,
            "note": "constraint says this cell is safe",
        }

    return None


def _guess_decision(
    snapshot: SolverSnapshot,
    action_index: int,
    mode: str,
    risk_weight: float,
    cols: int,
) -> dict[str, Any]:
    row, col = divmod(action_index, cols)
    return {
        "type": "agent_guess",
        "mode": mode,
        "risk_weight": risk_weight,
        "risk": float(snapshot.risk_map[row, col]),
        "best_guess": None if snapshot.best_guess is None else list(snapshot.best_guess),
        "best_guess_risk": snapshot.best_guess_risk,
    }


def _model_decision(
    action: Action,
    action_index: int,
    mode: str,
    risk_weight: float,
    legal_actions: int,
) -> dict[str, Any]:
    return {
        "type": f"agent_{action.kind.value}",
        "mode": mode,
        "risk_weight": risk_weight,
        "risk": None,
        "action_index": int(action_index),
        "legal_actions": int(legal_actions),
        "note": "model policy action",
    }


def _make_frame(
    game: MinesweeperGame,
    snapshot: SolverSnapshot | None,
    frame_index: int,
    event: str,
    action: Action | None,
    decision: dict[str, Any],
    reward: float,
    cumulative_reward: float,
    forced_steps: int,
    guess_steps: int,
    info: dict[str, Any],
) -> ReplayFrame:
    action_dict = None
    if action is not None:
        action_dict = {"kind": action.kind.value, "row": action.row, "col": action.col}

    risk_map, safe_mask, mine_mask, frontier_mask = _analysis_overlay(game, snapshot)
    return {
        "frame_index": frame_index,
        "event": event,
        "action": action_dict,
        "decision": decision,
        "reward": float(reward),
        "cumulative_reward": float(cumulative_reward),
        "info": _json_safe(info),
        "step_count": int(game.step_count),
        "summary": {
            "won": bool(game.won),
            "lost": bool(game.lost),
            "done": bool(game.done),
            "game_steps": int(game.step_count),
            "agent_steps": int(guess_steps),
            "guess_steps": int(guess_steps),
            "forced_steps": int(forced_steps),
            "revealed_safe_cells": int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum()),
            "flags": int(game.flagged.sum()),
            "reward": float(cumulative_reward),
            "seed": game.seed,
        },
        "board": {
            "revealed": _bool_matrix(game.revealed),
            "flagged": _bool_matrix(game.flagged),
            "numbers": _visible_numbers(game),
            "mines": _bool_matrix(game.mines),
            "risk": _float_matrix(risk_map),
            "safe": _bool_matrix(safe_mask),
            "mine": _bool_matrix(mine_mask),
            "frontier": _bool_matrix(frontier_mask),
        },
    }


def _analysis_overlay(
    game: MinesweeperGame,
    snapshot: SolverSnapshot | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if snapshot is not None:
        return snapshot.risk_map, snapshot.safe_mask, snapshot.mine_mask, snapshot.frontier_mask
    shape = (game.rows, game.cols)
    return (
        np.zeros(shape, dtype=np.float32),
        np.zeros(shape, dtype=bool),
        np.zeros(shape, dtype=bool),
        _visible_frontier(game),
    )


def _visible_frontier(game: MinesweeperGame) -> np.ndarray:
    frontier = np.zeros((game.rows, game.cols), dtype=bool)
    hidden = game.hidden_mask()
    for row, col in zip(*np.where(hidden)):
        frontier[row, col] = any(game.revealed[nr, nc] for nr, nc in game.neighbors(int(row), int(col)))
    return frontier


def _visible_numbers(game: MinesweeperGame) -> list[list[int]]:
    numbers = np.full((game.rows, game.cols), -1, dtype=np.int8)
    numbers[game.revealed] = game.adjacent[game.revealed]
    return numbers.astype(int).tolist()


def _bool_matrix(array: np.ndarray) -> list[list[int]]:
    return array.astype(np.int8).tolist()


def _float_matrix(array: np.ndarray) -> list[list[float]]:
    return np.round(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0), 4).astype(float).tolist()


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        return value.item()
    return value


def _empty_summary(seed: int) -> dict[str, Any]:
    return {
        "won": False,
        "lost": False,
        "done": False,
        "game_steps": 0,
        "agent_steps": 0,
        "guess_steps": 0,
        "forced_steps": 0,
        "revealed_safe_cells": 0,
        "flags": 0,
        "reward": 0.0,
        "seed": seed,
    }
