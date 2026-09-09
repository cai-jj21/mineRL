from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from minesweeper_rl.features import action_channel, action_to_index, encode_state
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.trainer import MinesweeperTrainer
from minesweeper_rl.types import Action, ActionType, EpisodeTransition


def load_windows_replay_transitions(
    trainer: MinesweeperTrainer,
    root: Path | str,
    *,
    limit: int | None = 8192,
    recursive: bool = True,
    per_game_limit: int | None = 256,
    sort_by_quality: bool = True,
    exclude_noisy_paths: bool = True,
) -> tuple[list[EpisodeTransition], dict[str, Any]]:
    root = Path(root)
    if not root.exists():
        raise FileNotFoundError(root)

    transitions: list[EpisodeTransition] = []
    preferred_paths = _preferred_windows_replay_paths(root, recursive=recursive)
    report: dict[str, Any] = {
        "root": str(root),
        "recursive": bool(recursive),
        "limit": None if limit is None else int(limit),
        "games_seen": 0,
        "games_loaded": 0,
        "games_skipped": 0,
        "transitions_loaded": 0,
        "behavior_labeled": 0,
        "solver_labeled": 0,
        "guess_labeled": 0,
        "no_progress_skipped": 0,
        "per_game_limit": None if per_game_limit is None else int(per_game_limit),
        "sort_by_quality": bool(sort_by_quality),
        "exclude_noisy_paths": bool(exclude_noisy_paths),
        "preferred_paths": len(preferred_paths),
        "preferred_paths_used": 0,
        "profiled_directories": 0,
        "frame_history_directories": 0,
        "final_only_directories": 0,
        "final_only_directories_skipped": 0,
        "sample_files": [],
    }

    candidates = list(_iter_windows_game_paths(root, recursive=recursive))
    if exclude_noisy_paths:
        candidates = [path for path in candidates if _is_replay_path_useful(path)]
    directory_profiles = _profile_windows_replay_directories(candidates)
    report["profiled_directories"] = len(directory_profiles)
    report["frame_history_directories"] = sum(1 for profile in directory_profiles.values() if profile["has_frame_history"])
    report["final_only_directories"] = sum(1 for profile in directory_profiles.values() if not profile["has_frame_history"])
    frame_history_directories = {directory for directory, profile in directory_profiles.items() if profile["has_frame_history"]}
    if frame_history_directories:
        candidates = [path for path in candidates if _directory_key(path) in frame_history_directories]
        report["final_only_directories_skipped"] = report["final_only_directories"]
    preferred_paths = {
        path_key
        for path_key in preferred_paths
        if directory_profiles.get(_directory_key_from_key(path_key), {}).get("has_frame_history", False)
    }
    report["preferred_paths_used"] = len(preferred_paths)
    if sort_by_quality:
        candidates.sort(
            key=lambda path: _replay_path_sort_key(
                path,
                preferred_paths=preferred_paths,
                directory_profiles=directory_profiles,
            ),
            reverse=True,
        )

    for path in candidates:
        if limit is not None and len(transitions) >= limit:
            break

        report["games_seen"] += 1
        try:
            remaining = None if limit is None else limit - len(transitions)
            game_limit = None if per_game_limit is None else per_game_limit
            if remaining is not None:
                game_limit = remaining if game_limit is None else min(game_limit, remaining)
            game_transitions, game_report = _load_windows_game(path, trainer, game_limit)
        except Exception as exc:
            report["games_skipped"] += 1
            if len(report["sample_files"]) < 20:
                report["sample_files"].append({"path": str(path), "status": "skipped", "error": str(exc)})
            continue

        if not game_transitions:
            report["games_skipped"] += 1
            if len(report["sample_files"]) < 20:
                report["sample_files"].append({"path": str(path), "status": "empty", **game_report})
            continue

        transitions.extend(game_transitions)
        report["games_loaded"] += 1
        report["transitions_loaded"] += len(game_transitions)
        report["behavior_labeled"] += int(game_report.get("behavior_labeled", 0))
        report["solver_labeled"] += int(game_report.get("solver_labeled", 0))
        report["guess_labeled"] += int(game_report.get("guess_labeled", 0))
        report["no_progress_skipped"] += int(game_report.get("no_progress_skipped", 0))
        if len(report["sample_files"]) < 20:
            report["sample_files"].append({"path": str(path), "status": "loaded", **game_report})

    return transitions, report


def _iter_windows_game_paths(root: Path, *, recursive: bool) -> Iterable[Path]:
    if root.is_file():
        if root.suffix.lower() == ".json":
            yield root
        return

    seen: set[str] = set()
    for path in _preferred_windows_summary_game_paths(root, recursive=recursive):
        key = path.resolve().as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        yield path

    pattern = "**/game_*.json" if recursive else "game_*.json"
    for path in sorted(root.glob(pattern)):
        key = path.resolve().as_posix().lower()
        if key in seen:
            continue
        seen.add(key)
        yield path


def _preferred_windows_replay_paths(root: Path, *, recursive: bool) -> set[str]:
    return {path.resolve().as_posix().lower() for path in _preferred_windows_summary_game_paths(root, recursive=recursive)}


def _preferred_windows_summary_game_paths(root: Path, *, recursive: bool) -> Iterable[Path]:
    if not root.exists() or root.is_file():
        return []

    summary_pattern = "**/per_game_summary.json" if recursive else "per_game_summary.json"
    preferred: list[Path] = []
    for summary_path in sorted(root.glob(summary_pattern)):
        try:
            summary = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        selected = summary.get("selected_range") if isinstance(summary, dict) else None
        games = selected.get("games") if isinstance(selected, dict) else None
        if not isinstance(games, list) or not games:
            continue
        base_dir = summary_path.parent
        for game in games:
            if not isinstance(game, dict):
                continue
            file_name = game.get("file")
            if not isinstance(file_name, str) or not file_name:
                continue
            candidate = base_dir / file_name
            if candidate.exists():
                preferred.append(candidate)
    return preferred


def _profile_windows_replay_directories(paths: Iterable[Path]) -> dict[str, dict[str, Any]]:
    directory_paths: dict[str, list[Path]] = {}
    for path in paths:
        directory_paths.setdefault(_directory_key(path), []).append(path)

    profiles: dict[str, dict[str, Any]] = {}
    for directory_key, directory_files in directory_paths.items():
        directory_files.sort(key=lambda path: (path.stat().st_size if path.exists() else 0, path.name), reverse=True)
        has_frame_history = False
        sampled = 0
        for candidate in directory_files[:3]:
            sampled += 1
            try:
                data = json.loads(candidate.read_text(encoding="utf-8"))
            except Exception:
                continue
            frames = data.get("frames") or []
            if len(frames) >= 2:
                has_frame_history = True
                break
        profiles[directory_key] = {
            "has_frame_history": has_frame_history,
            "sampled_paths": sampled,
            "path_count": len(directory_files),
        }
    return profiles


def _replay_path_sort_key(
    path: Path,
    *,
    preferred_paths: set[str],
    directory_profiles: dict[str, dict[str, Any]],
) -> tuple[int, int, int, str]:
    score, size, neg_length, text = _replay_path_priority(path)
    directory_profile = directory_profiles.get(_directory_key(path))
    frame_bonus = 10_000 if directory_profile and directory_profile.get("has_frame_history") else 0
    preferred_bonus = 500 if _path_key(path) in preferred_paths else 0
    return (frame_bonus + preferred_bonus + score, size, neg_length, text)


def _is_replay_path_useful(path: Path) -> bool:
    text = path.as_posix().lower()
    noisy_markers = ("debug", "probe", "audit", "click_", "absolute_")
    return not any(marker in text for marker in noisy_markers)


def _replay_path_priority(path: Path) -> tuple[int, int, int, str]:
    text = path.as_posix().lower()
    try:
        size = int(path.stat().st_size)
    except OSError:
        size = 0
    if not _is_replay_path_useful(path):
        return (size, -1000, 0, text)

    score = 0
    if "pure_rl_ensemble" in text:
        score += 500
    elif "pure_rl_fast3" in text:
        score += 470
    elif "pure_rl_fast2" in text:
        score += 440
    elif "pure_rlmix20_plus_refine_fast3" in text:
        score += 430
    elif "pure_rlmix20_refine_fast3" in text:
        score += 420
    elif "pure_rl" in text:
        score += 350

    if "streak" in text:
        score += 250
    if "manual_run" in text:
        score += 180
    if "run_" in text:
        score += 120
    if "benchmark" in text:
        score += 80
    if "validation" in text or "check" in text:
        score -= 60

    return (score, size, -len(text), text)


def _replay_source_quality(path: Path) -> float:
    text = path.as_posix().lower()
    quality = 1.0

    if "pure_rl_ensemble" in text:
        quality += 0.55
    elif "pure_rl_fast3" in text:
        quality += 0.45
    elif "pure_rl_fast2" in text:
        quality += 0.4
    elif "pure_rlmix20_plus_refine_fast3" in text:
        quality += 0.35
    elif "pure_rlmix20_refine_fast3" in text:
        quality += 0.3
    elif "pure_rl" in text:
        quality += 0.22

    if "streak" in text:
        quality += 0.15
    if "manual_run" in text:
        quality += 0.08
    if "run_" in text:
        quality += 0.05
    if "benchmark" in text:
        quality += 0.05
    if any(marker in text for marker in ("debug", "probe", "audit", "click_", "absolute_")):
        quality -= 0.2
    if "screen1" in text:
        quality -= 0.1
    if "play_current" in text:
        quality -= 0.25

    return float(np.clip(quality, 0.5, 1.6))


def _path_key(path: Path) -> str:
    return path.resolve().as_posix().lower()


def _directory_key(path: Path) -> str:
    return path.parent.resolve().as_posix().lower()


def _directory_key_from_key(path_key: str) -> str:
    return str(Path(path_key).parent.resolve().as_posix().lower())


def _load_windows_game(
    path: Path,
    trainer: MinesweeperTrainer,
    transition_limit: int | None,
) -> tuple[list[EpisodeTransition], dict[str, Any]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    frames = data.get("frames") or []
    actions = data.get("actions") or []
    rows = int(trainer.config.rows)
    cols = int(trainer.config.cols)
    source_quality = _replay_source_quality(path)
    if not frames or not actions:
        return [], {
            "frames": len(frames),
            "actions": len(actions),
            "reason": "missing_frames_or_actions",
            "source_quality": source_quality,
        }
    if len(frames) < 2:
        return [], {
            "frames": len(frames),
            "actions": len(actions),
            "reason": "insufficient_frame_history",
            "source_quality": source_quality,
        }

    usable = min(len(frames), len(actions))
    transitions: list[EpisodeTransition] = []
    behavior_labeled = 0
    solver_labeled = 0
    guess_labeled = 0
    no_progress_skipped = 0

    for index in _transition_indices(usable, transition_limit):
        if transition_limit is not None and len(transitions) >= transition_limit:
            break

        before_frame = frames[index]
        after_frame = frames[index + 1] if index + 1 < len(frames) else None
        action_record = actions[index]
        transition, label_info = _frame_action_to_transition(
            trainer=trainer,
            before_frame=before_frame,
            after_frame=after_frame,
            action_record=action_record,
            rows=rows,
            cols=cols,
            source_quality=source_quality,
        )
        if transition is None:
            no_progress_skipped += int(label_info.get("no_progress_skipped", 0))
            continue

        transitions.append(transition)
        behavior_labeled += int(label_info.get("behavior_labeled", 0))
        solver_labeled += int(label_info.get("solver_labeled", 0))
        guess_labeled += int(label_info.get("guess_labeled", 0))

    transition_density = (len(transitions) / usable) if usable else 0.0
    summary = data.get("summary") or {}
    terminal = bool(summary.get("done") or summary.get("won") or summary.get("lost"))
    if usable >= 64 and not terminal and len(transitions) < max(16, usable // 12):
        return [], {
            "frames": len(frames),
            "actions": len(actions),
            "usable_pairs": usable,
            "transitions": len(transitions),
            "transition_density": transition_density,
            "behavior_labeled": behavior_labeled,
            "solver_labeled": solver_labeled,
            "guess_labeled": guess_labeled,
            "no_progress_skipped": no_progress_skipped,
            "reason": "low_transition_density",
            "source_quality": source_quality,
        }

    return transitions, {
        "frames": len(frames),
        "actions": len(actions),
        "usable_pairs": usable,
        "transitions": len(transitions),
        "transition_density": transition_density,
        "behavior_labeled": behavior_labeled,
        "solver_labeled": solver_labeled,
        "guess_labeled": guess_labeled,
        "no_progress_skipped": no_progress_skipped,
        "source_quality": source_quality,
    }


def _transition_indices(usable: int, limit: int | None) -> list[int]:
    if limit is None or usable <= limit:
        return list(range(usable))
    head = limit // 2
    tail = limit - head
    return list(range(head)) + list(range(max(head, usable - tail), usable))


def _frame_action_to_transition(
    *,
    trainer: MinesweeperTrainer,
    before_frame: dict[str, Any],
    after_frame: dict[str, Any] | None,
    action_record: dict[str, Any],
    rows: int,
    cols: int,
    source_quality: float,
) -> tuple[EpisodeTransition | None, dict[str, int]]:
    action = _parse_action(action_record.get("action"))
    if action is None:
        return None, {}
    if action.row < 0 or action.row >= rows or action.col < 0 or action.col >= cols:
        return None, {}

    game = _game_from_frame(before_frame, rows=rows, cols=cols, mines=trainer.config.mines, safe_radius=trainer.config.safe_radius)
    if (
        after_frame is not None
        and not _frame_flag(after_frame, "done")
        and not _frame_flag(after_frame, "won")
        and not _frame_flag(after_frame, "lost")
        and not _frame_board_changed(before_frame, after_frame)
    ):
        return None, {"no_progress_skipped": 1}
    board, global_features, action_mask = encode_state(game)
    action_mask = trainer._decision_action_mask(action_mask)
    if not action_mask.any():
        return None, {}

    snapshot = trainer.solver.analyze(game) if game.mines_placed else None
    expert_action_mask = _window_expert_action_mask(trainer, game, action_mask, snapshot)
    if expert_action_mask is not None and not bool(expert_action_mask.any()):
        expert_action_mask = None

    action_index = action_to_index(action, rows, cols)
    reward = _estimate_reward(before_frame, after_frame, action, action_record)
    done = bool(_frame_flag(after_frame, "done") or _frame_flag(after_frame, "won") or _frame_flag(after_frame, "lost"))

    transition = EpisodeTransition(
        board=board,
        global_features=global_features,
        action_mask=action_mask,
        action_index=action_index,
        expert_action_index=action_index,
        reward=reward,
        done=done,
        expert_action_mask=expert_action_mask,
        # Screen replays do not contain the true mine layout.  The solver's
        # forced-mine mask is a pseudo-label for the current constraints, not
        # ground truth for the auxiliary mine classifier.
        mine_mask=None,
        risk_map=None if snapshot is None else snapshot.risk_map,
        expert_is_guess=bool(snapshot is not None and not snapshot.has_forced_moves),
        source_quality=source_quality,
    )
    label_info = {
        "behavior_labeled": 1,
        "solver_labeled": int(expert_action_mask is not None),
        "guess_labeled": int(snapshot is not None and not snapshot.has_forced_moves),
    }
    return transition, label_info


def _window_expert_action_mask(
    trainer: MinesweeperTrainer,
    game: MinesweeperGame,
    action_mask: np.ndarray,
    snapshot: Any | None,
) -> np.ndarray | None:
    if not game.mines_placed:
        return trainer._solver_expert_action_mask(game, action_mask, snapshot)
    if snapshot is None:
        return None

    expert_mask = np.zeros_like(action_mask, dtype=bool)
    mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
    expert_mask[action_channel(ActionType.FLAG)] = mine_cells

    safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
    expert_mask[action_channel(ActionType.OPEN)] = safe_cells
    if expert_mask.any():
        return expert_mask

    open_mask = action_mask[action_channel(ActionType.OPEN)]
    if not open_mask.any():
        return None
    guess_mask = trainer._topk_guess_open_mask(snapshot, open_mask)
    if guess_mask is None:
        return None
    expert_mask[action_channel(ActionType.OPEN)] = guess_mask
    return expert_mask


def _game_from_frame(
    frame: dict[str, Any],
    *,
    rows: int,
    cols: int,
    mines: int,
    safe_radius: int,
) -> MinesweeperGame:
    board = frame.get("board") or {}
    revealed = np.asarray(board.get("revealed", []), dtype=bool)
    flagged = np.asarray(board.get("flagged", []), dtype=bool)
    numbers = np.asarray(board.get("numbers", []), dtype=np.int16)
    if revealed.shape != (rows, cols) or flagged.shape != (rows, cols) or numbers.shape != (rows, cols):
        raise ValueError(f"unexpected board shape {revealed.shape}, {flagged.shape}, {numbers.shape}; expected {(rows, cols)}")

    game = MinesweeperGame(rows=rows, cols=cols, mines=mines, safe_radius=safe_radius)
    game.revealed = revealed.copy()
    game.flagged = flagged.copy()
    game.adjacent = np.where(revealed, np.clip(numbers, 0, 8), 0).astype(np.int8)
    game.mines = np.zeros((rows, cols), dtype=bool)
    game.mines_placed = bool(revealed.any() or flagged.any())
    game.done = bool(frame.get("done", False))
    game.won = bool(frame.get("won", False))
    game.lost = bool(frame.get("lost", False))
    game.step_count = int(frame.get("step", frame.get("step_count", 0)) or 0)
    return game


def _parse_action(payload: Any) -> Action | None:
    if not isinstance(payload, dict):
        return None
    kind_raw = payload.get("kind")
    row = payload.get("row")
    col = payload.get("col")
    if kind_raw is None or row is None or col is None:
        return None
    try:
        kind = ActionType(str(kind_raw))
    except ValueError:
        return None
    return Action(kind, int(row), int(col))


def _estimate_reward(
    before_frame: dict[str, Any],
    after_frame: dict[str, Any] | None,
    action: Action,
    action_record: dict[str, Any],
) -> float:
    before_summary = before_frame.get("summary") or before_frame
    after_summary = (after_frame.get("summary") or after_frame) if isinstance(after_frame, dict) else before_summary

    before_revealed = int(before_summary.get("revealed_safe_cells", 0) or 0)
    after_revealed = int(after_summary.get("revealed_safe_cells", before_revealed) or before_revealed)
    before_flags = int(before_summary.get("flags", 0) or 0)
    after_flags = int(after_summary.get("flags", before_flags) or before_flags)

    reward = -0.0005
    if bool(after_summary.get("lost")) and not bool(before_summary.get("lost")):
        return reward - 5.0
    if bool(after_summary.get("won")) and not bool(before_summary.get("won")):
        reward += 5.0

    if action.kind in {ActionType.OPEN, ActionType.CHORD}:
        if bool(action_record.get("no_progress")) or bool(action_record.get("blocked_repeat_open")):
            return reward - 0.02
        delta = max(0, after_revealed - before_revealed)
        reward += 0.01 * delta
        if delta <= 0:
            reward -= 0.02
    elif action.kind == ActionType.FLAG:
        if after_flags <= before_flags:
            reward -= 0.02
    elif action.kind == ActionType.UNFLAG:
        if after_flags >= before_flags:
            reward -= 0.02

    return float(reward)


def _frame_flag(frame: dict[str, Any] | None, key: str) -> bool:
    if not isinstance(frame, dict):
        return False
    if key in frame:
        return bool(frame.get(key))
    summary = frame.get("summary")
    if isinstance(summary, dict) and key in summary:
        return bool(summary.get(key))
    return False


def _frame_board_changed(before_frame: dict[str, Any], after_frame: dict[str, Any]) -> bool:
    before = before_frame.get("board") or {}
    after = after_frame.get("board") or {}
    for key in ("revealed", "flagged", "numbers"):
        if before.get(key) != after.get(key):
            return True
    return False
