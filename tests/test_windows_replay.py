from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from minesweeper_rl.features import action_channel, action_to_index
from minesweeper_rl.types import Action, ActionType
from minesweeper_rl.trainer import MinesweeperTrainer, TrainingConfig
from minesweeper_rl.windows_replay import _replay_path_priority, _replay_source_quality, load_windows_replay_transitions


def test_load_windows_replay_transitions(tmp_path: Path) -> None:
    run_dir = tmp_path / "windows"
    run_dir.mkdir()
    game_path = run_dir / "game_001.json"
    game_path.write_text(json.dumps(make_windows_game_log()), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, run_dir, limit=8)

    assert report["games_seen"] == 1
    assert report["games_loaded"] == 1
    assert report["transitions_loaded"] == 1
    assert report["behavior_labeled"] == 1
    assert report["solver_labeled"] == 1
    assert report["guess_labeled"] == 1

    transition = transitions[0]
    assert transition.board.shape == (20, 16, 30)
    assert transition.global_features.shape == (6,)
    assert transition.action_index == action_to_index(Action(ActionType.OPEN, 8, 15), 16, 30)
    assert transition.expert_action_index == transition.action_index
    assert transition.reward > 0.0
    assert transition.mine_mask is None
    assert transition.risk_map is not None
    assert transition.expert_action_mask is not None
    assert bool(transition.expert_action_mask.any())
    assert transition.expert_is_guess is True
    assert transition.done is False


def test_load_windows_replay_transitions_avoids_unflag_labels(tmp_path: Path) -> None:
    run_dir = tmp_path / "windows"
    run_dir.mkdir()
    game_path = run_dir / "game_001.json"
    game_path.write_text(json.dumps(make_windows_game_log_with_flag()), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, run_dir, limit=8)

    assert report["games_loaded"] == 1
    transition = transitions[0]
    assert transition.expert_action_mask is not None
    assert not bool(transition.expert_action_mask[action_channel(ActionType.UNFLAG)].any())


def test_load_windows_replay_transitions_skips_final_only_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "windows"
    run_dir.mkdir()
    payload = make_windows_game_log()
    payload["frames"] = payload["frames"][:1]
    payload["actions"] = payload["actions"] * 3
    (run_dir / "game_001.json").write_text(json.dumps(payload), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, run_dir, limit=8)

    assert transitions == []
    assert report["games_loaded"] == 0
    assert report["games_skipped"] == 1
    assert report["sample_files"][0]["reason"] == "insufficient_frame_history"


def test_load_windows_replay_transitions_skips_low_density_logs(tmp_path: Path) -> None:
    run_dir = tmp_path / "windows"
    run_dir.mkdir()
    game_path = run_dir / "game_001.json"
    game_path.write_text(json.dumps(make_low_density_windows_game_log()), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, run_dir, limit=8)

    assert transitions == []
    assert report["games_loaded"] == 0
    assert report["games_skipped"] == 1
    assert report["sample_files"][0]["reason"] == "low_transition_density"


def test_quality_priority_prefers_stronger_replay_family(tmp_path: Path) -> None:
    good = tmp_path / "pure_rl_ensemble_20_100best_1000"
    good.mkdir()
    good_path = good / "game_001.json"
    good_path.write_bytes(b"x" * 10)

    weak = tmp_path / "streak_fast_013"
    weak.mkdir()
    weak_path = weak / "game_001.json"
    weak_path.write_bytes(b"x" * 1000)

    assert _replay_path_priority(good_path) > _replay_path_priority(weak_path)


def test_source_quality_prefers_stronger_replay_family(tmp_path: Path) -> None:
    good = tmp_path / "pure_rl_ensemble_20_100best_1000" / "game_001.json"
    bad = tmp_path / "debug_memory_005" / "game_001.json"

    assert _replay_source_quality(good) > _replay_source_quality(bad)


def test_load_windows_replay_transitions_prefers_summary_selected_games(tmp_path: Path) -> None:
    root = tmp_path / "windows"
    root.mkdir()
    run_dir = root / "pure_rl_ensemble_20_100best_1000"
    run_dir.mkdir()
    summary = make_windows_summary()
    (run_dir / "per_game_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for game in summary["selected_range"]["games"]:
        (run_dir / game["file"]).write_text(json.dumps(make_windows_game_log()), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, root, limit=8)

    assert report["preferred_paths"] == len(summary["selected_range"]["games"])
    assert report["games_loaded"] == 8
    assert report["transitions_loaded"] == 8
    assert transitions
    selected_names = {game["file"] for game in summary["selected_range"]["games"]}
    loaded_names = {
        Path(sample["path"]).name
        for sample in report["sample_files"]
        if sample.get("status") == "loaded"
    }
    assert loaded_names
    assert loaded_names <= selected_names


def test_load_windows_replay_transitions_prefers_frame_history_directories(tmp_path: Path) -> None:
    root = tmp_path / "windows"
    root.mkdir()

    summary_dir = root / "pure_rl_ensemble_20_100best_1000"
    summary_dir.mkdir()
    summary = make_windows_summary()
    (summary_dir / "per_game_summary.json").write_text(json.dumps(summary), encoding="utf-8")
    for game in summary["selected_range"]["games"]:
        payload = make_windows_game_log()
        payload["frames"] = payload["frames"][:1]
        payload["actions"] = payload["actions"] * 3
        (summary_dir / game["file"]).write_text(json.dumps(payload), encoding="utf-8")

    full_frame_dir = root / "streak_fast_013"
    full_frame_dir.mkdir()
    (full_frame_dir / "game_001.json").write_text(json.dumps(make_windows_game_log()), encoding="utf-8")

    trainer = MinesweeperTrainer(TrainingConfig(seed=0))
    transitions, report = load_windows_replay_transitions(trainer, root, limit=1)

    assert transitions
    assert report["profiled_directories"] == 2
    assert report["frame_history_directories"] == 1
    assert report["final_only_directories"] == 1
    assert report["final_only_directories_skipped"] == 1
    assert report["preferred_paths_used"] == 0
    assert Path(report["sample_files"][0]["path"]).parent.name == "streak_fast_013"


def make_windows_game_log() -> dict[str, object]:
    before_revealed = zeros_board()
    before_revealed[8, 14] = 1
    before_numbers = np.full((16, 30), -1, dtype=int)
    before_numbers[8, 14] = 1

    after_revealed = zeros_board()
    after_revealed[8, 14] = 1
    after_revealed[8, 15] = 1
    after_numbers = np.full((16, 30), -1, dtype=int)
    after_numbers[8, 14] = 1
    after_numbers[8, 15] = 2

    return {
        "version": 1,
        "source": "windows_minesweeper",
        "checkpoint": "artifacts/full_rlmix_20.pt",
        "game_index": 1,
        "timing": {},
        "summary": {
            "won": False,
            "lost": False,
            "done": False,
            "revealed_safe_cells": 1,
            "flags": 0,
        },
        "actions": [
            {
                "step": 0,
                "action_index": action_to_index(Action(ActionType.OPEN, 8, 15), 16, 30),
                "action": {"kind": "open", "row": 8, "col": 15},
                "before_target": {"row": 8, "col": 15, "revealed": False, "flagged": False, "mine_like": False, "adjacent": 0},
                "after_target": {"row": 8, "col": 15, "revealed": True, "flagged": False, "mine_like": False, "adjacent": 2},
                "changed": True,
                "progress": True,
                "revealed_delta": 1,
            }
        ],
        "frames": [
            {
                "step": 0,
                "action": None,
                "won": False,
                "lost": False,
                "done": False,
                "revealed_safe_cells": 1,
                "flags": 0,
                "board": {
                    "revealed": before_revealed.tolist(),
                    "flagged": zeros_board().tolist(),
                    "numbers": before_numbers.tolist(),
                    "mine_like": zeros_board().tolist(),
                },
            },
            {
                "step": 1,
                "action": None,
                "won": False,
                "lost": False,
                "done": False,
                "revealed_safe_cells": 2,
                "flags": 0,
                "board": {
                    "revealed": after_revealed.tolist(),
                    "flagged": zeros_board().tolist(),
                    "numbers": after_numbers.tolist(),
                    "mine_like": zeros_board().tolist(),
                },
            },
        ],
    }


def make_windows_game_log_with_flag() -> dict[str, object]:
    revealed = zeros_board()
    revealed[8, 14] = 1
    after_revealed = revealed.copy()
    after_revealed[8, 15] = 1
    flagged = zeros_board()
    flagged[8, 13] = 1
    numbers = np.full((16, 30), -1, dtype=int)
    numbers[8, 14] = 2
    after_numbers = numbers.copy()
    after_numbers[8, 15] = 1

    return {
        "version": 1,
        "source": "windows_minesweeper",
        "checkpoint": "artifacts/full_rlmix_20.pt",
        "game_index": 1,
        "timing": {},
        "summary": {
            "won": False,
            "lost": False,
            "done": False,
            "revealed_safe_cells": 1,
            "flags": 1,
        },
        "actions": [
            {
                "step": 0,
                "action_index": action_to_index(Action(ActionType.OPEN, 8, 15), 16, 30),
                "action": {"kind": "open", "row": 8, "col": 15},
                "before_target": {"row": 8, "col": 15, "revealed": False, "flagged": False, "mine_like": False, "adjacent": 0},
                "after_target": {"row": 8, "col": 15, "revealed": True, "flagged": False, "mine_like": False, "adjacent": 1},
                "changed": True,
                "progress": True,
                "revealed_delta": 1,
            }
        ],
        "frames": [
            {
                "step": 0,
                "action": None,
                "won": False,
                "lost": False,
                "done": False,
                "revealed_safe_cells": 1,
                "flags": 1,
                "board": {
                    "revealed": after_revealed.tolist(),
                    "flagged": flagged.tolist(),
                    "numbers": after_numbers.tolist(),
                    "mine_like": zeros_board().tolist(),
                },
            },
            {
                "step": 1,
                "action": None,
                "won": False,
                "lost": False,
                "done": False,
                "revealed_safe_cells": 2,
                "flags": 1,
                "board": {
                    "revealed": revealed.tolist(),
                    "flagged": flagged.tolist(),
                    "numbers": numbers.tolist(),
                    "mine_like": zeros_board().tolist(),
                },
            },
        ],
    }


def make_low_density_windows_game_log() -> dict[str, object]:
    base = make_windows_game_log()
    changed_frame = base["frames"][1]
    base["frames"] = [base["frames"][0]] + [changed_frame for _ in range(100)]
    base["actions"] = [base["actions"][0] for _ in range(100)]
    return base


def make_windows_summary() -> dict[str, object]:
    games = [
        {"game_index": 747 + i, "file": f"game_{747 + i}.json"}
        for i in range(10)
    ]
    return {
        "file_count": 954,
        "win_rate_completed": 0.3960084033613445,
        "selected_range": {
            "start": 747,
            "end": 756,
            "file_count": 10,
            "completed_games": 10,
            "terminal_games": 10,
            "wins": 10,
            "losses": 0,
            "incomplete_games": 0,
            "win_rate_completed": 1.0,
            "games": games,
        },
    }


def zeros_board() -> np.ndarray:
    return np.zeros((16, 30), dtype=int)
