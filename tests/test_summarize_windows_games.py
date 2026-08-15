from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_windows_games.py"
SPEC = importlib.util.spec_from_file_location("summarize_windows_games_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
summarizer = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = summarizer
SPEC.loader.exec_module(summarizer)


def write_game(path: Path, index: int, **summary) -> None:
    payload = {"summary": summary}
    (path / f"game_{index:03d}.json").write_text(json.dumps(payload), encoding="utf-8")


def write_game_payload(path: Path, index: int, payload: dict) -> None:
    (path / f"game_{index:03d}.json").write_text(json.dumps(payload), encoding="utf-8")


def test_summarize_records_counts_completed_games_and_longest_streak(tmp_path: Path) -> None:
    write_game(tmp_path, 1, won=False, lost=True, done=True, elapsed_seconds=10, actions_per_second=5)
    write_game(tmp_path, 2, won=True, lost=False, done=True, elapsed_seconds=20, actions_per_second=10)
    write_game(tmp_path, 3, won=True, lost=False, done=True, elapsed_seconds=30, actions_per_second=15)
    write_game(tmp_path, 4, won=False, lost=False, done=False, elapsed_seconds=5, actions_per_second=1)
    write_game(tmp_path, 5, won=True, lost=False, done=True, elapsed_seconds=40, actions_per_second=20)

    records = summarizer.load_game_records(tmp_path)
    summary = summarizer.summarize_records(records)

    assert summary["file_count"] == 5
    assert summary["terminal_games"] == 4
    assert summary["completed_games"] == 4
    assert summary["wins"] == 3
    assert summary["losses"] == 1
    assert summary["incomplete_games"] == 1
    assert summary["win_rate_completed"] == 0.75
    assert summary["longest_streak"] == 2
    assert summary["longest_streak_start"] == 2
    assert summary["longest_streak_end"] == 3
    assert summary["avg_elapsed_seconds"] == 25.0
    assert summary["avg_actions_per_second"] == 12.5


def test_summarize_records_accumulates_execution_stability_signals(tmp_path: Path) -> None:
    write_game(
        tmp_path,
        7,
        won=True,
        lost=False,
        done=True,
        agent_steps=100,
        physical_open_actions=60,
        reclicks=1,
        click_unready_actions=2,
        unconfirmed_open_actions=3,
        read_recoveries=4,
    )
    write_game(
        tmp_path,
        8,
        won=False,
        lost=True,
        done=True,
        agent_steps=50,
        physical_open_actions=30,
        reclicks=5,
        click_unready_actions=6,
        unconfirmed_open_actions=7,
        read_recoveries=8,
    )

    summary = summarizer.summarize_records(summarizer.load_game_records(tmp_path))

    assert summary["avg_agent_steps"] == 75.0
    assert summary["avg_physical_open_actions"] == 45.0
    assert summary["total_reclicks"] == 6
    assert summary["total_click_unready_actions"] == 8
    assert summary["total_unconfirmed_open_actions"] == 10
    assert summary["total_read_recoveries"] == 12


def test_load_game_records_extracts_action_level_streak_fields(tmp_path: Path) -> None:
    write_game_payload(
        tmp_path,
        747,
        {
            "summary": {
                "won": True,
                "lost": False,
                "done": True,
                "agent_steps": 3,
                "physical_open_actions": 2,
                "flags": 1,
                "revealed_safe_cells": 380,
                "elapsed_seconds": 31.1,
                "actions_per_second": 9.7,
                "terminal_dialog": "win",
                "quick_number_read_actions": 2,
                "quick_number_read_fallbacks": 1,
                "avg_click_seconds": 0.09,
                "avg_after_read_seconds": 0.03,
                "open_zero_progress_actions": 0,
                "open_target_miss_with_progress": 0,
                "read_repairs": 0,
                "read_restores": 0,
                "first_open_action": {
                    "action": {"kind": "open", "row": 8, "col": 15},
                    "revealed_delta": 54,
                },
            },
            "actions": [
                {"step": 0, "action": {"kind": "open", "row": 8, "col": 15}},
                {"step": 1, "action": {"kind": "flag", "row": 7, "col": 17}, "virtual_only": True},
                {
                    "step": 2,
                    "action": {"kind": "open", "row": 13, "col": 29},
                    "terminal_dialog": "win",
                    "terminal_detected_at": "after_click",
                },
            ],
        },
    )

    record = summarizer.load_game_records(tmp_path)[0]

    assert record["first_open"] == {"kind": "open", "row": 8, "col": 15}
    assert record["first_open_revealed_delta"] == 54
    assert record["terminal_action"] == {"kind": "open", "row": 13, "col": 29}
    assert record["terminal_detected_at"] == "after_click"
    assert record["virtual_flag_actions"] == 1
    assert record["quick_number_read_actions"] == 2
    assert record["quick_number_read_fallbacks"] == 1
    assert record["avg_click_seconds"] == 0.09
    assert record["avg_after_read_seconds"] == 0.03


def test_render_streak_markdown_includes_game_table_and_anomaly_summary(tmp_path: Path) -> None:
    write_game_payload(
        tmp_path,
        1,
        {
            "summary": {
                "won": True,
                "lost": False,
                "done": True,
                "agent_steps": 10,
                "physical_open_actions": 7,
                "flags": 2,
                "revealed_safe_cells": 380,
                "elapsed_seconds": 12.5,
                "actions_per_second": 8.0,
                "terminal_dialog": "win",
                "first_open_action": {
                    "action": {"kind": "open", "row": 8, "col": 15},
                    "revealed_delta": 20,
                },
            },
            "actions": [
                {"step": 0, "action": {"kind": "open", "row": 8, "col": 15}},
                {"step": 1, "action": {"kind": "flag", "row": 0, "col": 0}, "virtual_only": True},
                {
                    "step": 9,
                    "action": {"kind": "open", "row": 0, "col": 29},
                    "terminal_dialog": "win",
                },
            ],
        },
    )
    records = summarizer.load_game_records(tmp_path)
    result = {"input": str(tmp_path), **summarizer.summarize_records(records)}
    result["selected_range"] = summarizer.build_selected_range(records, 1, 1)

    markdown = summarizer.render_streak_markdown(result)

    assert "# Windows Ten-Win Streak Review" in markdown
    assert "Range: `game_001.json` to `game_001.json`" in markdown
    assert "Execution anomalies: reclicks=0, unconfirmed_opens=0, read_recoveries=0, target_misses=0" in markdown
    assert "| 1 | win | r8c15 | 20 | 10 | 7 | 1 | 2 | 380 | 12.50 | 8.00 | r0c29 | win | 0 |" in markdown
