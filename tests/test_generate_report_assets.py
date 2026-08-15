from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "generate_report_assets.py"
SPEC = importlib.util.spec_from_file_location("generate_report_assets_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
assets = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = assets
SPEC.loader.exec_module(assets)


def write_json(path: Path, payload: dict) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_report_data_combines_internal_and_desktop_metrics(tmp_path: Path) -> None:
    candidate_eval = tmp_path / "candidate_eval.json"
    ensemble_eval = tmp_path / "ensemble_eval.json"
    desktop_single_summary = tmp_path / "desktop_single_summary.json"
    desktop_summary = tmp_path / "desktop_summary.json"

    write_json(
        candidate_eval,
        {
            "results": [
                {
                    "checkpoint": "artifacts/full_rlmix_20.pt",
                    "games": 200,
                    "win_rate": 0.415,
                    "longest_streak": 7,
                },
                {
                    "checkpoint": "artifacts/other.pt",
                    "games": 200,
                    "win_rate": 0.2,
                    "longest_streak": 2,
                },
            ]
        },
    )
    write_json(
        ensemble_eval,
        {
            "games": 1000,
            "wins": 430,
            "win_rate": 0.43,
            "longest_streak": 9,
        },
    )
    write_json(
        desktop_summary,
        {
            "terminal_games": 952,
            "wins": 377,
            "win_rate_completed": 0.3960084033613445,
            "longest_streak": 10,
            "avg_elapsed_seconds": 27.21186873066325,
            "selected_range": {
                "games": [
                    {"game_index": 747, "elapsed_seconds": 31.19},
                    {"game_index": 748, "elapsed_seconds": 28.8},
                ]
            },
        },
    )
    write_json(
        desktop_single_summary,
        {
            "terminal_games": 495,
            "wins": 200,
            "win_rate_completed": 200 / 495,
            "longest_streak": 7,
            "avg_elapsed_seconds": 55.8255825871169,
        },
    )

    metrics, streak_games = assets.build_report_data(
        candidate_eval=candidate_eval,
        ensemble_eval=ensemble_eval,
        desktop_single_summary=desktop_single_summary,
        desktop_summary=desktop_summary,
    )

    assert [metric.name for metric in metrics] == [
        "Internal single RL",
        "Internal RL ensemble",
        "Windows desktop single RL",
        "Windows desktop ensemble",
    ]
    assert metrics[0].games == 200
    assert metrics[0].wins == 83
    assert metrics[0].win_rate == 0.415
    assert metrics[1].wins == 430
    assert metrics[2].games == 495
    assert metrics[2].wins == 200
    assert metrics[2].win_rate == 200 / 495
    assert metrics[3].games == 952
    assert metrics[3].longest_streak == 10
    assert metrics[3].avg_seconds == 27.21186873066325
    assert [game["game_index"] for game in streak_games] == [747, 748]


def test_write_report_assets_emits_markdown_json_and_svgs(tmp_path: Path) -> None:
    metrics = [
        assets.ExperimentMetric(
            name="Internal RL ensemble",
            games=1000,
            wins=430,
            win_rate=0.43,
            longest_streak=9,
            avg_seconds=None,
            source="artifacts/ensemble.json",
        ),
        assets.ExperimentMetric(
            name="Windows desktop ensemble",
            games=952,
            wins=377,
            win_rate=0.396,
            longest_streak=10,
            avg_seconds=27.21,
            source="artifacts/desktop.json",
        ),
    ]
    streak_games = [
        {"game_index": 747, "elapsed_seconds": 31.19},
        {"game_index": 748, "elapsed_seconds": 28.8},
    ]

    outputs = assets.write_report_assets(metrics=metrics, streak_games=streak_games, output_dir=tmp_path)

    output_names = {Path(output).name for output in outputs}
    assert output_names == {
        "experiment_summary.md",
        "experiment_summary.json",
        "win_rate_comparison.svg",
        "longest_streak_comparison.svg",
        "ten_streak_times.svg",
    }
    assert "Internal RL ensemble" in (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert "43.00%" in (tmp_path / "experiment_summary.md").read_text(encoding="utf-8")
    assert '"streak_games"' in (tmp_path / "experiment_summary.json").read_text(encoding="utf-8")
    assert "<svg" in (tmp_path / "win_rate_comparison.svg").read_text(encoding="utf-8")
    assert "Win Rate Comparison" in (tmp_path / "win_rate_comparison.svg").read_text(encoding="utf-8")
    assert "Verified 10-Win Streak Game Times" in (tmp_path / "ten_streak_times.svg").read_text(encoding="utf-8")
