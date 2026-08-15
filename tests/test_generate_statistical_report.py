from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "generate_statistical_report.py"
SPEC = importlib.util.spec_from_file_location("generate_statistical_report_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
stats = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = stats
SPEC.loader.exec_module(stats)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_wilson_interval_matches_known_ensemble_result() -> None:
    low, high = stats.wilson_interval(wins=430, games=1000)

    assert low == pytest_approx(0.3996409199186558)
    assert high == pytest_approx(0.4608948262693219)


def test_build_statistical_report_combines_all_headline_sources(tmp_path: Path) -> None:
    candidate = tmp_path / "candidate.json"
    ensemble = tmp_path / "ensemble.json"
    desktop_single = tmp_path / "single.json"
    desktop_ensemble = tmp_path / "desktop.json"
    write_json(candidate, {"results": [{"checkpoint": "artifacts/full_rlmix_20.pt", "games": 200, "win_rate": 0.415}]})
    write_json(ensemble, {"games": 1000, "wins": 430, "win_rate": 0.43})
    write_json(desktop_single, {"terminal_games": 495, "wins": 200, "win_rate_completed": 200 / 495})
    write_json(desktop_ensemble, {"terminal_games": 952, "wins": 377, "win_rate_completed": 377 / 952})

    report = stats.build_statistical_report(
        candidate_eval=candidate,
        ensemble_eval=ensemble,
        desktop_single_summary=desktop_single,
        desktop_ensemble_summary=desktop_ensemble,
        target_win_rate=0.40,
    )

    rows = {row["name"]: row for row in report["experiments"]}
    assert list(rows) == [
        "Internal single RL",
        "Internal RL ensemble",
        "Windows desktop single RL",
        "Windows desktop ensemble",
    ]
    assert rows["Internal single RL"]["wins"] == 83
    assert rows["Windows desktop single RL"]["games"] == 495
    assert rows["Windows desktop single RL"]["win_rate"] == pytest_approx(200 / 495)
    assert rows["Windows desktop single RL"]["point_target_pass"] is True
    assert rows["Windows desktop single RL"]["wilson_lower_target_pass"] is False
    assert rows["Windows desktop ensemble"]["point_target_pass"] is False


def test_render_markdown_includes_intervals_and_target_explanation(tmp_path: Path) -> None:
    report = {
        "schema_version": 1,
        "confidence": 0.95,
        "target_win_rate": 0.40,
        "experiments": [
            {
                "name": "Example",
                "source": "example.json",
                "wins": 40,
                "games": 100,
                "win_rate": 0.40,
                "wilson_low": 0.31,
                "wilson_high": 0.50,
                "point_target_pass": True,
                "wilson_lower_target_pass": False,
            }
        ],
    }

    markdown = stats.render_markdown(report)

    assert "# Statistical Summary" in markdown
    assert "40 / 100" in markdown
    assert "31.00% - 50.00%" in markdown
    assert "pass | not proven" in markdown
    assert "Wilson 95% lower bound" in markdown


def pytest_approx(value: float):
    import pytest

    return pytest.approx(value)
