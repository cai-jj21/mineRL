from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_project_evidence.py"
SPEC = importlib.util.spec_from_file_location("validate_project_evidence_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = validator
SPEC.loader.exec_module(validator)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_valid_evidence(tmp_path: Path) -> validator.EvidencePaths:
    candidate_eval = tmp_path / "candidate_eval.json"
    ensemble_eval = tmp_path / "ensemble_eval.json"
    desktop_summary = tmp_path / "desktop_summary.json"
    report_summary = tmp_path / "experiment_summary.json"

    write_json(
        candidate_eval,
        {
            "results": [
                {
                    "checkpoint": "artifacts/full_rlmix_20.pt",
                    "games": 200,
                    "wins": 83,
                    "win_rate": 0.415,
                    "longest_streak": 7,
                }
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
    streak_games = [
        {"game_index": index, "won": True, "reclicks": 0, "unconfirmed_open_actions": 0, "read_recoveries": 0}
        for index in range(747, 757)
    ]
    write_json(
        desktop_summary,
        {
            "terminal_games": 952,
            "wins": 377,
            "win_rate_completed": 0.3960084033613445,
            "longest_streak": 10,
            "avg_elapsed_seconds": 27.21186873066325,
            "total_reclicks": 0,
            "total_click_unready_actions": 0,
            "total_unconfirmed_open_actions": 0,
            "total_read_recoveries": 0,
            "selected_range": {
                "start": 747,
                "end": 756,
                "games": streak_games,
            },
        },
    )
    write_json(
        report_summary,
        {
            "metrics": [
                {
                    "name": "Internal single RL",
                    "games": 200,
                    "wins": 83,
                    "win_rate": 0.415,
                    "longest_streak": 7,
                    "avg_seconds": None,
                    "source": str(candidate_eval),
                },
                {
                    "name": "Internal RL ensemble",
                    "games": 1000,
                    "wins": 430,
                    "win_rate": 0.43,
                    "longest_streak": 9,
                    "avg_seconds": None,
                    "source": str(ensemble_eval),
                },
                {
                    "name": "Windows desktop ensemble",
                    "games": 952,
                    "wins": 377,
                    "win_rate": 0.3960084033613445,
                    "longest_streak": 10,
                    "avg_seconds": 27.21186873066325,
                    "source": str(desktop_summary),
                },
            ],
            "streak_games": streak_games,
        },
    )

    return validator.EvidencePaths(
        candidate_eval=candidate_eval,
        ensemble_eval=ensemble_eval,
        desktop_summary=desktop_summary,
        report_summary=report_summary,
    )


def test_validate_project_evidence_accepts_complete_project_evidence(tmp_path: Path) -> None:
    paths = write_valid_evidence(tmp_path)

    report = validator.validate_project_evidence(paths, validator.EvidenceThresholds())

    assert report["ok"] is True
    assert report["summary"]["failed"] == 0
    assert report["summary"]["passed"] == report["summary"]["total"]


def test_validate_project_evidence_reports_failed_desktop_checks(tmp_path: Path) -> None:
    paths = write_valid_evidence(tmp_path)
    desktop = json.loads(paths.desktop_summary.read_text(encoding="utf-8"))
    desktop["longest_streak"] = 8
    desktop["total_reclicks"] = 1
    desktop["selected_range"]["games"][0]["won"] = False
    write_json(paths.desktop_summary, desktop)

    report = validator.validate_project_evidence(paths, validator.EvidenceThresholds())

    assert report["ok"] is False
    failed_ids = set(report["summary"]["failed_ids"])
    assert "desktop.longest_streak" in failed_ids
    assert "desktop.execution_anomalies" in failed_ids
    assert "desktop.streak_wins" in failed_ids
