from __future__ import annotations

import hashlib
import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "build_artifact_manifest.py"
SPEC = importlib.util.spec_from_file_location("build_artifact_manifest_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
manifest_builder = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = manifest_builder
SPEC.loader.exec_module(manifest_builder)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_manifest_fixture(tmp_path: Path) -> tuple[list[Path], dict[str, Path]]:
    candidate = tmp_path / "candidate.json"
    ensemble = tmp_path / "ensemble.json"
    desktop_single = tmp_path / "desktop_single.json"
    desktop = tmp_path / "desktop.json"
    report_dir = tmp_path / "report"
    failure_md = tmp_path / "FAILURE_ANALYSIS.md"
    streak_dir = tmp_path / "games"

    write_json(
        candidate,
        {"results": [{"checkpoint": "artifacts/full_rlmix_20.pt", "games": 200, "win_rate": 0.415, "longest_streak": 7}]},
    )
    write_json(ensemble, {"games": 1000, "wins": 430, "win_rate": 0.43, "longest_streak": 9})
    write_json(
        desktop_single,
        {
            "terminal_games": 495,
            "wins": 200,
            "win_rate_completed": 200 / 495,
            "longest_streak": 7,
            "avg_elapsed_seconds": 55.83,
        },
    )
    streak_games = [
        {"game_index": 1, "won": True, "elapsed_seconds": 31.0},
        {"game_index": 2, "won": True, "elapsed_seconds": 29.0},
    ]
    write_json(
        desktop,
        {
            "terminal_games": 952,
            "wins": 377,
            "win_rate_completed": 0.396,
            "longest_streak": 10,
            "avg_elapsed_seconds": 27.21,
            "total_reclicks": 0,
            "total_unconfirmed_open_actions": 0,
            "total_read_recoveries": 0,
            "selected_range": {"start": 1, "end": 2, "games": streak_games},
        },
    )
    write_text(report_dir / "experiment_summary.md", "# Experiment Summary\n")
    write_json(report_dir / "experiment_summary.json", {"metrics": [], "streak_games": streak_games})
    write_json(report_dir / "evidence_validation.json", {"summary": {"total": 25, "passed": 25, "failed": 0}})
    write_text(report_dir / "win_rate_comparison.svg", "<svg>win</svg>")
    write_text(report_dir / "longest_streak_comparison.svg", "<svg>streak</svg>")
    write_text(report_dir / "ten_streak_times.svg", "<svg>times</svg>")
    write_text(report_dir / "ten_streak_review.md", "# Ten-Win Streak Review\n")
    write_json(report_dir / "statistical_summary.json", {"experiments": []})
    write_text(report_dir / "statistical_summary.md", "# Statistical Summary\n")
    write_json(report_dir / "claim_audit.json", {"summary": {"total": 10, "passed": 10, "failed": 0}})
    write_json(
        report_dir / "failure_analysis_summary.json",
        {
            "loss_analysis": {
                "games": 500,
                "losses": 291,
                "wrong_flag_loss_rate": 0.23,
                "terminal_forced_rate": 0.316,
                "avg_target_risk": 0.287,
                "avg_best_guess_risk": 0.226,
            }
        },
    )
    write_text(failure_md, "# Failure Analysis\n")
    write_json(streak_dir / "game_001.json", {"summary": {"won": True}})
    write_json(streak_dir / "game_002.json", {"summary": {"won": True}})

    paths = manifest_builder.default_artifact_paths(
        candidate_eval=candidate,
        ensemble_eval=ensemble,
        desktop_single_summary=desktop_single,
        desktop_summary=desktop,
        report_dir=report_dir,
        failure_analysis_md=failure_md,
        streak_dir=streak_dir,
        streak_start=1,
        streak_end=2,
    )
    return paths, {
        "candidate": candidate,
        "ensemble": ensemble,
        "desktop_single": desktop_single,
        "desktop": desktop,
        "report_dir": report_dir,
        "failure_md": failure_md,
    }


def test_build_manifest_records_hashes_and_headline_metrics(tmp_path: Path) -> None:
    paths, refs = create_manifest_fixture(tmp_path)

    manifest = manifest_builder.build_manifest(
        paths=paths,
        candidate_eval=refs["candidate"],
        ensemble_eval=refs["ensemble"],
        desktop_single_summary=refs["desktop_single"],
        desktop_summary=refs["desktop"],
        report_dir=refs["report_dir"],
        failure_analysis_md=refs["failure_md"],
        streak_start=1,
        streak_end=2,
    )

    assert manifest["ok"] is True
    assert manifest["missing"] == []
    assert len(manifest["artifacts"]) == 18
    candidate_record = manifest["artifacts"][0]
    expected_hash = hashlib.sha256(refs["candidate"].read_bytes()).hexdigest()
    assert candidate_record["sha256"] == expected_hash
    assert manifest["metrics"]["single_rl"]["wins"] == 83
    assert manifest["metrics"]["internal_ensemble"]["win_rate"] == 0.43
    assert manifest["metrics"]["windows_desktop"]["single_wins"] == 200
    assert manifest["metrics"]["windows_desktop"]["terminal_games"] == 952
    assert manifest["metrics"]["verified_streak"]["wins"] == 2
    assert manifest["metrics"]["evidence_validation"]["passed"] == 25
    assert manifest["metrics"]["failure_analysis"]["losses"] == 291

    markdown = manifest_builder.render_manifest_markdown(manifest)
    assert "SHA-256" in markdown
    assert "43.00%" in markdown
    assert "game_001" in markdown
    assert "失败分析" in markdown


def test_build_manifest_marks_missing_artifacts(tmp_path: Path) -> None:
    missing = tmp_path / "missing.json"

    manifest = manifest_builder.build_manifest(
        paths=[missing],
        candidate_eval=missing,
        ensemble_eval=missing,
        desktop_single_summary=missing,
        desktop_summary=missing,
        report_dir=tmp_path,
        failure_analysis_md=missing,
        streak_start=1,
        streak_end=1,
    )

    assert manifest["ok"] is False
    assert manifest["missing"] == [missing.as_posix()]
    assert manifest["artifacts"][0]["exists"] is False
