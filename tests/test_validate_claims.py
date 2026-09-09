from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_claims.py"
SPEC = importlib.util.spec_from_file_location("validate_claims_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
claims = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = claims
SPEC.loader.exec_module(claims)


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def create_claim_fixture(root: Path) -> Path:
    stats = root / "artifacts" / "report_assets" / "statistical_summary.json"
    write_json(
        stats,
        {
            "experiments": [
                {
                    "name": "Internal single RL",
                    "wins": 83,
                    "games": 200,
                    "win_rate": 0.415,
                    "wilson_low": 0.34894285449757567,
                    "wilson_high": 0.4842608509865596,
                },
                {
                    "name": "Internal RL ensemble",
                    "wins": 430,
                    "games": 1000,
                    "win_rate": 0.43,
                    "wilson_low": 0.3996409199186558,
                    "wilson_high": 0.4608948262693219,
                },
                {
                    "name": "Windows desktop single RL",
                    "wins": 200,
                    "games": 495,
                    "win_rate": 200 / 495,
                    "wilson_low": 0.36171167721067227,
                    "wilson_high": 0.44784705468665525,
                },
                {
                    "name": "Windows desktop ensemble",
                    "wins": 377,
                    "games": 952,
                    "win_rate": 377 / 952,
                    "wilson_low": 0.3654191588045586,
                    "wilson_high": 0.4274335175751726,
                },
            ]
        },
    )
    write_text(root / "README.md", "43.0% 40.40% 39.60% 495 完成局 952 完成局")
    write_text(root / "PROJECT_ONE_PAGER.md", "43.0% 40.40% 39.60% 495 完成局 952 完成局")
    write_text(root / "INTERVIEW_QA.md", "43.0% 40.40% 39.60% 495 完成局 952 完成局 不是稳定超过 40%")
    write_text(root / "DATA_DEVELOPMENT_CASE.md", "43.0% 40.40% 39.60% 495 完成局 952 完成局 不是稳定超过 40%")
    write_text(root / "ABLATION_STUDY.md", "41.50% 43.0% 40.40% 39.60% 495 完成局 952 完成局")
    write_text(root / "PROJECT_COMPLETION_AUDIT.md", "41.50% 43.0% 40.40% 39.60% 495 完成局 952 完成局")
    write_text(root / "PROJECT_REPORT.md", "43.0% 40.40% 39.60% 495 个完成局 952 个完成局")
    write_text(root / "RESUME_PROJECT_CARD.md", "43.0% 40.40% 39.60% 495 完成局")
    write_text(root / "MODEL_CARD.md", "40.40% 39.60%")
    write_text(root / "DEMO_GUIDE.md", "39.60% 40.40% 不是稳定超过 40%")
    write_text(
        root / "EVALUATION_PROTOCOL.md",
        "\n".join(
            [
                "430 / 1000 43.00% 39.96% - 46.09%",
                "200 / 495 40.40% 36.17% - 44.78%",
                "377 / 952 39.60% 36.54% - 42.74%",
                "避免 稳定超过 40% Windows 胜率",
            ]
        ),
    )
    write_text(
        root / "WINDOWS_AGENT_RESULTS.md",
        "terminal_games: 495\nwin_rate_completed: 40.40%\ntotal_read_recoveries: 22\n",
    )
    return stats


def test_validate_claims_accepts_current_audited_language(tmp_path: Path) -> None:
    stats = create_claim_fixture(tmp_path)

    report = claims.validate_claims(root=tmp_path, stats_summary=stats)

    assert report["ok"] is True
    assert report["summary"]["failed"] == 0


def test_validate_claims_rejects_stale_win_rate(tmp_path: Path) -> None:
    stats = create_claim_fixture(tmp_path)
    write_text(tmp_path / "README.md", "43.0% 42.4% 39.60% 495 完成局 952 完成局")

    report = claims.validate_claims(root=tmp_path, stats_summary=stats)

    assert report["ok"] is False
    assert any(failed_id.startswith("stale.README.md") for failed_id in report["summary"]["failed_ids"])


def test_validate_claims_rejects_unnegated_stable_windows_claim(tmp_path: Path) -> None:
    stats = create_claim_fixture(tmp_path)
    write_text(tmp_path / "DEMO_GUIDE.md", "39.60% 40.40% 稳定超过 40% Windows 胜率")

    report = claims.validate_claims(root=tmp_path, stats_summary=stats)

    assert report["ok"] is False
    assert any(failed_id.startswith("overclaim.DEMO_GUIDE.md") for failed_id in report["summary"]["failed_ids"])
