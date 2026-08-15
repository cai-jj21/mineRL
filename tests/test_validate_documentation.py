from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_documentation.py"
SPEC = importlib.util.spec_from_file_location("validate_documentation_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
doc_validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = doc_validator
SPEC.loader.exec_module(doc_validator)


def write_text(path: Path, text: str = "ok") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")


def create_docs(root: Path, test_count: int = 147) -> None:
    for path in doc_validator.REQUIRED_FILES:
        write_text(root / path)
    write_text(
        root / "README.md",
        "\n".join(
            [
                "[PROJECT_REPORT.md](PROJECT_REPORT.md)",
                "[ARCHITECTURE.md](ARCHITECTURE.md)",
                "[DEMO_GUIDE.md](DEMO_GUIDE.md)",
                "[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)",
                "[PROJECT_ONE_PAGER.md](PROJECT_ONE_PAGER.md)",
                "[REPRODUCIBILITY.md](REPRODUCIBILITY.md)",
                "[RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md)",
                "ARCHITECTURE.md DEMO_GUIDE.md EVALUATION_PROTOCOL.md PROJECT_ONE_PAGER.md PROJECT_REPORT.md FAILURE_ANALYSIS.md REPRODUCIBILITY.md MODEL_CARD.md ARTIFACTS.md",
                "EXPERIMENT_MANIFEST.md RELEASE_CHECKLIST.md",
                "scripts/validate_documentation.py",
                "scripts/validate_release.py",
                "scripts/generate_statistical_report.py",
                "scripts/validate_claims.py",
                f"当前 `{test_count}` 项测试",
            ]
        ),
    )
    write_text(
        root / "ARCHITECTURE.md",
        "src/minesweeper_rl/game.py src/minesweeper_rl/features.py src/minesweeper_rl/model.py "
        "src/minesweeper_rl/trainer.py src/minesweeper_rl/solver.py "
        "scripts/windows_minesweeper_agent.py scripts/validate_project_evidence.py "
        "artifacts/report_assets solver 不参与最终决策 PROJECT_REPORT.md",
    )
    write_text(
        root / "DEMO_GUIDE.md",
        "30 秒开场 5 分钟演示路线 scripts/summarize_windows_games.py "
        "scripts/validate_project_evidence.py scripts/validate_documentation.py scripts/validate_release.py "
        "--solver-assist none --solver-safety-filter none solver 不参与动作选择 "
        "longest_streak = 10 不要过度声称 RESUME_PROJECT_CARD.md",
    )
    write_text(
        root / "EVALUATION_PROTOCOL.md",
        "16 x 30 99 win_rate_completed terminal_games Wilson 95% 39%+ "
        "--solver-assist none --solver-safety-filter none solver 不参与动作选择 "
        "game_747.json game_756.json scripts/validate_project_evidence.py scripts/validate_release.py REPRODUCIBILITY.md",
    )
    write_text(
        root / "MODEL_CARD.md",
        "full_rlmix_20.pt full_rlmix_100_best.pt solver 不参与动作选择 39.60% FAILURE_ANALYSIS.md ARTIFACTS.md",
    )
    write_text(
        root / "ARTIFACTS.md",
        "artifacts/full_rlmix_20.pt artifacts/full_rlmix_100_best.pt EXPERIMENT_MANIFEST.md .gitignore",
    )
    write_text(
        root / "PROJECT_REPORT.md",
        f"ARCHITECTURE.md DEMO_GUIDE.md EVALUATION_PROTOCOL.md MODEL_CARD.md ARTIFACTS.md FAILURE_ANALYSIS.md scripts/summarize_failure_analysis.py scripts/generate_statistical_report.py scripts/validate_claims.py 25 / 25 10 连胜 {test_count}",
    )
    write_text(
        root / "REPRODUCIBILITY.md",
        "\n".join(
            [
                "ARCHITECTURE.md DEMO_GUIDE.md EVALUATION_PROTOCOL.md scripts/validate_project_evidence.py scripts/build_artifact_manifest.py scripts/validate_release.py",
                "scripts/summarize_failure_analysis.py scripts/generate_statistical_report.py scripts/validate_claims.py .github/workflows/ci.yml",
                "RELEASE_CHECKLIST.md ARTIFACTS.md 25",
                f"{test_count} passed",
            ]
        ),
    )
    write_text(
        root / "RESUME_PROJECT_CARD.md",
        f"ARCHITECTURE.md DEMO_GUIDE.md EVALUATION_PROTOCOL.md MODEL_CARD.md ARTIFACTS.md FAILURE_ANALYSIS.md REPRODUCIBILITY.md RELEASE_CHECKLIST.md scripts/validate_project_evidence.py scripts/build_artifact_manifest.py {test_count}",
    )
    write_text(
        root / "RELEASE_CHECKLIST.md",
        f"ARCHITECTURE.md DEMO_GUIDE.md EVALUATION_PROTOCOL.md scripts/validate_documentation.py scripts/validate_release.py scripts/validate_claims.py MODEL_CARD.md ARTIFACTS.md EXPERIMENT_MANIFEST.md FAILURE_ANALYSIS.md {test_count}",
    )


def test_validate_documentation_accepts_complete_docs(tmp_path: Path) -> None:
    create_docs(tmp_path, test_count=147)

    report = doc_validator.validate_documentation(tmp_path, expected_tests=147)

    assert report["ok"] is True
    assert report["summary"]["failed"] == 0


def test_validate_documentation_reports_missing_links_and_stale_counts(tmp_path: Path) -> None:
    create_docs(tmp_path, test_count=130)
    (tmp_path / "README.md").write_text(
        (tmp_path / "README.md").read_text(encoding="utf-8") + "\n[Missing](MISSING.md)\n",
        encoding="utf-8",
    )

    report = doc_validator.validate_documentation(tmp_path, expected_tests=147)

    assert report["ok"] is False
    failed_ids = set(report["summary"]["failed_ids"])
    assert any("missing.md" in failed_id for failed_id in failed_ids)
    assert any(failed_id.startswith("test_count.") for failed_id in failed_ids)


def test_validate_documentation_checks_artifact_manifest_count_mentions(tmp_path: Path) -> None:
    create_docs(tmp_path, test_count=147)
    manifest_path = tmp_path / "artifacts" / "report_assets" / "artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(
        json.dumps({"artifacts": [{"path": "a"}, {"path": "b"}, {"path": "c"}]}),
        encoding="utf-8",
    )
    (tmp_path / "REPRODUCIBILITY.md").write_text(
        (tmp_path / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
        + "\n当前清单覆盖 `3` 个关键产物\n",
        encoding="utf-8",
    )
    (tmp_path / "ARTIFACTS.md").write_text(
        (tmp_path / "ARTIFACTS.md").read_text(encoding="utf-8")
        + "\nbuild_artifact_manifest: 3 artifacts, missing []\n",
        encoding="utf-8",
    )

    report = doc_validator.validate_documentation(tmp_path, expected_tests=147, check_artifacts=True)

    assert report["ok"] is True


def test_validate_documentation_rejects_stale_artifact_manifest_count(tmp_path: Path) -> None:
    create_docs(tmp_path, test_count=147)
    manifest_path = tmp_path / "artifacts" / "report_assets" / "artifact_manifest.json"
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps({"artifacts": [{"path": "a"}, {"path": "b"}]}), encoding="utf-8")
    (tmp_path / "REPRODUCIBILITY.md").write_text(
        (tmp_path / "REPRODUCIBILITY.md").read_text(encoding="utf-8")
        + "\n当前清单覆盖 `1` 个关键产物\n",
        encoding="utf-8",
    )
    (tmp_path / "ARTIFACTS.md").write_text(
        (tmp_path / "ARTIFACTS.md").read_text(encoding="utf-8")
        + "\nbuild_artifact_manifest: 1 artifacts, missing []\n",
        encoding="utf-8",
    )

    report = doc_validator.validate_documentation(tmp_path, expected_tests=147, check_artifacts=True)

    assert report["ok"] is False
    failed_ids = set(report["summary"]["failed_ids"])
    assert "artifact_manifest_count.REPRODUCIBILITY.md" in failed_ids
    assert "artifact_manifest_count.ARTIFACTS.md" in failed_ids
