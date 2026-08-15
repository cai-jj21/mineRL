from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path
from typing import Sequence


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "validate_release.py"
SPEC = importlib.util.spec_from_file_location("validate_release_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
release_validator = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = release_validator
SPEC.loader.exec_module(release_validator)


def completed(command: Sequence[str], returncode: int = 0) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(list(command), returncode, stdout="ok\n", stderr="")


def test_build_release_checks_includes_full_gate_by_default() -> None:
    checks = release_validator.build_release_checks(
        expected_tests=147,
        skip_tests=False,
        skip_evidence=False,
        skip_claims=False,
        skip_manifest=False,
        skip_artifact_links=False,
    )

    assert [check.id for check in checks] == [
        "py_compile",
        "pytest",
        "documentation",
        "evidence",
        "statistics",
        "claims",
        "artifact_manifest",
    ]
    doc_command = checks[2].command
    assert "--expected-tests" in doc_command
    assert "147" in doc_command
    assert "--check-artifacts" in doc_command
    assert "scripts/validate_release.py" in checks[0].command
    assert "scripts/generate_statistical_report.py" in checks[0].command
    assert "scripts/validate_claims.py" in checks[0].command
    assert checks[-2].id == "claims"
    assert "artifacts/report_assets/claim_audit.json" in checks[-2].command


def test_validate_release_reports_fail_fast_result() -> None:
    seen: list[list[str]] = []

    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        seen.append(list(command))
        if len(seen) == 2:
            return subprocess.CompletedProcess(list(command), 1, stdout="", stderr="pytest failed")
        return completed(command)

    report = release_validator.validate_release(
        expected_tests=147,
        fail_fast=True,
        runner=runner,
    )

    assert report["ok"] is False
    assert report["summary"]["total"] == 2
    assert report["summary"]["failed_ids"] == ["pytest"]
    assert report["checks"][1]["stderr_tail"] == "pytest failed"


def test_validate_release_dry_run_skips_commands() -> None:
    def runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
        raise AssertionError(f"unexpected command: {command}")

    report = release_validator.validate_release(
        expected_tests=147,
        skip_tests=True,
        skip_evidence=True,
        skip_claims=True,
        skip_manifest=True,
        dry_run=True,
        runner=runner,
    )

    assert report["ok"] is True
    assert report["summary"]["skipped"] == 2
    assert report["summary"]["skipped_ids"] == ["py_compile", "documentation"]
