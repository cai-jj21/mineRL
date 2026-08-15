from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable, Sequence


COMPILE_TARGETS = [
    Path("scripts/build_artifact_manifest.py"),
    Path("scripts/validate_project_evidence.py"),
    Path("scripts/validate_documentation.py"),
    Path("scripts/validate_release.py"),
    Path("scripts/generate_report_assets.py"),
    Path("scripts/generate_statistical_report.py"),
    Path("scripts/summarize_failure_analysis.py"),
    Path("scripts/summarize_windows_games.py"),
    Path("scripts/validate_claims.py"),
    Path("scripts/evaluate_ensemble.py"),
    Path("scripts/hard_loss_refine.py"),
    Path("scripts/windows_minesweeper_agent.py"),
]


@dataclass(frozen=True)
class ReleaseCheckSpec:
    id: str
    command: list[str]
    detail: str


@dataclass
class ReleaseCheckResult:
    id: str
    ok: bool
    command: list[str]
    returncode: int | None
    elapsed_seconds: float
    detail: str
    skipped: bool = False
    stdout_tail: str = ""
    stderr_tail: str = ""


Runner = Callable[[Sequence[str]], subprocess.CompletedProcess[str]]


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Run the release-level validation gate for the Minesweeper RL project.")
    parser.add_argument("--expected-tests", type=int, default=147)
    parser.add_argument("--skip-tests", action="store_true", help="skip pytest")
    parser.add_argument("--skip-evidence", action="store_true", help="skip local experiment evidence validation")
    parser.add_argument("--skip-claims", action="store_true", help="skip report claim audit")
    parser.add_argument("--skip-manifest", action="store_true", help="skip artifact manifest generation")
    parser.add_argument(
        "--skip-artifact-links",
        action="store_true",
        help="do not ask validate_documentation.py to check artifact links",
    )
    parser.add_argument("--fail-fast", action="store_true", help="stop after the first failed check")
    parser.add_argument("--dry-run", action="store_true", help="print planned checks without executing them")
    parser.add_argument("--output", type=Path, help="optional JSON output path")
    args = parser.parse_args()

    report = validate_release(
        expected_tests=args.expected_tests,
        skip_tests=args.skip_tests,
        skip_evidence=args.skip_evidence,
        skip_claims=args.skip_claims,
        skip_manifest=args.skip_manifest,
        skip_artifact_links=args.skip_artifact_links,
        fail_fast=args.fail_fast,
        dry_run=args.dry_run,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


def validate_release(
    *,
    expected_tests: int,
    skip_tests: bool = False,
    skip_evidence: bool = False,
    skip_claims: bool = False,
    skip_manifest: bool = False,
    skip_artifact_links: bool = False,
    fail_fast: bool = False,
    dry_run: bool = False,
    runner: Runner | None = None,
) -> dict[str, Any]:
    checks = build_release_checks(
        expected_tests=expected_tests,
        skip_tests=skip_tests,
        skip_evidence=skip_evidence,
        skip_claims=skip_claims,
        skip_manifest=skip_manifest,
        skip_artifact_links=skip_artifact_links,
    )
    results = run_release_checks(checks, fail_fast=fail_fast, dry_run=dry_run, runner=runner)
    return {
        "ok": all(result.ok for result in results),
        "dry_run": dry_run,
        "checks": [asdict(result) for result in results],
        "summary": summarize_results(results),
    }


def build_release_checks(
    *,
    expected_tests: int,
    skip_tests: bool,
    skip_evidence: bool,
    skip_claims: bool,
    skip_manifest: bool,
    skip_artifact_links: bool,
) -> list[ReleaseCheckSpec]:
    checks = [
        ReleaseCheckSpec(
            id="py_compile",
            command=[sys.executable, "-m", "py_compile", *[path.as_posix() for path in COMPILE_TARGETS]],
            detail="compile release helper scripts",
        )
    ]
    if not skip_tests:
        checks.append(
            ReleaseCheckSpec(
                id="pytest",
                command=[sys.executable, "-m", "pytest", "-q"],
                detail="run the full unit test suite",
            )
        )
    doc_command = [
        sys.executable,
        "scripts/validate_documentation.py",
        "--expected-tests",
        str(expected_tests),
    ]
    if not skip_artifact_links:
        doc_command.append("--check-artifacts")
    checks.append(
        ReleaseCheckSpec(
            id="documentation",
            command=doc_command,
            detail="validate documentation links, required mentions, and test count",
        )
    )
    if not skip_evidence:
        checks.append(
            ReleaseCheckSpec(
                id="evidence",
                command=[
                    sys.executable,
                    "scripts/validate_project_evidence.py",
                    "--output",
                    "artifacts/report_assets/evidence_validation.json",
                ],
                detail="validate report metrics against local experiment artifacts",
            )
        )
    if not skip_manifest:
        checks.append(
            ReleaseCheckSpec(
                id="statistics",
                command=[sys.executable, "scripts/generate_statistical_report.py"],
                detail="regenerate Wilson confidence interval report",
            )
        )
    if not skip_claims:
        checks.append(
            ReleaseCheckSpec(
                id="claims",
                command=[sys.executable, "scripts/validate_claims.py", "--output", "artifacts/report_assets/claim_audit.json"],
                detail="audit report and resume claims against current evidence",
            )
        )
    if not skip_manifest:
        checks.append(
            ReleaseCheckSpec(
                id="artifact_manifest",
                command=[sys.executable, "scripts/build_artifact_manifest.py"],
                detail="rebuild artifact checksum manifest",
            )
        )
    return checks


def run_release_checks(
    checks: list[ReleaseCheckSpec],
    *,
    fail_fast: bool,
    dry_run: bool,
    runner: Runner | None,
) -> list[ReleaseCheckResult]:
    results: list[ReleaseCheckResult] = []
    actual_runner = runner or default_runner
    for spec in checks:
        if dry_run:
            results.append(
                ReleaseCheckResult(
                    id=spec.id,
                    ok=True,
                    command=spec.command,
                    returncode=None,
                    elapsed_seconds=0.0,
                    detail=spec.detail,
                    skipped=True,
                )
            )
            continue
        started = time.perf_counter()
        completed = actual_runner(spec.command)
        elapsed = time.perf_counter() - started
        ok = completed.returncode == 0
        results.append(
            ReleaseCheckResult(
                id=spec.id,
                ok=ok,
                command=spec.command,
                returncode=completed.returncode,
                elapsed_seconds=elapsed,
                detail=spec.detail,
                stdout_tail=tail(completed.stdout),
                stderr_tail=tail(completed.stderr),
            )
        )
        if fail_fast and not ok:
            break
    return results


def default_runner(command: Sequence[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, encoding="utf-8", errors="replace", capture_output=True, check=False)


def summarize_results(results: list[ReleaseCheckResult]) -> dict[str, Any]:
    failed = [result.id for result in results if not result.ok]
    skipped = [result.id for result in results if result.skipped]
    return {
        "total": len(results),
        "passed": sum(1 for result in results if result.ok and not result.skipped),
        "failed": len(failed),
        "skipped": len(skipped),
        "failed_ids": failed,
        "skipped_ids": skipped,
    }


def tail(text: str | None, max_chars: int = 4000) -> str:
    if text is None:
        return ""
    if len(text) <= max_chars:
        return text
    return text[-max_chars:]


if __name__ == "__main__":
    main()
