from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


REQUIRED_FILES = [
    Path("README.md"),
    Path("ARCHITECTURE.md"),
    Path("DEMO_GUIDE.md"),
    Path("EVALUATION_PROTOCOL.md"),
    Path("PROJECT_REPORT.md"),
    Path("REPRODUCIBILITY.md"),
    Path("RESUME_PROJECT_CARD.md"),
    Path("WINDOWS_AGENT_RESULTS.md"),
    Path("MODEL_CARD.md"),
    Path("ARTIFACTS.md"),
    Path("FAILURE_ANALYSIS.md"),
    Path("EXPERIMENT_MANIFEST.md"),
    Path("RELEASE_CHECKLIST.md"),
    Path(".github/workflows/ci.yml"),
    Path("scripts/generate_report_assets.py"),
    Path("scripts/generate_statistical_report.py"),
    Path("scripts/summarize_failure_analysis.py"),
    Path("scripts/validate_claims.py"),
    Path("scripts/validate_project_evidence.py"),
    Path("scripts/build_artifact_manifest.py"),
    Path("scripts/validate_documentation.py"),
    Path("scripts/validate_release.py"),
]

REQUIRED_MENTIONS = {
    "README.md": [
        "ARCHITECTURE.md",
        "DEMO_GUIDE.md",
        "EVALUATION_PROTOCOL.md",
        "PROJECT_REPORT.md",
        "FAILURE_ANALYSIS.md",
        "REPRODUCIBILITY.md",
        "MODEL_CARD.md",
        "ARTIFACTS.md",
        "EXPERIMENT_MANIFEST.md",
        "RELEASE_CHECKLIST.md",
        "scripts/validate_documentation.py",
        "scripts/validate_release.py",
        "scripts/generate_statistical_report.py",
        "scripts/validate_claims.py",
    ],
    "DEMO_GUIDE.md": [
        "30 秒开场",
        "5 分钟演示路线",
        "scripts/summarize_windows_games.py",
        "scripts/validate_project_evidence.py",
        "scripts/validate_documentation.py",
        "scripts/validate_release.py",
        "--solver-assist none --solver-safety-filter none",
        "solver 不参与动作选择",
        "longest_streak = 10",
        "不要过度声称",
        "RESUME_PROJECT_CARD.md",
    ],
    "EVALUATION_PROTOCOL.md": [
        "16 x 30",
        "99",
        "win_rate_completed",
        "terminal_games",
        "Wilson 95%",
        "39%+",
        "--solver-assist none --solver-safety-filter none",
        "solver 不参与动作选择",
        "game_747.json",
        "game_756.json",
        "scripts/validate_project_evidence.py",
        "scripts/validate_release.py",
        "REPRODUCIBILITY.md",
    ],
    "ARCHITECTURE.md": [
        "src/minesweeper_rl/game.py",
        "src/minesweeper_rl/features.py",
        "src/minesweeper_rl/model.py",
        "src/minesweeper_rl/trainer.py",
        "src/minesweeper_rl/solver.py",
        "scripts/windows_minesweeper_agent.py",
        "scripts/validate_project_evidence.py",
        "artifacts/report_assets",
        "solver 不参与最终决策",
        "PROJECT_REPORT.md",
    ],
    "MODEL_CARD.md": [
        "full_rlmix_20.pt",
        "full_rlmix_100_best.pt",
        "solver 不参与动作选择",
        "39.60%",
        "FAILURE_ANALYSIS.md",
        "ARTIFACTS.md",
    ],
    "ARTIFACTS.md": [
        "artifacts/full_rlmix_20.pt",
        "artifacts/full_rlmix_100_best.pt",
        "EXPERIMENT_MANIFEST.md",
        ".gitignore",
    ],
    "PROJECT_REPORT.md": [
        "ARCHITECTURE.md",
        "DEMO_GUIDE.md",
        "EVALUATION_PROTOCOL.md",
        "MODEL_CARD.md",
        "ARTIFACTS.md",
        "FAILURE_ANALYSIS.md",
        "scripts/summarize_failure_analysis.py",
        "scripts/generate_statistical_report.py",
        "scripts/validate_claims.py",
        "25 / 25",
        "10 连胜",
    ],
    "REPRODUCIBILITY.md": [
        "ARCHITECTURE.md",
        "DEMO_GUIDE.md",
        "EVALUATION_PROTOCOL.md",
        "scripts/validate_project_evidence.py",
        "scripts/build_artifact_manifest.py",
        "scripts/validate_release.py",
        "scripts/summarize_failure_analysis.py",
        "scripts/generate_statistical_report.py",
        "scripts/validate_claims.py",
        ".github/workflows/ci.yml",
        "RELEASE_CHECKLIST.md",
        "ARTIFACTS.md",
        "25",
    ],
    "RESUME_PROJECT_CARD.md": [
        "ARCHITECTURE.md",
        "DEMO_GUIDE.md",
        "EVALUATION_PROTOCOL.md",
        "MODEL_CARD.md",
        "ARTIFACTS.md",
        "FAILURE_ANALYSIS.md",
        "REPRODUCIBILITY.md",
        "RELEASE_CHECKLIST.md",
        "scripts/validate_project_evidence.py",
        "scripts/build_artifact_manifest.py",
    ],
    "RELEASE_CHECKLIST.md": [
        "ARCHITECTURE.md",
        "DEMO_GUIDE.md",
        "EVALUATION_PROTOCOL.md",
        "scripts/validate_documentation.py",
        "scripts/validate_release.py",
        "scripts/validate_claims.py",
        "MODEL_CARD.md",
        "ARTIFACTS.md",
        "EXPERIMENT_MANIFEST.md",
        "FAILURE_ANALYSIS.md",
    ],
}

TEST_COUNT_MENTION_FILES = {
    "README.md",
    "PROJECT_REPORT.md",
    "REPRODUCIBILITY.md",
    "RESUME_PROJECT_CARD.md",
    "RELEASE_CHECKLIST.md",
}

TEST_COUNT_SCAN_FILES = {
    "README.md",
    "REPRODUCIBILITY.md",
    "RESUME_PROJECT_CARD.md",
}

ARTIFACT_MANIFEST_PATH = Path("artifacts/report_assets/artifact_manifest.json")

MARKDOWN_LINK_RE = re.compile(r"\[[^\]]+\]\(([^)]+)\)")
PASSED_RE = re.compile(r"(?<!\d)(\d+)[ \t]+passed\b")
TEST_COUNT_RE = re.compile(r"`(\d+)`\s*项测试")


@dataclass
class DocumentationCheck:
    id: str
    ok: bool
    observed: Any
    expected: Any
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate local project documentation links and key evidence mentions.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--expected-tests", type=int, default=147)
    parser.add_argument("--check-artifacts", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    report = validate_documentation(
        root=args.root,
        expected_tests=args.expected_tests,
        check_artifacts=args.check_artifacts,
    )
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


def validate_documentation(root: Path, expected_tests: int, check_artifacts: bool = False) -> dict[str, Any]:
    root = root.resolve()
    checks: list[DocumentationCheck] = []
    checks.extend(check_required_files(root))
    checks.extend(check_required_mentions(root, expected_tests))
    checks.extend(check_markdown_links(root, check_artifacts=check_artifacts))
    if check_artifacts:
        checks.extend(check_artifact_manifest_mentions(root))
    checks.extend(check_test_count_mentions(root, expected_tests=expected_tests))
    return {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
        "summary": summarize_checks(checks),
    }


def check_required_files(root: Path) -> list[DocumentationCheck]:
    checks: list[DocumentationCheck] = []
    for path in REQUIRED_FILES:
        checks.append(
            DocumentationCheck(
                id=f"required_file.{path.as_posix()}",
                ok=(root / path).exists(),
                observed=path.as_posix(),
                expected="exists",
                detail="required project documentation or helper script exists",
            )
        )
    return checks


def check_required_mentions(root: Path, expected_tests: int) -> list[DocumentationCheck]:
    checks: list[DocumentationCheck] = []
    for rel_path, mentions in REQUIRED_MENTIONS.items():
        path = root / rel_path
        text = read_text(path)
        expected_mentions = list(mentions)
        if rel_path in TEST_COUNT_MENTION_FILES:
            expected_mentions.append(str(expected_tests))
        for mention in expected_mentions:
            checks.append(
                DocumentationCheck(
                    id=f"mention.{rel_path}.{slug(mention)}",
                    ok=mention in text,
                    observed=mention if mention in text else None,
                    expected=mention,
                    detail=f"{rel_path} mentions {mention}",
                )
            )
    return checks


def check_markdown_links(root: Path, check_artifacts: bool) -> list[DocumentationCheck]:
    checks: list[DocumentationCheck] = []
    for markdown_path in sorted(root.glob("*.md")):
        text = read_text(markdown_path)
        for target in MARKDOWN_LINK_RE.findall(text):
            normalized = normalize_markdown_target(target)
            if normalized is None:
                continue
            rel_target = normalized.as_posix()
            if not check_artifacts and is_artifact_path(normalized):
                continue
            exists = (markdown_path.parent / normalized).exists()
            checks.append(
                DocumentationCheck(
                    id=f"link.{markdown_path.name}.{slug(rel_target)}",
                    ok=exists,
                    observed=rel_target,
                    expected="existing local file",
                    detail=f"{markdown_path.name} local markdown link exists",
                )
            )
    return checks


def check_test_count_mentions(root: Path, expected_tests: int) -> list[DocumentationCheck]:
    checks: list[DocumentationCheck] = []
    for rel_path in sorted(TEST_COUNT_SCAN_FILES):
        markdown_path = root / rel_path
        text = read_text(markdown_path)
        for pattern_name, pattern in [("passed", PASSED_RE), ("test_count", TEST_COUNT_RE)]:
            for match in pattern.finditer(text):
                observed = int(match.group(1))
                checks.append(
                    DocumentationCheck(
                        id=f"test_count.{markdown_path.name}.{pattern_name}.{match.start()}",
                        ok=observed == expected_tests,
                        observed=observed,
                        expected=expected_tests,
                        detail=f"{markdown_path.name} test count mention is current",
                )
            )
    return checks


def check_artifact_manifest_mentions(root: Path) -> list[DocumentationCheck]:
    manifest_path = root / ARTIFACT_MANIFEST_PATH
    checks: list[DocumentationCheck] = []
    if not manifest_path.exists():
        return [
            DocumentationCheck(
                id="artifact_manifest.exists",
                ok=False,
                observed=ARTIFACT_MANIFEST_PATH.as_posix(),
                expected="existing artifact manifest JSON",
                detail="artifact-aware documentation validation needs the generated manifest",
            )
        ]

    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        return [
            DocumentationCheck(
                id="artifact_manifest.valid_json",
                ok=False,
                observed=str(exc),
                expected="valid JSON",
                detail="artifact manifest should be parseable JSON",
            )
        ]

    artifact_count = len(manifest.get("artifacts", []))
    expected_mentions = {
        "REPRODUCIBILITY.md": f"当前清单覆盖 `{artifact_count}` 个关键产物",
        "ARTIFACTS.md": f"build_artifact_manifest: {artifact_count} artifacts, missing []",
    }
    for rel_path, mention in expected_mentions.items():
        text = read_text(root / rel_path)
        checks.append(
            DocumentationCheck(
                id=f"artifact_manifest_count.{rel_path}",
                ok=mention in text,
                observed=mention if mention in text else None,
                expected=mention,
                detail=f"{rel_path} describes the current artifact manifest count",
            )
        )
    return checks


def normalize_markdown_target(target: str) -> Path | None:
    target = target.strip()
    if not target or target.startswith("#"):
        return None
    lower = target.lower()
    if lower.startswith(("http://", "https://", "mailto:", "file:", "vscode:")):
        return None
    target = target.split("#", 1)[0]
    target = target.split("?", 1)[0]
    if not target:
        return None
    return Path(target)


def is_artifact_path(path: Path) -> bool:
    parts = path.parts
    return bool(parts and parts[0].replace("\\", "/") == "artifacts")


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def summarize_checks(checks: list[DocumentationCheck]) -> dict[str, Any]:
    failed = [check.id for check in checks if not check.ok]
    return {
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_ids": failed,
    }


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


if __name__ == "__main__":
    main()
