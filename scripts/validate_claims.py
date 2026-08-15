from __future__ import annotations

import argparse
import json
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_STATS_SUMMARY = Path("artifacts/report_assets/statistical_summary.json")
DEFAULT_OUTPUT = Path("artifacts/report_assets/claim_audit.json")

CLAIM_FILES = [
    Path("README.md"),
    Path("PROJECT_ONE_PAGER.md"),
    Path("ABLATION_STUDY.md"),
    Path("PROJECT_COMPLETION_AUDIT.md"),
    Path("PROJECT_REPORT.md"),
    Path("RESUME_PROJECT_CARD.md"),
    Path("MODEL_CARD.md"),
    Path("DEMO_GUIDE.md"),
    Path("EVALUATION_PROTOCOL.md"),
    Path("WINDOWS_AGENT_RESULTS.md"),
]

STALE_GLOBAL_PATTERNS = [
    re.compile(r"42\.4(?:0)?%"),
    re.compile(r"212\s*/\s*500"),
]

OVERCLAIM_PATTERNS = [
    re.compile(r"稳定\s*(?:超过|超|>|>=)?\s*40%?\+?\s*Windows\s*胜率", re.IGNORECASE),
    re.compile(r"稳定\s*40%\+\s*Windows", re.IGNORECASE),
    re.compile(r"stable\s*(?:above|over|>=)?\s*40%?\+?\s*Windows", re.IGNORECASE),
]

NEGATION_MARKERS = [
    "不说",
    "不要",
    "避免",
    "不是",
    "不应",
    "不能",
    "不等于",
    "略低于",
    "not ",
    "avoid",
    "do not",
    "don't",
    "cannot",
]


@dataclass
class ClaimCheck:
    id: str
    ok: bool
    observed: Any
    expected: Any
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Audit report and resume claims against current experiment evidence.")
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--stats-summary", type=Path, default=DEFAULT_STATS_SUMMARY)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    report = validate_claims(root=args.root, stats_summary=args.stats_summary)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


def validate_claims(root: Path, stats_summary: Path) -> dict[str, Any]:
    root = root.resolve()
    stats_path = resolve_path(root, stats_summary)
    stats = load_json(stats_path)
    checks: list[ClaimCheck] = []
    checks.append(
        ClaimCheck(
            id="stats.exists",
            ok=stats_path.exists(),
            observed=str(stats_summary),
            expected="existing statistical summary JSON",
            detail="claim audit reads generated Wilson interval evidence",
        )
    )
    checks.extend(check_claim_files(root))
    checks.extend(check_stale_patterns(root))
    checks.extend(check_overclaims(root))
    if stats:
        checks.extend(check_required_current_claims(root, stats))
        checks.extend(check_wilson_table(root, stats))
    return {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
        "summary": summarize_checks(checks),
    }


def check_claim_files(root: Path) -> list[ClaimCheck]:
    return [
        ClaimCheck(
            id=f"claim_file.{path.as_posix()}",
            ok=(root / path).exists(),
            observed=path.as_posix(),
            expected="exists",
            detail="claim-bearing document exists",
        )
        for path in CLAIM_FILES
    ]


def check_stale_patterns(root: Path) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    for path in CLAIM_FILES:
        text = read_text(root / path)
        for pattern in STALE_GLOBAL_PATTERNS:
            matches = line_matches(text, pattern)
            allowed = allowed_stale_matches(path, matches)
            unexpected = [match for match in matches if match not in allowed]
            checks.append(
                ClaimCheck(
                    id=f"stale.{path.as_posix()}.{slug(pattern.pattern)}",
                    ok=not unexpected,
                    observed=unexpected,
                    expected="no stale metric claims",
                    detail=f"{path.as_posix()} does not use outdated result pattern {pattern.pattern}",
                )
            )
    return checks


def check_overclaims(root: Path) -> list[ClaimCheck]:
    checks: list[ClaimCheck] = []
    for path in CLAIM_FILES:
        text = read_text(root / path)
        for pattern in OVERCLAIM_PATTERNS:
            matches = line_matches(text, pattern)
            unexpected = [match for match in matches if not has_negation_context(match["line"] + " " + match.get("context", ""))]
            checks.append(
                ClaimCheck(
                    id=f"overclaim.{path.as_posix()}.{slug(pattern.pattern)}",
                    ok=not unexpected,
                    observed=unexpected,
                    expected="only negated or avoided near-40% stability claims",
                    detail=f"{path.as_posix()} avoids overclaim pattern {pattern.pattern}",
                )
            )
    return checks


def check_required_current_claims(root: Path, stats: dict[str, Any]) -> list[ClaimCheck]:
    experiments = {str(row.get("name")): row for row in stats.get("experiments", [])}
    required: dict[Path, list[str]] = {
        Path("README.md"): [
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 完成局",
            "952 完成局",
        ],
        Path("PROJECT_REPORT.md"): [
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 个完成局",
            "952 个完成局",
        ],
        Path("PROJECT_ONE_PAGER.md"): [
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 完成局",
            "952 完成局",
        ],
        Path("ABLATION_STUDY.md"): [
            report_percent(experiments["Internal single RL"]["win_rate"]),
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 完成局",
            "952 完成局",
        ],
        Path("PROJECT_COMPLETION_AUDIT.md"): [
            report_percent(experiments["Internal single RL"]["win_rate"]),
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 完成局",
            "952 完成局",
        ],
        Path("RESUME_PROJECT_CARD.md"): [
            report_percent(experiments["Internal RL ensemble"]["win_rate"]),
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "495 完成局",
        ],
        Path("MODEL_CARD.md"): [
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
        ],
        Path("DEMO_GUIDE.md"): [
            report_percent(experiments["Windows desktop single RL"]["win_rate"]),
            report_percent(experiments["Windows desktop ensemble"]["win_rate"]),
            "不是稳定超过 40%",
        ],
        Path("WINDOWS_AGENT_RESULTS.md"): [
            "terminal_games: 495",
            "win_rate_completed: 40.40%",
            "total_read_recoveries: 22",
        ],
    }
    checks: list[ClaimCheck] = []
    for path, mentions in required.items():
        text = read_text(root / path)
        for mention in mentions:
            checks.append(
                ClaimCheck(
                    id=f"current_claim.{path.as_posix()}.{slug(mention)}",
                    ok=mention in text,
                    observed=mention if mention in text else None,
                    expected=mention,
                    detail=f"{path.as_posix()} carries current audited claim {mention}",
                )
            )
    return checks


def check_wilson_table(root: Path, stats: dict[str, Any]) -> list[ClaimCheck]:
    protocol = read_text(root / "EVALUATION_PROTOCOL.md")
    checks: list[ClaimCheck] = []
    for row in stats.get("experiments", []):
        name = str(row.get("name"))
        if name not in {"Internal RL ensemble", "Windows desktop single RL", "Windows desktop ensemble"}:
            continue
        expected_fragments = [
            f"{int(row['wins'])} / {int(row['games'])}",
            f"{format_percent(row['win_rate'])}",
            f"{format_percent(row['wilson_low'])} - {format_percent(row['wilson_high'])}",
        ]
        for fragment in expected_fragments:
            checks.append(
                ClaimCheck(
                    id=f"wilson.EVALUATION_PROTOCOL.md.{slug(name)}.{slug(fragment)}",
                    ok=fragment in protocol,
                    observed=fragment if fragment in protocol else None,
                    expected=fragment,
                    detail=f"EVALUATION_PROTOCOL.md Wilson table matches {name}",
                )
            )
    return checks


def allowed_stale_matches(path: Path, matches: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if path != Path("WINDOWS_AGENT_RESULTS.md"):
        return []
    return [match for match in matches if "not supported by the current local per-game" in match["line"]]


def has_negation_context(line: str) -> bool:
    lowered = line.lower()
    return any(marker in lowered for marker in NEGATION_MARKERS)


def line_matches(text: str, pattern: re.Pattern[str]) -> list[dict[str, Any]]:
    matches: list[dict[str, Any]] = []
    lines = text.splitlines()
    for index, line in enumerate(lines, start=1):
        if pattern.search(line):
            context = " ".join(item.strip() for item in lines[max(0, index - 3) : index - 1])
            matches.append({"line_number": index, "line": line.strip(), "context": context})
    return matches


def summarize_checks(checks: list[ClaimCheck]) -> dict[str, Any]:
    failed = [check.id for check in checks if not check.ok]
    return {
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_ids": failed,
    }


def resolve_path(root: Path, path: Path) -> Path:
    return path if path.is_absolute() else root / path


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_text(path: Path) -> str:
    if not path.exists():
        return ""
    return path.read_text(encoding="utf-8")


def format_percent(value: Any) -> str:
    return f"{float(value) * 100.0:.2f}%"


def report_percent(value: Any) -> str:
    scaled = float(value) * 100.0
    if abs(scaled - round(scaled)) < 1e-9:
        return f"{scaled:.1f}%"
    return f"{scaled:.2f}%"


def slug(value: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_.-]+", "_", value).strip("_").lower()


if __name__ == "__main__":
    main()
