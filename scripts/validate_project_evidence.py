from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_EVAL = Path("artifacts/candidate_eval_200_seed0.json")
DEFAULT_ENSEMBLE_EVAL = Path("artifacts/ensemble_20_100best_eval_1000_seed0.json")
DEFAULT_DESKTOP_SUMMARY = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json")
DEFAULT_REPORT_SUMMARY = Path("artifacts/report_assets/experiment_summary.json")


@dataclass(frozen=True)
class EvidenceThresholds:
    single_checkpoint: str = "full_rlmix_20.pt"
    min_single_games: int = 200
    min_single_win_rate: float = 0.40
    min_single_longest_streak: int = 7
    min_ensemble_games: int = 1000
    min_ensemble_wins: int = 430
    min_ensemble_win_rate: float = 0.43
    min_ensemble_longest_streak: int = 9
    min_desktop_terminal_games: int = 900
    min_desktop_wins: int = 370
    min_desktop_win_rate: float = 0.39
    min_desktop_longest_streak: int = 10
    max_desktop_avg_seconds: float = 35.0
    expected_streak_start: int = 747
    expected_streak_end: int = 756
    expected_streak_wins: int = 10
    max_execution_anomalies: int = 0


@dataclass(frozen=True)
class EvidencePaths:
    candidate_eval: Path = DEFAULT_CANDIDATE_EVAL
    ensemble_eval: Path = DEFAULT_ENSEMBLE_EVAL
    desktop_summary: Path = DEFAULT_DESKTOP_SUMMARY
    report_summary: Path = DEFAULT_REPORT_SUMMARY


@dataclass
class EvidenceCheck:
    id: str
    ok: bool
    observed: Any
    expected: Any
    detail: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate the project report evidence against local JSON artifacts.")
    parser.add_argument("--candidate-eval", type=Path, default=DEFAULT_CANDIDATE_EVAL)
    parser.add_argument("--ensemble-eval", type=Path, default=DEFAULT_ENSEMBLE_EVAL)
    parser.add_argument("--desktop-summary", type=Path, default=DEFAULT_DESKTOP_SUMMARY)
    parser.add_argument("--report-summary", type=Path, default=DEFAULT_REPORT_SUMMARY)
    parser.add_argument("--single-checkpoint", default=EvidenceThresholds.single_checkpoint)
    parser.add_argument("--min-single-win-rate", type=float, default=EvidenceThresholds.min_single_win_rate)
    parser.add_argument("--min-ensemble-win-rate", type=float, default=EvidenceThresholds.min_ensemble_win_rate)
    parser.add_argument("--min-desktop-win-rate", type=float, default=EvidenceThresholds.min_desktop_win_rate)
    parser.add_argument("--max-desktop-avg-seconds", type=float, default=EvidenceThresholds.max_desktop_avg_seconds)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    paths = EvidencePaths(
        candidate_eval=args.candidate_eval,
        ensemble_eval=args.ensemble_eval,
        desktop_summary=args.desktop_summary,
        report_summary=args.report_summary,
    )
    thresholds = EvidenceThresholds(
        single_checkpoint=args.single_checkpoint,
        min_single_win_rate=args.min_single_win_rate,
        min_ensemble_win_rate=args.min_ensemble_win_rate,
        min_desktop_win_rate=args.min_desktop_win_rate,
        max_desktop_avg_seconds=args.max_desktop_avg_seconds,
    )
    report = validate_project_evidence(paths=paths, thresholds=thresholds)
    text = json.dumps(report, ensure_ascii=False, indent=2)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text + "\n", encoding="utf-8")
    print(text)
    raise SystemExit(0 if report["ok"] else 1)


def validate_project_evidence(paths: EvidencePaths, thresholds: EvidenceThresholds) -> dict[str, Any]:
    checks: list[EvidenceCheck] = []
    data: dict[str, dict[str, Any]] = {}

    for key, path in [
        ("candidate_eval", paths.candidate_eval),
        ("ensemble_eval", paths.ensemble_eval),
        ("desktop_summary", paths.desktop_summary),
        ("report_summary", paths.report_summary),
    ]:
        loaded = load_optional_json(path)
        data[key] = loaded or {}
        checks.append(
            EvidenceCheck(
                id=f"{key}.exists",
                ok=loaded is not None,
                observed=str(path),
                expected="existing UTF-8 JSON file",
                detail="source artifact is available",
            )
        )

    if data["candidate_eval"]:
        checks.extend(validate_single_candidate(data["candidate_eval"], thresholds))
    if data["ensemble_eval"]:
        checks.extend(validate_ensemble_eval(data["ensemble_eval"], thresholds))
    if data["desktop_summary"]:
        checks.extend(validate_desktop_summary(data["desktop_summary"], thresholds))
    if data["report_summary"] and data["candidate_eval"] and data["ensemble_eval"] and data["desktop_summary"]:
        checks.extend(validate_report_summary(data, thresholds))

    return {
        "ok": all(check.ok for check in checks),
        "checks": [asdict(check) for check in checks],
        "summary": summarize_checks(checks),
    }


def validate_single_candidate(data: dict[str, Any], thresholds: EvidenceThresholds) -> list[EvidenceCheck]:
    row = find_candidate(data, thresholds.single_checkpoint)
    if row is None:
        return [
            EvidenceCheck(
                id="single_rl_checkpoint.present",
                ok=False,
                observed=[Path(str(item.get("checkpoint", ""))).name for item in data.get("results", [])],
                expected=thresholds.single_checkpoint,
                detail="single-model evaluation row is present",
            )
        ]

    games = int(row.get("games", 0))
    win_rate = float(row.get("win_rate", 0.0))
    longest_streak = int(row.get("longest_streak", 0))
    return [
        EvidenceCheck(
            id="single_rl.games",
            ok=games >= thresholds.min_single_games,
            observed=games,
            expected=f">= {thresholds.min_single_games}",
            detail="single RL candidate has enough evaluation games",
        ),
        EvidenceCheck(
            id="single_rl.win_rate",
            ok=win_rate + 1e-12 >= thresholds.min_single_win_rate,
            observed=win_rate,
            expected=f">= {thresholds.min_single_win_rate}",
            detail="single RL candidate clears the report threshold",
        ),
        EvidenceCheck(
            id="single_rl.longest_streak",
            ok=longest_streak >= thresholds.min_single_longest_streak,
            observed=longest_streak,
            expected=f">= {thresholds.min_single_longest_streak}",
            detail="single RL candidate retains a meaningful streak",
        ),
    ]


def validate_ensemble_eval(data: dict[str, Any], thresholds: EvidenceThresholds) -> list[EvidenceCheck]:
    games = int(float(data.get("games", 0)))
    wins = int(data.get("wins", 0))
    win_rate = float(data.get("win_rate", 0.0))
    longest_streak = int(data.get("longest_streak", 0))
    return [
        EvidenceCheck(
            id="ensemble_rl.games",
            ok=games >= thresholds.min_ensemble_games,
            observed=games,
            expected=f">= {thresholds.min_ensemble_games}",
            detail="ensemble internal evaluation has enough games",
        ),
        EvidenceCheck(
            id="ensemble_rl.wins",
            ok=wins >= thresholds.min_ensemble_wins,
            observed=wins,
            expected=f">= {thresholds.min_ensemble_wins}",
            detail="ensemble internal evaluation reaches the documented win count",
        ),
        EvidenceCheck(
            id="ensemble_rl.win_rate",
            ok=win_rate + 1e-12 >= thresholds.min_ensemble_win_rate,
            observed=win_rate,
            expected=f">= {thresholds.min_ensemble_win_rate}",
            detail="ensemble internal evaluation reaches the documented win rate",
        ),
        EvidenceCheck(
            id="ensemble_rl.longest_streak",
            ok=longest_streak >= thresholds.min_ensemble_longest_streak,
            observed=longest_streak,
            expected=f">= {thresholds.min_ensemble_longest_streak}",
            detail="ensemble internal evaluation reaches the documented streak",
        ),
    ]


def validate_desktop_summary(data: dict[str, Any], thresholds: EvidenceThresholds) -> list[EvidenceCheck]:
    terminal_games = int(data.get("terminal_games", data.get("completed_games", 0)))
    wins = int(data.get("wins", 0))
    win_rate = float(data.get("win_rate_completed", 0.0))
    longest_streak = int(data.get("longest_streak", 0))
    avg_seconds = float(data.get("avg_elapsed_seconds", 0.0))
    anomaly_fields = [
        "total_reclicks",
        "total_click_unready_actions",
        "total_unconfirmed_open_actions",
        "total_read_recoveries",
    ]
    anomaly_total = sum(int(data.get(field, 0)) for field in anomaly_fields)
    selected = data.get("selected_range", {})
    selected_games = list(selected.get("games", []))
    selected_wins = sum(1 for game in selected_games if game.get("won"))
    selected_anomalies = sum(
        int(game.get("reclicks", 0) or 0)
        + int(game.get("click_unready_actions", 0) or 0)
        + int(game.get("unconfirmed_open_actions", 0) or 0)
        + int(game.get("read_recoveries", 0) or 0)
        for game in selected_games
    )

    return [
        EvidenceCheck(
            id="desktop.terminal_games",
            ok=terminal_games >= thresholds.min_desktop_terminal_games,
            observed=terminal_games,
            expected=f">= {thresholds.min_desktop_terminal_games}",
            detail="desktop evaluation has enough terminal games",
        ),
        EvidenceCheck(
            id="desktop.wins",
            ok=wins >= thresholds.min_desktop_wins,
            observed=wins,
            expected=f">= {thresholds.min_desktop_wins}",
            detail="desktop evaluation reaches the documented win count",
        ),
        EvidenceCheck(
            id="desktop.win_rate",
            ok=win_rate + 1e-12 >= thresholds.min_desktop_win_rate,
            observed=win_rate,
            expected=f">= {thresholds.min_desktop_win_rate}",
            detail="desktop evaluation reaches the report threshold",
        ),
        EvidenceCheck(
            id="desktop.longest_streak",
            ok=longest_streak >= thresholds.min_desktop_longest_streak,
            observed=longest_streak,
            expected=f">= {thresholds.min_desktop_longest_streak}",
            detail="desktop evaluation contains the verified streak",
        ),
        EvidenceCheck(
            id="desktop.avg_seconds",
            ok=avg_seconds <= thresholds.max_desktop_avg_seconds,
            observed=avg_seconds,
            expected=f"<= {thresholds.max_desktop_avg_seconds}",
            detail="desktop execution is fast enough for the report claim",
        ),
        EvidenceCheck(
            id="desktop.execution_anomalies",
            ok=anomaly_total <= thresholds.max_execution_anomalies,
            observed={field: int(data.get(field, 0)) for field in anomaly_fields},
            expected=f"total <= {thresholds.max_execution_anomalies}",
            detail="desktop run has no aggregate execution anomalies",
        ),
        EvidenceCheck(
            id="desktop.streak_range",
            ok=selected.get("start") == thresholds.expected_streak_start
            and selected.get("end") == thresholds.expected_streak_end,
            observed={"start": selected.get("start"), "end": selected.get("end")},
            expected={"start": thresholds.expected_streak_start, "end": thresholds.expected_streak_end},
            detail="selected ten-win streak range matches the report",
        ),
        EvidenceCheck(
            id="desktop.streak_wins",
            ok=len(selected_games) == thresholds.expected_streak_wins and selected_wins == thresholds.expected_streak_wins,
            observed={"games": len(selected_games), "wins": selected_wins},
            expected={"games": thresholds.expected_streak_wins, "wins": thresholds.expected_streak_wins},
            detail="selected streak contains exactly ten wins",
        ),
        EvidenceCheck(
            id="desktop.streak_execution_anomalies",
            ok=selected_anomalies <= thresholds.max_execution_anomalies,
            observed=selected_anomalies,
            expected=f"<= {thresholds.max_execution_anomalies}",
            detail="selected ten-win streak has no execution anomalies",
        ),
    ]


def validate_report_summary(data: dict[str, dict[str, Any]], thresholds: EvidenceThresholds) -> list[EvidenceCheck]:
    metrics = {str(metric.get("name")): metric for metric in data["report_summary"].get("metrics", [])}
    candidate = find_candidate(data["candidate_eval"], thresholds.single_checkpoint) or {}
    ensemble = data["ensemble_eval"]
    desktop = data["desktop_summary"]

    expected = {
        "Internal single RL": {
            "games": int(candidate.get("games", 0)),
            "win_rate": float(candidate.get("win_rate", 0.0)),
            "longest_streak": int(candidate.get("longest_streak", 0)),
        },
        "Internal RL ensemble": {
            "games": int(float(ensemble.get("games", 0))),
            "wins": int(ensemble.get("wins", 0)),
            "win_rate": float(ensemble.get("win_rate", 0.0)),
            "longest_streak": int(ensemble.get("longest_streak", 0)),
        },
        "Windows desktop ensemble": {
            "games": int(desktop.get("terminal_games", desktop.get("completed_games", 0))),
            "wins": int(desktop.get("wins", 0)),
            "win_rate": float(desktop.get("win_rate_completed", 0.0)),
            "longest_streak": int(desktop.get("longest_streak", 0)),
            "avg_seconds": float(desktop.get("avg_elapsed_seconds", 0.0)),
        },
    }

    checks: list[EvidenceCheck] = []
    checks.append(
        EvidenceCheck(
            id="report.metrics.present",
            ok=set(expected).issubset(metrics),
            observed=sorted(metrics),
            expected=sorted(expected),
            detail="report asset contains all headline experiments",
        )
    )
    for name, expected_values in expected.items():
        observed = metrics.get(name, {})
        comparisons: dict[str, bool] = {}
        for key, expected_value in expected_values.items():
            comparisons[key] = values_close(observed.get(key), expected_value)
        checks.append(
            EvidenceCheck(
                id=f"report.metrics.{slug(name)}",
                ok=bool(observed) and all(comparisons.values()),
                observed={key: observed.get(key) for key in expected_values},
                expected=expected_values,
                detail=f"report metric for {name} matches source artifacts",
            )
        )

    selected = data["desktop_summary"].get("selected_range", {})
    report_streak_games = list(data["report_summary"].get("streak_games", []))
    checks.append(
        EvidenceCheck(
            id="report.streak_games",
            ok=[game.get("game_index") for game in report_streak_games]
            == [game.get("game_index") for game in selected.get("games", [])],
            observed=[game.get("game_index") for game in report_streak_games],
            expected=[game.get("game_index") for game in selected.get("games", [])],
            detail="report asset carries the same ten-win streak games",
        )
    )
    return checks


def summarize_checks(checks: list[EvidenceCheck]) -> dict[str, Any]:
    failed = [check.id for check in checks if not check.ok]
    return {
        "total": len(checks),
        "passed": len(checks) - len(failed),
        "failed": len(failed),
        "failed_ids": failed,
    }


def find_candidate(data: dict[str, Any], checkpoint_name: str) -> dict[str, Any] | None:
    for row in data.get("results", []):
        checkpoint = Path(str(row.get("checkpoint", ""))).name
        if checkpoint == checkpoint_name:
            return row
    return None


def load_optional_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def values_close(observed: Any, expected: Any, tolerance: float = 1e-9) -> bool:
    if observed is None:
        return False
    if isinstance(expected, float):
        return abs(float(observed) - expected) <= tolerance
    return observed == expected


def slug(value: str) -> str:
    return value.lower().replace(" ", "_")


if __name__ == "__main__":
    main()
