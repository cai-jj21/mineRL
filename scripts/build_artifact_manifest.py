from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_EVAL = Path("artifacts/candidate_eval_200_seed0.json")
DEFAULT_ENSEMBLE_EVAL = Path("artifacts/ensemble_20_100best_eval_1000_seed0.json")
DEFAULT_DESKTOP_SINGLE_SUMMARY = Path("artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json")
DEFAULT_DESKTOP_SUMMARY = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json")
DEFAULT_REPORT_DIR = Path("artifacts/report_assets")
DEFAULT_STREAK_DIR = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000")
DEFAULT_OUTPUT_JSON = DEFAULT_REPORT_DIR / "artifact_manifest.json"
DEFAULT_OUTPUT_MD = Path("EXPERIMENT_MANIFEST.md")
DEFAULT_FAILURE_ANALYSIS_MD = Path("FAILURE_ANALYSIS.md")


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Build a checksum manifest for report and experiment artifacts.")
    parser.add_argument("--candidate-eval", type=Path, default=DEFAULT_CANDIDATE_EVAL)
    parser.add_argument("--ensemble-eval", type=Path, default=DEFAULT_ENSEMBLE_EVAL)
    parser.add_argument("--desktop-single-summary", type=Path, default=DEFAULT_DESKTOP_SINGLE_SUMMARY)
    parser.add_argument("--desktop-summary", type=Path, default=DEFAULT_DESKTOP_SUMMARY)
    parser.add_argument("--report-dir", type=Path, default=DEFAULT_REPORT_DIR)
    parser.add_argument("--failure-analysis-md", type=Path, default=DEFAULT_FAILURE_ANALYSIS_MD)
    parser.add_argument("--streak-dir", type=Path, default=DEFAULT_STREAK_DIR)
    parser.add_argument("--streak-start", type=int, default=747)
    parser.add_argument("--streak-end", type=int, default=756)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    paths = default_artifact_paths(
        candidate_eval=args.candidate_eval,
        ensemble_eval=args.ensemble_eval,
        desktop_single_summary=args.desktop_single_summary,
        desktop_summary=args.desktop_summary,
        report_dir=args.report_dir,
        failure_analysis_md=args.failure_analysis_md,
        streak_dir=args.streak_dir,
        streak_start=args.streak_start,
        streak_end=args.streak_end,
    )
    manifest = build_manifest(
        paths=paths,
        candidate_eval=args.candidate_eval,
        ensemble_eval=args.ensemble_eval,
        desktop_single_summary=args.desktop_single_summary,
        desktop_summary=args.desktop_summary,
        report_dir=args.report_dir,
        failure_analysis_md=args.failure_analysis_md,
        streak_start=args.streak_start,
        streak_end=args.streak_end,
    )
    write_manifest(manifest=manifest, output_json=args.output_json, output_md=args.output_md)
    print(
        json.dumps(
            {
                "ok": manifest["ok"],
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "artifacts": len(manifest["artifacts"]),
                "missing": manifest["missing"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    raise SystemExit(0 if manifest["ok"] else 1)


def default_artifact_paths(
    candidate_eval: Path,
    ensemble_eval: Path,
    desktop_single_summary: Path,
    desktop_summary: Path,
    report_dir: Path,
    failure_analysis_md: Path,
    streak_dir: Path,
    streak_start: int,
    streak_end: int,
) -> list[Path]:
    report_files = [
        report_dir / "experiment_summary.md",
        report_dir / "experiment_summary.json",
        report_dir / "evidence_validation.json",
        report_dir / "win_rate_comparison.svg",
        report_dir / "longest_streak_comparison.svg",
        report_dir / "ten_streak_times.svg",
        report_dir / "ten_streak_review.md",
        report_dir / "statistical_summary.json",
        report_dir / "statistical_summary.md",
        report_dir / "claim_audit.json",
        report_dir / "failure_analysis_summary.json",
        failure_analysis_md,
    ]
    streak_files = [streak_dir / f"game_{index:03d}.json" for index in range(streak_start, streak_end + 1)]
    return [candidate_eval, ensemble_eval, desktop_single_summary, desktop_summary, *report_files, *streak_files]


def build_manifest(
    paths: list[Path],
    candidate_eval: Path,
    ensemble_eval: Path,
    desktop_single_summary: Path,
    desktop_summary: Path,
    report_dir: Path,
    failure_analysis_md: Path,
    streak_start: int,
    streak_end: int,
) -> dict[str, Any]:
    artifacts = [file_record(path) for path in paths]
    missing = [record["path"] for record in artifacts if not record["exists"]]
    return {
        "schema_version": 1,
        "ok": not missing,
        "missing": missing,
        "metrics": extract_metrics(
            candidate_eval=candidate_eval,
            ensemble_eval=ensemble_eval,
            desktop_single_summary=desktop_single_summary,
            desktop_summary=desktop_summary,
            report_dir=report_dir,
            failure_analysis_md=failure_analysis_md,
            streak_start=streak_start,
            streak_end=streak_end,
        ),
        "artifacts": artifacts,
    }


def write_manifest(manifest: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_manifest_markdown(manifest), encoding="utf-8")


def file_record(path: Path) -> dict[str, Any]:
    exists = path.exists()
    record: dict[str, Any] = {
        "path": normalize_path(path),
        "exists": exists,
        "size_bytes": path.stat().st_size if exists else None,
        "sha256": sha256_file(path) if exists else None,
    }
    return record


def extract_metrics(
    candidate_eval: Path,
    ensemble_eval: Path,
    desktop_single_summary: Path,
    desktop_summary: Path,
    report_dir: Path,
    failure_analysis_md: Path,
    streak_start: int,
    streak_end: int,
) -> dict[str, Any]:
    candidate_data = load_json(candidate_eval)
    ensemble_data = load_json(ensemble_eval)
    desktop_single_data = load_json(desktop_single_summary)
    desktop_data = load_json(desktop_summary)
    validation_data = load_json(report_dir / "evidence_validation.json")
    failure_data = load_json(report_dir / "failure_analysis_summary.json")
    single = find_candidate(candidate_data, "full_rlmix_20.pt") if candidate_data else None
    selected = desktop_data.get("selected_range", {}) if desktop_data else {}
    selected_games = list(selected.get("games", []))

    return {
        "single_rl": {
            "checkpoint": "full_rlmix_20.pt",
            "games": int(single.get("games", 0)) if single else None,
            "wins": round(float(single.get("win_rate", 0.0)) * int(single.get("games", 0))) if single else None,
            "win_rate": float(single.get("win_rate", 0.0)) if single else None,
            "longest_streak": int(single.get("longest_streak", 0)) if single else None,
        },
        "internal_ensemble": {
            "games": int(float(ensemble_data.get("games", 0))) if ensemble_data else None,
            "wins": int(ensemble_data.get("wins", 0)) if ensemble_data else None,
            "win_rate": float(ensemble_data.get("win_rate", 0.0)) if ensemble_data else None,
            "longest_streak": int(ensemble_data.get("longest_streak", 0)) if ensemble_data else None,
        },
        "windows_desktop": {
            "single_terminal_games": int(desktop_single_data.get("terminal_games", 0)) if desktop_single_data else None,
            "single_wins": int(desktop_single_data.get("wins", 0)) if desktop_single_data else None,
            "single_win_rate_completed": float(desktop_single_data.get("win_rate_completed", 0.0))
            if desktop_single_data
            else None,
            "terminal_games": int(desktop_data.get("terminal_games", 0)) if desktop_data else None,
            "wins": int(desktop_data.get("wins", 0)) if desktop_data else None,
            "win_rate_completed": float(desktop_data.get("win_rate_completed", 0.0)) if desktop_data else None,
            "longest_streak": int(desktop_data.get("longest_streak", 0)) if desktop_data else None,
            "avg_elapsed_seconds": float(desktop_data.get("avg_elapsed_seconds", 0.0)) if desktop_data else None,
            "total_reclicks": int(desktop_data.get("total_reclicks", 0)) if desktop_data else None,
            "total_unconfirmed_open_actions": int(desktop_data.get("total_unconfirmed_open_actions", 0))
            if desktop_data
            else None,
            "total_read_recoveries": int(desktop_data.get("total_read_recoveries", 0)) if desktop_data else None,
        },
        "verified_streak": {
            "start": selected.get("start") if selected else None,
            "end": selected.get("end") if selected else None,
            "expected_start": streak_start,
            "expected_end": streak_end,
            "games": len(selected_games),
            "wins": sum(1 for game in selected_games if game.get("won")),
            "avg_elapsed_seconds": mean(
                [float(game.get("elapsed_seconds", 0.0)) for game in selected_games if game.get("elapsed_seconds")]
            ),
        },
        "evidence_validation": validation_data.get("summary") if validation_data else None,
        "failure_analysis": summarize_failure_metrics(failure_data, failure_analysis_md),
    }


def render_manifest_markdown(manifest: dict[str, Any]) -> str:
    metrics = manifest.get("metrics", {})
    desktop = metrics.get("windows_desktop", {})
    ensemble = metrics.get("internal_ensemble", {})
    single = metrics.get("single_rl", {})
    streak = metrics.get("verified_streak", {})
    validation = metrics.get("evidence_validation") or {}
    failure = metrics.get("failure_analysis") or {}

    lines = [
        "# 实验产物清单",
        "",
        "该清单记录项目报告所依赖的本地实验产物、核心指标和 SHA-256 校验值。`artifacts/` 默认被 git 忽略，因此此文件用于保留可提交的证据索引。",
        "",
        "## 核心指标",
        "",
        "| 证据 | 指标 |",
        "| --- | --- |",
        f"| 单模型内部评估 | {single.get('games', '-')} 局，胜率 {format_percent(single.get('win_rate'))}，最长 {single.get('longest_streak', '-')} 连胜 |",
        f"| 内部 RL 集成 | {ensemble.get('games', '-')} 局，{ensemble.get('wins', '-')} 胜，胜率 {format_percent(ensemble.get('win_rate'))}，最长 {ensemble.get('longest_streak', '-')} 连胜 |",
        f"| Windows 桌面集成 | {desktop.get('terminal_games', '-')} 完成局，{desktop.get('wins', '-')} 胜，胜率 {format_percent(desktop.get('win_rate_completed'))}，平均 {format_seconds(desktop.get('avg_elapsed_seconds'))} |",
        f"| 十连胜区间 | game_{int_or_dash(streak.get('start'))} 到 game_{int_or_dash(streak.get('end'))}，{streak.get('wins', '-')} / {streak.get('games', '-')} 胜 |",
        f"| 证据校验 | {validation.get('passed', '-')} / {validation.get('total', '-')} 通过，失败 {validation.get('failed', '-')} 项 |",
        f"| 失败分析 | {failure.get('games', '-')} 局对照，{failure.get('losses', '-')} 输局，错旗输局 {format_percent(failure.get('wrong_flag_loss_rate'))}，终局 forced move {format_percent(failure.get('terminal_forced_rate'))} |",
        "",
        "## 产物哈希",
        "",
        "| Path | Size | SHA-256 |",
        "| --- | ---: | --- |",
    ]
    for record in manifest.get("artifacts", []):
        size = "-" if record.get("size_bytes") is None else str(record["size_bytes"])
        sha = "-" if record.get("sha256") is None else record["sha256"]
        lines.append(f"| `{record['path']}` | {size} | `{sha}` |")
    lines.extend(
        [
            "",
            "重新生成：",
            "",
            "```powershell",
            "python scripts/build_artifact_manifest.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def find_candidate(data: dict[str, Any], checkpoint_name: str) -> dict[str, Any] | None:
    for row in data.get("results", []):
        checkpoint = Path(str(row.get("checkpoint", ""))).name
        if checkpoint == checkpoint_name:
            return row
    return None


def summarize_failure_metrics(data: dict[str, Any], failure_analysis_md: Path) -> dict[str, Any] | None:
    if not data:
        return None
    loss = data.get("loss_analysis", {})
    return {
        "path": normalize_path(failure_analysis_md),
        "games": loss.get("games"),
        "losses": loss.get("losses"),
        "wrong_flag_loss_rate": loss.get("wrong_flag_loss_rate"),
        "terminal_forced_rate": loss.get("terminal_forced_rate"),
        "avg_target_risk": loss.get("avg_target_risk"),
        "avg_best_guess_risk": loss.get("avg_best_guess_risk"),
    }


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def normalize_path(path: Path) -> str:
    return path.as_posix()


def mean(values: list[float]) -> float | None:
    if not values:
        return None
    return sum(values) / len(values)


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f} 秒"


def int_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value):03d}"


if __name__ == "__main__":
    main()
