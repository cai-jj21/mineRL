from __future__ import annotations

import argparse
import json
import math
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any


DEFAULT_CANDIDATE_EVAL = Path("artifacts/candidate_eval_200_seed0.json")
DEFAULT_ENSEMBLE_EVAL = Path("artifacts/ensemble_20_100best_eval_1000_seed0.json")
DEFAULT_DESKTOP_SINGLE_SUMMARY = Path("artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json")
DEFAULT_DESKTOP_ENSEMBLE_SUMMARY = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json")
DEFAULT_OUTPUT_JSON = Path("artifacts/report_assets/statistical_summary.json")
DEFAULT_OUTPUT_MD = Path("artifacts/report_assets/statistical_summary.md")
DEFAULT_TARGET_WIN_RATE = 0.40
Z_95 = 1.959963984540054


@dataclass(frozen=True)
class ExperimentStats:
    name: str
    source: str
    wins: int
    games: int
    win_rate: float
    wilson_low: float
    wilson_high: float
    target_win_rate: float
    point_target_pass: bool
    wilson_lower_target_pass: bool
    note: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Wilson confidence intervals for headline experiments.")
    parser.add_argument("--candidate-eval", type=Path, default=DEFAULT_CANDIDATE_EVAL)
    parser.add_argument("--ensemble-eval", type=Path, default=DEFAULT_ENSEMBLE_EVAL)
    parser.add_argument("--desktop-single-summary", type=Path, default=DEFAULT_DESKTOP_SINGLE_SUMMARY)
    parser.add_argument("--desktop-ensemble-summary", type=Path, default=DEFAULT_DESKTOP_ENSEMBLE_SUMMARY)
    parser.add_argument("--target-win-rate", type=float, default=DEFAULT_TARGET_WIN_RATE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    report = build_statistical_report(
        candidate_eval=args.candidate_eval,
        ensemble_eval=args.ensemble_eval,
        desktop_single_summary=args.desktop_single_summary,
        desktop_ensemble_summary=args.desktop_ensemble_summary,
        target_win_rate=args.target_win_rate,
    )
    write_outputs(report, output_json=args.output_json, output_md=args.output_md)
    print(
        json.dumps(
            {
                "ok": True,
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "experiments": len(report["experiments"]),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_statistical_report(
    *,
    candidate_eval: Path,
    ensemble_eval: Path,
    desktop_single_summary: Path,
    desktop_ensemble_summary: Path,
    target_win_rate: float,
) -> dict[str, Any]:
    experiments: list[ExperimentStats] = []

    if candidate_eval.exists():
        candidate = find_candidate(load_json(candidate_eval), "full_rlmix_20.pt")
        if candidate is not None:
            games = int(candidate.get("games", 0))
            wins = int(candidate.get("wins", round(float(candidate.get("win_rate", 0.0)) * games)))
            experiments.append(
                make_stats(
                    name="Internal single RL",
                    source=candidate_eval,
                    wins=wins,
                    games=games,
                    target_win_rate=target_win_rate,
                    note="Single checkpoint validation run used for candidate screening.",
                )
            )

    if ensemble_eval.exists():
        ensemble = load_json(ensemble_eval)
        experiments.append(
            make_stats(
                name="Internal RL ensemble",
                source=ensemble_eval,
                wins=int(ensemble.get("wins", 0)),
                games=int(float(ensemble.get("games", 0))),
                target_win_rate=target_win_rate,
                note="Pure RL probability ensemble in the internal simulator.",
            )
        )

    if desktop_single_summary.exists():
        single = load_json(desktop_single_summary)
        experiments.append(
            make_stats(
                name="Windows desktop single RL",
                source=desktop_single_summary,
                wins=int(single.get("wins", 0)),
                games=int(single.get("terminal_games", single.get("completed_games", 0))),
                target_win_rate=target_win_rate,
                note="Verified from available per-game Windows JSON logs.",
            )
        )

    if desktop_ensemble_summary.exists():
        desktop = load_json(desktop_ensemble_summary)
        experiments.append(
            make_stats(
                name="Windows desktop ensemble",
                source=desktop_ensemble_summary,
                wins=int(desktop.get("wins", 0)),
                games=int(desktop.get("terminal_games", desktop.get("completed_games", 0))),
                target_win_rate=target_win_rate,
                note="High-speed Windows run with the verified ten-win streak.",
            )
        )

    return {
        "schema_version": 1,
        "confidence": 0.95,
        "z": Z_95,
        "target_win_rate": target_win_rate,
        "experiments": [asdict(experiment) for experiment in experiments],
        "interpretation": [
            "point_target_pass checks only the observed point estimate.",
            "wilson_lower_target_pass is stricter and asks whether the 95% lower bound clears the target.",
            "A point estimate near 40% should be reported with sample size and confidence interval.",
        ],
    }


def make_stats(
    *,
    name: str,
    source: Path,
    wins: int,
    games: int,
    target_win_rate: float,
    note: str,
) -> ExperimentStats:
    low, high = wilson_interval(wins=wins, games=games)
    win_rate = wins / games if games else 0.0
    return ExperimentStats(
        name=name,
        source=source.as_posix(),
        wins=wins,
        games=games,
        win_rate=win_rate,
        wilson_low=low,
        wilson_high=high,
        target_win_rate=target_win_rate,
        point_target_pass=win_rate + 1e-12 >= target_win_rate,
        wilson_lower_target_pass=low + 1e-12 >= target_win_rate,
        note=note,
    )


def wilson_interval(wins: int, games: int, z: float = Z_95) -> tuple[float, float]:
    if games <= 0:
        return 0.0, 0.0
    phat = wins / games
    denominator = 1.0 + z * z / games
    center = (phat + z * z / (2.0 * games)) / denominator
    half_width = z * math.sqrt((phat * (1.0 - phat) + z * z / (4.0 * games)) / games) / denominator
    return max(0.0, center - half_width), min(1.0, center + half_width)


def write_outputs(report: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_markdown(report), encoding="utf-8")


def render_markdown(report: dict[str, Any]) -> str:
    target = float(report["target_win_rate"])
    lines = [
        "# Statistical Summary",
        "",
        f"Wilson score intervals use {float(report['confidence']) * 100:.0f}% confidence and target win rate {format_percent(target)}.",
        "",
        "| Experiment | Wins / Games | Win rate | Wilson 95% interval | Point target | Lower-bound target | Source |",
        "| --- | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for row in report["experiments"]:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row["name"]),
                    f"{int(row['wins'])} / {int(row['games'])}",
                    format_percent(row["win_rate"]),
                    f"{format_percent(row['wilson_low'])} - {format_percent(row['wilson_high'])}",
                    pass_label(row["point_target_pass"]),
                    pass_label(row["wilson_lower_target_pass"]),
                    f"`{row['source']}`",
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "Interpretation:",
            "",
            "- `Point target` means the observed win rate is at or above the target.",
            "- `Lower-bound target` is stricter: the Wilson 95% lower bound is at or above the target.",
            "- Near-threshold results should be described with their sample size and interval, not only the point estimate.",
            "",
            "Generated by `scripts/generate_statistical_report.py`.",
            "",
        ]
    )
    return "\n".join(lines)


def find_candidate(data: dict[str, Any], checkpoint_name: str) -> dict[str, Any] | None:
    for row in data.get("results", []):
        if Path(str(row.get("checkpoint", ""))).name == checkpoint_name:
            return row
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def format_percent(value: Any) -> str:
    return f"{float(value) * 100.0:.2f}%"


def pass_label(value: Any) -> str:
    return "pass" if bool(value) else "not proven"


if __name__ == "__main__":
    main()
