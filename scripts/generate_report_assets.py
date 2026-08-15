from __future__ import annotations

import argparse
import html
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class ExperimentMetric:
    name: str
    games: int
    wins: int | None
    win_rate: float
    longest_streak: int | None
    avg_seconds: float | None
    source: str


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate Markdown and SVG assets for the project report.")
    parser.add_argument("--candidate-eval", type=Path, default=Path("artifacts/candidate_eval_200_seed0.json"))
    parser.add_argument(
        "--ensemble-eval",
        type=Path,
        default=Path("artifacts/ensemble_20_100best_eval_1000_seed0.json"),
    )
    parser.add_argument(
        "--desktop-summary",
        type=Path,
        default=Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json"),
    )
    parser.add_argument(
        "--desktop-single-summary",
        type=Path,
        default=Path("artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json"),
    )
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/report_assets"))
    args = parser.parse_args()

    metrics, streak_games = build_report_data(
        candidate_eval=args.candidate_eval,
        ensemble_eval=args.ensemble_eval,
        desktop_single_summary=args.desktop_single_summary,
        desktop_summary=args.desktop_summary,
    )
    outputs = write_report_assets(metrics=metrics, streak_games=streak_games, output_dir=args.output_dir)
    print(json.dumps({"output_dir": str(args.output_dir), "outputs": outputs}, ensure_ascii=False, indent=2))


def build_report_data(
    candidate_eval: Path,
    ensemble_eval: Path,
    desktop_single_summary: Path,
    desktop_summary: Path,
) -> tuple[list[ExperimentMetric], list[dict[str, Any]]]:
    metrics: list[ExperimentMetric] = []

    if candidate_eval.exists():
        candidate_data = load_json(candidate_eval)
        single = find_candidate(candidate_data, "full_rlmix_20.pt")
        if single is not None:
            games = int(single.get("games", 0))
            win_rate = float(single.get("win_rate", 0.0))
            metrics.append(
                ExperimentMetric(
                    name="Internal single RL",
                    games=games,
                    wins=round(win_rate * games),
                    win_rate=win_rate,
                    longest_streak=none_or_int(single.get("longest_streak")),
                    avg_seconds=None,
                    source=str(candidate_eval),
                )
            )

    if ensemble_eval.exists():
        ensemble_data = load_json(ensemble_eval)
        metrics.append(
            ExperimentMetric(
                name="Internal RL ensemble",
                games=int(float(ensemble_data.get("games", 0))),
                wins=none_or_int(ensemble_data.get("wins")),
                win_rate=float(ensemble_data.get("win_rate", 0.0)),
                longest_streak=none_or_int(ensemble_data.get("longest_streak")),
                avg_seconds=None,
                source=str(ensemble_eval),
            )
        )

    if desktop_single_summary.exists():
        desktop_single = load_json(desktop_single_summary)
        terminal_games = int(desktop_single.get("terminal_games", desktop_single.get("completed_games", 0)))
        metrics.append(
            ExperimentMetric(
                name="Windows desktop single RL",
                games=terminal_games,
                wins=none_or_int(desktop_single.get("wins")),
                win_rate=float(desktop_single.get("win_rate_completed", 0.0)),
                longest_streak=none_or_int(desktop_single.get("longest_streak")),
                avg_seconds=float(desktop_single.get("avg_elapsed_seconds", 0.0)),
                source=str(desktop_single_summary),
            )
        )

    streak_games: list[dict[str, Any]] = []
    if desktop_summary.exists():
        desktop_data = load_json(desktop_summary)
        terminal_games = int(desktop_data.get("terminal_games", desktop_data.get("completed_games", 0)))
        metrics.append(
            ExperimentMetric(
                name="Windows desktop ensemble",
                games=terminal_games,
                wins=none_or_int(desktop_data.get("wins")),
                win_rate=float(desktop_data.get("win_rate_completed", 0.0)),
                longest_streak=none_or_int(desktop_data.get("longest_streak")),
                avg_seconds=float(desktop_data.get("avg_elapsed_seconds", 0.0)),
                source=str(desktop_summary),
            )
        )
        selected = desktop_data.get("selected_range", {})
        streak_games = list(selected.get("games", []))

    return metrics, streak_games


def write_report_assets(
    metrics: list[ExperimentMetric],
    streak_games: list[dict[str, Any]],
    output_dir: Path,
) -> list[str]:
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs: list[str] = []

    summary_path = output_dir / "experiment_summary.md"
    summary_path.write_text(render_experiment_table(metrics), encoding="utf-8")
    outputs.append(str(summary_path))

    json_path = output_dir / "experiment_summary.json"
    json_path.write_text(
        json.dumps(
            {
                "metrics": [metric.__dict__ for metric in metrics],
                "streak_games": streak_games,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    outputs.append(str(json_path))

    if metrics:
        win_rate_svg = render_bar_chart_svg(
            title="Win Rate Comparison",
            labels=[metric.name for metric in metrics],
            values=[metric.win_rate * 100.0 for metric in metrics],
            suffix="%",
            max_value=max(50.0, max(metric.win_rate * 100.0 for metric in metrics) * 1.15),
        )
        win_rate_path = output_dir / "win_rate_comparison.svg"
        win_rate_path.write_text(win_rate_svg, encoding="utf-8")
        outputs.append(str(win_rate_path))

        streak_metrics = [metric for metric in metrics if metric.longest_streak is not None]
        if streak_metrics:
            streak_svg = render_bar_chart_svg(
                title="Longest Win Streak",
                labels=[metric.name for metric in streak_metrics],
                values=[float(metric.longest_streak or 0) for metric in streak_metrics],
                suffix="",
                max_value=max(10.0, max(float(metric.longest_streak or 0) for metric in streak_metrics) * 1.2),
            )
            streak_path = output_dir / "longest_streak_comparison.svg"
            streak_path.write_text(streak_svg, encoding="utf-8")
            outputs.append(str(streak_path))

    if streak_games:
        seconds_svg = render_bar_chart_svg(
            title="Verified 10-Win Streak Game Times",
            labels=[str(game.get("game_index", "")) for game in streak_games],
            values=[float(game.get("elapsed_seconds", 0.0)) for game in streak_games],
            suffix="s",
            max_value=max(40.0, max(float(game.get("elapsed_seconds", 0.0)) for game in streak_games) * 1.15),
        )
        seconds_path = output_dir / "ten_streak_times.svg"
        seconds_path.write_text(seconds_svg, encoding="utf-8")
        outputs.append(str(seconds_path))

    return outputs


def render_experiment_table(metrics: list[ExperimentMetric]) -> str:
    lines = [
        "# Experiment Summary",
        "",
        "| Experiment | Games | Wins | Win rate | Longest streak | Avg seconds | Source |",
        "| --- | ---: | ---: | ---: | ---: | ---: | --- |",
    ]
    for metric in metrics:
        wins = "-" if metric.wins is None else str(metric.wins)
        streak = "-" if metric.longest_streak is None else str(metric.longest_streak)
        avg_seconds = "-" if metric.avg_seconds is None else f"{metric.avg_seconds:.2f}"
        lines.append(
            "| "
            + " | ".join(
                [
                    metric.name,
                    str(metric.games),
                    wins,
                    f"{metric.win_rate * 100.0:.2f}%",
                    streak,
                    avg_seconds,
                    f"`{metric.source}`",
                ]
            )
            + " |"
        )
    lines.append("")
    lines.append("Generated by `scripts/generate_report_assets.py`.")
    lines.append("")
    return "\n".join(lines)


def render_bar_chart_svg(
    title: str,
    labels: list[str],
    values: list[float],
    suffix: str,
    max_value: float,
) -> str:
    width = 920
    height = 420
    margin_left = 80
    margin_right = 40
    margin_top = 64
    margin_bottom = 96
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom
    count = max(1, len(values))
    slot = plot_width / count
    bar_width = min(120.0, slot * 0.58)
    max_value = max(max_value, 1e-6)

    parts = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        "text{font-family:Segoe UI,Arial,sans-serif;fill:#182026}",
        ".axis{stroke:#57606a;stroke-width:1}",
        ".grid{stroke:#d8dee4;stroke-width:1}",
        ".bar{fill:#2f81f7}",
        ".label{font-size:13px}",
        ".value{font-size:14px;font-weight:600}",
        ".title{font-size:22px;font-weight:700}",
        "</style>",
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        f'<text class="title" x="{width / 2:.1f}" y="34" text-anchor="middle">{escape(title)}</text>',
    ]

    for fraction in (0.0, 0.25, 0.5, 0.75, 1.0):
        y = margin_top + plot_height * (1.0 - fraction)
        value = max_value * fraction
        parts.append(f'<line class="grid" x1="{margin_left}" y1="{y:.1f}" x2="{width - margin_right}" y2="{y:.1f}"/>')
        parts.append(f'<text class="label" x="{margin_left - 12}" y="{y + 4:.1f}" text-anchor="end">{value:.0f}{escape(suffix)}</text>')

    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}"/>')
    parts.append(f'<line class="axis" x1="{margin_left}" y1="{margin_top + plot_height}" x2="{width - margin_right}" y2="{margin_top + plot_height}"/>')

    for index, (label, value) in enumerate(zip(labels, values)):
        bar_height = plot_height * max(0.0, value) / max_value
        x = margin_left + slot * index + (slot - bar_width) / 2.0
        y = margin_top + plot_height - bar_height
        parts.append(f'<rect class="bar" x="{x:.1f}" y="{y:.1f}" width="{bar_width:.1f}" height="{bar_height:.1f}" rx="3"/>')
        parts.append(f'<text class="value" x="{x + bar_width / 2:.1f}" y="{y - 8:.1f}" text-anchor="middle">{value:.2f}{escape(suffix)}</text>')
        parts.append(
            f'<text class="label" x="{x + bar_width / 2:.1f}" y="{margin_top + plot_height + 28:.1f}" '
            f'text-anchor="middle">{escape(label)}</text>'
        )

    parts.append("</svg>")
    return "\n".join(parts)


def find_candidate(data: dict[str, Any], checkpoint_name: str) -> dict[str, Any] | None:
    for row in data.get("results", []):
        checkpoint = Path(str(row.get("checkpoint", ""))).name
        if checkpoint == checkpoint_name:
            return row
    return None


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def none_or_int(value: Any) -> int | None:
    if value is None:
        return None
    return int(float(value))


def escape(value: object) -> str:
    return html.escape(str(value), quote=True)


if __name__ == "__main__":
    main()
