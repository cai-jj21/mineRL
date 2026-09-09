from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize gated specialist evaluation results.")
    parser.add_argument("--result", dest="results", action="append", type=Path, required=True)
    parser.add_argument("--output-json", type=Path, required=True)
    parser.add_argument("--output-md", type=Path, required=True)
    args = parser.parse_args()

    rows = []
    for path in args.results:
        payload = json.loads(path.read_text(encoding="utf-8"))
        rows.append(
            {
                "path": str(path),
                "win_rate": float(payload.get("win_rate", 0.0)),
                "wins": int(payload.get("wins", 0)),
                "games": int(payload.get("games", 0)),
                "longest_streak": int(payload.get("longest_streak", 0)),
                "specialist_checkpoint": payload.get("specialist_checkpoint"),
                "specialist_checkpoints": payload.get("specialist_checkpoints"),
                "base_checkpoints": payload.get("base_checkpoints", payload.get("base_checkpoint")),
                "safe_left_threshold": payload.get("safe_left_threshold"),
                "specialist_weight": payload.get("specialist_weight"),
                "blend_scope": payload.get("blend_scope", "all"),
                "specialist_decision_fraction": payload.get("specialist_decision_fraction"),
                "specialist_applied_fraction": payload.get("specialist_applied_fraction"),
            }
        )
    rows.sort(key=lambda row: (row["win_rate"], row["wins"]), reverse=True)
    summary = {"best": rows[0] if rows else None, "results": rows}

    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.write_text(render_markdown(summary), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


def render_markdown(summary: dict[str, Any]) -> str:
    lines = [
        "# Specialist Evaluation Summary",
        "",
        "| rank | win rate | wins/games | streak | checkpoint | gate | weight | scope | decision fraction | result |",
        "|---:|---:|---:|---:|---|---:|---:|---|---:|---|",
    ]
    for index, row in enumerate(summary["results"], start=1):
        fraction = row["specialist_decision_fraction"]
        applied_fraction = row.get("specialist_applied_fraction")
        specialist_name = row.get("specialist_checkpoints") or row.get("specialist_checkpoint")
        lines.append(
            "| "
            + " | ".join(
                [
                    str(index),
                    f"{row['win_rate']:.3f}",
                    f"{row['wins']}/{row['games']}",
                    str(row["longest_streak"]),
                    str(specialist_name),
                    str(row["safe_left_threshold"]),
                    str(row["specialist_weight"]),
                    str(row["blend_scope"]),
                    (
                        "" if fraction is None else f"{float(fraction):.3f}"
                    )
                    + (
                        "" if applied_fraction is None else f" / applied {float(applied_fraction):.4f}"
                    ),
                    str(row["path"]),
                ]
            )
            + " |"
        )
    lines.append("")
    return "\n".join(lines)


if __name__ == "__main__":
    main()
