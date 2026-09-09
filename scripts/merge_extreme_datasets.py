from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import load_extreme_dataset, save_extreme_dataset


def main() -> None:
    parser = argparse.ArgumentParser(description="Merge compatible extreme replay datasets.")
    parser.add_argument("--input", dest="inputs", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--family-filter",
        action="append",
        default=[],
        help="Optionally keep only specific extreme families. Repeat to include multiple families.",
    )
    args = parser.parse_args()

    transitions = []
    reports = []
    shapes = set()
    family_filter = set(args.family_filter or [])
    for path in args.inputs:
        rows, report = load_extreme_dataset(path)
        if not rows:
            continue
        shapes.add((tuple(rows[0].board.shape), tuple(rows[0].action_mask.shape)))
        selected_rows = [
            row
            for row in rows
            if not family_filter or (row.extreme_family or "unclassified") in family_filter
        ]
        transitions.extend(selected_rows)
        reports.append(
            {
                "path": str(path),
                "records": len(rows),
                "selected_records": len(selected_rows),
                "manifest": report,
            }
        )

    if len(shapes) > 1:
        raise ValueError(f"incompatible dataset shapes: {sorted(shapes)}")
    if args.max_records > 0:
        transitions = transitions[: args.max_records]
    if not transitions:
        raise RuntimeError("no transitions were loaded")

    families = Counter(row.extreme_family or "unclassified" for row in transitions)
    manifest = save_extreme_dataset(
        args.output,
        transitions,
        metadata={
            "source": "merged_extreme_replay_assets",
            "inputs": [str(path) for path in args.inputs],
            "family_filter": sorted(family_filter),
            "input_reports": reports,
            "records": len(transitions),
            "families": dict(sorted(families.items())),
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "records": len(transitions),
                "families": dict(sorted(families.items())),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
