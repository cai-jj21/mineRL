from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean, median
from typing import Any, Iterable

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel
from minesweeper_rl.types import ActionType, EpisodeTransition


SAFE_LEFT_BUCKETS = (
    (0, 60, "000_060"),
    (61, 140, "061_140"),
    (141, 240, "141_240"),
    (241, 391, "241_391"),
)
OPEN_CANDIDATE_BUCKETS = (
    (0, 20, "000_020"),
    (21, 80, "021_080"),
    (81, 200, "081_200"),
    (201, 480, "201_480"),
)
REGRET_BUCKETS = (
    (float("-inf"), 1.0, "lt_1"),
    (1.0, 4.0, "1_4"),
    (4.0, 8.0, "4_8"),
    (8.0, float("inf"), "gte_8"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Profile extreme replay datasets for data-asset driven training."
    )
    parser.add_argument("--dataset", action="append", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--records-output",
        type=Path,
        default=None,
        help="Optional CSV path for one row per transition.",
    )
    parser.add_argument("--total-safe", type=int, default=391)
    args = parser.parse_args()

    report = profile_datasets(args.dataset, total_safe=int(args.total_safe))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    if args.records_output is not None:
        write_record_csv(args.records_output, report["records"])

    compact = {
        "ok": True,
        "output": str(args.output),
        "records_output": None if args.records_output is None else str(args.records_output),
        "datasets": report["datasets"],
        "record_count": report["summary"]["record_count"],
        "families": report["summary"]["families"],
        "regions": report["summary"]["regions"],
        "avg_behavior_regret": report["summary"]["avg_behavior_regret"],
        "high_regret_rate": report["summary"]["high_regret_rate"],
    }
    print(json.dumps(compact, ensure_ascii=False, indent=2))


def profile_datasets(paths: Iterable[Path], *, total_safe: int = 391) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    dataset_reports: list[dict[str, Any]] = []
    for path in paths:
        transitions, manifest = load_extreme_dataset(path)
        dataset_reports.append(
            {
                "path": str(path),
                "records": len(transitions),
                "manifest_records": manifest.get("records"),
                "manifest_families": manifest.get("families", {}),
                "manifest_label_summary": manifest.get("label_summary", {}),
            }
        )
        for index, transition in enumerate(transitions):
            rows.append(profile_transition(transition, dataset=str(path), index=index, total_safe=total_safe))

    return {
        "datasets": dataset_reports,
        "summary": summarize_rows(rows),
        "by_family": summarize_groups(rows, "family"),
        "by_region": summarize_groups(rows, "region"),
        "by_safe_left_bucket": summarize_groups(rows, "safe_left_bucket"),
        "by_open_candidate_bucket": summarize_groups(rows, "open_candidate_bucket"),
        "by_behavior_regret_bucket": summarize_groups(rows, "behavior_regret_bucket"),
        "records": rows,
    }


def profile_transition(
    transition: EpisodeTransition,
    *,
    dataset: str,
    index: int,
    total_safe: int = 391,
) -> dict[str, Any]:
    rows, cols = _transition_shape(transition)
    action = _decode_open_action(transition.action_index, rows=rows, cols=cols)
    action_row = None if action is None else action[0]
    action_col = None if action is None else action[1]
    region = _region(action_row, action_col, rows=rows, cols=cols)
    safe_left = _safe_left(transition, total_safe=total_safe)
    open_candidate_count = _open_candidate_count(transition.action_mask)

    values = transition.counterfactual_open_values
    label_count = 0
    negative_count = 0
    positive_count = 0
    best_value = None
    behavior_value = None
    behavior_regret = None
    behavior_mine = _behavior_mine(transition, action_row, action_col)
    if values is not None:
        values_array = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(values_array)
        if bool(finite.any()):
            label_count = int(finite.sum())
            negative_count = int((values_array[finite] < 0.0).sum())
            positive_count = int((values_array[finite] > 0.0).sum())
            best_value = float(np.max(values_array[finite]))
            if action_row is not None and action_col is not None:
                value = float(values_array[action_row, action_col])
                if np.isfinite(value):
                    behavior_value = value
                    behavior_regret = float(best_value - behavior_value)

    return {
        "dataset": dataset,
        "index": int(index),
        "family": transition.extreme_family or "unclassified",
        "expert_is_guess": bool(transition.expert_is_guess),
        "source_quality": float(transition.source_quality),
        "extreme_score": float(transition.extreme_score),
        "rows": int(rows),
        "cols": int(cols),
        "action_row": action_row,
        "action_col": action_col,
        "region": region,
        "safe_left": safe_left,
        "safe_left_bucket": bucket_value(safe_left, SAFE_LEFT_BUCKETS, unknown="unknown"),
        "open_candidate_count": int(open_candidate_count),
        "open_candidate_bucket": bucket_value(
            open_candidate_count,
            OPEN_CANDIDATE_BUCKETS,
            unknown="unknown",
        ),
        "counterfactual_label_count": int(label_count),
        "negative_counterfactual_candidates": int(negative_count),
        "positive_counterfactual_candidates": int(positive_count),
        "negative_counterfactual_rate": _safe_rate(negative_count, label_count),
        "best_counterfactual_value": best_value,
        "behavior_counterfactual_value": behavior_value,
        "behavior_regret": behavior_regret,
        "behavior_regret_bucket": bucket_value(
            behavior_regret,
            REGRET_BUCKETS,
            unknown="unlabelled",
            upper_inclusive=False,
        ),
        "behavior_mine": behavior_mine,
    }


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    regrets = [float(row["behavior_regret"]) for row in rows if row["behavior_regret"] is not None]
    label_counts = [int(row["counterfactual_label_count"]) for row in rows]
    negative_counts = [int(row["negative_counterfactual_candidates"]) for row in rows]
    known_mines = [bool(row["behavior_mine"]) for row in rows if row["behavior_mine"] is not None]
    high_regret = [regret for regret in regrets if regret >= 4.0]
    return {
        "record_count": len(rows),
        "families": dict(sorted(Counter(row["family"] for row in rows).items())),
        "regions": dict(sorted(Counter(row["region"] for row in rows).items())),
        "safe_left_buckets": dict(sorted(Counter(row["safe_left_bucket"] for row in rows).items())),
        "open_candidate_buckets": dict(sorted(Counter(row["open_candidate_bucket"] for row in rows).items())),
        "behavior_regret_buckets": dict(sorted(Counter(row["behavior_regret_bucket"] for row in rows).items())),
        "counterfactual_records": int(sum(count > 0 for count in label_counts)),
        "counterfactual_candidate_labels": int(sum(label_counts)),
        "negative_counterfactual_candidates": int(sum(negative_counts)),
        "negative_counterfactual_rate": _safe_rate(sum(negative_counts), sum(label_counts)),
        "avg_behavior_regret": _mean_or_none(regrets),
        "median_behavior_regret": _median_or_none(regrets),
        "max_behavior_regret": max(regrets) if regrets else None,
        "high_regret_records": len(high_regret),
        "high_regret_rate": _safe_rate(len(high_regret), len(regrets)),
        "known_behavior_mine_records": int(sum(known_mines)),
        "behavior_mine_rate": _safe_rate(sum(known_mines), len(known_mines)),
    }


def summarize_groups(rows: list[dict[str, Any]], key: str) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[str(row[key])].append(row)
    return {name: summarize_rows(group_rows) for name, group_rows in sorted(groups.items())}


def write_record_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "index",
        "family",
        "expert_is_guess",
        "region",
        "safe_left",
        "safe_left_bucket",
        "open_candidate_count",
        "open_candidate_bucket",
        "counterfactual_label_count",
        "negative_counterfactual_rate",
        "best_counterfactual_value",
        "behavior_counterfactual_value",
        "behavior_regret",
        "behavior_regret_bucket",
        "behavior_mine",
        "action_row",
        "action_col",
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field) for field in fieldnames})


def bucket_value(
    value: float | int | None,
    buckets: Iterable[tuple[float, float, str]],
    *,
    unknown: str,
    upper_inclusive: bool = True,
) -> str:
    if value is None:
        return unknown
    numeric = float(value)
    for lower, upper, label in buckets:
        if upper_inclusive:
            if float(lower) <= numeric <= float(upper):
                return label
        elif float(lower) <= numeric < float(upper):
            return label
    return unknown


def _transition_shape(transition: EpisodeTransition) -> tuple[int, int]:
    if transition.counterfactual_open_values is not None:
        values = np.asarray(transition.counterfactual_open_values)
        if values.ndim == 2:
            return int(values.shape[0]), int(values.shape[1])
    if transition.action_mask.ndim >= 2:
        return int(transition.action_mask.shape[-2]), int(transition.action_mask.shape[-1])
    raise ValueError("transition does not expose board dimensions")


def _decode_open_action(action_index: int, *, rows: int, cols: int) -> tuple[int, int] | None:
    cells = int(rows) * int(cols)
    kind_index, cell_index = divmod(int(action_index), cells)
    if kind_index != action_channel(ActionType.OPEN):
        return None
    row, col = divmod(cell_index, int(cols))
    if not (0 <= row < int(rows) and 0 <= col < int(cols)):
        return None
    return int(row), int(col)


def _region(row: int | None, col: int | None, *, rows: int, cols: int) -> str:
    if row is None or col is None:
        return "non_open"
    corner = bool(row in {0, rows - 1} and col in {0, cols - 1})
    if corner:
        return "corner"
    edge = bool(row in {0, rows - 1} or col in {0, cols - 1})
    return "edge" if edge else "interior"


def _safe_left(transition: EpisodeTransition, *, total_safe: int) -> int | None:
    features = np.asarray(transition.global_features, dtype=np.float32).reshape(-1)
    if features.size < 2:
        return None
    progress = float(np.clip(features[1], 0.0, 1.0))
    return int(round((1.0 - progress) * int(total_safe)))


def _open_candidate_count(action_mask: np.ndarray) -> int:
    mask = np.asarray(action_mask)
    if mask.ndim != 3:
        return 0
    return int(np.asarray(mask[action_channel(ActionType.OPEN)], dtype=bool).sum())


def _behavior_mine(
    transition: EpisodeTransition,
    row: int | None,
    col: int | None,
) -> bool | None:
    if row is None or col is None or transition.mine_mask is None:
        return None
    mines = np.asarray(transition.mine_mask, dtype=bool)
    if mines.ndim != 2 or not (0 <= row < mines.shape[0] and 0 <= col < mines.shape[1]):
        return None
    return bool(mines[row, col])


def _safe_rate(numerator: int | float, denominator: int | float) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)


def _mean_or_none(values: list[float]) -> float | None:
    return float(mean(values)) if values else None


def _median_or_none(values: list[float]) -> float | None:
    return float(median(values)) if values else None


if __name__ == "__main__":
    main()
