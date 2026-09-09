from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel
from minesweeper_rl.trainer import load_checkpoint
from minesweeper_rl.types import ActionType, EpisodeTransition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how the counterfactual value head ranks labelled OPEN candidates."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--near-best-margin", type=float, default=0.25)
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    transitions, metadata = load_extreme_dataset(args.dataset)
    if args.max_records > 0:
        transitions = transitions[: args.max_records]
    transitions = [
        transition
        for transition in transitions
        if transition.counterfactual_open_values is not None
        and np.isfinite(transition.counterfactual_open_values).any()
    ]
    if not transitions:
        raise ValueError("dataset has no counterfactual-labelled rows")

    trainer = load_checkpoint(args.checkpoint, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.model.eval()

    started_at = time.time()
    result = analyze_value_alignment(
        transitions=transitions,
        trainer=trainer,
        batch_size=max(1, int(args.batch_size)),
        use_flip_ensemble=bool(args.inference_flips),
        near_best_margin=float(args.near_best_margin),
    )
    payload = {
        "dataset": str(args.dataset),
        "dataset_records": len(transitions),
        "dataset_metadata": metadata.get("metadata", metadata),
        "checkpoint": str(args.checkpoint),
        "device": str(trainer.device),
        "inference_augment_flips": bool(args.inference_flips),
        "near_best_margin": float(args.near_best_margin),
        "elapsed_seconds": time.time() - started_at,
        **result,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def analyze_value_alignment(
    *,
    transitions: list[EpisodeTransition],
    trainer,
    batch_size: int,
    use_flip_ensemble: bool,
    near_best_margin: float,
) -> dict[str, Any]:
    overall = _new_stats()
    family_stats: dict[str, dict[str, Any]] = defaultdict(_new_stats)
    open_channel = action_channel(ActionType.OPEN)

    with torch.inference_mode():
        for start in range(0, len(transitions), batch_size):
            batch = transitions[start : start + batch_size]
            boards = np.stack([transition.board for transition in batch])
            globals_batch = np.stack([transition.global_features for transition in batch])
            predicted = trainer._predict_counterfactual_values_batch(
                boards=boards,
                global_features_batch=globals_batch,
                use_flip_ensemble=use_flip_ensemble,
            ).detach().cpu().numpy()
            for index, transition in enumerate(batch):
                values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
                legal = np.asarray(transition.action_mask[open_channel], dtype=bool).reshape(-1)
                valid = legal & np.isfinite(values)
                if not bool(valid.any()):
                    continue
                metrics = _score_row(values, predicted[index].reshape(-1), valid, near_best_margin)
                _add(overall, metrics)
                _add(family_stats[transition.extreme_family or "unclassified"], metrics)

    return {
        "counterfactual_records": int(overall["records"]),
        "overall": _finalize(overall),
        "families": {
            family: _finalize(stats)
            for family, stats in sorted(family_stats.items())
        },
    }


def _score_row(
    labels: np.ndarray,
    predictions: np.ndarray,
    valid: np.ndarray,
    near_best_margin: float,
) -> dict[str, float]:
    labelled = labels.copy()
    labelled[~valid] = -np.inf
    predicted = predictions.copy()
    predicted[~valid] = -np.inf
    best = int(np.argmax(labelled))
    chosen = int(np.argmax(predicted))
    best_value = float(labelled[best])
    chosen_value = float(labelled[chosen])
    valid_indices = np.flatnonzero(valid)
    sorted_by_prediction = valid_indices[np.argsort(predicted[valid_indices])[::-1]]
    best_rank = int(np.flatnonzero(sorted_by_prediction == best)[0]) + 1
    regret = best_value - chosen_value
    return {
        "best_match": float(chosen == best),
        "near_best": float(regret <= near_best_margin),
        "chosen_negative": float(chosen_value < 0.0),
        "regret": float(regret),
        "chosen_value": chosen_value,
        "best_value": best_value,
        "best_rank": float(best_rank),
    }


def _new_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "best_match": 0.0,
        "near_best": 0.0,
        "chosen_negative": 0.0,
        "regret": [],
        "chosen_value": [],
        "best_value": [],
        "best_rank": [],
    }


def _add(stats: dict[str, Any], metrics: dict[str, float]) -> None:
    stats["records"] += 1
    for key in ("best_match", "near_best", "chosen_negative"):
        stats[key] += float(metrics[key])
    for key in ("regret", "chosen_value", "best_value", "best_rank"):
        stats[key].append(float(metrics[key]))


def _finalize(stats: dict[str, Any]) -> dict[str, Any]:
    records = int(stats["records"])
    if records <= 0:
        return {"records": 0}
    return {
        "records": records,
        "best_match_rate": stats["best_match"] / records,
        "near_best_rate": stats["near_best"] / records,
        "chosen_negative_rate": stats["chosen_negative"] / records,
        "avg_regret": _mean(stats["regret"]),
        "median_regret": _median(stats["regret"]),
        "p90_regret": _percentile(stats["regret"], 90.0),
        "avg_chosen_value": _mean(stats["chosen_value"]),
        "avg_best_value": _mean(stats["best_value"]),
        "avg_best_rank": _mean(stats["best_rank"]),
        "median_best_rank": _median(stats["best_rank"]),
    }


def _mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def _percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


if __name__ == "__main__":
    main()
