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
from minesweeper_rl.features import action_channel, decode_action_index
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint
from minesweeper_rl.types import ActionType, EpisodeTransition


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Measure how model policy rankings align with counterfactual labels on extreme states."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--model-weight", action="append", type=float, default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument(
        "--family-filter",
        action="append",
        default=[],
        help="Keep only the requested extreme family. Repeat to include multiple families.",
    )
    parser.add_argument(
        "--safe-left-min",
        type=int,
        default=None,
        help="Keep only states with at least this many safe cells left.",
    )
    parser.add_argument(
        "--safe-left-max",
        type=int,
        default=None,
        help="Keep only states with at most this many safe cells left.",
    )
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--near-best-margin", type=float, default=0.25)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    transitions, metadata = load_extreme_dataset(args.dataset)
    if args.max_records > 0:
        transitions = transitions[: args.max_records]
    transitions = filter_transitions(
        transitions,
        family_filter=args.family_filter,
        safe_left_min=args.safe_left_min,
        safe_left_max=args.safe_left_max,
    )
    trainers = [load_eval_trainer(path, args) for path in args.checkpoint]
    weights = normalize_weights(args.model_weight, len(trainers))

    started_at = time.time()
    result = analyze_alignment(
        transitions=transitions,
        trainers=trainers,
        model_weights=weights,
        batch_size=args.batch_size,
        near_best_margin=float(args.near_best_margin),
    )
    payload = {
        "dataset": str(args.dataset),
        "dataset_records": len(transitions),
        "dataset_metadata": metadata.get("metadata", metadata),
        "family_filter": list(args.family_filter or []),
        "safe_left_min": args.safe_left_min,
        "safe_left_max": args.safe_left_max,
        "checkpoints": [str(path) for path in args.checkpoint],
        "model_weights": weights,
        "device": str(trainers[0].device),
        "inference_augment_flips": bool(args.inference_flips),
        "inference_ensemble": args.inference_ensemble,
        "risk_head_weight": float(args.risk_head_weight),
        "near_best_margin": float(args.near_best_margin),
        "elapsed_seconds": time.time() - started_at,
        **result,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def filter_transitions(
    transitions: list[EpisodeTransition],
    *,
    family_filter: list[str] | tuple[str, ...] | None = None,
    safe_left_min: int | None = None,
    safe_left_max: int | None = None,
) -> list[EpisodeTransition]:
    families = set(family_filter or [])
    filtered: list[EpisodeTransition] = []
    for transition in transitions:
        family = transition.extreme_family or "unclassified"
        if families and family not in families:
            continue
        safe_left = _safe_left_from_transition(transition)
        if safe_left_min is not None and safe_left < int(safe_left_min):
            continue
        if safe_left_max is not None and safe_left > int(safe_left_max):
            continue
        filtered.append(transition)
    return filtered


def _safe_left_from_transition(transition: EpisodeTransition, total_safe: int = 391) -> int:
    progress = float(np.clip(transition.global_features[1], 0.0, 1.0))
    return int(round((1.0 - progress) * int(total_safe)))


def load_eval_trainer(path: Path, args: argparse.Namespace) -> MinesweeperTrainer:
    trainer = load_checkpoint(path, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = 600
    trainer.config.decision_actions = "full"
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = float(args.risk_head_weight)
    trainer.model.eval()
    return trainer


def normalize_weights(weights: list[float], expected: int) -> list[float]:
    if expected <= 0:
        raise ValueError("at least one checkpoint is required")
    if not weights:
        return [1.0 / float(expected) for _ in range(expected)]
    if len(weights) != expected:
        raise ValueError(f"model-weight count must match checkpoint count: {len(weights)} != {expected}")
    array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(array).all() or (array < 0.0).any() or float(array.sum()) <= 0.0:
        raise ValueError("model weights must be finite, non-negative, and sum positive")
    return (array / float(array.sum())).astype(float).tolist()


def analyze_alignment(
    *,
    transitions: list[EpisodeTransition],
    trainers: list[MinesweeperTrainer],
    model_weights: list[float],
    batch_size: int,
    near_best_margin: float,
) -> dict[str, Any]:
    counterfactual_rows = [
        transition
        for transition in transitions
        if transition.counterfactual_open_values is not None
        and np.isfinite(transition.counterfactual_open_values).any()
    ]
    if not counterfactual_rows:
        raise ValueError("dataset has no counterfactual-labelled rows")

    family_stats: dict[str, dict[str, Any]] = defaultdict(new_stats)
    overall = new_stats()
    batch_size = max(1, int(batch_size))
    base = trainers[0]

    with torch.no_grad():
        for start in range(0, len(counterfactual_rows), batch_size):
            batch = counterfactual_rows[start : start + batch_size]
            boards = np.stack([transition.board for transition in batch])
            global_features = np.stack([transition.global_features for transition in batch])
            action_masks = np.stack([transition.action_mask for transition in batch])
            scores = None
            for weight, trainer in zip(model_weights, trainers, strict=True):
                part = trainer._predict_policy_scores_batch(
                    boards=boards,
                    global_features_batch=global_features,
                    action_masks=action_masks,
                    use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                )
                weighted = part * float(weight)
                scores = weighted if scores is None else scores + weighted
            assert scores is not None
            scores_np = scores.detach().cpu().numpy()

            for local_index, transition in enumerate(batch):
                metrics = score_transition(
                    transition,
                    scores_np[local_index],
                    rows=base.config.rows,
                    cols=base.config.cols,
                    near_best_margin=near_best_margin,
                )
                if metrics is None:
                    continue
                family = transition.extreme_family or "unclassified"
                add_metrics(overall, metrics)
                add_metrics(family_stats[family], metrics)

    return {
        "counterfactual_records": int(overall["records"]),
        "overall": finalize_stats(overall),
        "families": {
            family: finalize_stats(stats)
            for family, stats in sorted(family_stats.items())
        },
    }


def score_transition(
    transition: EpisodeTransition,
    scores: np.ndarray,
    *,
    rows: int,
    cols: int,
    near_best_margin: float,
) -> dict[str, Any] | None:
    cf_values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
    action_mask = np.asarray(transition.action_mask, dtype=bool)
    open_mask = action_mask[action_channel(ActionType.OPEN)].reshape(-1)
    valid = open_mask & np.isfinite(cf_values)
    if not valid.any():
        return None

    open_start = action_channel(ActionType.OPEN) * rows * cols
    open_stop = open_start + rows * cols
    open_scores = scores[open_start:open_stop].copy()
    open_scores[~valid] = -np.inf
    chosen_open = int(np.argmax(open_scores))

    masked_cf = cf_values.copy()
    masked_cf[~valid] = -np.inf
    best_open = int(np.argmax(masked_cf))
    best_value = float(masked_cf[best_open])
    chosen_value = float(masked_cf[chosen_open])
    regret = best_value - chosen_value

    full_scores = scores.copy()
    full_valid = action_mask.reshape(-1)
    full_scores[~full_valid] = -np.inf
    full_choice = int(np.argmax(full_scores))
    full_action = decode_action_index(full_choice, rows, cols)
    behavior_action = decode_action_index(int(transition.action_index), rows, cols)
    behavior_open = (
        behavior_action.row * cols + behavior_action.col
        if behavior_action.kind == ActionType.OPEN
        else None
    )
    behavior_value = (
        float(masked_cf[behavior_open])
        if behavior_open is not None and bool(valid[behavior_open])
        else None
    )

    mine_mask = transition.mine_mask.reshape(-1) if transition.mine_mask is not None else None
    chosen_is_mine = bool(mine_mask[chosen_open]) if mine_mask is not None else False
    best_is_mine = bool(mine_mask[best_open]) if mine_mask is not None else False
    behavior_is_mine = (
        bool(mine_mask[behavior_open])
        if mine_mask is not None and behavior_open is not None and bool(valid[behavior_open])
        else False
    )
    valid_indices = np.flatnonzero(valid)
    sorted_by_score = valid_indices[np.argsort(open_scores[valid_indices])[::-1]]
    best_rank = int(np.flatnonzero(sorted_by_score == best_open)[0]) + 1

    return {
        "best_match": float(chosen_open == best_open),
        "near_best": float(regret <= near_best_margin),
        "regret": float(regret),
        "chosen_value": chosen_value,
        "best_value": best_value,
        "chosen_negative": float(chosen_value < 0.0),
        "chosen_mine": float(chosen_is_mine),
        "best_mine": float(best_is_mine),
        "full_choice_open": float(full_action.kind == ActionType.OPEN),
        "full_choice_matches_open_choice": float(
            full_action.kind == ActionType.OPEN
            and full_action.row * cols + full_action.col == chosen_open
        ),
        "replays_behavior_open": float(behavior_open == chosen_open),
        "behavior_regret": (best_value - behavior_value) if behavior_value is not None else None,
        "behavior_mine": float(behavior_is_mine),
        "best_rank": float(best_rank),
    }


def new_stats() -> dict[str, Any]:
    return {
        "records": 0,
        "best_match": 0.0,
        "near_best": 0.0,
        "regret": [],
        "chosen_value": [],
        "best_value": [],
        "chosen_negative": 0.0,
        "chosen_mine": 0.0,
        "best_mine": 0.0,
        "full_choice_open": 0.0,
        "full_choice_matches_open_choice": 0.0,
        "replays_behavior_open": 0.0,
        "behavior_regret": [],
        "behavior_mine": 0.0,
        "best_rank": [],
    }


def add_metrics(stats: dict[str, Any], metrics: dict[str, Any]) -> None:
    stats["records"] += 1
    for key in (
        "best_match",
        "near_best",
        "chosen_negative",
        "chosen_mine",
        "best_mine",
        "full_choice_open",
        "full_choice_matches_open_choice",
        "replays_behavior_open",
        "behavior_mine",
    ):
        stats[key] += float(metrics[key])
    for key in ("regret", "chosen_value", "best_value", "best_rank"):
        stats[key].append(float(metrics[key]))
    if metrics["behavior_regret"] is not None:
        stats["behavior_regret"].append(float(metrics["behavior_regret"]))


def finalize_stats(stats: dict[str, Any]) -> dict[str, Any]:
    records = int(stats["records"])
    if records <= 0:
        return {"records": 0}
    return {
        "records": records,
        "best_match_rate": stats["best_match"] / records,
        "near_best_rate": stats["near_best"] / records,
        "chosen_negative_rate": stats["chosen_negative"] / records,
        "chosen_mine_rate": stats["chosen_mine"] / records,
        "best_mine_rate": stats["best_mine"] / records,
        "full_choice_open_rate": stats["full_choice_open"] / records,
        "full_choice_matches_open_choice_rate": stats["full_choice_matches_open_choice"] / records,
        "replays_behavior_open_rate": stats["replays_behavior_open"] / records,
        "behavior_mine_rate": stats["behavior_mine"] / records,
        "avg_regret": mean(stats["regret"]),
        "median_regret": median(stats["regret"]),
        "p90_regret": percentile(stats["regret"], 90.0),
        "avg_behavior_regret": mean(stats["behavior_regret"]),
        "avg_chosen_value": mean(stats["chosen_value"]),
        "avg_best_value": mean(stats["best_value"]),
        "avg_best_rank": mean(stats["best_rank"]),
        "median_best_rank": median(stats["best_rank"]),
    }


def mean(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def median(values: list[float]) -> float | None:
    return float(np.median(values)) if values else None


def percentile(values: list[float], q: float) -> float | None:
    return float(np.percentile(values, q)) if values else None


if __name__ == "__main__":
    main()
