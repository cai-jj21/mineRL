from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel, decode_action_index
from minesweeper_rl.gate_calibration import FEATURE_NAMES, build_gate_features
from minesweeper_rl.types import ActionType, EpisodeTransition


GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_for_extreme_gate", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a specialist gate calibration table directly from counterfactual extreme records."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--base-ensemble-checkpoint",
        dest="base_ensemble_checkpoints",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--base-model-weight",
        dest="base_model_weights",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument(
        "--specialist-ensemble-checkpoint",
        dest="specialist_ensemble_checkpoints",
        action="append",
        type=Path,
        default=[],
    )
    parser.add_argument(
        "--specialist-model-weight",
        dest="specialist_model_weights",
        action="append",
        type=float,
        default=[],
    )
    parser.add_argument("--model-ensemble-reduction", choices=["mean", "geomean"], default="mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--label-margin", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--family-filter",
        action="append",
        default=[],
        help="Keep only the requested extreme family. Repeat to include multiple families.",
    )
    parser.add_argument("--safe-left-min", type=int, default=None)
    parser.add_argument("--safe-left-max", type=int, default=None)
    parser.add_argument(
        "--include-agreements",
        action="store_true",
        help="Also keep rows where base and specialist choose the same top OPEN cell.",
    )
    parser.add_argument(
        "--require-base-open",
        action="store_true",
        help="Keep only rows where the base model's full top action is OPEN.",
    )
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--base-risk-head-weight", type=float)
    parser.add_argument("--specialist-risk-head-weight", type=float)
    args = parser.parse_args()

    base_paths = [args.base_checkpoint, *args.base_ensemble_checkpoints]
    specialist_paths = [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
    base_weights = gated.normalize_ensemble_weights(
        args.base_model_weights,
        expected=len(base_paths),
        label="base-model-weight",
    )
    specialist_weights = gated.normalize_ensemble_weights(
        args.specialist_model_weights,
        expected=len(specialist_paths),
        label="specialist-model-weight",
    )
    eval_args = SimpleNamespace(
        device=args.device,
        max_steps=600,
        decision_actions="full",
        inference_flips=bool(args.inference_flips),
        inference_ensemble=args.inference_ensemble,
        risk_head_weight=float(args.risk_head_weight),
    )
    base_risk_head_weight = (
        float(args.risk_head_weight)
        if args.base_risk_head_weight is None
        else float(args.base_risk_head_weight)
    )
    specialist_risk_head_weight = (
        float(args.risk_head_weight)
        if args.specialist_risk_head_weight is None
        else float(args.specialist_risk_head_weight)
    )
    bases = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=base_risk_head_weight)
        for path in base_paths
    ]
    specialists = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=specialist_risk_head_weight)
        for path in specialist_paths
    ]
    transitions, metadata = load_extreme_dataset(args.dataset)
    transitions = _filter_transitions(
        transitions,
        family_filter=args.family_filter,
        safe_left_min=args.safe_left_min,
        safe_left_max=args.safe_left_max,
        total_safe=bases[0].config.rows * bases[0].config.cols - bases[0].config.mines,
    )
    if args.max_records > 0:
        transitions = transitions[: int(args.max_records)]
    started_at = time.time()
    payload = build_dataset(
        transitions=transitions,
        bases=bases,
        base_weights=base_weights,
        specialists=specialists,
        specialist_weights=specialist_weights,
        batch_size=args.batch_size,
        model_ensemble_reduction=args.model_ensemble_reduction,
        label_margin=float(args.label_margin),
        include_agreements=bool(args.include_agreements),
        require_base_open=bool(args.require_base_open),
        scores_are_probabilities=bool(args.inference_flips and args.inference_ensemble == "probs"),
        seed=int(args.seed),
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, **payload["arrays"])
    report = {
        "ok": True,
        "dataset": str(args.dataset),
        "output": str(args.output),
        "feature_names": list(FEATURE_NAMES),
        "records": int(payload["records"]),
        "positive_records": int(payload["positive_records"]),
        "negative_records": int(payload["negative_records"]),
        "positive_rate": float(payload["positive_rate"]),
        "mean_value_delta": payload["mean_value_delta"],
        "elapsed_seconds": time.time() - started_at,
        "metadata": {
            "source": "extreme_counterfactual_specialist_gate_labels",
            "input_metadata": metadata.get("metadata", metadata),
            "base_checkpoints": [str(path) for path in base_paths],
            "base_model_weights": base_weights,
            "specialist_checkpoints": [str(path) for path in specialist_paths],
            "specialist_model_weights": specialist_weights,
            "model_ensemble_reduction": args.model_ensemble_reduction,
            "risk_head_weight": float(args.risk_head_weight),
            "base_risk_head_weight": base_risk_head_weight,
            "specialist_risk_head_weight": specialist_risk_head_weight,
            "label_margin": float(args.label_margin),
            "family_filter": list(args.family_filter or []),
            "safe_left_min": args.safe_left_min,
            "safe_left_max": args.safe_left_max,
            "include_agreements": bool(args.include_agreements),
            "require_base_open": bool(args.require_base_open),
            "inference_flips": bool(args.inference_flips),
            "inference_ensemble": args.inference_ensemble,
            "source_records": len(transitions),
            "skipped_records": payload["skipped_records"],
        },
    }
    report_path = args.output.with_suffix(".json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def build_dataset(
    *,
    transitions: list[EpisodeTransition],
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    batch_size: int,
    model_ensemble_reduction: str,
    label_margin: float,
    include_agreements: bool,
    require_base_open: bool,
    scores_are_probabilities: bool,
    seed: int,
) -> dict[str, Any]:
    base = bases[0]
    rows, cols = base.config.rows, base.config.cols
    batch_size = max(1, int(batch_size))
    features: list[np.ndarray] = []
    labels: list[int] = []
    value_delta: list[float] = []
    base_value: list[float] = []
    specialist_value: list[float] = []
    source_index: list[int] = []
    safe_left_values: list[int] = []
    base_actions: list[int] = []
    specialist_actions: list[int] = []
    base_full_actions: list[int] = []
    specialist_full_actions: list[int] = []
    skipped: dict[str, int] = {
        "unlabelled": 0,
        "agreement": 0,
        "base_not_open": 0,
        "invalid_choice": 0,
    }

    counterfactual_rows = [
        (index, transition)
        for index, transition in enumerate(transitions)
        if transition.counterfactual_open_values is not None
        and np.isfinite(transition.counterfactual_open_values).any()
    ]
    skipped["unlabelled"] = len(transitions) - len(counterfactual_rows)

    with torch.inference_mode():
        for start in range(0, len(counterfactual_rows), batch_size):
            indexed_batch = counterfactual_rows[start : start + batch_size]
            batch = [transition for _, transition in indexed_batch]
            boards = np.stack([transition.board for transition in batch])
            globals_batch = np.stack([transition.global_features for transition in batch])
            masks = np.stack([transition.action_mask for transition in batch])
            flat_mask = torch.tensor(
                masks.reshape(len(batch), -1),
                dtype=torch.bool,
                device=base.device,
            )
            base_scores = gated.combine_model_scores(
                [
                    trainer._predict_policy_scores_batch(
                        boards=boards,
                        global_features_batch=globals_batch,
                        action_masks=masks,
                        use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                    )
                    for trainer in bases
                ],
                base_weights,
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            specialist_scores = gated.combine_model_scores(
                [
                    trainer._predict_policy_scores_batch(
                        boards=boards,
                        global_features_batch=globals_batch,
                        action_masks=masks,
                        use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                    )
                    for trainer in specialists
                ],
                specialist_weights,
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            base_np = base_scores.detach().cpu().numpy()
            specialist_np = specialist_scores.detach().cpu().numpy()
            batch_features = build_gate_features(
                base_scores=base_np,
                specialist_scores=specialist_np,
                global_features=globals_batch,
                boards=boards,
                action_masks=masks,
                scores_are_probabilities=scores_are_probabilities,
            )
            for local_index, (original_index, transition) in enumerate(indexed_batch):
                label = _gate_label_for_transition(
                    transition=transition,
                    base_scores=base_np[local_index],
                    specialist_scores=specialist_np[local_index],
                    rows=rows,
                    cols=cols,
                    label_margin=label_margin,
                    include_agreements=include_agreements,
                    require_base_open=require_base_open,
                )
                if label is None:
                    reason = _skip_reason_for_transition(
                        transition=transition,
                        base_scores=base_np[local_index],
                        specialist_scores=specialist_np[local_index],
                        rows=rows,
                        cols=cols,
                        include_agreements=include_agreements,
                        require_base_open=require_base_open,
                    )
                    skipped[reason] = skipped.get(reason, 0) + 1
                    continue
                features.append(batch_features[local_index])
                labels.append(int(label["label"]))
                value_delta.append(float(label["value_delta"]))
                base_value.append(float(label["base_value"]))
                specialist_value.append(float(label["specialist_value"]))
                source_index.append(int(original_index))
                safe_left_values.append(_safe_left_from_transition(transition, rows * cols - base.config.mines))
                base_actions.append(int(label["base_open_action"]))
                specialist_actions.append(int(label["specialist_open_action"]))
                base_full_actions.append(int(label["base_full_action"]))
                specialist_full_actions.append(int(label["specialist_full_action"]))

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(0, len(FEATURE_NAMES))
    labels_array = np.asarray(labels, dtype=np.int64)
    rng = np.random.default_rng(seed)
    game_index = rng.permutation(max(1, len(labels_array)))[: len(labels_array)].astype(np.int64)
    deltas = np.asarray(value_delta, dtype=np.float32)
    return {
        "records": int(len(labels_array)),
        "positive_records": int(labels_array.sum()),
        "negative_records": int(len(labels_array) - labels_array.sum()),
        "positive_rate": float(labels_array.mean()) if len(labels_array) else 0.0,
        "mean_value_delta": float(deltas.mean()) if len(deltas) else None,
        "skipped_records": skipped,
        "arrays": {
            "features": feature_array,
            "labels": labels_array,
            "value_delta": deltas,
            "base_value": np.asarray(base_value, dtype=np.float32),
            "specialist_value": np.asarray(specialist_value, dtype=np.float32),
            "game_index": game_index,
            "step": np.zeros(len(labels_array), dtype=np.int64),
            "safe_left": np.asarray(safe_left_values, dtype=np.int64),
            "base_action": np.asarray(base_actions, dtype=np.int64),
            "specialist_action": np.asarray(specialist_actions, dtype=np.int64),
            "base_full_action": np.asarray(base_full_actions, dtype=np.int64),
            "specialist_full_action": np.asarray(specialist_full_actions, dtype=np.int64),
            "source_index": np.asarray(source_index, dtype=np.int64),
        },
    }


def _gate_label_for_transition(
    *,
    transition: EpisodeTransition,
    base_scores: np.ndarray,
    specialist_scores: np.ndarray,
    rows: int,
    cols: int,
    label_margin: float,
    include_agreements: bool,
    require_base_open: bool,
) -> dict[str, float | int] | None:
    values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
    action_mask = np.asarray(transition.action_mask, dtype=bool)
    open_channel = action_channel(ActionType.OPEN)
    cells = rows * cols
    open_mask = action_mask[open_channel].reshape(-1)
    valid = open_mask & np.isfinite(values)
    if not bool(valid.any()):
        return None

    base_full_action = int(np.argmax(np.where(action_mask.reshape(-1), base_scores, -np.inf)))
    specialist_full_action = int(np.argmax(np.where(action_mask.reshape(-1), specialist_scores, -np.inf)))
    if require_base_open and decode_action_index(base_full_action, rows, cols).kind != ActionType.OPEN:
        return None

    open_start = open_channel * cells
    open_stop = open_start + cells
    base_open = np.asarray(base_scores[open_start:open_stop], dtype=np.float32).copy()
    specialist_open = np.asarray(specialist_scores[open_start:open_stop], dtype=np.float32).copy()
    base_open[~valid] = -np.inf
    specialist_open[~valid] = -np.inf
    base_choice = int(np.argmax(base_open))
    specialist_choice = int(np.argmax(specialist_open))
    if not include_agreements and base_choice == specialist_choice:
        return None

    if not np.isfinite(values[base_choice]) or not np.isfinite(values[specialist_choice]):
        return None
    base_value = float(values[base_choice])
    specialist_value = float(values[specialist_choice])
    delta = specialist_value - base_value
    return {
        "label": int(delta > float(label_margin)),
        "value_delta": float(delta),
        "base_value": base_value,
        "specialist_value": specialist_value,
        "base_open_action": base_choice,
        "specialist_open_action": specialist_choice,
        "base_full_action": base_full_action,
        "specialist_full_action": specialist_full_action,
    }


def _skip_reason_for_transition(
    *,
    transition: EpisodeTransition,
    base_scores: np.ndarray,
    specialist_scores: np.ndarray,
    rows: int,
    cols: int,
    include_agreements: bool,
    require_base_open: bool,
) -> str:
    values = transition.counterfactual_open_values
    if values is None or not np.isfinite(values).any():
        return "unlabelled"
    action_mask = np.asarray(transition.action_mask, dtype=bool)
    if require_base_open:
        base_full_action = int(np.argmax(np.where(action_mask.reshape(-1), base_scores, -np.inf)))
        if decode_action_index(base_full_action, rows, cols).kind != ActionType.OPEN:
            return "base_not_open"
    if not include_agreements:
        cells = rows * cols
        open_channel = action_channel(ActionType.OPEN)
        valid = action_mask[open_channel].reshape(-1) & np.isfinite(np.asarray(values).reshape(-1))
        if bool(valid.any()):
            start = open_channel * cells
            stop = start + cells
            base_open = np.asarray(base_scores[start:stop], dtype=np.float32).copy()
            specialist_open = np.asarray(specialist_scores[start:stop], dtype=np.float32).copy()
            base_open[~valid] = -np.inf
            specialist_open[~valid] = -np.inf
            if int(np.argmax(base_open)) == int(np.argmax(specialist_open)):
                return "agreement"
    return "invalid_choice"


def _filter_transitions(
    transitions: list[EpisodeTransition],
    *,
    family_filter: list[str] | tuple[str, ...] | None,
    safe_left_min: int | None,
    safe_left_max: int | None,
    total_safe: int,
) -> list[EpisodeTransition]:
    families = set(family_filter or [])
    filtered: list[EpisodeTransition] = []
    for transition in transitions:
        if families and (transition.extreme_family or "unclassified") not in families:
            continue
        safe_left = _safe_left_from_transition(transition, total_safe)
        if safe_left_min is not None and safe_left < int(safe_left_min):
            continue
        if safe_left_max is not None and safe_left > int(safe_left_max):
            continue
        filtered.append(transition)
    return filtered


def _safe_left_from_transition(transition: EpisodeTransition, total_safe: int) -> int:
    progress = float(np.clip(transition.global_features[1], 0.0, 1.0))
    return int(round((1.0 - progress) * int(total_safe)))


if __name__ == "__main__":
    main()
