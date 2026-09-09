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

from minesweeper_rl.candidate_calibration import (
    candidate_policy_probabilities,
    load_candidate_calibrator,
)
from minesweeper_rl.candidate_gate import (
    FEATURE_NAMES,
    CandidateGateCalibrator,
    build_candidate_gate_features,
)
from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel
from minesweeper_rl.types import ActionType


EVALUATE_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
EVALUATE_SPEC = importlib.util.spec_from_file_location(
    "evaluate_gated_for_candidate_gate_training",
    EVALUATE_PATH,
)
if EVALUATE_SPEC is None or EVALUATE_SPEC.loader is None:
    raise RuntimeError(f"could not load {EVALUATE_PATH}")
evaluate_module = importlib.util.module_from_spec(EVALUATE_SPEC)
sys.modules[EVALUATE_SPEC.name] = evaluate_module
EVALUATE_SPEC.loader.exec_module(evaluate_module)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Train a visible-state gate for deciding whether to apply a candidate policy."
    )
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument("--base-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--base-model-weight", action="append", type=float, default=[])
    parser.add_argument("--specialist-checkpoint", type=Path, action="append", required=True)
    parser.add_argument("--specialist-model-weight", action="append", type=float, default=[])
    parser.add_argument("--candidate-calibrator", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument("--min-improvement", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=300)
    parser.add_argument("--learning-rate", type=float, default=2e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.25)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    args = parser.parse_args()

    if args.batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    if args.candidate_topk <= 0:
        raise ValueError("--candidate-topk must be positive")
    if args.min_improvement < 0.0:
        raise ValueError("--min-improvement must be non-negative")
    if not 0.0 < float(args.validation_fraction) < 1.0:
        raise ValueError("--validation-fraction must be in (0, 1)")

    np.random.seed(int(args.seed))
    torch.manual_seed(int(args.seed))
    transitions, manifests = load_transitions(args.dataset, max_records=int(args.max_records))
    if not transitions:
        raise ValueError("datasets contain no transitions")

    candidate = load_candidate_calibrator(args.candidate_calibrator)
    eval_args = SimpleNamespace(
        device=args.device,
        max_steps=600,
        decision_actions="full",
        inference_flips=args.inference_flips,
        inference_ensemble=args.inference_ensemble,
        risk_head_weight=args.risk_head_weight,
    )
    bases = [
        evaluate_module.load_eval_trainer(path, eval_args, risk_head_weight=args.risk_head_weight)
        for path in args.base_checkpoint
    ]
    specialists = [
        evaluate_module.load_eval_trainer(path, eval_args, risk_head_weight=args.risk_head_weight)
        for path in args.specialist_checkpoint
    ]
    base_weights = evaluate_module.normalize_ensemble_weights(
        args.base_model_weight,
        expected=len(bases),
        label="base-model-weight",
    )
    specialist_weights = evaluate_module.normalize_ensemble_weights(
        args.specialist_model_weight,
        expected=len(specialists),
        label="specialist-model-weight",
    )

    started_at = time.time()
    features, deltas, labels, row_metadata = collect_gate_rows(
        transitions=transitions,
        bases=bases,
        base_weights=base_weights,
        specialists=specialists,
        specialist_weights=specialist_weights,
        candidate_calibrator=candidate,
        batch_size=int(args.batch_size),
        candidate_topk=int(args.candidate_topk),
        min_improvement=float(args.min_improvement),
        scores_are_probabilities=args.inference_ensemble == "probs",
    )
    gate, fit_metrics = fit_gate(
        features=features,
        deltas=deltas,
        labels=labels,
        validation_fraction=float(args.validation_fraction),
        epochs=int(args.epochs),
        learning_rate=float(args.learning_rate),
        weight_decay=float(args.weight_decay),
        seed=int(args.seed),
    )
    validation_indices = np.asarray(
        fit_metrics.pop("_validation_indices"),
        dtype=np.int64,
    )
    gate.save(args.output)
    threshold_metrics = select_threshold(
        gate=gate,
        features=features,
        deltas=deltas,
        labels=labels,
        validation_indices=validation_indices,
        min_improvement=float(args.min_improvement),
    )
    gate.threshold = float(threshold_metrics["selected_threshold"])
    gate.save(args.output)
    report = {
        "ok": True,
        "output": str(args.output),
        "datasets": [str(path) for path in args.dataset],
        "dataset_manifests": manifests,
        "records": int(len(features)),
        "positive_records": int(labels.sum()),
        "positive_rate": float(labels.mean()) if len(labels) else 0.0,
        "candidate_calibrator": str(args.candidate_calibrator),
        "candidate_topk": int(args.candidate_topk),
        "min_improvement": float(args.min_improvement),
        "feature_names": list(FEATURE_NAMES),
        "base_checkpoints": [str(path) for path in args.base_checkpoint],
        "base_model_weights": base_weights,
        "specialist_checkpoints": [str(path) for path in args.specialist_checkpoint],
        "specialist_model_weights": specialist_weights,
        "device": str(bases[0].device),
        "inference_flips": bool(args.inference_flips),
        "inference_ensemble": args.inference_ensemble,
        "threshold": float(gate.threshold),
        "elapsed_seconds": float(time.time() - started_at),
        "delta_summary": summarize_deltas(deltas, labels),
        "row_metadata": row_metadata,
        **fit_metrics,
        "threshold_selection": threshold_metrics,
    }
    report_path = args.output.with_name(args.output.name + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))


def load_transitions(paths: list[Path], *, max_records: int) -> tuple[list[Any], list[dict[str, Any]]]:
    transitions: list[Any] = []
    manifests: list[dict[str, Any]] = []
    for path in paths:
        loaded, manifest = load_extreme_dataset(path)
        transitions.extend(loaded)
        manifests.append(
            {
                "path": str(path),
                "records": int(len(loaded)),
                "metadata": manifest.get("metadata", manifest),
            }
        )
    if max_records > 0:
        transitions = transitions[:max_records]
    return transitions, manifests


def collect_gate_rows(
    *,
    transitions,
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    candidate_calibrator,
    batch_size: int,
    candidate_topk: int,
    min_improvement: float,
    scores_are_probabilities: bool,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any]]:
    if not bases or not specialists:
        raise ValueError("at least one base and one specialist checkpoint is required")
    base = bases[0]
    rows: list[np.ndarray] = []
    deltas: list[float] = []
    labels: list[float] = []
    family_counts: dict[str, int] = {}
    skipped = 0
    open_channel = action_channel(ActionType.OPEN)
    batch_size = max(1, int(batch_size))

    with torch.inference_mode():
        for start in range(0, len(transitions), batch_size):
            batch = transitions[start : start + batch_size]
            boards = np.stack([transition.board for transition in batch])
            global_features = np.stack([transition.global_features for transition in batch])
            action_masks = np.stack([transition.action_mask for transition in batch])
            flat_mask = torch.tensor(
                action_masks.reshape(len(batch), -1),
                dtype=torch.bool,
                device=base.device,
            )
            base_scores = evaluate_module.combine_model_scores(
                [
                    trainer._predict_policy_scores_batch(
                        boards=boards,
                        global_features_batch=global_features,
                        action_masks=action_masks,
                        use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                    )
                    for trainer in bases
                ],
                base_weights,
                reduction="mean",
                flat_mask=flat_mask,
            )
            specialist_scores = evaluate_module.combine_model_scores(
                [
                    trainer._predict_policy_scores_batch(
                        boards=boards,
                        global_features_batch=global_features,
                        action_masks=action_masks,
                        use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                    )
                    for trainer in specialists
                ],
                specialist_weights,
                reduction="mean",
                flat_mask=flat_mask,
            )
            risk_maps = transition_risk_maps(batch, boards.shape[-2:])
            candidate_probs = candidate_policy_probabilities(
                candidate_calibrator,
                base_scores=base_scores.detach().cpu().numpy(),
                specialist_scores=specialist_scores.detach().cpu().numpy(),
                global_features=global_features,
                boards=boards,
                action_masks=action_masks,
                scores_are_probabilities=scores_are_probabilities,
                risk_maps=risk_maps if "solver_risk" in candidate_calibrator.feature_names else None,
            )
            shortlist = evaluate_module._candidate_shortlist_mask(
                base_scores=base_scores,
                specialist_scores=specialist_scores,
                action_masks=action_masks,
                rows=boards.shape[-2],
                cols=boards.shape[-1],
                topk=int(candidate_topk),
                candidate_probabilities=candidate_probs,
            )
            shortlist_probs = np.where(shortlist, np.maximum(candidate_probs, 0.0), 0.0)
            shortlist_probs /= np.maximum(shortlist_probs.sum(axis=1, keepdims=True), 1e-8)
            gate_features = build_candidate_gate_features(
                base_scores=base_scores.detach().cpu().numpy(),
                candidate_probabilities=candidate_probs,
                global_features=global_features,
                action_masks=action_masks,
                risk_maps=risk_maps,
                scores_are_probabilities=scores_are_probabilities,
            )
            for index, transition in enumerate(batch):
                family = transition.extreme_family or "unclassified"
                family_counts[family] = family_counts.get(family, 0) + 1
                values = transition.counterfactual_open_values
                if values is None:
                    skipped += 1
                    continue
                values = np.asarray(values, dtype=np.float32).reshape(-1)
                legal = np.asarray(transition.action_mask[open_channel], dtype=bool).reshape(-1)
                valid = legal & np.isfinite(values) & shortlist[index]
                if not valid.any():
                    skipped += 1
                    continue
                base_open = base_scores[index, : values.size].detach().cpu().numpy()
                base_choice = int(np.argmax(np.where(legal & np.isfinite(values), base_open, -np.inf)))
                candidate_choice = int(np.argmax(np.where(valid, shortlist_probs[index], -np.inf)))
                if not np.isfinite(base_open[base_choice]):
                    skipped += 1
                    continue
                base_value = float(values[base_choice])
                candidate_value = float(values[candidate_choice])
                if not np.isfinite(base_value) or not np.isfinite(candidate_value):
                    skipped += 1
                    continue
                delta = candidate_value - base_value
                rows.append(gate_features[index])
                deltas.append(delta)
                labels.append(float(delta > float(min_improvement)))

    if not rows:
        raise ValueError("no gate-labelled rows were produced")
    feature_array = np.stack(rows).astype(np.float32)
    delta_array = np.asarray(deltas, dtype=np.float32)
    label_array = np.asarray(labels, dtype=np.float32)
    if not label_array.any() or bool(label_array.all()):
        raise ValueError("gate labels contain only one class; add more varied counterfactual data")
    return (
        feature_array,
        delta_array,
        label_array,
        {
            "family_counts": dict(sorted(family_counts.items())),
            "skipped": int(skipped),
            "delta_mean": float(delta_array.mean()),
            "delta_std": float(delta_array.std()),
            "delta_positive_rate": float(np.mean(delta_array > 0.0)),
        },
    )


def transition_risk_maps(transitions, shape: tuple[int, int]) -> np.ndarray:
    maps = []
    for transition in transitions:
        risk = getattr(transition, "risk_map", None)
        if risk is None:
            maps.append(np.zeros(shape, dtype=np.float32))
            continue
        array = np.asarray(risk, dtype=np.float32)
        if array.shape != shape:
            raise ValueError(f"risk_map shape {array.shape} does not match board shape {shape}")
        maps.append(np.nan_to_num(array, nan=0.0, posinf=1.0, neginf=0.0))
    return np.stack(maps).astype(np.float32)


def fit_gate(
    *,
    features: np.ndarray,
    deltas: np.ndarray,
    labels: np.ndarray,
    validation_fraction: float,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    seed: int,
) -> tuple[CandidateGateCalibrator, dict[str, Any]]:
    deltas = np.asarray(deltas, dtype=np.float32)
    labels = np.asarray(labels, dtype=np.float32)
    count = len(features)
    split = max(1, min(count - 1, int(round(count * (1.0 - validation_fraction)))))
    rng = np.random.default_rng(seed)
    order = rng.permutation(count)
    train_index = order[:split]
    validation_index = order[split:]
    means = features[train_index].mean(axis=0)
    scales = np.maximum(features[train_index].std(axis=0), 1e-5)
    train_x = torch.tensor((features[train_index] - means) / scales, dtype=torch.float32)
    train_y = torch.tensor(labels[train_index], dtype=torch.float32)
    sample_weights = 1.0 + np.clip(np.abs(deltas[train_index]), 0.0, 4.0)
    sample_weights *= np.where(train_y.numpy() > 0.5, 1.5, 1.0)
    train_w = torch.tensor(sample_weights, dtype=torch.float32)

    torch.manual_seed(seed)
    linear = torch.nn.Linear(train_x.shape[1], 1)
    optimizer = torch.optim.AdamW(
        linear.parameters(),
        lr=float(learning_rate),
        weight_decay=float(weight_decay),
    )
    positive = max(float(train_y.sum()), 1.0)
    negative = max(float(len(train_y) - train_y.sum()), 1.0)
    class_weight = torch.tensor(negative / positive, dtype=torch.float32)
    losses: list[float] = []
    for _ in range(max(1, int(epochs))):
        optimizer.zero_grad(set_to_none=True)
        logits = linear(train_x).squeeze(1)
        loss = torch.nn.functional.binary_cross_entropy_with_logits(
            logits,
            train_y,
            weight=train_w,
            pos_weight=class_weight,
        )
        loss.backward()
        optimizer.step()
        losses.append(float(loss.detach()))

    with torch.inference_mode():
        train_prob = torch.sigmoid(linear(train_x).squeeze(1)).cpu().numpy()
        validation_x = torch.tensor((features[validation_index] - means) / scales, dtype=torch.float32)
        validation_prob = torch.sigmoid(linear(validation_x).squeeze(1)).cpu().numpy()
    gate = CandidateGateCalibrator(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        weights=linear.weight.detach().cpu().numpy().reshape(-1).astype(np.float32),
        bias=float(linear.bias.detach().cpu().item()),
        threshold=0.5,
    )
    return gate, {
        "train_records": int(len(train_index)),
        "validation_records": int(len(validation_index)),
        "train_positive_rate": float(labels[train_index].mean()),
        "validation_positive_rate": float(labels[validation_index].mean()),
        "final_loss": float(losses[-1]),
        "train_auc": binary_auc(labels[train_index], train_prob),
        "validation_auc": binary_auc(labels[validation_index], validation_prob),
        "train_apply_rate_at_05": float(np.mean(train_prob >= 0.5)),
        "validation_apply_rate_at_05": float(np.mean(validation_prob >= 0.5)),
        "_validation_indices": validation_index.tolist(),
    }


def select_threshold(
    *,
    gate: CandidateGateCalibrator,
    features: np.ndarray,
    deltas: np.ndarray,
    labels: np.ndarray,
    validation_indices: np.ndarray,
    min_improvement: float,
) -> dict[str, Any]:
    validation_index = np.asarray(validation_indices, dtype=np.int64)
    probabilities = gate.predict_proba_features(features[validation_index])
    validation_deltas = deltas[validation_index]
    validation_labels = labels[validation_index]
    thresholds = np.linspace(0.35, 0.95, 61)
    candidates = []
    for threshold in thresholds:
        applied = probabilities >= threshold
        if not applied.any():
            continue
        applied_deltas = validation_deltas[applied]
        positive = applied_deltas > float(min_improvement)
        candidates.append(
            {
                "threshold": float(threshold),
                "apply_rate": float(applied.mean()),
                "precision": float(positive.mean()),
                "mean_applied_delta": float(applied_deltas.mean()),
                "total_applied_delta": float(applied_deltas.sum()),
                "positive_label_recall": float(
                    ((applied & (validation_labels > 0.5)).sum())
                    / max(1, int((validation_labels > 0.5).sum()))
                ),
            }
        )
    if not candidates:
        selected = {
            "threshold": 0.95,
            "apply_rate": 0.0,
            "precision": 0.0,
            "mean_applied_delta": 0.0,
            "total_applied_delta": 0.0,
            "positive_label_recall": 0.0,
        }
    else:
        selected = max(
            candidates,
            key=lambda row: (
                row["total_applied_delta"],
                row["precision"],
                -row["apply_rate"],
            ),
        )
    return {
        "selected_threshold": float(selected["threshold"]),
        "selected": selected,
        "candidates": candidates,
        "validation_records": int(len(validation_index)),
    }


def summarize_deltas(deltas: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    deltas = np.asarray(deltas, dtype=np.float32)
    labels = np.asarray(labels, dtype=bool)
    return {
        "mean": float(deltas.mean()),
        "std": float(deltas.std()),
        "min": float(deltas.min()),
        "max": float(deltas.max()),
        "positive_label_mean": float(deltas[labels].mean()) if labels.any() else 0.0,
        "negative_label_mean": float(deltas[~labels].mean()) if (~labels).any() else 0.0,
    }


def binary_auc(labels: np.ndarray, probabilities: np.ndarray) -> float:
    labels = np.asarray(labels, dtype=bool)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    positives = probabilities[labels]
    negatives = probabilities[~labels]
    if len(positives) == 0 or len(negatives) == 0:
        return 0.5
    comparisons = (positives[:, None] > negatives[None, :]).mean()
    ties = (positives[:, None] == negatives[None, :]).mean()
    return float(comparisons + 0.5 * ties)


if __name__ == "__main__":
    main()
