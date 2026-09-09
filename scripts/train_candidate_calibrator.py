from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import replace
import importlib.util
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.candidate_calibration import (
    CANDIDATE_FEATURE_NAMES,
    RISK_AWARE_CANDIDATE_FEATURE_NAMES,
    CandidateValueCalibrator,
    NeuralCandidateValueCalibrator,
    build_candidate_feature_names,
    build_candidate_features,
)
from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel
from minesweeper_rl.trainer import load_checkpoint
from minesweeper_rl.types import ActionType


GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_for_candidate_calibrator", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train a visible-feature candidate value calibrator.")
    parser.add_argument("--dataset", type=Path, action="append", required=True)
    parser.add_argument(
        "--dataset-weight",
        dest="dataset_weights",
        action="append",
        type=float,
        default=[],
        help="Optional per-dataset training weight, in the same order as --dataset.",
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-ensemble-checkpoint", dest="base_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--base-model-weight", dest="base_model_weights", action="append", type=float, default=[])
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-ensemble-checkpoint", dest="specialist_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--specialist-model-weight", dest="specialist_model_weights", action="append", type=float, default=[])
    parser.add_argument("--model-ensemble-reduction", choices=["mean", "geomean"], default="mean")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument(
        "--fit-model",
        choices=["ridge", "mlp", "qvalue", "softmax", "pairwise"],
        default="ridge",
    )
    parser.add_argument("--ridge", type=float, default=0.05)
    parser.add_argument("--temperature", type=float, default=0.35)
    parser.add_argument("--target-clip", type=float, default=4.0)
    parser.add_argument("--hidden-layers", type=int, default=2)
    parser.add_argument("--hidden-units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=160)
    parser.add_argument("--learning-rate", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--validation-fraction", type=float, default=0.2)
    parser.add_argument("--negative-weight", type=float, default=2.0)
    parser.add_argument(
        "--negative-mass-coef",
        type=float,
        default=0.0,
        help="Extra softmax loss coefficient that penalizes probability mass on negative-labelled candidates.",
    )
    parser.add_argument("--positive-weight", type=float, default=1.0)
    parser.add_argument(
        "--qvalue-huber-beta",
        type=float,
        default=1.0,
        help="Huber transition point for --fit-model qvalue.",
    )
    parser.add_argument("--target-temperature", type=float, default=1.0)
    parser.add_argument(
        "--pairwise-negative-count",
        type=int,
        default=32,
        help="Number of hard lower-value candidates paired against the best candidate per state.",
    )
    parser.add_argument(
        "--pairwise-min-gap",
        type=float,
        default=0.25,
        help="Ignore label ties and near-ties smaller than this value in pairwise training.",
    )
    parser.add_argument(
        "--pairwise-temperature",
        type=float,
        default=0.5,
        help="Temperature used by the pairwise logistic ranking loss.",
    )
    parser.add_argument(
        "--pairwise-bad-negative-fraction",
        type=float,
        default=0.25,
        help="Fraction of pairwise negatives reserved for lowest-value candidates.",
    )
    parser.add_argument(
        "--pairwise-score-l2",
        type=float,
        default=1e-4,
        help="Light score-magnitude penalty for pairwise rankers so online softmax does not over-sharpen.",
    )
    parser.add_argument(
        "--pairwise-policy-topk",
        type=int,
        default=0,
        help="Restrict pairwise labels to the base/specialist OPEN top-K union. Use 0 to keep every legal OPEN cell.",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--base-risk-head-weight", type=float)
    parser.add_argument("--specialist-risk-head-weight", type=float)
    parser.add_argument(
        "--solver-risk-features",
        action="store_true",
        help="Include visible solver risk and frontier features in the candidate ranker.",
    )
    parser.add_argument(
        "--local-context-radius",
        type=int,
        default=0,
        help="Append visible board context around every candidate cell. Radius 2 means a 5x5 context.",
    )
    args = parser.parse_args()
    if int(args.local_context_radius) < 0:
        raise ValueError("--local-context-radius must be non-negative")
    feature_names = build_candidate_feature_names(
        solver_risk_features=bool(args.solver_risk_features),
        local_context_radius=int(args.local_context_radius),
    )

    transitions = []
    source_metadata = []
    dataset_weights = normalize_dataset_weights(args.dataset_weights, expected=len(args.dataset))
    for path, dataset_weight in zip(args.dataset, dataset_weights, strict=True):
        loaded, report = load_extreme_dataset(path)
        transitions.extend(weighted_transitions(loaded, dataset_weight=float(dataset_weight)))
        metadata = dict(report.get("metadata", report))
        metadata["dataset_weight"] = float(dataset_weight)
        source_metadata.append(metadata)
    source_transition_records = len(transitions)
    transitions = [
        transition
        for transition in transitions
        if transition.counterfactual_open_values is not None
        and np.isfinite(transition.counterfactual_open_values).any()
    ]
    counterfactual_transition_records = len(transitions)
    transitions = filter_training_transitions(
        transitions,
        family_filter=args.family_filter,
        safe_left_min=args.safe_left_min,
        safe_left_max=args.safe_left_max,
    )
    if args.max_records > 0:
        transitions = transitions[: args.max_records]
    if not transitions:
        raise ValueError("no counterfactual-labelled transitions were found")

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
        inference_flips=args.inference_flips,
        inference_ensemble=args.inference_ensemble,
        risk_head_weight=args.risk_head_weight,
    )
    base_risk_head_weight = (
        args.risk_head_weight if args.base_risk_head_weight is None else args.base_risk_head_weight
    )
    specialist_risk_head_weight = (
        args.risk_head_weight
        if args.specialist_risk_head_weight is None
        else args.specialist_risk_head_weight
    )
    bases = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=base_risk_head_weight)
        for path in base_paths
    ]
    specialists = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=specialist_risk_head_weight)
        for path in specialist_paths
    ]
    if args.fit_model == "ridge":
        features, targets = collect_candidate_rows(
            transitions=transitions,
            bases=bases,
            base_weights=base_weights,
            specialists=specialists,
            specialist_weights=specialist_weights,
            model_ensemble_reduction=args.model_ensemble_reduction,
            batch_size=args.batch_size,
            scores_are_probabilities=args.inference_ensemble == "probs",
            include_solver_risk_features=bool(args.solver_risk_features),
            local_context_radius=int(args.local_context_radius),
        )
        targets = np.clip(targets, -float(args.target_clip), float(args.target_clip)).astype(np.float32)
        calibrator, fit_metrics = fit_ridge(
            features,
            targets,
            ridge=float(args.ridge),
            temperature=float(args.temperature),
            feature_names=feature_names,
        )
    elif args.fit_model in {"mlp", "qvalue"}:
        features, targets = collect_candidate_rows(
            transitions=transitions,
            bases=bases,
            base_weights=base_weights,
            specialists=specialists,
            specialist_weights=specialist_weights,
            model_ensemble_reduction=args.model_ensemble_reduction,
            batch_size=args.batch_size,
            scores_are_probabilities=args.inference_ensemble == "probs",
            include_solver_risk_features=bool(args.solver_risk_features),
            local_context_radius=int(args.local_context_radius),
        )
        targets = np.clip(targets, -float(args.target_clip), float(args.target_clip)).astype(np.float32)
        calibrator, fit_metrics = fit_mlp(
            features,
            targets,
            hidden_layers=int(args.hidden_layers),
            hidden_units=int(args.hidden_units),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            validation_fraction=float(args.validation_fraction),
            negative_weight=float(args.negative_weight),
            positive_weight=float(args.positive_weight),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            temperature=float(args.temperature),
            feature_names=feature_names,
            loss_kind="huber" if args.fit_model == "qvalue" else "mse",
            huber_beta=float(args.qvalue_huber_beta),
        )
    elif args.fit_model == "softmax":
        (
            transition_feature_batches,
            transition_valid_masks,
            transition_targets,
            transition_values,
            transition_weights,
        ) = collect_candidate_transition_batches(
            transitions=transitions,
            bases=bases,
            base_weights=base_weights,
            specialists=specialists,
            specialist_weights=specialist_weights,
            model_ensemble_reduction=args.model_ensemble_reduction,
            batch_size=args.batch_size,
            scores_are_probabilities=args.inference_ensemble == "probs",
            target_temperature=float(args.target_temperature),
            target_clip=float(args.target_clip),
            include_solver_risk_features=bool(args.solver_risk_features),
            local_context_radius=int(args.local_context_radius),
        )
        calibrator, fit_metrics = fit_softmax(
            transition_features=transition_feature_batches,
            transition_valid_masks=transition_valid_masks,
            transition_targets=transition_targets,
            transition_values=transition_values,
            transition_weights=transition_weights,
            hidden_layers=int(args.hidden_layers),
            hidden_units=int(args.hidden_units),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            validation_fraction=float(args.validation_fraction),
            negative_weight=float(args.negative_weight),
            negative_mass_coef=float(args.negative_mass_coef),
            positive_weight=float(args.positive_weight),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            temperature=float(args.temperature),
            target_temperature=float(args.target_temperature),
            feature_names=feature_names,
        )
    else:
        (
            transition_feature_batches,
            transition_valid_masks,
            _transition_targets,
            transition_values,
            transition_weights,
        ) = collect_candidate_transition_batches(
            transitions=transitions,
            bases=bases,
            base_weights=base_weights,
            specialists=specialists,
            specialist_weights=specialist_weights,
            model_ensemble_reduction=args.model_ensemble_reduction,
            batch_size=args.batch_size,
            scores_are_probabilities=args.inference_ensemble == "probs",
            target_temperature=float(args.target_temperature),
            target_clip=float(args.target_clip),
            include_solver_risk_features=bool(args.solver_risk_features),
            local_context_radius=int(args.local_context_radius),
            policy_topk=int(args.pairwise_policy_topk),
        )
        calibrator, fit_metrics = fit_pairwise(
            transition_features=transition_feature_batches,
            transition_valid_masks=transition_valid_masks,
            transition_values=transition_values,
            transition_weights=transition_weights,
            hidden_layers=int(args.hidden_layers),
            hidden_units=int(args.hidden_units),
            epochs=int(args.epochs),
            learning_rate=float(args.learning_rate),
            weight_decay=float(args.weight_decay),
            validation_fraction=float(args.validation_fraction),
            seed=int(args.seed),
            batch_size=max(1, int(args.batch_size)),
            temperature=float(args.temperature),
            feature_names=feature_names,
            negative_count=int(args.pairwise_negative_count),
            min_gap=float(args.pairwise_min_gap),
            pairwise_temperature=float(args.pairwise_temperature),
            bad_negative_fraction=float(args.pairwise_bad_negative_fraction),
            score_l2=float(args.pairwise_score_l2),
        )
    calibrator.save(args.output)
    candidate_records = (
        int(len(targets))
        if args.fit_model in {"ridge", "mlp", "qvalue"}
        else int(np.asarray(transition_valid_masks, dtype=bool).sum())
    )
    report = {
        "ok": True,
        "output": str(args.output),
        "datasets": [str(path) for path in args.dataset],
        "dataset_weights": [float(weight) for weight in dataset_weights],
        "source_metadata": source_metadata,
        "source_transition_records": int(source_transition_records),
        "counterfactual_transition_records": int(counterfactual_transition_records),
        "transition_records": int(len(transitions)),
        "candidate_records": candidate_records,
        "family_filter": list(args.family_filter or []),
        "safe_left_min": args.safe_left_min,
        "safe_left_max": args.safe_left_max,
        "training_family_counts": transition_family_counts(transitions),
        "feature_names": list(feature_names),
        "base_checkpoints": [str(path) for path in base_paths],
        "base_model_weights": base_weights,
        "specialist_checkpoints": [str(path) for path in specialist_paths],
        "specialist_model_weights": specialist_weights,
        "model_ensemble_reduction": args.model_ensemble_reduction,
        "inference_flips": bool(args.inference_flips),
        "inference_ensemble": args.inference_ensemble,
        "risk_head_weight": float(args.risk_head_weight),
        "base_risk_head_weight": float(base_risk_head_weight),
        "specialist_risk_head_weight": float(specialist_risk_head_weight),
        "solver_risk_features": bool(args.solver_risk_features),
        "local_context_radius": int(args.local_context_radius),
        "fit_model": args.fit_model,
        "ridge": float(args.ridge),
        "temperature": float(args.temperature),
        "hidden_layers": int(args.hidden_layers),
        "hidden_units": int(args.hidden_units),
        "epochs": int(args.epochs),
        "learning_rate": float(args.learning_rate),
        "weight_decay": float(args.weight_decay),
        "validation_fraction": float(args.validation_fraction),
        "negative_weight": float(args.negative_weight),
        "negative_mass_coef": float(args.negative_mass_coef),
        "positive_weight": float(args.positive_weight),
        "qvalue_huber_beta": float(args.qvalue_huber_beta),
        "pairwise_negative_count": int(args.pairwise_negative_count),
        "pairwise_min_gap": float(args.pairwise_min_gap),
        "pairwise_temperature": float(args.pairwise_temperature),
        "pairwise_bad_negative_fraction": float(args.pairwise_bad_negative_fraction),
        "pairwise_score_l2": float(args.pairwise_score_l2),
        "pairwise_policy_topk": int(args.pairwise_policy_topk),
        "seed": int(args.seed),
        **fit_metrics,
    }
    report_path = args.output.with_name(args.output.name + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**report, "report": str(report_path)}, ensure_ascii=False, indent=2))


def filter_training_transitions(
    transitions,
    *,
    family_filter: list[str] | tuple[str, ...] | None = None,
    safe_left_min: int | None = None,
    safe_left_max: int | None = None,
):
    families = set(family_filter or [])
    filtered = []
    for transition in transitions:
        family = transition.extreme_family or "unclassified"
        if families and family not in families:
            continue
        safe_left = safe_left_from_transition(transition)
        if safe_left_min is not None and safe_left < int(safe_left_min):
            continue
        if safe_left_max is not None and safe_left > int(safe_left_max):
            continue
        filtered.append(transition)
    return filtered


def normalize_dataset_weights(weights: list[float], *, expected: int) -> list[float]:
    if expected <= 0:
        raise ValueError("at least one dataset is required")
    if not weights:
        return [1.0 for _ in range(expected)]
    if len(weights) != expected:
        raise ValueError(f"dataset-weight count must match dataset count: {len(weights)} != {expected}")
    normalized: list[float] = []
    for weight in weights:
        value = float(weight)
        if not np.isfinite(value) or value <= 0.0:
            raise ValueError("dataset weights must be finite positive values")
        normalized.append(value)
    return normalized


def weighted_transitions(transitions, *, dataset_weight: float):
    weight = float(dataset_weight)
    return [
        replace(
            transition,
            source_quality=float(transition.source_quality) * weight,
        )
        for transition in transitions
    ]


def safe_left_from_transition(transition, total_safe: int = 391) -> int:
    progress = float(np.clip(transition.global_features[1], 0.0, 1.0))
    return int(round((1.0 - progress) * int(total_safe)))


def transition_family_counts(transitions) -> dict[str, int]:
    counts = Counter(transition.extreme_family or "unclassified" for transition in transitions)
    return dict(sorted(counts.items()))


def _transition_risk_maps(transitions) -> np.ndarray:
    maps: list[np.ndarray] = []
    for transition in transitions:
        expected = tuple(np.asarray(transition.board).shape[-2:])
        risk_map = getattr(transition, "risk_map", None)
        if risk_map is None:
            maps.append(np.zeros(expected, dtype=np.float32))
            continue
        risk_array = np.asarray(risk_map, dtype=np.float32)
        if risk_array.shape != expected:
            raise ValueError(
                "transition risk_map shape does not match board: "
                f"{risk_array.shape} != {expected}"
            )
        maps.append(np.nan_to_num(risk_array, nan=0.0, posinf=1.0, neginf=0.0))
    return np.stack(maps).astype(np.float32)


def collect_candidate_rows(
    *,
    transitions,
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    model_ensemble_reduction: str,
    batch_size: int,
    scores_are_probabilities: bool,
    include_solver_risk_features: bool = False,
    local_context_radius: int = 0,
) -> tuple[np.ndarray, np.ndarray]:
    rows: list[np.ndarray] = []
    targets: list[np.ndarray] = []
    base = bases[0]
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
            base_scores = gated.combine_model_scores(
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
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            specialist_scores = gated.combine_model_scores(
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
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            feature_batch = build_candidate_features(
                base_scores=base_scores.detach().cpu().numpy(),
                specialist_scores=specialist_scores.detach().cpu().numpy(),
                global_features=global_features,
                boards=boards,
                action_masks=action_masks,
                scores_are_probabilities=scores_are_probabilities,
                risk_maps=(
                    _transition_risk_maps(batch)
                    if include_solver_risk_features
                    else None
                ),
                local_context_radius=int(local_context_radius),
            )
            for index, transition in enumerate(batch):
                values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
                legal = np.asarray(transition.action_mask[open_channel], dtype=bool).reshape(-1)
                valid = legal & np.isfinite(values)
                if valid.any():
                    rows.append(feature_batch[index, valid])
                    targets.append(values[valid])
    if not rows:
        raise ValueError("counterfactual records contain no legal OPEN candidates")
    return (
        np.concatenate(rows, axis=0).astype(np.float32),
        np.concatenate(targets, axis=0).astype(np.float32),
    )


def collect_candidate_transition_batches(
    *,
    transitions,
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    model_ensemble_reduction: str,
    batch_size: int,
    scores_are_probabilities: bool,
    target_temperature: float,
    target_clip: float,
    include_solver_risk_features: bool = False,
    local_context_radius: int = 0,
    policy_topk: int = 0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    feature_batches: list[np.ndarray] = []
    valid_masks: list[np.ndarray] = []
    target_batches: list[np.ndarray] = []
    sample_weights: list[float] = []
    value_batches: list[np.ndarray] = []
    base = bases[0]
    open_channel = action_channel(ActionType.OPEN)
    batch_size = max(1, int(batch_size))
    target_temperature = max(float(target_temperature), 1e-6)
    target_clip = max(float(target_clip), 1e-6)

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
            base_scores = gated.combine_model_scores(
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
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            specialist_scores = gated.combine_model_scores(
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
                reduction=model_ensemble_reduction,
                flat_mask=flat_mask,
            )
            feature_batch = build_candidate_features(
                base_scores=base_scores.detach().cpu().numpy(),
                specialist_scores=specialist_scores.detach().cpu().numpy(),
                global_features=global_features,
                boards=boards,
                action_masks=action_masks,
                scores_are_probabilities=scores_are_probabilities,
                risk_maps=(
                    _transition_risk_maps(batch)
                    if include_solver_risk_features
                    else None
                ),
                local_context_radius=int(local_context_radius),
            )
            policy_masks = (
                _policy_topk_open_masks(
                    base_scores=base_scores,
                    specialist_scores=specialist_scores,
                    action_masks=action_masks,
                    rows=boards.shape[-2],
                    cols=boards.shape[-1],
                    topk=int(policy_topk),
                )
                if int(policy_topk) > 0
                else None
            )
            for index, transition in enumerate(batch):
                values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
                legal = np.asarray(transition.action_mask[open_channel], dtype=bool).reshape(-1)
                valid = legal & np.isfinite(values)
                if policy_masks is not None:
                    valid &= policy_masks[index]
                if not valid.any():
                    continue
                target = _softmax_target_distribution(
                    values,
                    valid,
                    temperature=target_temperature,
                    target_clip=target_clip,
                )
                feature_batches.append(feature_batch[index])
                valid_masks.append(valid)
                target_batches.append(target)
                value_batches.append(values)
                sample_weights.append(
                    _transition_rank_weight(
                        transition=transition,
                        values=values,
                        valid=valid,
                        target_clip=target_clip,
                    )
                )

    if not feature_batches:
        raise ValueError("counterfactual records contain no legal OPEN candidates")
    return (
        np.stack(feature_batches).astype(np.float32),
        np.stack(valid_masks).astype(bool),
        np.stack(target_batches).astype(np.float32),
        np.stack(value_batches).astype(np.float32),
        np.asarray(sample_weights, dtype=np.float32),
    )


def _policy_topk_open_masks(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    topk: int,
) -> np.ndarray:
    """Return the base/specialist OPEN shortlist used by policy-only reranking."""

    if int(topk) <= 0:
        return np.asarray(
            action_masks[:, action_channel(ActionType.OPEN)],
            dtype=bool,
        ).reshape(-1, int(rows) * int(cols))
    return gated._candidate_shortlist_mask(
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        action_masks=action_masks,
        rows=int(rows),
        cols=int(cols),
        topk=int(topk),
        candidate_probabilities=None,
    )


def _softmax_target_distribution(
    values: np.ndarray,
    valid: np.ndarray,
    *,
    temperature: float,
    target_clip: float,
) -> np.ndarray:
    scores = np.asarray(values, dtype=np.float32).copy()
    scores[~valid] = -np.inf
    scores = np.clip(scores, -float(target_clip), float(target_clip))
    finite = np.isfinite(scores)
    if not finite.any():
        target = valid.astype(np.float32)
        target /= max(float(target.sum()), 1.0)
        return target
    masked = scores / max(float(temperature), 1e-6)
    masked[~finite] = -1e9
    masked -= float(np.max(masked[finite]))
    exponentials = np.exp(np.clip(masked, -80.0, 0.0)) * valid.astype(np.float32)
    total = float(exponentials.sum())
    if total <= 1e-8:
        target = valid.astype(np.float32)
        target /= max(float(target.sum()), 1.0)
        return target
    return (exponentials / total).astype(np.float32)


def _transition_rank_weight(
    *,
    transition,
    values: np.ndarray,
    valid: np.ndarray,
    target_clip: float,
) -> float:
    masked = np.asarray(values, dtype=np.float32).copy()
    masked[~valid] = -np.inf
    if not np.isfinite(masked).any():
        return 1.0
    best_value = float(np.max(masked[np.isfinite(masked)]))
    behavior_value = _behavior_open_value(transition, values, valid)
    regret = 0.0 if behavior_value is None else max(best_value - behavior_value, 0.0)
    source_quality = float(getattr(transition, "source_quality", 1.0))
    if not np.isfinite(source_quality) or source_quality <= 0.0:
        source_quality = 1.0
    return float(source_quality * (1.0 + np.clip(regret / max(target_clip, 1e-6), 0.0, 1.0)))


def _behavior_open_value(
    transition,
    values: np.ndarray,
    valid: np.ndarray,
) -> float | None:
    if transition.action_mask.ndim != 3:
        return None
    rows, cols = transition.action_mask.shape[1:]
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != 0:
        return None
    if not (0 <= cell_index < cells):
        return None
    flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
    if not bool(valid[cell_index]):
        return None
    value = float(flat_values[cell_index])
    return value if np.isfinite(value) else None


def fit_ridge(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    ridge: float,
    temperature: float,
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES,
) -> tuple[CandidateValueCalibrator, dict[str, float]]:
    means = features.mean(axis=0)
    scales = np.maximum(features.std(axis=0), 1e-5)
    x = (features - means) / scales
    y = targets.astype(np.float64)
    design = np.concatenate([x.astype(np.float64), np.ones((x.shape[0], 1), dtype=np.float64)], axis=1)
    penalty = np.eye(design.shape[1], dtype=np.float64) * float(ridge)
    penalty[-1, -1] = 0.0
    params = np.linalg.solve(design.T @ design + penalty, design.T @ y)
    predictions = design @ params
    errors = predictions - y
    return (
        CandidateValueCalibrator(
            means=means.astype(np.float32),
            scales=scales.astype(np.float32),
            weights=params[:-1].astype(np.float32),
            bias=float(params[-1]),
            temperature=temperature,
            feature_names=feature_names,
        ),
        {
            "target_mean": float(y.mean()),
            "target_std": float(y.std()),
            "prediction_mean": float(predictions.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "positive_target_rate": float(np.mean(y > 0.0)),
            "mine_target_rate": float(np.mean(y < 0.0)),
        },
    )


class _CandidateMLP(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_layers: int, hidden_units: int) -> None:
        super().__init__()
        hidden_layers = max(0, int(hidden_layers))
        hidden_units = max(4, int(hidden_units))
        layers: list[torch.nn.Module] = []
        width = int(input_dim)
        for _ in range(hidden_layers):
            layers.append(torch.nn.Linear(width, hidden_units))
            layers.append(torch.nn.SiLU())
            width = hidden_units
        layers.append(torch.nn.Linear(width, 1))
        self.network = torch.nn.Sequential(*layers)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.network(features).squeeze(-1)


def fit_mlp(
    features: np.ndarray,
    targets: np.ndarray,
    *,
    hidden_layers: int,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    negative_weight: float,
    positive_weight: float,
    seed: int,
    batch_size: int,
    temperature: float,
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES,
    loss_kind: str = "mse",
    huber_beta: float = 1.0,
) -> tuple[NeuralCandidateValueCalibrator, dict[str, float]]:
    if features.ndim != 2:
        raise ValueError("features must be a 2D candidate matrix")
    if targets.ndim != 1 or targets.shape[0] != features.shape[0]:
        raise ValueError("targets must be a 1D vector matching features")
    if loss_kind not in {"mse", "huber"}:
        raise ValueError(f"unknown pointwise regression loss {loss_kind!r}")
    huber_beta = max(float(huber_beta), 1e-6)

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(features.shape[0])
    validation_fraction = float(np.clip(validation_fraction, 0.0, 0.5))
    validation_count = int(round(features.shape[0] * validation_fraction))
    if validation_count <= 0 and features.shape[0] >= 10:
        validation_count = max(1, features.shape[0] // 5)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if train_indices.size == 0:
        train_indices = indices
        validation_indices = indices[:0]

    train_features = features[train_indices]
    means = train_features.mean(axis=0)
    scales = np.maximum(train_features.std(axis=0), 1e-5)
    normalized = ((features - means) / scales).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _CandidateMLP(
        input_dim=features.shape[1],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(learning_rate), 1e-6),
        weight_decay=max(float(weight_decay), 0.0),
    )
    x_all = torch.tensor(normalized, dtype=torch.float32)
    y_all = torch.tensor(targets.astype(np.float32), dtype=torch.float32)
    sample_weights = torch.where(
        y_all < 0.0,
        torch.full_like(y_all, max(float(negative_weight), 0.0)),
        torch.full_like(y_all, max(float(positive_weight), 0.0)),
    ).clamp_min(1e-3)
    x_train = x_all[train_indices].to(device)
    y_train = y_all[train_indices].to(device)
    weight_train = sample_weights[train_indices].to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    final_train_loss = 0.0
    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0
        seen = 0
        for start in range(0, x_train.shape[0], batch_size):
            batch_index = order[start : start + batch_size]
            predictions = model(x_train[batch_index])
            if loss_kind == "huber":
                losses = torch.nn.functional.smooth_l1_loss(
                    predictions,
                    y_train[batch_index],
                    reduction="none",
                    beta=huber_beta,
                )
            else:
                losses = (predictions - y_train[batch_index]).square()
            losses = losses * weight_train[batch_index]
            loss = losses.mean()
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * int(batch_index.numel())
            seen += int(batch_index.numel())
        final_train_loss = total_loss / max(1, seen)

        model.eval()
        with torch.inference_mode():
            if validation_indices.size > 0:
                validation_predictions = model(x_all[validation_indices].to(device)).detach().cpu().numpy()
                validation_error = validation_predictions - targets[validation_indices]
                if loss_kind == "huber":
                    absolute_error = np.abs(validation_error)
                    quadratic = np.minimum(absolute_error, huber_beta)
                    linear = absolute_error - quadratic
                    validation_score = float(
                        np.mean(0.5 * quadratic**2 + huber_beta * linear)
                    )
                else:
                    validation_score = float(np.sqrt(np.mean(validation_error**2)))
            else:
                validation_score = final_train_loss
        if validation_score < best_validation:
            best_validation = validation_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(x_all.to(device)).detach().cpu().numpy().astype(np.float32)

    errors = predictions.astype(np.float64) - targets.astype(np.float64)
    train_predictions = predictions[train_indices]
    train_errors = train_predictions.astype(np.float64) - targets[train_indices].astype(np.float64)
    validation_errors = (
        predictions[validation_indices].astype(np.float64) - targets[validation_indices].astype(np.float64)
        if validation_indices.size > 0
        else np.asarray([], dtype=np.float64)
    )
    calibrator = NeuralCandidateValueCalibrator(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        layers=_export_mlp_layers(model),
        temperature=temperature,
        feature_names=feature_names,
    )
    return (
        calibrator,
        {
            "target_mean": float(targets.mean()),
            "target_std": float(targets.std()),
            "prediction_mean": float(predictions.mean()),
            "rmse": float(np.sqrt(np.mean(errors**2))),
            "mae": float(np.mean(np.abs(errors))),
            "train_rmse": float(np.sqrt(np.mean(train_errors**2))),
            "train_mae": float(np.mean(np.abs(train_errors))),
            "validation_rmse": float(np.sqrt(np.mean(validation_errors**2))) if validation_errors.size else 0.0,
            "validation_mae": float(np.mean(np.abs(validation_errors))) if validation_errors.size else 0.0,
            "best_validation_rmse": float(best_validation),
            "final_train_weighted_mse": float(final_train_loss),
            "loss_kind": loss_kind,
            "huber_beta": float(huber_beta),
            "positive_target_rate": float(np.mean(targets > 0.0)),
            "mine_target_rate": float(np.mean(targets < 0.0)),
        },
    )


def fit_softmax(
    *,
    transition_features: np.ndarray,
    transition_valid_masks: np.ndarray,
    transition_targets: np.ndarray,
    transition_values: np.ndarray,
    transition_weights: np.ndarray,
    hidden_layers: int,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    negative_weight: float,
    negative_mass_coef: float,
    positive_weight: float,
    seed: int,
    batch_size: int,
    temperature: float,
    target_temperature: float,
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES,
) -> tuple[NeuralCandidateValueCalibrator, dict[str, float]]:
    if transition_features.ndim != 3:
        raise ValueError("transition_features must have shape [batch, cells, features]")
    if transition_valid_masks.shape != transition_targets.shape:
        raise ValueError("transition_valid_masks and transition_targets must have the same shape")
    if transition_values.shape != transition_targets.shape:
        raise ValueError("transition_values and transition_targets must have the same shape")
    if transition_weights.ndim != 1 or transition_weights.shape[0] != transition_features.shape[0]:
        raise ValueError("transition_weights must match the number of transitions")

    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(transition_features.shape[0])
    validation_fraction = float(np.clip(validation_fraction, 0.0, 0.5))
    validation_count = int(round(transition_features.shape[0] * validation_fraction))
    if validation_count <= 0 and transition_features.shape[0] >= 10:
        validation_count = max(1, transition_features.shape[0] // 5)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if train_indices.size == 0:
        train_indices = indices
        validation_indices = indices[:0]

    train_valid = transition_valid_masks[train_indices]
    train_features = transition_features[train_indices][train_valid]
    means = train_features.mean(axis=0)
    scales = np.maximum(train_features.std(axis=0), 1e-5)
    normalized = ((transition_features - means) / scales).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _CandidateMLP(
        input_dim=transition_features.shape[-1],
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(learning_rate), 1e-6),
        weight_decay=max(float(weight_decay), 0.0),
    )

    best_values = np.max(np.where(transition_valid_masks, transition_values, -np.inf), axis=1)
    label_weights = np.where(best_values < 0.0, float(negative_weight), float(positive_weight)).astype(np.float32)
    sample_weights = np.asarray(transition_weights, dtype=np.float32) * label_weights
    sample_weights = np.clip(sample_weights, 1e-3, None)
    target_dists = np.stack(
        [
            _softmax_target_distribution(
                values=row_values,
                valid=row_valid,
                temperature=target_temperature,
                target_clip=float(np.max(np.abs(row_values[np.isfinite(row_values)]))) if np.isfinite(row_values).any() else 1.0,
            )
            for row_values, row_valid in zip(transition_values, transition_valid_masks, strict=True)
        ],
        axis=0,
    )

    x_all = torch.tensor(normalized, dtype=torch.float32)
    valid_all = torch.tensor(transition_valid_masks, dtype=torch.bool)
    target_all = torch.tensor(target_dists, dtype=torch.float32)
    negative_all = torch.tensor(
        np.asarray(transition_valid_masks, dtype=bool) & (np.asarray(transition_values, dtype=np.float32) < 0.0),
        dtype=torch.bool,
    )
    weight_all = torch.tensor(sample_weights, dtype=torch.float32)
    x_train = x_all[train_indices].to(device)
    valid_train = valid_all[train_indices].to(device)
    target_train = target_all[train_indices].to(device)
    negative_train = negative_all[train_indices].to(device)
    weight_train = weight_all[train_indices].to(device)

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    final_train_loss = 0.0
    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    cells = transition_features.shape[1]
    feature_count = transition_features.shape[2]

    for _epoch in range(epochs):
        model.train()
        order = torch.randperm(x_train.shape[0], device=device)
        total_loss = 0.0
        seen = 0
        for start in range(0, x_train.shape[0], batch_size):
            batch_index = order[start : start + batch_size]
            batch_features = x_train[batch_index].reshape(-1, feature_count)
            logits = model(batch_features).reshape(-1, cells)
            masked_logits = logits.masked_fill(~valid_train[batch_index], -1e9)
            log_probs = masked_logits - torch.logsumexp(masked_logits, dim=1, keepdim=True)
            loss = _weighted_softmax_rank_loss(
                log_probs=log_probs,
                targets=target_train[batch_index],
                negative_mask=negative_train[batch_index],
                weights=weight_train[batch_index],
                negative_mass_coef=float(negative_mass_coef),
            )
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().cpu().item()) * int(batch_index.numel())
            seen += int(batch_index.numel())
        final_train_loss = total_loss / max(1, seen)

        model.eval()
        with torch.inference_mode():
            if validation_indices.size > 0:
                validation_features = x_all[validation_indices].to(device).reshape(-1, feature_count)
                validation_logits = model(validation_features).reshape(-1, cells)
                validation_masked = validation_logits.masked_fill(~valid_all[validation_indices].to(device), -1e9)
                validation_log_probs = validation_masked - torch.logsumexp(validation_masked, dim=1, keepdim=True)
                validation_loss = _weighted_softmax_rank_loss(
                    log_probs=validation_log_probs,
                    targets=target_all[validation_indices].to(device),
                    negative_mask=negative_all[validation_indices].to(device),
                    weights=weight_all[validation_indices].to(device),
                    negative_mass_coef=float(negative_mass_coef),
                )
                validation_score = float(validation_loss.detach().cpu().item())
            else:
                validation_score = final_train_loss
        if validation_score < best_validation:
            best_validation = validation_score
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(x_all.to(device).reshape(-1, feature_count)).reshape(-1, cells).detach().cpu().numpy()

    metrics = _score_softmax_ranker(
        predictions=predictions,
        values=transition_values,
        valid_masks=transition_valid_masks,
    )
    calibrator = NeuralCandidateValueCalibrator(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        layers=_export_mlp_layers(model),
        temperature=temperature,
        feature_names=feature_names,
    )
    return (
        calibrator,
        {
            "target_mean": float(transition_targets.mean()),
            "target_std": float(transition_targets.std()),
            "prediction_mean": float(predictions.mean()),
            "train_loss": float(final_train_loss),
            "validation_loss": float(best_validation),
            "best_validation_loss": float(best_validation),
            "positive_target_rate": float(np.mean(best_values > 0.0)),
            "mine_target_rate": float(np.mean(best_values < 0.0)),
            **metrics,
        },
    )


def fit_pairwise(
    *,
    transition_features: np.ndarray,
    transition_valid_masks: np.ndarray,
    transition_values: np.ndarray,
    transition_weights: np.ndarray,
    hidden_layers: int,
    hidden_units: int,
    epochs: int,
    learning_rate: float,
    weight_decay: float,
    validation_fraction: float,
    seed: int,
    batch_size: int,
    temperature: float,
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES,
    negative_count: int = 32,
    min_gap: float = 0.25,
    pairwise_temperature: float = 0.5,
    bad_negative_fraction: float = 0.25,
    score_l2: float = 1e-4,
) -> tuple[NeuralCandidateValueCalibrator, dict[str, float]]:
    """Train a state-local ranker using best-candidate versus hard-negative pairs."""

    transition_features = np.asarray(transition_features, dtype=np.float32)
    transition_valid_masks = np.asarray(transition_valid_masks, dtype=bool)
    transition_values = np.asarray(transition_values, dtype=np.float32)
    transition_weights = np.asarray(transition_weights, dtype=np.float32).reshape(-1)
    if transition_features.ndim != 3:
        raise ValueError("transition_features must have shape [batch, cells, features]")
    if transition_valid_masks.shape != transition_values.shape:
        raise ValueError("transition_valid_masks and transition_values must have the same shape")
    if transition_valid_masks.shape != transition_features.shape[:2]:
        raise ValueError("transition_valid_masks must match [batch, cells]")
    if transition_weights.shape[0] != transition_features.shape[0]:
        raise ValueError("transition_weights must match the number of transitions")
    if int(negative_count) <= 0:
        raise ValueError("negative_count must be positive")
    if float(pairwise_temperature) <= 0.0:
        raise ValueError("pairwise_temperature must be positive")
    bad_negative_fraction = float(np.clip(bad_negative_fraction, 0.0, 1.0))
    score_l2 = max(float(score_l2), 0.0)

    transition_count, cells, feature_count = transition_features.shape
    rng = np.random.default_rng(int(seed))
    indices = rng.permutation(transition_count)
    validation_fraction = float(np.clip(validation_fraction, 0.0, 0.5))
    validation_count = int(round(transition_count * validation_fraction))
    if validation_count <= 0 and transition_count >= 10:
        validation_count = max(1, transition_count // 5)
    validation_indices = indices[:validation_count]
    train_indices = indices[validation_count:]
    if train_indices.size == 0:
        train_indices = indices
        validation_indices = indices[:0]

    positive_indices = np.zeros(transition_count, dtype=np.int64)
    negative_indices = np.full(
        (transition_count, max(1, int(negative_count))),
        -1,
        dtype=np.int64,
    )
    pair_mask = np.zeros_like(negative_indices, dtype=bool)
    pair_weights = np.zeros_like(negative_indices, dtype=np.float32)
    for index in range(transition_count):
        valid = transition_valid_masks[index] & np.isfinite(transition_values[index])
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            continue
        values = transition_values[index]
        best_index = int(valid_indices[np.argmax(values[valid_indices])])
        positive_indices[index] = best_index
        lower = valid_indices[values[valid_indices] < values[best_index] - float(min_gap)]
        if lower.size == 0:
            lower = valid_indices[values[valid_indices] < values[best_index]]
        if lower.size == 0:
            continue
        ordered_hard = lower[np.argsort(values[lower])[::-1]]
        ordered_bad = lower[np.argsort(values[lower])]
        limit = min(int(negative_count), ordered_hard.size)
        bad_limit = min(limit, int(round(limit * bad_negative_fraction)))
        hard_limit = limit - bad_limit
        selected_list: list[int] = []
        for candidate in ordered_hard[:hard_limit]:
            selected_list.append(int(candidate))
        for candidate in ordered_bad[:bad_limit]:
            candidate = int(candidate)
            if candidate not in selected_list:
                selected_list.append(candidate)
        if len(selected_list) < limit:
            for candidate in ordered_hard:
                candidate = int(candidate)
                if candidate in selected_list:
                    continue
                selected_list.append(candidate)
                if len(selected_list) >= limit:
                    break
        selected = np.asarray(selected_list[:limit], dtype=np.int64)
        negative_indices[index, :limit] = selected
        pair_mask[index, :limit] = True
        gaps = values[best_index] - values[selected]
        pair_weights[index, :limit] = np.clip(
            transition_weights[index] * (0.5 + gaps),
            0.5,
            5.0,
        )

    usable = pair_mask.any(axis=1)
    if not bool(usable.any()):
        raise ValueError("counterfactual records contain no usable pairwise ranking targets")
    pair_weights[usable] /= max(float(pair_weights[usable].mean()), 1e-6)

    train_features = transition_features[train_indices][transition_valid_masks[train_indices]]
    if train_features.size == 0:
        raise ValueError("pairwise training split contains no legal candidates")
    means = train_features.mean(axis=0)
    scales = np.maximum(train_features.std(axis=0), 1e-5)
    normalized = ((transition_features - means) / scales).astype(np.float32)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = _CandidateMLP(
        input_dim=feature_count,
        hidden_layers=hidden_layers,
        hidden_units=hidden_units,
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=max(float(learning_rate), 1e-6),
        weight_decay=max(float(weight_decay), 0.0),
    )
    x_all = torch.tensor(normalized, dtype=torch.float32)
    positive_all = torch.tensor(positive_indices, dtype=torch.long)
    negative_all = torch.tensor(negative_indices, dtype=torch.long)
    pair_mask_all = torch.tensor(pair_mask, dtype=torch.bool)
    pair_weights_all = torch.tensor(pair_weights, dtype=torch.float32)

    def batch_loss(batch_index: torch.Tensor) -> torch.Tensor:
        batch_features = x_all[batch_index].to(device).reshape(-1, feature_count)
        predictions = model(batch_features).reshape(-1, cells)
        positive = predictions.gather(1, positive_all[batch_index].to(device).view(-1, 1))
        negative_index = negative_all[batch_index].to(device).clamp_min(0)
        negative = predictions.gather(1, negative_index)
        logits = (positive - negative) / max(float(pairwise_temperature), 1e-6)
        losses = torch.nn.functional.softplus(-logits)
        mask = pair_mask_all[batch_index].to(device)
        weights = pair_weights_all[batch_index].to(device)
        pair_loss = (losses * mask.to(dtype=losses.dtype) * weights).sum() / mask.sum().clamp_min(1)
        if score_l2 > 0.0:
            pair_loss = pair_loss + score_l2 * predictions.square().mean()
        return pair_loss

    best_state: dict[str, torch.Tensor] | None = None
    best_validation = float("inf")
    final_train_loss = 0.0
    epochs = max(1, int(epochs))
    batch_size = max(1, int(batch_size))
    for _epoch in range(epochs):
        model.train()
        order = torch.as_tensor(
            rng.permutation(train_indices),
            dtype=torch.long,
        )
        total_loss = 0.0
        seen_pairs = 0
        for start in range(0, order.numel(), batch_size):
            batch_index = order[start : start + batch_size]
            if not bool(pair_mask_all[batch_index].any()):
                continue
            loss = batch_loss(batch_index)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            pair_count = int(pair_mask_all[batch_index].sum().item())
            total_loss += float(loss.detach().cpu().item()) * pair_count
            seen_pairs += pair_count
        final_train_loss = total_loss / max(1, seen_pairs)

        model.eval()
        with torch.inference_mode():
            if validation_indices.size > 0 and bool(pair_mask_all[validation_indices].any()):
                validation_index = torch.as_tensor(validation_indices, dtype=torch.long)
                validation_loss = float(batch_loss(validation_index).detach().cpu().item())
            else:
                validation_loss = final_train_loss
        if validation_loss < best_validation:
            best_validation = validation_loss
            best_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }

    if best_state is not None:
        model.load_state_dict(best_state)
    model.eval()
    with torch.inference_mode():
        predictions = model(x_all.to(device).reshape(-1, feature_count))
        predictions = predictions.reshape(-1, cells).detach().cpu().numpy().astype(np.float32)

    metrics = _score_softmax_ranker(
        predictions=predictions,
        values=transition_values,
        valid_masks=transition_valid_masks,
    )
    train_pair_count = int(pair_mask[train_indices].sum())
    validation_pair_count = int(pair_mask[validation_indices].sum()) if validation_indices.size else 0
    calibrator = NeuralCandidateValueCalibrator(
        means=means.astype(np.float32),
        scales=scales.astype(np.float32),
        layers=_export_mlp_layers(model),
        temperature=temperature,
        feature_names=feature_names,
    )
    return (
        calibrator,
        {
            "target_mean": float(transition_values[transition_valid_masks].mean()),
            "target_std": float(transition_values[transition_valid_masks].std()),
            "prediction_mean": float(predictions.mean()),
            "train_loss": float(final_train_loss),
            "validation_loss": float(best_validation),
            "best_validation_loss": float(best_validation),
            "train_pair_count": float(train_pair_count),
            "validation_pair_count": float(validation_pair_count),
            "pairwise_negative_count": float(negative_count),
            "pairwise_min_gap": float(min_gap),
            "pairwise_temperature": float(pairwise_temperature),
            "pairwise_bad_negative_fraction": float(bad_negative_fraction),
            "pairwise_score_l2": float(score_l2),
            **metrics,
        },
    )


def _weighted_softmax_rank_loss(
    *,
    log_probs: torch.Tensor,
    targets: torch.Tensor,
    negative_mask: torch.Tensor,
    weights: torch.Tensor,
    negative_mass_coef: float,
) -> torch.Tensor:
    loss = -(targets * log_probs).sum(dim=1)
    if float(negative_mass_coef) > 0.0:
        loss = loss + float(negative_mass_coef) * _negative_probability_mass_penalty(
            log_probs=log_probs,
            negative_mask=negative_mask,
        )
    return (loss * weights).mean()


def _negative_probability_mass_penalty(
    *,
    log_probs: torch.Tensor,
    negative_mask: torch.Tensor,
) -> torch.Tensor:
    if negative_mask.shape != log_probs.shape:
        raise ValueError("negative_mask must match log_probs")
    probabilities = log_probs.exp()
    negative_mass = (probabilities * negative_mask.to(dtype=probabilities.dtype)).sum(dim=1)
    negative_mass = negative_mass.clamp(0.0, 1.0 - 1e-6)
    return -torch.log1p(-negative_mass)


def _score_softmax_ranker(
    *,
    predictions: np.ndarray,
    values: np.ndarray,
    valid_masks: np.ndarray,
    near_best_margin: float = 0.25,
) -> dict[str, float]:
    best_match = 0.0
    near_best = 0.0
    chosen_negative = 0.0
    regrets: list[float] = []
    best_ranks: list[float] = []
    negative_masses: list[float] = []
    states_with_negative_candidates = 0.0
    for predicted_row, value_row, valid in zip(predictions, values, valid_masks, strict=True):
        valid = np.asarray(valid, dtype=bool)
        if not valid.any():
            continue
        predicted = np.asarray(predicted_row, dtype=np.float32).copy()
        labelled = np.asarray(value_row, dtype=np.float32).copy()
        predicted[~valid] = -np.inf
        labelled[~valid] = -np.inf
        best = int(np.argmax(labelled))
        chosen = int(np.argmax(predicted))
        best_value = float(labelled[best])
        chosen_value = float(labelled[chosen])
        regret = best_value - chosen_value
        valid_indices = np.flatnonzero(valid)
        sorted_by_prediction = valid_indices[np.argsort(predicted[valid_indices])[::-1]]
        best_rank = int(np.flatnonzero(sorted_by_prediction == best)[0]) + 1
        negative = valid & (np.asarray(value_row, dtype=np.float32) < 0.0)
        negative_mass = 0.0
        if bool(negative.any()):
            states_with_negative_candidates += 1.0
            shifted = predicted[valid_indices] - float(np.max(predicted[valid_indices]))
            probabilities = np.exp(np.clip(shifted, -80.0, 0.0))
            probabilities = probabilities / max(float(probabilities.sum()), 1e-8)
            negative_mass = float(probabilities[np.isin(valid_indices, np.flatnonzero(negative))].sum())
        best_match += float(chosen == best)
        near_best += float(regret <= near_best_margin)
        chosen_negative += float(chosen_value < 0.0)
        regrets.append(float(regret))
        best_ranks.append(float(best_rank))
        negative_masses.append(negative_mass)
    records = max(len(regrets), 1)
    return {
        "best_match_rate": best_match / records,
        "near_best_rate": near_best / records,
        "chosen_negative_rate": chosen_negative / records,
        "avg_regret": float(np.mean(regrets)) if regrets else None,
        "avg_best_rank": float(np.mean(best_ranks)) if best_ranks else None,
        "avg_negative_probability_mass": float(np.mean(negative_masses)) if negative_masses else None,
        "states_with_negative_candidates_rate": states_with_negative_candidates / records,
    }


def _export_mlp_layers(model: _CandidateMLP) -> list[dict[str, object]]:
    exported: list[dict[str, object]] = []
    linear_layers = [module for module in model.network if isinstance(module, torch.nn.Linear)]
    for index, layer in enumerate(linear_layers):
        activation = "identity" if index == len(linear_layers) - 1 else "silu"
        exported.append(
            {
                "weights": layer.weight.detach().cpu().numpy().T.astype(np.float32),
                "bias": layer.bias.detach().cpu().numpy().astype(np.float32),
                "activation": activation,
            }
        )
    return exported


if __name__ == "__main__":
    main()
