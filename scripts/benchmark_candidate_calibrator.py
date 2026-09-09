from __future__ import annotations

import argparse
import importlib.util
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

from minesweeper_rl.candidate_calibration import (
    candidate_feature_options,
    candidate_policy_probabilities,
    load_candidate_calibrator,
)
from minesweeper_rl.candidate_gate import (
    CandidateGateCalibrator,
    build_candidate_gate_features,
)
from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.features import action_channel
from minesweeper_rl.trainer import load_checkpoint
from minesweeper_rl.types import ActionType


GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_for_candidate_benchmark", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


SAFE_LEFT_BUCKETS = (
    (1, 60, "1_60"),
    (61, 140, "61_140"),
    (141, 240, "141_240"),
    (241, 391, "241_391"),
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Benchmark a candidate ranker on fixed counterfactual hard-case states."
    )
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--base-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--base-model-weight", action="append", type=float, default=[])
    parser.add_argument(
        "--risk-head-weight",
        type=float,
        default=0.0,
        help="Default learned-risk-head weight for base and specialist policy scoring.",
    )
    parser.add_argument(
        "--base-risk-head-weight",
        type=float,
        default=None,
        help="Override the learned-risk-head weight for base checkpoints.",
    )
    parser.add_argument("--specialist-checkpoint", action="append", type=Path, required=True)
    parser.add_argument("--specialist-model-weight", action="append", type=float, default=[])
    parser.add_argument(
        "--specialist-risk-head-weight",
        type=float,
        default=None,
        help="Override the learned-risk-head weight for specialist checkpoints.",
    )
    parser.add_argument("--candidate-calibrator", type=Path, required=True)
    parser.add_argument("--candidate-gate-calibrator", type=Path, default=None)
    parser.add_argument("--candidate-gate-threshold", type=float, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--max-records", type=int, default=0)
    parser.add_argument("--near-best-margin", type=float, default=0.25)
    parser.add_argument(
        "--candidate-weight",
        type=float,
        default=0.0,
        help="Also evaluate the candidate after reproducing the online score blend.",
    )
    parser.add_argument("--candidate-mode", choices=["blend", "replace"], default="blend")
    parser.add_argument("--candidate-topk", type=int, default=64)
    parser.add_argument(
        "--candidate-shortlist-source",
        choices=["all", "policy"],
        default="all",
        help="Use candidate top-K cells in the rerank shortlist, or restrict reranking to base/specialist policy candidates.",
    )
    parser.add_argument("--candidate-policy-margin-max", type=float, default=None)
    parser.add_argument("--candidate-ranker-margin-min", type=float, default=None)
    parser.add_argument(
        "--candidate-allow-non-open",
        action="store_true",
        help="Allow candidate reranking when the current full-policy action is FLAG/other, not only OPEN.",
    )
    parser.add_argument(
        "--candidate-risk-safety-filter",
        choices=["none", "not-higher", "strictly-lower"],
        default="none",
    )
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    transitions, dataset_manifest = load_extreme_dataset(args.dataset)
    if int(args.max_records) > 0:
        transitions = transitions[: int(args.max_records)]
    if not transitions:
        raise ValueError("dataset contains no transitions")

    calibrator = load_candidate_calibrator(args.candidate_calibrator)
    candidate_gate = (
        CandidateGateCalibrator.load(args.candidate_gate_calibrator)
        if args.candidate_gate_calibrator
        else None
    )
    if candidate_gate is not None and args.candidate_gate_threshold is not None:
        candidate_gate.threshold = float(np.clip(args.candidate_gate_threshold, 0.0, 1.0))
    feature_options = candidate_feature_options(calibrator.feature_names)
    eval_args = argparse.Namespace(
        device=args.device,
        max_steps=600,
        decision_actions="full",
        inference_flips=args.inference_flips,
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
        for path in args.base_checkpoint
    ]
    specialists = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=specialist_risk_head_weight)
        for path in args.specialist_checkpoint
    ]
    base_weights = gated.normalize_ensemble_weights(
        args.base_model_weight,
        expected=len(bases),
        label="base-model-weight",
    )
    specialist_weights = gated.normalize_ensemble_weights(
        args.specialist_model_weight,
        expected=len(specialists),
        label="specialist-model-weight",
    )

    started_at = time.time()
    result = benchmark_transitions(
        transitions=transitions,
        bases=bases,
        base_weights=base_weights,
        specialists=specialists,
        specialist_weights=specialist_weights,
        calibrator=calibrator,
        model_ensemble_reduction="mean",
        batch_size=max(1, int(args.batch_size)),
        scores_are_probabilities=args.inference_ensemble == "probs",
        near_best_margin=float(args.near_best_margin),
        include_solver_risk_features=bool(feature_options["solver_risk_features"]),
        candidate_weight=float(args.candidate_weight),
        candidate_mode=args.candidate_mode,
        candidate_topk=int(args.candidate_topk),
        candidate_shortlist_source=args.candidate_shortlist_source,
        candidate_policy_margin_max=args.candidate_policy_margin_max,
        candidate_ranker_margin_min=args.candidate_ranker_margin_min,
        candidate_risk_safety_filter=args.candidate_risk_safety_filter,
        candidate_require_open=not bool(args.candidate_allow_non_open),
        candidate_gate=candidate_gate,
    )
    payload = {
        "ok": True,
        "dataset": str(args.dataset),
        "dataset_records": len(transitions),
        "dataset_manifest": dataset_manifest,
        "base_checkpoints": [str(path) for path in args.base_checkpoint],
        "base_model_weights": base_weights,
        "specialist_checkpoints": [str(path) for path in args.specialist_checkpoint],
        "specialist_model_weights": specialist_weights,
        "risk_head_weight": float(args.risk_head_weight),
        "base_risk_head_weight": base_risk_head_weight,
        "specialist_risk_head_weight": specialist_risk_head_weight,
        "candidate_calibrator": str(args.candidate_calibrator),
        "candidate_gate_calibrator": (
            str(args.candidate_gate_calibrator)
            if args.candidate_gate_calibrator
            else None
        ),
        "candidate_gate_threshold": (
            None if candidate_gate is None else float(candidate_gate.threshold)
        ),
        "candidate_feature_options": feature_options,
        "device": str(bases[0].device),
        "inference_augment_flips": bool(args.inference_flips),
        "inference_ensemble": args.inference_ensemble,
        "near_best_margin": float(args.near_best_margin),
        "candidate_weight": float(args.candidate_weight),
        "candidate_mode": args.candidate_mode,
        "candidate_topk": int(args.candidate_topk),
        "candidate_shortlist_source": args.candidate_shortlist_source,
        "candidate_policy_margin_max": args.candidate_policy_margin_max,
        "candidate_ranker_margin_min": args.candidate_ranker_margin_min,
        "candidate_risk_safety_filter": args.candidate_risk_safety_filter,
        "candidate_require_open": not bool(args.candidate_allow_non_open),
        "elapsed_seconds": time.time() - started_at,
        **result,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(payload, ensure_ascii=False, indent=2))


def benchmark_transitions(
    *,
    transitions,
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    calibrator,
    model_ensemble_reduction: str,
    batch_size: int,
    scores_are_probabilities: bool,
    near_best_margin: float,
    include_solver_risk_features: bool,
    candidate_weight: float = 0.0,
    candidate_mode: str = "blend",
    candidate_topk: int = 64,
    candidate_shortlist_source: str = "all",
    candidate_policy_margin_max: float | None = None,
    candidate_ranker_margin_min: float | None = None,
    candidate_risk_safety_filter: str = "none",
    candidate_require_open: bool = True,
    candidate_gate: CandidateGateCalibrator | None = None,
) -> dict[str, Any]:
    if not bases or not specialists:
        raise ValueError("at least one base and one specialist checkpoint are required")
    rows = []
    blend_rows = []
    for start in range(0, len(transitions), max(1, int(batch_size))):
        batch = transitions[start : start + max(1, int(batch_size))]
        boards = np.stack([transition.board for transition in batch])
        global_features = np.stack([transition.global_features for transition in batch])
        action_masks = np.stack([transition.action_mask for transition in batch])
        flat_mask = torch.tensor(
            action_masks.reshape(len(batch), -1),
            dtype=torch.bool,
            device=bases[0].device,
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
        risk_maps = None
        if include_solver_risk_features:
            risk_maps = np.stack(
                [
                    np.asarray(transition.risk_map, dtype=np.float32)
                    if transition.risk_map is not None
                    else np.zeros(boards.shape[-2:], dtype=np.float32)
                    for transition in batch
                ]
            )
        candidate_probs = candidate_policy_probabilities(
            calibrator,
            base_scores=base_scores.detach().cpu().numpy(),
            specialist_scores=specialist_scores.detach().cpu().numpy(),
            global_features=global_features,
            boards=boards,
            action_masks=action_masks,
            scores_are_probabilities=scores_are_probabilities,
            risk_maps=risk_maps,
        )
        base_open_scores = base_scores.detach().cpu().numpy()[:, : boards.shape[-2] * boards.shape[-1]]
        if float(candidate_weight) > 0.0:
            blended_open_scores = _blend_candidate_open_scores_for_batch(
                base_scores=base_scores,
                specialist_scores=specialist_scores,
                candidate_probabilities=candidate_probs,
                action_masks=action_masks,
                risk_maps=risk_maps,
                candidate_weight=float(candidate_weight),
                candidate_mode=candidate_mode,
                candidate_topk=int(candidate_topk),
                candidate_shortlist_source=candidate_shortlist_source,
                candidate_policy_margin_max=candidate_policy_margin_max,
                candidate_ranker_margin_min=candidate_ranker_margin_min,
                candidate_risk_safety_filter=candidate_risk_safety_filter,
                candidate_require_open=bool(candidate_require_open),
                candidate_gate=candidate_gate,
                global_features=global_features,
                boards=boards,
                transitions=batch,
                rows=boards.shape[-2],
                cols=boards.shape[-1],
                scores_are_probabilities=scores_are_probabilities,
                )
        else:
            blended_open_scores = None
        for index, transition in enumerate(batch):
            base_row = score_transition(
                transition=transition,
                base_open_scores=base_open_scores[index],
                candidate_probabilities=candidate_probs[index],
                near_best_margin=near_best_margin,
            )
            rows.append(base_row)
            if blended_open_scores is not None:
                blend_rows.append(
                    score_transition(
                        transition=transition,
                        base_open_scores=base_open_scores[index],
                        candidate_probabilities=candidate_probs[index],
                        near_best_margin=near_best_margin,
                        blended_open_scores=blended_open_scores[index],
                    )
                )
    result = summarize_benchmark_rows(rows)
    if blend_rows:
        result["policy_blend"] = summarize_benchmark_rows(blend_rows)
    return result


def _blend_candidate_open_scores_for_batch(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    candidate_probabilities: np.ndarray,
    action_masks: np.ndarray,
    risk_maps: np.ndarray | None,
    candidate_weight: float,
    candidate_mode: str,
    candidate_topk: int,
    candidate_shortlist_source: str,
    candidate_policy_margin_max: float | None,
    candidate_ranker_margin_min: float | None,
    candidate_risk_safety_filter: str,
    candidate_require_open: bool,
    candidate_gate: CandidateGateCalibrator | None,
    global_features: np.ndarray,
    boards: np.ndarray,
    transitions,
    rows: int,
    cols: int,
    scores_are_probabilities: bool,
) -> np.ndarray:
    cells = int(rows) * int(cols)
    base_cpu = base_scores.detach().cpu().numpy()
    base_action_is_open = np.argmax(base_cpu, axis=1) < cells
    transition_guess = np.asarray(
        [bool(getattr(transition, "expert_is_guess", False)) for transition in transitions],
        dtype=bool,
    )
    active = transition_guess
    if candidate_require_open:
        active &= base_action_is_open
    if candidate_policy_margin_max is not None:
        active &= gated._counterfactual_value_policy_margin_gate(
            scores=base_scores,
            action_masks=action_masks,
            rows=rows,
            cols=cols,
            max_margin=float(candidate_policy_margin_max),
        )

    shortlist = gated._candidate_shortlist_mask(
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        action_masks=action_masks,
        rows=rows,
        cols=cols,
        topk=max(1, int(candidate_topk)),
        candidate_probabilities=(
            candidate_probabilities
            if candidate_shortlist_source == "all"
            else None
        ),
    )
    active &= shortlist.any(axis=1)
    if candidate_ranker_margin_min is not None:
        active &= gated._candidate_ranker_margin_gate(
            candidate_probabilities=candidate_probabilities,
            action_masks=action_masks,
            min_margin=float(candidate_ranker_margin_min),
        )
    if candidate_gate is not None and bool(active.any()):
        gate_features = build_candidate_gate_features(
            base_scores=base_scores.detach().cpu().numpy(),
            candidate_probabilities=candidate_probabilities,
            global_features=global_features,
            action_masks=action_masks,
            risk_maps=risk_maps,
            scores_are_probabilities=scores_are_probabilities,
        )
        active &= candidate_gate.should_apply_features(gate_features)
    active_tensor = torch.tensor(active, dtype=torch.bool, device=base_scores.device)
    adjusted = gated._blend_candidate_open_scores(
        scores=base_scores,
        candidate_probabilities=torch.tensor(
            candidate_probabilities,
            dtype=base_scores.dtype,
            device=base_scores.device,
        ),
        candidate_tail=active_tensor,
        candidate_mask=torch.tensor(shortlist, dtype=torch.bool, device=base_scores.device),
        weight=float(candidate_weight),
        rows=rows,
        cols=cols,
        scores_are_probabilities=scores_are_probabilities,
        mode=candidate_mode,
    )

    if candidate_risk_safety_filter != "none":
        if risk_maps is None:
            risk_maps = np.zeros((len(transitions), rows, cols), dtype=np.float32)
        violations = gated._candidate_risk_safety_violations(
            before_scores=base_scores,
            after_scores=adjusted,
            action_masks=action_masks,
            risk_maps=risk_maps,
            active=active_tensor,
            rows=rows,
            cols=cols,
            mode=candidate_risk_safety_filter,
        )
        adjusted = torch.where(violations.view(-1, 1), base_scores, adjusted)

    return adjusted[:, :cells].detach().cpu().numpy()


def score_transition(
    *,
    transition,
    base_open_scores: np.ndarray,
    candidate_probabilities: np.ndarray,
    near_best_margin: float,
    blended_open_scores: np.ndarray | None = None,
) -> dict[str, Any] | None:
    values = np.asarray(transition.counterfactual_open_values, dtype=np.float32).reshape(-1)
    legal = np.asarray(transition.action_mask[action_channel(ActionType.OPEN)], dtype=bool).reshape(-1)
    valid = legal & np.isfinite(values)
    if not bool(valid.any()):
        return None
    base_choice = int(np.argmax(np.where(valid, base_open_scores, -np.inf)))
    candidate_choice = int(np.argmax(np.where(valid, candidate_probabilities, -np.inf)))
    selected_scores = candidate_probabilities if blended_open_scores is None else blended_open_scores
    selected_choice = int(np.argmax(np.where(valid, selected_scores, -np.inf)))
    best_choice = int(np.argmax(np.where(valid, values, -np.inf)))
    best_value = float(values[best_choice])
    base_value = float(values[base_choice])
    candidate_value = float(values[selected_choice])
    base_regret = best_value - base_value
    candidate_regret = best_value - candidate_value
    safe_left = int(round((1.0 - float(np.clip(transition.global_features[1], 0.0, 1.0))) * 391))
    return {
        "family": transition.extreme_family or "unclassified",
        "safe_left": safe_left,
        "base_best": float(base_choice == best_choice),
        "candidate_best": float(selected_choice == best_choice),
        "base_near_best": float(base_regret <= near_best_margin),
        "candidate_near_best": float(candidate_regret <= near_best_margin),
        "base_negative": float(base_value < 0.0),
        "candidate_negative": float(candidate_value < 0.0),
        "base_regret": float(base_regret),
        "candidate_regret": float(candidate_regret),
        "candidate_improved": float(candidate_regret < base_regret - 1e-6),
        "candidate_worsened": float(candidate_regret > base_regret + 1e-6),
        "candidate_unchanged": float(abs(candidate_regret - base_regret) <= 1e-6),
        "base_choice": base_choice,
        "candidate_choice": selected_choice,
        "candidate_rank_choice": candidate_choice,
        "best_choice": best_choice,
    }


def summarize_benchmark_rows(rows: list[dict[str, Any] | None]) -> dict[str, Any]:
    valid_rows = [row for row in rows if row is not None]
    return {
        "records": len(valid_rows),
        "overall": summarize_group(valid_rows),
        "families": summarize_groups(valid_rows, lambda row: str(row["family"])),
        "safe_left_buckets": summarize_groups(valid_rows, lambda row: safe_left_bucket(int(row["safe_left"]))),
    }


def summarize_groups(rows: list[dict[str, Any]], key_fn) -> dict[str, Any]:
    groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[key_fn(row)].append(row)
    return {key: summarize_group(group) for key, group in sorted(groups.items())}


def summarize_group(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if not rows:
        return {"records": 0}
    return {
        "records": len(rows),
        "base_best_match_rate": float(np.mean([row["base_best"] for row in rows])),
        "candidate_best_match_rate": float(np.mean([row["candidate_best"] for row in rows])),
        "base_near_best_rate": float(np.mean([row["base_near_best"] for row in rows])),
        "candidate_near_best_rate": float(np.mean([row["candidate_near_best"] for row in rows])),
        "base_negative_rate": float(np.mean([row["base_negative"] for row in rows])),
        "candidate_negative_rate": float(np.mean([row["candidate_negative"] for row in rows])),
        "base_avg_regret": float(np.mean([row["base_regret"] for row in rows])),
        "candidate_avg_regret": float(np.mean([row["candidate_regret"] for row in rows])),
        "candidate_improved_rate": float(np.mean([row["candidate_improved"] for row in rows])),
        "candidate_worsened_rate": float(np.mean([row["candidate_worsened"] for row in rows])),
        "candidate_unchanged_rate": float(np.mean([row["candidate_unchanged"] for row in rows])),
    }


def safe_left_bucket(safe_left: int) -> str:
    for lower, upper, name in SAFE_LEFT_BUCKETS:
        if lower <= int(safe_left) <= upper:
            return name
    return "outside"


if __name__ == "__main__":
    main()
