from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.candidate_calibration import (
    CandidateCalibrator,
    candidate_policy_probabilities,
    load_candidate_calibrator,
)
from minesweeper_rl.candidate_gate import (
    CandidateGateCalibrator,
    build_candidate_gate_features,
)
from minesweeper_rl.features import action_channel, decode_action_index, encode_state
from minesweeper_rl.gate_calibration import GateCalibrator, build_gate_features
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint, make_game
from minesweeper_rl.types import ActionType


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate a pure-RL specialist only on late/endgame states."
    )
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
        help="Optional per-base checkpoint ensemble weights, in checkpoint order.",
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
        help="Optional per-specialist checkpoint ensemble weights, in checkpoint order.",
    )
    parser.add_argument(
        "--model-ensemble-reduction",
        choices=["mean", "geomean"],
        default="mean",
        help="How to combine scores from multiple checkpoints.",
    )
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games", type=int, default=200)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--decision-actions", choices=["open", "full"], default="full")
    parser.add_argument("--safe-left-threshold", type=int, default=100)
    parser.add_argument(
        "--adjustment-gate",
        choices=["safe-left", "solver-guess", "safe-left-and-solver-guess"],
        default="safe-left",
        help=(
            "When specialist/candidate/value adjustments are allowed. "
            "solver-guess modes are diagnostic: solver only decides whether the state has forced moves."
        ),
    )
    parser.add_argument(
        "--gate-exact-limit",
        type=int,
        default=32,
        help="Exact solver limit used only by solver-guess adjustment gates.",
    )
    parser.add_argument(
        "--specialist-weight",
        type=float,
        default=1.0,
        help="Blend weight for specialist scores on gated states, in [0, 1].",
    )
    parser.add_argument(
        "--blend-scope",
        choices=["all", "open", "open-disagree-margin"],
        default="all",
        help="Whether the specialist blend applies to all actions or only the open channel.",
    )
    parser.add_argument("--specialist-margin", type=float, default=0.0)
    parser.add_argument(
        "--specialist-region-gate",
        choices=[
            "any",
            "current-edge-or-corner",
            "specialist-edge-or-corner",
            "either-edge-or-corner",
            "current-interior",
            "specialist-interior",
            "both-interior",
        ],
        default="any",
        help="Optionally restrict specialist blending by the base/specialist OPEN choice geometry.",
    )
    parser.add_argument("--gate-calibrator", type=Path, default=None)
    parser.add_argument(
        "--gate-threshold",
        type=float,
        default=None,
        help="Optional runtime override for the offline gate calibrator threshold.",
    )
    parser.add_argument("--candidate-calibrator", type=Path, default=None)
    parser.add_argument("--candidate-gate-calibrator", type=Path, default=None)
    parser.add_argument("--candidate-gate-threshold", type=float, default=None)
    parser.add_argument("--candidate-weight", type=float, default=0.0)
    parser.add_argument(
        "--candidate-mode",
        choices=["blend", "replace"],
        default="blend",
        help="Blend candidate scores with the current OPEN scores, or replace them on the candidate shortlist.",
    )
    parser.add_argument("--candidate-safe-left-threshold", type=int, default=None)
    parser.add_argument("--candidate-topk", type=int, default=16)
    parser.add_argument(
        "--candidate-shortlist-source",
        choices=["all", "policy"],
        default="all",
        help="Use candidate top-K cells in the rerank shortlist, or restrict reranking to base/specialist policy candidates.",
    )
    parser.add_argument(
        "--candidate-adjustment-gate",
        choices=["safe-left", "solver-guess", "safe-left-and-solver-guess"],
        default=None,
        help=(
            "Optional gate for candidate reranking. Defaults to --adjustment-gate; "
            "solver-guess uses the solver only to identify uncertain states."
        ),
    )
    parser.add_argument(
        "--candidate-policy-margin-max",
        type=float,
        default=None,
        help="Only apply candidate reranking when the current policy's top-two OPEN score margin is at most this value.",
    )
    parser.add_argument(
        "--candidate-ranker-margin-min",
        type=float,
        default=None,
        help="Only apply candidate reranking when the candidate ranker's top-two legal OPEN probability gap reaches this value.",
    )
    parser.add_argument(
        "--candidate-allow-non-open",
        action="store_true",
        help="Allow candidate reranking when the current full-policy action is FLAG/other, not only OPEN.",
    )
    parser.add_argument(
        "--candidate-risk-safety-filter",
        choices=["none", "not-higher", "strictly-lower"],
        default="none",
        help="Optionally reject candidate reranking when its final OPEN choice has higher solver-visible risk.",
    )
    parser.add_argument(
        "--counterfactual-value-checkpoint",
        dest="counterfactual_value_checkpoints",
        action="append",
        type=Path,
        default=[],
        help="Optional checkpoint whose counterfactual value head reranks OPEN candidates.",
    )
    parser.add_argument("--counterfactual-value-weight", type=float, default=0.0)
    parser.add_argument("--counterfactual-value-temperature", type=float, default=0.35)
    parser.add_argument("--counterfactual-value-safe-left-threshold", type=int, default=None)
    parser.add_argument("--counterfactual-value-min-safe-left", type=int, default=None)
    parser.add_argument("--counterfactual-value-topk", type=int, default=16)
    parser.add_argument(
        "--counterfactual-value-open-gate",
        choices=["top-open", "any"],
        default="top-open",
        help="Whether value reranking requires the current top action to already be OPEN.",
    )
    parser.add_argument(
        "--counterfactual-value-region-gate",
        choices=[
            "any",
            "current-edge-or-corner",
            "value-edge-or-corner",
            "either-edge-or-corner",
            "current-interior",
            "value-interior",
            "both-interior",
        ],
        default="any",
        help="Optionally restrict value reranking by the current/value OPEN choice geometry.",
    )
    parser.add_argument(
        "--counterfactual-value-disagree-margin",
        type=float,
        default=0.0,
        help="Only apply value reranking when its best OPEN cell differs from the current policy and its top-two value gap reaches this margin.",
    )
    parser.add_argument(
        "--counterfactual-value-policy-margin-max",
        type=float,
        default=None,
        help="Only apply value reranking when the current policy's top-two OPEN score margin is at most this value.",
    )
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument(
        "--risk-head-weight",
        type=float,
        default=0.0,
        help="Weight for each model's learned risk head during policy scoring.",
    )
    parser.add_argument("--base-risk-head-weight", type=float)
    parser.add_argument("--specialist-risk-head-weight", type=float)
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    base_checkpoints = [args.base_checkpoint, *args.base_ensemble_checkpoints]
    base_model_weights = normalize_ensemble_weights(
        args.base_model_weights,
        expected=len(base_checkpoints),
        label="base-model-weight",
    )
    base_risk_head_weight = (
        args.risk_head_weight if args.base_risk_head_weight is None else args.base_risk_head_weight
    )
    specialist_risk_head_weight = (
        args.risk_head_weight
        if args.specialist_risk_head_weight is None
        else args.specialist_risk_head_weight
    )
    bases = [load_eval_trainer(path, args, risk_head_weight=base_risk_head_weight) for path in base_checkpoints]
    specialist_checkpoints = [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
    specialist_model_weights = normalize_ensemble_weights(
        args.specialist_model_weights,
        expected=len(specialist_checkpoints),
        label="specialist-model-weight",
    )
    specialists = [
        load_eval_trainer(path, args, risk_head_weight=specialist_risk_head_weight)
        for path in specialist_checkpoints
    ]
    started_at = time.time()
    gate_calibrator = GateCalibrator.load(args.gate_calibrator) if args.gate_calibrator else None
    if gate_calibrator is not None and args.gate_threshold is not None:
        gate_calibrator.threshold = float(np.clip(args.gate_threshold, 0.0, 1.0))
    candidate_calibrator = load_candidate_calibrator(args.candidate_calibrator) if args.candidate_calibrator else None
    candidate_gate = (
        CandidateGateCalibrator.load(args.candidate_gate_calibrator)
        if args.candidate_gate_calibrator
        else None
    )
    if candidate_gate is not None and args.candidate_gate_threshold is not None:
        candidate_gate.threshold = float(np.clip(args.candidate_gate_threshold, 0.0, 1.0))
    if args.counterfactual_value_weight > 0.0 and not args.counterfactual_value_checkpoints:
        parser.error("--counterfactual-value-weight requires at least one --counterfactual-value-checkpoint")
    counterfactual_value_trainers = [
        load_eval_trainer(path, args, risk_head_weight=0.0)
        for path in args.counterfactual_value_checkpoints
    ]
    metrics = evaluate_gated(
        bases=bases,
        base_model_weights=base_model_weights,
        specialists=specialists,
        specialist_model_weights=specialist_model_weights,
        model_ensemble_reduction=args.model_ensemble_reduction,
        games=args.games,
        seed=args.seed,
        batch_size=args.batch_size,
        max_steps=args.max_steps,
        safe_left_threshold=args.safe_left_threshold,
        adjustment_gate=args.adjustment_gate,
        gate_exact_limit=args.gate_exact_limit,
        specialist_weight=args.specialist_weight,
        blend_scope=args.blend_scope,
        specialist_margin=args.specialist_margin,
        specialist_region_gate=args.specialist_region_gate,
        gate_calibrator=gate_calibrator,
        candidate_calibrator=candidate_calibrator,
        candidate_gate=candidate_gate,
        candidate_weight=args.candidate_weight,
        candidate_mode=args.candidate_mode,
        candidate_topk=args.candidate_topk,
        candidate_shortlist_source=args.candidate_shortlist_source,
        candidate_adjustment_gate=args.candidate_adjustment_gate,
        candidate_safe_left_threshold=(
            args.safe_left_threshold
            if args.candidate_safe_left_threshold is None
            else args.candidate_safe_left_threshold
        ),
        candidate_policy_margin_max=args.candidate_policy_margin_max,
        candidate_ranker_margin_min=args.candidate_ranker_margin_min,
        candidate_risk_safety_filter=args.candidate_risk_safety_filter,
        candidate_require_open=not bool(args.candidate_allow_non_open),
        counterfactual_value_trainers=counterfactual_value_trainers,
        counterfactual_value_weight=args.counterfactual_value_weight,
        counterfactual_value_temperature=args.counterfactual_value_temperature,
        counterfactual_value_topk=args.counterfactual_value_topk,
        counterfactual_value_open_gate=args.counterfactual_value_open_gate,
        counterfactual_value_region_gate=args.counterfactual_value_region_gate,
        counterfactual_value_disagree_margin=args.counterfactual_value_disagree_margin,
        counterfactual_value_policy_margin_max=args.counterfactual_value_policy_margin_max,
        counterfactual_value_safe_left_threshold=(
            args.safe_left_threshold
            if args.counterfactual_value_safe_left_threshold is None
            else args.counterfactual_value_safe_left_threshold
        ),
        counterfactual_value_min_safe_left=args.counterfactual_value_min_safe_left,
    )
    result = {
        "base_checkpoints": [str(path) for path in base_checkpoints],
        "base_model_weights": base_model_weights,
        "specialist_checkpoints": [str(path) for path in specialist_checkpoints],
        "specialist_model_weights": specialist_model_weights,
        "model_ensemble_reduction": args.model_ensemble_reduction,
        "device": str(bases[0].device),
        "inference_augment_flips": args.inference_flips,
        "inference_ensemble": args.inference_ensemble,
        "decision_actions": args.decision_actions,
        "risk_head_weight": args.risk_head_weight,
        "base_risk_head_weight": base_risk_head_weight,
        "specialist_risk_head_weight": specialist_risk_head_weight,
        "safe_left_threshold": args.safe_left_threshold,
        "adjustment_gate": args.adjustment_gate,
        "gate_exact_limit": int(args.gate_exact_limit),
        "specialist_weight": args.specialist_weight,
        "blend_scope": args.blend_scope,
        "specialist_margin": args.specialist_margin,
        "specialist_region_gate": args.specialist_region_gate,
        "gate_calibrator": str(args.gate_calibrator) if args.gate_calibrator else None,
        "gate_threshold": None if gate_calibrator is None else float(gate_calibrator.threshold),
        "candidate_calibrator": str(args.candidate_calibrator) if args.candidate_calibrator else None,
        "candidate_gate_calibrator": (
            str(args.candidate_gate_calibrator)
            if args.candidate_gate_calibrator
            else None
        ),
        "candidate_gate_threshold": (
            None if candidate_gate is None else float(candidate_gate.threshold)
        ),
        "candidate_weight": float(args.candidate_weight),
        "candidate_mode": args.candidate_mode,
        "candidate_topk": int(args.candidate_topk),
        "candidate_shortlist_source": args.candidate_shortlist_source,
        "candidate_adjustment_gate": (
            args.adjustment_gate
            if args.candidate_adjustment_gate is None
            else args.candidate_adjustment_gate
        ),
        "candidate_safe_left_threshold": (
            args.safe_left_threshold
            if args.candidate_safe_left_threshold is None
            else args.candidate_safe_left_threshold
        ),
        "candidate_policy_margin_max": args.candidate_policy_margin_max,
        "candidate_ranker_margin_min": args.candidate_ranker_margin_min,
        "candidate_risk_safety_filter": args.candidate_risk_safety_filter,
        "candidate_require_open": not bool(args.candidate_allow_non_open),
        "counterfactual_value_checkpoints": [
            str(path) for path in args.counterfactual_value_checkpoints
        ],
        "counterfactual_value_weight": float(args.counterfactual_value_weight),
        "counterfactual_value_temperature": float(args.counterfactual_value_temperature),
        "counterfactual_value_topk": int(args.counterfactual_value_topk),
        "counterfactual_value_open_gate": args.counterfactual_value_open_gate,
        "counterfactual_value_region_gate": args.counterfactual_value_region_gate,
        "counterfactual_value_disagree_margin": float(args.counterfactual_value_disagree_margin),
        "counterfactual_value_policy_margin_max": (
            None
            if args.counterfactual_value_policy_margin_max is None
            else float(args.counterfactual_value_policy_margin_max)
        ),
        "counterfactual_value_safe_left_threshold": (
            args.safe_left_threshold
            if args.counterfactual_value_safe_left_threshold is None
            else args.counterfactual_value_safe_left_threshold
        ),
        "counterfactual_value_min_safe_left": args.counterfactual_value_min_safe_left,
        "solver_decision": False,
        "elapsed_seconds": time.time() - started_at,
        **metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_eval_trainer(
    path: Path,
    args: argparse.Namespace,
    *,
    risk_head_weight: float | None = None,
) -> MinesweeperTrainer:
    trainer = load_checkpoint(path, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = args.decision_actions
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = float(
        args.risk_head_weight if risk_head_weight is None else risk_head_weight
    )
    trainer.model.eval()
    return trainer


def normalize_ensemble_weights(
    weights: list[float],
    *,
    expected: int,
    label: str,
) -> list[float]:
    if expected <= 0:
        raise ValueError(f"{label} requires at least one checkpoint")
    if not weights:
        return [1.0 / float(expected) for _ in range(expected)]
    if len(weights) != expected:
        raise ValueError(f"{label} count must match checkpoint count: {len(weights)} != {expected}")
    array = np.asarray(weights, dtype=np.float64)
    if not np.isfinite(array).all():
        raise ValueError(f"{label} must be finite")
    if (array < 0.0).any():
        raise ValueError(f"{label} must be non-negative")
    total = float(array.sum())
    if total <= 0.0:
        raise ValueError(f"{label} sum must be positive")
    return (array / total).astype(float).tolist()


def evaluate_gated(
    *,
    bases: list[MinesweeperTrainer],
    base_model_weights: list[float] | None = None,
    specialists: list[MinesweeperTrainer],
    specialist_model_weights: list[float] | None = None,
    model_ensemble_reduction: str = "mean",
    games: int,
    seed: int,
    batch_size: int,
    max_steps: int,
    safe_left_threshold: int,
    specialist_weight: float,
    blend_scope: str,
    specialist_margin: float,
    specialist_region_gate: str = "any",
    adjustment_gate: str = "safe-left",
    gate_exact_limit: int = 32,
    gate_calibrator: GateCalibrator | None = None,
    candidate_calibrator: CandidateCalibrator | None = None,
    candidate_gate: CandidateGateCalibrator | None = None,
    candidate_weight: float = 0.0,
    candidate_mode: str = "blend",
    candidate_topk: int = 16,
    candidate_shortlist_source: str = "all",
    candidate_adjustment_gate: str | None = None,
    candidate_safe_left_threshold: int | None = None,
    candidate_policy_margin_max: float | None = None,
    candidate_ranker_margin_min: float | None = None,
    candidate_risk_safety_filter: str = "none",
    candidate_require_open: bool = True,
    counterfactual_value_trainers: list[MinesweeperTrainer] | None = None,
    counterfactual_value_weight: float = 0.0,
    counterfactual_value_temperature: float = 0.35,
    counterfactual_value_topk: int = 16,
    counterfactual_value_open_gate: str = "top-open",
    counterfactual_value_region_gate: str = "any",
    counterfactual_value_disagree_margin: float = 0.0,
    counterfactual_value_policy_margin_max: float | None = None,
    counterfactual_value_safe_left_threshold: int | None = None,
    counterfactual_value_min_safe_left: int | None = None,
) -> dict[str, Any]:
    if games < 0:
        raise ValueError("games must be non-negative")
    specialist_weight = float(np.clip(specialist_weight, 0.0, 1.0))
    if not bases:
        raise ValueError("at least one base trainer is required")
    if not specialists:
        raise ValueError("at least one specialist trainer is required")
    base = bases[0]
    if model_ensemble_reduction not in {"mean", "geomean"}:
        raise ValueError(f"unknown model_ensemble_reduction {model_ensemble_reduction!r}")
    if model_ensemble_reduction == "geomean" and base.config.inference_ensemble != "probs":
        raise ValueError("geomean model ensemble requires inference_ensemble='probs'")
    if adjustment_gate not in {"safe-left", "solver-guess", "safe-left-and-solver-guess"}:
        raise ValueError(f"unknown adjustment_gate {adjustment_gate!r}")
    if candidate_adjustment_gate is None:
        candidate_adjustment_gate = adjustment_gate
    if candidate_adjustment_gate not in {"safe-left", "solver-guess", "safe-left-and-solver-guess"}:
        raise ValueError(f"unknown candidate_adjustment_gate {candidate_adjustment_gate!r}")
    if candidate_shortlist_source not in {"all", "policy"}:
        raise ValueError(f"unknown candidate_shortlist_source {candidate_shortlist_source!r}")
    if candidate_risk_safety_filter not in {"none", "not-higher", "strictly-lower"}:
        raise ValueError(f"unknown candidate_risk_safety_filter {candidate_risk_safety_filter!r}")
    if counterfactual_value_open_gate not in {"top-open", "any"}:
        raise ValueError(f"unknown counterfactual_value_open_gate {counterfactual_value_open_gate!r}")
    if counterfactual_value_policy_margin_max is not None and float(counterfactual_value_policy_margin_max) < 0.0:
        raise ValueError("counterfactual_value_policy_margin_max must be non-negative")
    if counterfactual_value_region_gate not in {
        "any",
        "current-edge-or-corner",
        "value-edge-or-corner",
        "either-edge-or-corner",
        "current-interior",
        "value-interior",
        "both-interior",
    }:
        raise ValueError(f"unknown counterfactual_value_region_gate {counterfactual_value_region_gate!r}")
    if specialist_region_gate not in {
        "any",
        "current-edge-or-corner",
        "specialist-edge-or-corner",
        "either-edge-or-corner",
        "current-interior",
        "specialist-interior",
        "both-interior",
    }:
        raise ValueError(f"unknown specialist_region_gate {specialist_region_gate!r}")
    base_model_weights = (
        normalize_ensemble_weights([], expected=len(bases), label="base-model-weight")
        if base_model_weights is None
        else normalize_ensemble_weights(base_model_weights, expected=len(bases), label="base-model-weight")
    )
    specialist_model_weights = (
        normalize_ensemble_weights([], expected=len(specialists), label="specialist-model-weight")
        if specialist_model_weights is None
        else normalize_ensemble_weights(
            specialist_model_weights,
            expected=len(specialists),
            label="specialist-model-weight",
        )
    )
    if any(
        trainer.config.rows != base.config.rows or trainer.config.cols != base.config.cols
        for trainer in bases
    ):
        raise ValueError("base checkpoints must use the same board shape")
    if any(
        trainer.config.rows != base.config.rows or trainer.config.cols != base.config.cols
        for trainer in specialists
    ):
        raise ValueError("specialist checkpoints must use the same board shape")

    wins: list[bool] = []
    rewards: list[float] = []
    steps: list[int] = []
    revealed_safe_cells: list[int] = []
    tail_decisions = 0
    base_decisions = 0
    specialist_decisions = 0
    specialist_applied_decisions = 0
    candidate_adjusted_decisions = 0
    candidate_changed_decisions = 0
    candidate_gate_considered_decisions = 0
    candidate_gate_applied_decisions = 0
    candidate_gate_rejected_decisions = 0
    counterfactual_value_adjusted_decisions = 0
    counterfactual_value_changed_decisions = 0
    solver_guess_decisions = 0
    solver_forced_decisions = 0
    batch_size = max(1, int(batch_size))
    total_safe = base.config.rows * base.config.cols - base.config.mines
    counterfactual_value_trainers = list(counterfactual_value_trainers or [])
    solver_gate = (
        MinesweeperSolver(exact_limit=max(1, int(gate_exact_limit)))
        if _uses_solver_gate(adjustment_gate) or _uses_solver_gate(candidate_adjustment_gate)
        else None
    )

    with torch.no_grad():
        for batch_start in range(0, games, batch_size):
            current_batch = min(batch_size, games - batch_start)
            game_batch = [make_game(base.config, seed=seed + batch_start + offset) for offset in range(current_batch)]
            batch_rewards = [0.0 for _ in range(current_batch)]
            batch_steps = [0 for _ in range(current_batch)]
            active = set(range(current_batch))

            while active:
                ready: list[int] = []
                boards: list[np.ndarray] = []
                global_features: list[np.ndarray] = []
                action_masks: list[np.ndarray] = []
                tail_mask: list[bool] = []
                solver_guess_mask: list[bool] = []
                solver_snapshot_batch: list[Any | None] = []
                safe_left_batch: list[int] = []

                for index in list(active):
                    game = game_batch[index]
                    if game.done or batch_steps[index] >= max_steps:
                        active.remove(index)
                        continue
                    board, global_feature, action_mask = encode_state(game)
                    action_mask = base._decision_action_mask(action_mask)
                    if not action_mask.any():
                        active.remove(index)
                        continue

                    progress = float(np.clip(global_feature[1], 0.0, 1.0))
                    safe_left = int(round((1.0 - progress) * total_safe))
                    needs_solver_guess = (
                        solver_gate is not None
                        and (
                            adjustment_gate == "solver-guess"
                            or candidate_adjustment_gate == "solver-guess"
                            or candidate_adjustment_gate == "safe-left-and-solver-guess"
                            or safe_left <= int(safe_left_threshold)
                        )
                    )
                    solver_snapshot = (
                        solver_gate.analyze(game)
                        if needs_solver_guess and solver_gate is not None
                        else None
                    )
                    wrong_flags = int((game.flagged & ~game.mines).sum())
                    solver_guess = bool(
                        solver_snapshot is not None
                        and not solver_snapshot.has_forced_moves
                        and wrong_flags == 0
                    )
                    use_specialist = _adjustment_gate_active(
                        adjustment_gate,
                        safe_left=safe_left,
                        threshold=int(safe_left_threshold),
                        solver_guess=solver_guess,
                    )
                    ready.append(index)
                    boards.append(board)
                    global_features.append(global_feature)
                    action_masks.append(action_mask)
                    tail_mask.append(use_specialist)
                    solver_guess_mask.append(solver_guess)
                    solver_snapshot_batch.append(solver_snapshot)
                    safe_left_batch.append(safe_left)

                if not ready:
                    continue

                board_batch = np.stack(boards)
                global_batch = np.stack(global_features)
                mask_batch = np.stack(action_masks)
                safe_left_array = np.asarray(safe_left_batch, dtype=np.int32)
                solver_guess_array = np.asarray(solver_guess_mask, dtype=bool)
                flat_mask_tensor = torch.tensor(mask_batch.reshape(mask_batch.shape[0], -1), dtype=torch.bool, device=base.device)
                base_parts: list[torch.Tensor] = []
                for model_weight, base_trainer in zip(base_model_weights, bases, strict=True):
                    part = base_trainer._predict_policy_scores_batch(
                        boards=board_batch,
                        global_features_batch=global_batch,
                        action_masks=mask_batch,
                        use_flip_ensemble=bool(base_trainer.config.inference_augment_flips),
                    )
                    base_parts.append(part)
                base_scores = combine_model_scores(
                    base_parts,
                    base_model_weights,
                    reduction=model_ensemble_reduction,
                    flat_mask=flat_mask_tensor,
                )
                scores_are_probabilities = bool(
                    base.config.inference_augment_flips
                    and base.config.inference_ensemble == "probs"
                )
                specialist_parts: list[torch.Tensor] = []
                for model_weight, specialist in zip(specialist_model_weights, specialists, strict=True):
                    part = specialist._predict_policy_scores_batch(
                        boards=board_batch,
                        global_features_batch=global_batch,
                        action_masks=mask_batch,
                        use_flip_ensemble=bool(specialist.config.inference_augment_flips),
                    )
                    specialist_parts.append(part)
                specialist_scores = combine_model_scores(
                    specialist_parts,
                    specialist_model_weights,
                    reduction=model_ensemble_reduction,
                    flat_mask=flat_mask_tensor,
                )
                gate = _specialist_gate(
                    tail_mask=tail_mask,
                    base_scores=base_scores,
                    specialist_scores=specialist_scores,
                    blend_scope=blend_scope,
                    specialist_margin=float(specialist_margin),
                    specialist_region_gate=specialist_region_gate,
                    rows=base.config.rows,
                    cols=base.config.cols,
                    gate_calibrator=gate_calibrator,
                    board_batch=board_batch,
                    global_batch=global_batch,
                    mask_batch=mask_batch,
                    scores_are_probabilities=scores_are_probabilities,
                    device=base.device,
                )
                blended_scores = _blend_scores(
                    base_scores=base_scores,
                    specialist_scores=specialist_scores,
                    specialist_weight=specialist_weight,
                    blend_scope=blend_scope,
                    rows=base.config.rows,
                    cols=base.config.cols,
                )
                scores = torch.where(gate, blended_scores, base_scores)
                if candidate_calibrator is not None and float(candidate_weight) > 0.0:
                    candidate_threshold = (
                        int(safe_left_threshold)
                        if candidate_safe_left_threshold is None
                        else int(candidate_safe_left_threshold)
                    )
                    candidate_tail = _adjustment_gate_batch(
                        candidate_adjustment_gate,
                        safe_left=safe_left_array,
                        threshold=candidate_threshold,
                        solver_guess=solver_guess_array,
                    )
                    base_action_is_open = (
                        base_scores.argmax(dim=1) < base.config.rows * base.config.cols
                    ).detach().cpu().numpy()
                    if candidate_require_open:
                        candidate_tail = candidate_tail & base_action_is_open
                    if candidate_policy_margin_max is not None:
                        candidate_tail = candidate_tail & _counterfactual_value_policy_margin_gate(
                            scores=scores,
                            action_masks=mask_batch,
                            rows=base.config.rows,
                            cols=base.config.cols,
                            max_margin=float(candidate_policy_margin_max),
                        )
                    if bool(candidate_tail.any()):
                        candidate_probs = candidate_policy_probabilities(
                            candidate_calibrator,
                            base_scores=base_scores.detach().cpu().numpy(),
                            specialist_scores=specialist_scores.detach().cpu().numpy(),
                            global_features=global_batch,
                            boards=board_batch,
                            action_masks=mask_batch,
                            scores_are_probabilities=scores_are_probabilities,
                            risk_maps=(
                                np.stack(
                                    [
                                        snapshot.risk_map
                                        if snapshot is not None
                                        else np.zeros((base.config.rows, base.config.cols), dtype=np.float32)
                                        for snapshot in solver_snapshot_batch
                                    ]
                                )
                                if "solver_risk" in candidate_calibrator.feature_names
                                else None
                            ),
                        )
                        if candidate_ranker_margin_min is not None:
                            candidate_tail = candidate_tail & _candidate_ranker_margin_gate(
                                candidate_probabilities=candidate_probs,
                                action_masks=mask_batch,
                                min_margin=float(candidate_ranker_margin_min),
                            )
                        if candidate_gate is not None:
                            candidate_gate_considered_decisions += int(candidate_tail.sum())
                            candidate_gate_features = build_candidate_gate_features(
                                base_scores=base_scores.detach().cpu().numpy(),
                                candidate_probabilities=candidate_probs,
                                global_features=global_batch,
                                action_masks=mask_batch,
                                risk_maps=np.stack(
                                    [
                                        snapshot.risk_map
                                        if snapshot is not None
                                        else np.zeros(
                                            (base.config.rows, base.config.cols),
                                            dtype=np.float32,
                                        )
                                        for snapshot in solver_snapshot_batch
                                    ]
                                ),
                                scores_are_probabilities=scores_are_probabilities,
                            )
                            candidate_gate_accept = candidate_gate.should_apply_features(
                                candidate_gate_features
                            )
                            candidate_gate_rejected_decisions += int(
                                (candidate_tail & ~candidate_gate_accept).sum()
                            )
                            candidate_tail = candidate_tail & candidate_gate_accept
                            candidate_gate_applied_decisions += int(candidate_tail.sum())
                        candidate_shortlist = _candidate_shortlist_mask(
                            base_scores=base_scores,
                            specialist_scores=specialist_scores,
                            action_masks=mask_batch,
                            rows=base.config.rows,
                            cols=base.config.cols,
                            topk=int(candidate_topk),
                            candidate_probabilities=(
                                candidate_probs
                                if candidate_shortlist_source == "all"
                                else None
                            ),
                        )
                        candidate_tail = candidate_tail & candidate_shortlist.any(axis=1)
                        before_candidate_actions = scores.argmax(dim=1)
                        scores_before_candidate = scores.clone()
                        candidate_tail_tensor = torch.tensor(
                            candidate_tail,
                            dtype=torch.bool,
                            device=scores.device,
                        )
                        scores = _blend_candidate_open_scores(
                            scores=scores,
                            candidate_probabilities=torch.tensor(
                                candidate_probs,
                                dtype=scores.dtype,
                                device=scores.device,
                            ),
                            candidate_tail=candidate_tail_tensor,
                            candidate_mask=torch.tensor(
                                candidate_shortlist,
                                dtype=torch.bool,
                                device=scores.device,
                            ),
                            weight=float(candidate_weight),
                            rows=base.config.rows,
                            cols=base.config.cols,
                            scores_are_probabilities=scores_are_probabilities,
                            mode=candidate_mode,
                        )
                        if candidate_risk_safety_filter != "none":
                            candidate_risk_maps = np.stack(
                                [
                                    snapshot.risk_map
                                    if snapshot is not None
                                    else np.zeros((base.config.rows, base.config.cols), dtype=np.float32)
                                    for snapshot in solver_snapshot_batch
                                ]
                            )
                            risk_violations = _candidate_risk_safety_violations(
                                before_scores=scores_before_candidate,
                                after_scores=scores,
                                action_masks=mask_batch,
                                risk_maps=candidate_risk_maps,
                                active=candidate_tail_tensor,
                                rows=base.config.rows,
                                cols=base.config.cols,
                                mode=candidate_risk_safety_filter,
                            )
                            scores = torch.where(
                                risk_violations.view(-1, 1),
                                scores_before_candidate,
                                scores,
                            )
                            candidate_tail_tensor = candidate_tail_tensor & ~risk_violations
                        candidate_adjusted_decisions += int(candidate_tail_tensor.sum().item())
                        after_candidate_actions = scores.argmax(dim=1)
                        candidate_changed_decisions += int(
                            ((before_candidate_actions != after_candidate_actions) & candidate_tail_tensor).sum().item()
                        )
                if counterfactual_value_trainers and float(counterfactual_value_weight) > 0.0:
                    value_threshold = (
                        int(safe_left_threshold)
                        if counterfactual_value_safe_left_threshold is None
                        else int(counterfactual_value_safe_left_threshold)
                    )
                    value_tail = _adjustment_gate_batch(
                        adjustment_gate,
                        safe_left=safe_left_array,
                        threshold=value_threshold,
                        solver_guess=solver_guess_array,
                    )
                    if counterfactual_value_min_safe_left is not None:
                        value_tail = value_tail & (
                            safe_left_array >= int(counterfactual_value_min_safe_left)
                        )
                    cells = base.config.rows * base.config.cols
                    if counterfactual_value_open_gate == "top-open":
                        top_action_is_open = (scores.argmax(dim=1) < cells).detach().cpu().numpy()
                        value_tail = value_tail & top_action_is_open
                    if counterfactual_value_policy_margin_max is not None:
                        value_tail = value_tail & _counterfactual_value_policy_margin_gate(
                            scores=scores,
                            action_masks=mask_batch,
                            rows=base.config.rows,
                            cols=base.config.cols,
                            max_margin=float(counterfactual_value_policy_margin_max),
                        )
                    if bool(value_tail.any()):
                        value_parts: list[torch.Tensor] = []
                        for value_trainer in counterfactual_value_trainers:
                            values = value_trainer._predict_counterfactual_values_batch(
                                boards=board_batch,
                                global_features_batch=global_batch,
                                use_flip_ensemble=bool(value_trainer.config.inference_augment_flips),
                            )
                            value_parts.append(values.reshape(values.shape[0], -1))
                        counterfactual_values = torch.stack(value_parts, dim=0).mean(dim=0)
                        if float(counterfactual_value_disagree_margin) > 0.0:
                            value_tail = value_tail & _counterfactual_value_disagree_gate(
                                scores=scores,
                                counterfactual_values=counterfactual_values,
                                action_masks=mask_batch,
                                rows=base.config.rows,
                                cols=base.config.cols,
                                margin=float(counterfactual_value_disagree_margin),
                            )
                        if counterfactual_value_region_gate != "any":
                            value_tail = value_tail & _counterfactual_value_region_gate(
                                scores=scores,
                                counterfactual_values=counterfactual_values,
                                action_masks=mask_batch,
                                rows=base.config.rows,
                                cols=base.config.cols,
                                region_gate=counterfactual_value_region_gate,
                            )
                        value_shortlist = _counterfactual_value_shortlist_mask(
                            base_scores=base_scores,
                            specialist_scores=specialist_scores,
                            counterfactual_values=counterfactual_values,
                            action_masks=mask_batch,
                            rows=base.config.rows,
                            cols=base.config.cols,
                            topk=int(counterfactual_value_topk),
                        )
                        value_tail = value_tail & value_shortlist.any(axis=1)
                        before_value_actions = scores.argmax(dim=1)
                        value_tail_tensor = torch.tensor(
                            value_tail,
                            dtype=torch.bool,
                            device=scores.device,
                        )
                        scores = _blend_counterfactual_value_open_scores(
                            scores=scores,
                            counterfactual_values=counterfactual_values,
                            value_tail=value_tail_tensor,
                            value_mask=torch.tensor(
                                value_shortlist,
                                dtype=torch.bool,
                                device=scores.device,
                            ),
                            weight=float(counterfactual_value_weight),
                            temperature=float(counterfactual_value_temperature),
                            scores_are_probabilities=scores_are_probabilities,
                            rows=base.config.rows,
                            cols=base.config.cols,
                        )
                        counterfactual_value_adjusted_decisions += int(value_tail.sum())
                        after_value_actions = scores.argmax(dim=1)
                        counterfactual_value_changed_decisions += int(
                            ((before_value_actions != after_value_actions) & value_tail_tensor).sum().item()
                        )
                action_indices = scores.argmax(dim=1).detach().cpu().numpy()

                tail_count = sum(tail_mask)
                gate_count = int(gate.sum().item())
                tail_decisions += tail_count
                solver_guess_decisions += int(solver_guess_array.sum())
                solver_forced_decisions += int((~solver_guess_array).sum()) if solver_gate is not None else 0
                specialist_decisions += tail_count
                specialist_applied_decisions += gate_count
                base_decisions += len(tail_mask) - gate_count
                for local_index, game_index in enumerate(ready):
                    game = game_batch[game_index]
                    action = decode_action_index(int(action_indices[local_index]), base.config.rows, base.config.cols)
                    _, reward, _, _ = game.step(action)
                    batch_rewards[game_index] += float(reward)
                    batch_steps[game_index] += 1
                    if game.done or batch_steps[game_index] >= max_steps:
                        active.discard(game_index)

            for index, game in enumerate(game_batch):
                wins.append(bool(game.won))
                rewards.append(batch_rewards[index])
                steps.append(batch_steps[index])
                revealed_safe_cells.append(
                    int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum())
                )

    return {
        "games": int(games),
        "seed": int(seed),
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "wins": int(sum(wins)),
        "longest_streak": longest_streak(wins),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_agent_steps": float(np.mean(steps)) if steps else 0.0,
        "avg_revealed_safe_cells": float(np.mean(revealed_safe_cells)) if revealed_safe_cells else 0.0,
        "tail_decisions": int(tail_decisions),
        "base_decisions": int(base_decisions),
        "specialist_decisions": int(specialist_decisions),
        "specialist_applied_decisions": int(specialist_applied_decisions),
        "candidate_adjusted_decisions": int(candidate_adjusted_decisions),
        "candidate_changed_decisions": int(candidate_changed_decisions),
        "candidate_gate_considered_decisions": int(candidate_gate_considered_decisions),
        "candidate_gate_applied_decisions": int(candidate_gate_applied_decisions),
        "candidate_gate_rejected_decisions": int(candidate_gate_rejected_decisions),
        "counterfactual_value_adjusted_decisions": int(counterfactual_value_adjusted_decisions),
        "counterfactual_value_changed_decisions": int(counterfactual_value_changed_decisions),
        "solver_guess_decisions": int(solver_guess_decisions),
        "solver_forced_decisions": int(solver_forced_decisions),
        "specialist_decision_fraction": float(specialist_decisions / max(1, tail_decisions + base_decisions)),
        "specialist_applied_fraction": float(specialist_applied_decisions / max(1, tail_decisions + base_decisions)),
    }


def _uses_solver_gate(adjustment_gate: str) -> bool:
    return adjustment_gate in {"solver-guess", "safe-left-and-solver-guess"}


def _is_solver_guess_state(game: Any, solver: MinesweeperSolver | None) -> bool:
    if solver is None or not getattr(game, "mines_placed", False):
        return False
    snapshot = solver.analyze(game)
    wrong_flags = int((game.flagged & ~game.mines).sum())
    return bool(not snapshot.has_forced_moves and wrong_flags == 0)


def _adjustment_gate_active(
    adjustment_gate: str,
    *,
    safe_left: int,
    threshold: int,
    solver_guess: bool,
) -> bool:
    if adjustment_gate == "safe-left":
        return int(safe_left) <= int(threshold)
    if adjustment_gate == "solver-guess":
        return bool(solver_guess)
    if adjustment_gate == "safe-left-and-solver-guess":
        return int(safe_left) <= int(threshold) and bool(solver_guess)
    raise ValueError(f"unknown adjustment_gate {adjustment_gate!r}")


def _adjustment_gate_batch(
    adjustment_gate: str,
    *,
    safe_left: np.ndarray,
    threshold: int,
    solver_guess: np.ndarray,
) -> np.ndarray:
    safe_left = np.asarray(safe_left)
    solver_guess = np.asarray(solver_guess, dtype=bool)
    if adjustment_gate == "safe-left":
        return safe_left <= int(threshold)
    if adjustment_gate == "solver-guess":
        return solver_guess.copy()
    if adjustment_gate == "safe-left-and-solver-guess":
        return (safe_left <= int(threshold)) & solver_guess
    raise ValueError(f"unknown adjustment_gate {adjustment_gate!r}")


def _candidate_ranker_margin_gate(
    *,
    candidate_probabilities: np.ndarray,
    action_masks: np.ndarray,
    min_margin: float,
) -> np.ndarray:
    """Apply a candidate ranker only when its legal OPEN ranking is decisive."""

    probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    masks = np.asarray(action_masks, dtype=bool)
    if probabilities.ndim != 2 or masks.ndim != 4:
        raise ValueError("candidate probabilities and action masks have invalid ranks")
    batch, cells = probabilities.shape
    if masks.shape[0] != batch or masks.shape[2] * masks.shape[3] != cells:
        raise ValueError("candidate probabilities and action masks have incompatible shapes")
    open_mask = masks[:, action_channel(ActionType.OPEN)].reshape(batch, cells)
    legal_counts = open_mask.sum(axis=1)
    if np.any(legal_counts <= 0):
        raise ValueError("each candidate state must have at least one legal OPEN cell")
    masked = np.where(open_mask, probabilities, -np.inf)
    if cells == 1:
        top_gap = np.full(batch, np.inf, dtype=np.float32)
    else:
        top_two = np.partition(masked, kth=cells - 2, axis=1)[:, -2:]
        top_two.sort(axis=1)
        top_gap = top_two[:, 1] - top_two[:, 0]
    return (legal_counts <= 1) | (top_gap >= float(min_margin))


def combine_model_scores(
    parts: list[torch.Tensor],
    weights: list[float],
    *,
    reduction: str,
    flat_mask: torch.Tensor,
) -> torch.Tensor:
    if not parts:
        raise ValueError("at least one score tensor is required")
    if len(parts) != len(weights):
        raise ValueError("score tensor and weight counts must match")
    if reduction == "mean":
        combined = parts[0] * float(weights[0])
        for part, weight in zip(parts[1:], weights[1:], strict=True):
            combined = combined + part * float(weight)
        return combined
    if reduction == "geomean":
        if not all(torch.isfinite(part[flat_mask]).all() for part in parts):
            raise ValueError("geomean model scores must be finite on legal actions")
        stacked = torch.stack([part.masked_fill(~flat_mask, 1.0).clamp_min(1e-8) for part in parts], dim=0)
        weight_tensor = torch.tensor(weights, dtype=stacked.dtype, device=stacked.device).view(-1, 1, 1)
        combined = torch.exp((stacked.log() * weight_tensor).sum(dim=0))
        return combined.masked_fill(~flat_mask, -1.0)
    raise ValueError(f"unknown model ensemble reduction {reduction!r}")


def longest_streak(wins: list[bool]) -> int:
    best = 0
    current = 0
    for won in wins:
        current = current + 1 if won else 0
        best = max(best, current)
    return best


def _blend_scores(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    specialist_weight: float,
    blend_scope: str,
    rows: int,
    cols: int,
) -> torch.Tensor:
    specialist_weight = float(np.clip(specialist_weight, 0.0, 1.0))
    if specialist_weight <= 0.0:
        return base_scores
    if blend_scope == "all":
        return base_scores * (1.0 - specialist_weight) + specialist_scores * specialist_weight
    if blend_scope == "open":
        blended = base_scores.clone()
        open_channel = action_channel(ActionType.OPEN)
        start = open_channel * rows * cols
        stop = start + rows * cols
        blended[:, start:stop] = (
            base_scores[:, start:stop] * (1.0 - specialist_weight)
            + specialist_scores[:, start:stop] * specialist_weight
        )
        return blended
    if blend_scope == "open-disagree-margin":
        return _blend_scores(
            base_scores=base_scores,
            specialist_scores=specialist_scores,
            specialist_weight=specialist_weight,
            blend_scope="open",
            rows=rows,
            cols=cols,
        )
    raise ValueError(f"unknown blend_scope {blend_scope!r}")


def _open_disagree_margin_gate(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    margin: float,
    rows: int,
    cols: int,
) -> torch.Tensor:
    start = action_channel(ActionType.OPEN) * rows * cols
    stop = start + rows * cols
    base_open = base_scores[:, start:stop]
    specialist_open = specialist_scores[:, start:stop]
    base_choice = base_open.argmax(dim=1)
    specialist_top2 = torch.topk(specialist_open, k=min(2, specialist_open.shape[1]), dim=1)
    specialist_choice = specialist_top2.indices[:, 0]
    if specialist_top2.values.shape[1] == 1:
        specialist_gap = torch.full_like(specialist_top2.values[:, 0], float("inf"))
    else:
        specialist_gap = specialist_top2.values[:, 0] - specialist_top2.values[:, 1]
    disagree = specialist_choice != base_choice
    confident = specialist_gap >= float(margin)
    return (disagree & confident).view(-1, 1)


def _specialist_gate(
    *,
    tail_mask: list[bool],
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    blend_scope: str,
    specialist_margin: float,
    specialist_region_gate: str = "any",
    rows: int,
    cols: int,
    gate_calibrator: GateCalibrator | None = None,
    board_batch: np.ndarray | None = None,
    global_batch: np.ndarray | None = None,
    mask_batch: np.ndarray | None = None,
    scores_are_probabilities: bool = False,
    device: torch.device | None = None,
) -> torch.Tensor:
    gate = torch.tensor(tail_mask, dtype=torch.bool, device=device or base_scores.device).view(-1, 1)
    if blend_scope != "open-disagree-margin":
        if specialist_region_gate == "any":
            return gate
        if mask_batch is None:
            raise ValueError("specialist region gate requires action masks")
        region = _specialist_region_gate(
            base_scores=base_scores,
            specialist_scores=specialist_scores,
            action_masks=mask_batch,
            rows=rows,
            cols=cols,
            region_gate=specialist_region_gate,
        )
        return gate & torch.tensor(region, dtype=torch.bool, device=gate.device).view(-1, 1)

    margin_gate = _open_disagree_margin_gate(
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        margin=float(specialist_margin),
        rows=rows,
        cols=cols,
    )
    gate = gate & margin_gate
    if specialist_region_gate != "any":
        if mask_batch is None:
            raise ValueError("specialist region gate requires action masks")
        region = _specialist_region_gate(
            base_scores=base_scores,
            specialist_scores=specialist_scores,
            action_masks=mask_batch,
            rows=rows,
            cols=cols,
            region_gate=specialist_region_gate,
        )
        gate = gate & torch.tensor(region, dtype=torch.bool, device=gate.device).view(-1, 1)
    if gate_calibrator is None:
        return gate

    if board_batch is None or global_batch is None or mask_batch is None:
        raise ValueError("gate calibrator requires board, global, and mask batches")
    gate_features = build_gate_features(
        base_scores=base_scores.detach().cpu().numpy(),
        specialist_scores=specialist_scores.detach().cpu().numpy(),
        global_features=global_batch,
        boards=board_batch,
        action_masks=mask_batch,
        scores_are_probabilities=scores_are_probabilities,
    )
    calibrated = torch.tensor(
        gate_calibrator.should_apply_features(gate_features),
        dtype=torch.bool,
        device=gate.device,
    ).view(-1, 1)
    return gate & calibrated


def _specialist_region_gate(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    region_gate: str,
) -> np.ndarray:
    if region_gate == "any":
        return np.ones(base_scores.shape[0], dtype=bool)
    if region_gate not in {
        "current-edge-or-corner",
        "specialist-edge-or-corner",
        "either-edge-or-corner",
        "current-interior",
        "specialist-interior",
        "both-interior",
    }:
        raise ValueError(f"unknown specialist_region_gate {region_gate!r}")

    cells = rows * cols
    open_channel = action_channel(ActionType.OPEN)
    legal = torch.tensor(
        np.asarray(action_masks[:, open_channel], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=base_scores.device,
    )
    start = open_channel * cells
    stop = start + cells
    current_open = base_scores[:, start:stop]
    specialist_open = specialist_scores[:, start:stop]
    current_choice = current_open.masked_fill(~legal, -1e9).argmax(dim=1)
    specialist_choice = specialist_open.masked_fill(~legal, -1e9).argmax(dim=1)
    current_region = _flat_open_indices_are_edge_or_corner(current_choice, rows=rows, cols=cols)
    specialist_region = _flat_open_indices_are_edge_or_corner(
        specialist_choice,
        rows=rows,
        cols=cols,
    )
    if region_gate == "current-edge-or-corner":
        return current_region.detach().cpu().numpy()
    if region_gate == "specialist-edge-or-corner":
        return specialist_region.detach().cpu().numpy()
    if region_gate == "either-edge-or-corner":
        return (current_region | specialist_region).detach().cpu().numpy()
    current_interior = ~current_region
    specialist_interior = ~specialist_region
    if region_gate == "current-interior":
        return current_interior.detach().cpu().numpy()
    if region_gate == "specialist-interior":
        return specialist_interior.detach().cpu().numpy()
    return (current_interior & specialist_interior).detach().cpu().numpy()


def _blend_candidate_open_scores(
    *,
    scores: torch.Tensor,
    candidate_probabilities: torch.Tensor,
    candidate_tail: torch.Tensor,
    candidate_mask: torch.Tensor,
    weight: float,
    rows: int,
    cols: int,
    scores_are_probabilities: bool = True,
    mode: str = "blend",
) -> torch.Tensor:
    """Blend candidate rankings without changing the OPEN score scale."""

    if mode not in {"blend", "replace"}:
        raise ValueError(f"unknown candidate score mode {mode!r}")
    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return scores
    start = action_channel(ActionType.OPEN) * rows * cols
    stop = start + rows * cols
    adjusted = scores.clone()
    open_scores = adjusted[:, start:stop]
    candidate_scores = candidate_probabilities.to(dtype=scores.dtype, device=scores.device)
    active_mask = candidate_tail.view(-1, 1) & candidate_mask
    if scores_are_probabilities:
        # Candidate probabilities sum to one over legal OPEN cells, while the
        # policy scores contain probability mass for every action channel.
        # Match the OPEN-channel scale before blending so the candidate model
        # cannot dominate the full policy.
        open_mass = open_scores.clamp_min(0.0).sum(dim=1, keepdim=True)
        candidate_scores = candidate_scores * open_mass
        if mode == "replace":
            candidate_scores = candidate_scores.masked_fill(~active_mask, 0.0)
            candidate_scores = candidate_scores / candidate_scores.sum(dim=1, keepdim=True).clamp_min(1e-8)
            blended = candidate_scores * open_mass
        else:
            blended = open_scores * (1.0 - weight) + candidate_scores * weight
    else:
        # In logit mode, use a centered ranking perturbation whose magnitude
        # follows the current OPEN score spread.
        candidate_logits = candidate_scores.clamp_min(1e-8).log()
        active_count = active_mask.sum(dim=1, keepdim=True).clamp_min(1)
        centered = candidate_logits - (
            candidate_logits.masked_fill(~active_mask, 0.0).sum(dim=1, keepdim=True)
            / active_count
        )
        open_mean = open_scores.masked_fill(~active_mask, 0.0).sum(dim=1, keepdim=True) / active_count
        open_variance = (
            (open_scores - open_mean).masked_fill(~active_mask, 0.0).square().sum(dim=1, keepdim=True)
            / active_count
        )
        candidate_scores = open_mean + centered * open_variance.sqrt().clamp_min(1e-3)
        blended = (
            candidate_scores
            if mode == "replace"
            else open_scores * (1.0 - weight) + candidate_scores * weight
        )
    adjusted[:, start:stop] = torch.where(active_mask, blended, open_scores)
    return adjusted


def _candidate_risk_safety_violations(
    *,
    before_scores: torch.Tensor,
    after_scores: torch.Tensor,
    action_masks: np.ndarray,
    risk_maps: np.ndarray,
    active: torch.Tensor,
    rows: int,
    cols: int,
    mode: str,
) -> torch.Tensor:
    if mode not in {"not-higher", "strictly-lower"}:
        raise ValueError(f"unknown candidate risk safety mode {mode!r}")
    risk_maps = np.asarray(risk_maps, dtype=np.float32)
    if risk_maps.shape != (before_scores.shape[0], rows, cols):
        raise ValueError(
            "risk_maps must have shape [batch, rows, cols], "
            f"got {risk_maps.shape}"
        )
    open_mask = np.asarray(
        action_masks[:, action_channel(ActionType.OPEN)],
        dtype=bool,
    ).reshape(-1, rows * cols)
    before_open = before_scores[:, : rows * cols].masked_fill(
        ~torch.tensor(open_mask, dtype=torch.bool, device=before_scores.device),
        -1e9,
    ).argmax(dim=1)
    after_open = after_scores[:, : rows * cols].masked_fill(
        ~torch.tensor(open_mask, dtype=torch.bool, device=after_scores.device),
        -1e9,
    ).argmax(dim=1)
    risk_tensor = torch.tensor(
        risk_maps.reshape(-1, rows * cols),
        dtype=before_scores.dtype,
        device=before_scores.device,
    )
    before_risk = risk_tensor.gather(1, before_open.view(-1, 1)).squeeze(1)
    after_risk = risk_tensor.gather(1, after_open.view(-1, 1)).squeeze(1)
    epsilon = 1e-6
    if mode == "not-higher":
        unsafe = after_risk > before_risk + epsilon
    else:
        unsafe = after_risk >= before_risk - epsilon
    return active.to(dtype=torch.bool, device=before_scores.device) & unsafe


def _blend_counterfactual_value_open_scores(
    *,
    scores: torch.Tensor,
    counterfactual_values: torch.Tensor,
    value_tail: torch.Tensor,
    value_mask: torch.Tensor,
    weight: float,
    temperature: float,
    scores_are_probabilities: bool,
    rows: int,
    cols: int,
) -> torch.Tensor:
    """Rerank a small OPEN shortlist using model-only counterfactual values."""

    weight = float(np.clip(weight, 0.0, 1.0))
    if weight <= 0.0:
        return scores
    start = action_channel(ActionType.OPEN) * rows * cols
    stop = start + rows * cols
    adjusted = scores.clone()
    open_scores = adjusted[:, start:stop]
    values = counterfactual_values.to(dtype=scores.dtype, device=scores.device)
    active_mask = value_tail.view(-1, 1) & value_mask
    if scores_are_probabilities:
        masked_values = values.masked_fill(~active_mask, -1e9)
        value_probs = torch.softmax(masked_values / max(float(temperature), 1e-6), dim=1)
        value_probs = value_probs.masked_fill(~active_mask, 0.0)
        blended = open_scores * (1.0 - weight) + value_probs * weight
    else:
        centered = values - torch.where(active_mask, values, torch.zeros_like(values)).sum(
            dim=1,
            keepdim=True,
        ) / active_mask.sum(dim=1, keepdim=True).clamp_min(1)
        variance = torch.where(active_mask, centered.square(), torch.zeros_like(centered)).sum(
            dim=1,
            keepdim=True,
        ) / active_mask.sum(dim=1, keepdim=True).clamp_min(1)
        normalized = centered / variance.sqrt().clamp_min(1e-6)
        blended = open_scores + weight * normalized
    adjusted[:, start:stop] = torch.where(active_mask, blended, open_scores)
    return adjusted


def _candidate_shortlist_mask(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    topk: int,
    candidate_probabilities: np.ndarray | None = None,
) -> np.ndarray:
    cells = rows * cols
    topk = max(1, min(int(topk), cells))
    legal = torch.tensor(
        np.asarray(action_masks[:, action_channel(ActionType.OPEN)], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=base_scores.device,
    )
    shortlist = torch.zeros_like(legal)
    for scores in (base_scores[:, :cells], specialist_scores[:, :cells]):
        _, indices = torch.topk(scores.masked_fill(~legal, -1e9), k=topk, dim=1)
        shortlist.scatter_(1, indices, True)
    if candidate_probabilities is not None:
        candidate_scores = torch.tensor(
            np.asarray(candidate_probabilities, dtype=np.float32).reshape(-1, cells),
            dtype=base_scores.dtype,
            device=base_scores.device,
        )
        if candidate_scores.shape != legal.shape:
            raise ValueError(
                "candidate probabilities shape does not match the legal OPEN mask: "
                f"{tuple(candidate_scores.shape)} != {tuple(legal.shape)}"
            )
        _, candidate_indices = torch.topk(
            candidate_scores.masked_fill(~legal, -1e9),
            k=topk,
            dim=1,
        )
        shortlist.scatter_(1, candidate_indices, True)
    return (shortlist & legal).detach().cpu().numpy()


def _counterfactual_value_shortlist_mask(
    *,
    base_scores: torch.Tensor,
    specialist_scores: torch.Tensor,
    counterfactual_values: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    topk: int,
) -> np.ndarray:
    cells = rows * cols
    topk = max(1, min(int(topk), cells))
    legal = torch.tensor(
        np.asarray(action_masks[:, action_channel(ActionType.OPEN)], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=base_scores.device,
    )
    shortlist = torch.zeros_like(legal)
    for scores in (base_scores[:, :cells], specialist_scores[:, :cells], counterfactual_values):
        _, indices = torch.topk(scores.masked_fill(~legal, -1e9), k=topk, dim=1)
        shortlist.scatter_(1, indices, True)
    return (shortlist & legal).detach().cpu().numpy()


def _counterfactual_value_disagree_gate(
    *,
    scores: torch.Tensor,
    counterfactual_values: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    margin: float,
) -> np.ndarray:
    """Return rows where the value head makes a confident different choice."""

    cells = rows * cols
    open_channel = action_channel(ActionType.OPEN)
    legal = torch.tensor(
        np.asarray(action_masks[:, open_channel], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=scores.device,
    )
    current_open = scores[:, open_channel * cells : (open_channel + 1) * cells]
    current_choice = current_open.masked_fill(~legal, -1e9).argmax(dim=1)
    value_open = counterfactual_values.masked_fill(~legal, -1e9)
    value_top2 = torch.topk(value_open, k=min(2, cells), dim=1)
    value_choice = value_top2.indices[:, 0]
    if value_top2.values.shape[1] == 1:
        value_gap = torch.full_like(value_top2.values[:, 0], float("inf"))
    else:
        value_gap = value_top2.values[:, 0] - value_top2.values[:, 1]
    return (
        (current_choice != value_choice)
        & (value_gap >= float(margin))
    ).detach().cpu().numpy()


def _counterfactual_value_policy_margin_gate(
    *,
    scores: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    max_margin: float,
) -> np.ndarray:
    """Return rows where the current OPEN policy is uncertain enough to rerank."""

    cells = rows * cols
    open_channel = action_channel(ActionType.OPEN)
    legal = torch.tensor(
        np.asarray(action_masks[:, open_channel], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=scores.device,
    )
    legal_counts = legal.sum(dim=1)
    current_open = scores[:, open_channel * cells : (open_channel + 1) * cells]
    topk = min(2, cells)
    top_values = torch.topk(current_open.masked_fill(~legal, -1e9), k=topk, dim=1).values
    if topk < 2:
        return np.zeros(scores.shape[0], dtype=bool)
    margin = top_values[:, 0] - top_values[:, 1]
    return (
        (legal_counts >= 2)
        & torch.isfinite(margin)
        & (margin <= float(max_margin))
    ).detach().cpu().numpy()


def _counterfactual_value_region_gate(
    *,
    scores: torch.Tensor,
    counterfactual_values: torch.Tensor,
    action_masks: np.ndarray,
    rows: int,
    cols: int,
    region_gate: str,
) -> np.ndarray:
    """Return rows where the current or value OPEN choice is on the boundary."""

    if region_gate == "any":
        return np.ones(scores.shape[0], dtype=bool)
    if region_gate not in {
        "current-edge-or-corner",
        "value-edge-or-corner",
        "either-edge-or-corner",
        "current-interior",
        "value-interior",
        "both-interior",
    }:
        raise ValueError(f"unknown counterfactual_value_region_gate {region_gate!r}")

    cells = rows * cols
    open_channel = action_channel(ActionType.OPEN)
    legal = torch.tensor(
        np.asarray(action_masks[:, open_channel], dtype=bool).reshape(-1, cells),
        dtype=torch.bool,
        device=scores.device,
    )
    current_open = scores[:, open_channel * cells : (open_channel + 1) * cells]
    current_choice = current_open.masked_fill(~legal, -1e9).argmax(dim=1)
    value_choice = counterfactual_values.masked_fill(~legal, -1e9).argmax(dim=1)
    current_region = _flat_open_indices_are_edge_or_corner(current_choice, rows=rows, cols=cols)
    value_region = _flat_open_indices_are_edge_or_corner(value_choice, rows=rows, cols=cols)
    if region_gate == "current-edge-or-corner":
        return current_region.detach().cpu().numpy()
    if region_gate == "value-edge-or-corner":
        return value_region.detach().cpu().numpy()
    if region_gate == "either-edge-or-corner":
        return (current_region | value_region).detach().cpu().numpy()
    current_interior = ~current_region
    value_interior = ~value_region
    if region_gate == "current-interior":
        return current_interior.detach().cpu().numpy()
    if region_gate == "value-interior":
        return value_interior.detach().cpu().numpy()
    return (current_interior & value_interior).detach().cpu().numpy()


def _flat_open_indices_are_edge_or_corner(
    indices: torch.Tensor,
    *,
    rows: int,
    cols: int,
) -> torch.Tensor:
    row = indices // int(cols)
    col = indices % int(cols)
    return (row == 0) | (row == int(rows) - 1) | (col == 0) | (col == int(cols) - 1)


if __name__ == "__main__":
    main()
