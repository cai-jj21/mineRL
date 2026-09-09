from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import save_extreme_dataset
from minesweeper_rl.gate_calibration import GateCalibrator
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import load_checkpoint, make_game


HARD_REFINE_PATH = ROOT / "scripts" / "hard_loss_refine.py"
HARD_SPEC = importlib.util.spec_from_file_location("hard_loss_refine_for_gated_dataset", HARD_REFINE_PATH)
if HARD_SPEC is None or HARD_SPEC.loader is None:
    raise RuntimeError(f"could not load {HARD_REFINE_PATH}")
hard_loss_refine = importlib.util.module_from_spec(HARD_SPEC)
sys.modules[HARD_SPEC.name] = hard_loss_refine
HARD_SPEC.loader.exec_module(hard_loss_refine)

GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_specialist_for_dataset", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


GUESS_FAMILIES = {
    "guess",
    "guess_tail",
    "edge_guess",
    "corner_guess",
    "edge_guess_tail",
    "corner_guess_tail",
}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Mine solver-labelled hard states from failures of a gated pure-RL policy."
    )
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-ensemble-checkpoint", dest="base_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--base-model-weight", dest="base_model_weights", action="append", type=float, default=[])
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-ensemble-checkpoint", dest="specialist_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--specialist-model-weight", dest="specialist_model_weights", action="append", type=float, default=[])
    parser.add_argument("--model-ensemble-reduction", choices=["mean", "geomean"], default="mean")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--base-risk-head-weight", type=float)
    parser.add_argument("--specialist-risk-head-weight", type=float)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=130000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--tail-states", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument(
        "--save-every-records",
        type=int,
        default=0,
        help="Write a partial dataset after every N selected output records. Use 0 to disable.",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=None,
        help="Optional path for periodic partial saves. Defaults to <output>.partial.npz.",
    )
    parser.add_argument(
        "--progress-every-seconds",
        type=float,
        default=30.0,
        help="Print a compact JSON progress row every N seconds. Use 0 to disable.",
    )
    parser.add_argument("--endgame-safe-left", type=int, default=100)
    parser.add_argument("--exact-limit", type=int, default=32)
    parser.add_argument("--guess-topk", type=int, default=8)
    parser.add_argument("--counterfactual-labels", action="store_true")
    parser.add_argument("--counterfactual-topk", type=int, default=64)
    parser.add_argument(
        "--counterfactual-all-open",
        action="store_true",
        help="Label every legal OPEN candidate in each guess state.",
    )
    parser.add_argument(
        "--counterfactual-model-topk",
        type=int,
        default=None,
        help="Number of gated-policy top OPEN candidates to force into offline labels.",
    )
    parser.add_argument("--sample-family", choices=["guess", "all-hard"], default="guess")
    parser.add_argument(
        "--family-filter",
        action="append",
        choices=sorted(GUESS_FAMILIES),
        default=[],
        help="Optionally keep only specific extreme families. Repeat to include multiple families.",
    )
    parser.add_argument(
        "--collect-guess-states",
        action="store_true",
        help="Label true no-forced OPEN guess states during play, instead of relying only on failure tails.",
    )
    parser.add_argument(
        "--collect-guess-safe-left-min",
        type=int,
        default=0,
        help="Minimum safe cells left for live guess-state collection.",
    )
    parser.add_argument(
        "--collect-guess-safe-left-max",
        type=int,
        default=None,
        help="Maximum safe cells left for live guess-state collection. Omit for no upper bound.",
    )
    parser.add_argument(
        "--collect-guess-max-per-game",
        type=int,
        default=0,
        help="Maximum live guess states to keep per game. Use 0 for unlimited.",
    )
    parser.add_argument(
        "--collect-guess-prefilter",
        choices=["exact", "basic", "none"],
        default="exact",
        help="Forced-move prefilter used before expensive live guess labelling.",
    )
    parser.add_argument(
        "--collect-guess-policy-margin-max",
        type=float,
        default=None,
        help="Only collect live guess candidates when the current OPEN top-two score margin is at most this value.",
    )
    parser.add_argument(
        "--collect-guess-open-candidates-min",
        type=int,
        default=0,
        help="Only collect live guess states with at least N legal OPEN candidates.",
    )
    parser.add_argument(
        "--collect-guess-open-candidates-max",
        type=int,
        default=None,
        help="Only collect live guess states with at most N legal OPEN candidates. Omit for no upper bound.",
    )
    parser.add_argument(
        "--collect-guess-min-behavior-regret",
        type=float,
        default=0.0,
        help="After counterfactual labelling, keep only states where the selected OPEN action has at least this regret.",
    )
    parser.add_argument(
        "--collect-guess-behavior-mine-only",
        action="store_true",
        help="After counterfactual labelling, keep only states where the selected OPEN action is an actual mine.",
    )
    parser.add_argument(
        "--collect-guess-region",
        choices=["any", "current-interior", "current-edge", "current-corner", "current-edge-or-corner"],
        default="any",
        help="Restrict live guess collection by the policy's current OPEN action geometry.",
    )
    parser.add_argument("--safe-left-threshold", type=int, default=60)
    parser.add_argument("--specialist-weight", type=float, default=0.5)
    parser.add_argument("--blend-scope", choices=["all", "open", "open-disagree-margin"], default="open-disagree-margin")
    parser.add_argument("--specialist-margin", type=float, default=0.05)
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
    )
    parser.add_argument("--gate-calibrator", type=Path, default=None)
    parser.add_argument("--gate-threshold", type=float, default=None)
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--no-inference-flips", action="store_true")
    args = parser.parse_args()
    if (
        float(args.collect_guess_min_behavior_regret) > 0.0
        or bool(args.collect_guess_behavior_mine_only)
    ) and not bool(args.counterfactual_labels):
        raise ValueError(
            "--collect-guess-min-behavior-regret and --collect-guess-behavior-mine-only require "
            "--counterfactual-labels"
        )

    base_checkpoints = [args.base_checkpoint, *args.base_ensemble_checkpoints]
    specialist_checkpoints = [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
    base_weights = gated.normalize_ensemble_weights(
        args.base_model_weights,
        expected=len(base_checkpoints),
        label="base-model-weight",
    )
    specialist_weights = gated.normalize_ensemble_weights(
        args.specialist_model_weights,
        expected=len(specialist_checkpoints),
        label="specialist-model-weight",
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
    specialists = [
        load_eval_trainer(path, args, risk_head_weight=specialist_risk_head_weight)
        for path in specialist_checkpoints
    ]
    label_trainer = bases[0]
    label_trainer.solver = MinesweeperSolver(exact_limit=args.exact_limit)
    gate_calibrator = GateCalibrator.load(args.gate_calibrator) if args.gate_calibrator else None
    if gate_calibrator is not None and args.gate_threshold is not None:
        gate_calibrator.threshold = float(np.clip(args.gate_threshold, 0.0, 1.0))

    started_at = time.time()
    transitions, mining_metrics = mine_gated_failures(
        label_trainer=label_trainer,
        bases=bases,
        base_weights=base_weights,
        specialists=specialists,
        specialist_weights=specialist_weights,
        args=args,
        gate_calibrator=gate_calibrator,
    )
    family_filter = set(args.family_filter or [])
    selected = []
    source_families: Counter[str] = Counter()
    for transition in transitions:
        family = transition.extreme_family or "unclassified"
        source_families[family] += 1
        if args.sample_family == "guess" and (not transition.expert_is_guess or family not in GUESS_FAMILIES):
            continue
        if family_filter and family not in family_filter:
            continue
        selected.append(transition)
        if len(selected) >= args.max_records:
            break

    if not selected:
        raise RuntimeError("no gated failure transitions were selected; increase --games or --tail-states")

    selected_counts = Counter(transition.extreme_family or "unclassified" for transition in selected)
    manifest = save_extreme_dataset(
        args.output,
        selected,
        metadata={
            "source": "gated_policy_live_guess_and_failure_tail_solver_labels"
            if args.collect_guess_states
            else "gated_policy_failure_tail_solver_labels",
            "base_checkpoints": [str(path) for path in base_checkpoints],
            "base_model_weights": base_weights,
            "risk_head_weight": float(args.risk_head_weight),
            "base_risk_head_weight": float(base_risk_head_weight),
            "specialist_risk_head_weight": float(specialist_risk_head_weight),
            "specialist_checkpoints": [str(path) for path in specialist_checkpoints],
            "specialist_model_weights": specialist_weights,
            "model_ensemble_reduction": args.model_ensemble_reduction,
            "games_requested": int(args.games),
            "seed": int(args.seed),
            "tail_states": int(args.tail_states),
            "max_steps": int(args.max_steps),
            "save_every_records": int(args.save_every_records),
            "partial_output": str(_partial_output_path(args)),
            "progress_every_seconds": float(args.progress_every_seconds),
            "endgame_safe_left": int(args.endgame_safe_left),
            "exact_limit": int(args.exact_limit),
            "guess_topk": int(args.guess_topk),
            "counterfactual_labels": bool(args.counterfactual_labels),
            "counterfactual_topk": int(args.counterfactual_topk),
            "counterfactual_all_open": bool(args.counterfactual_all_open),
            "counterfactual_model_topk": hard_loss_refine._counterfactual_model_topk(args),
            "sample_family": args.sample_family,
            "family_filter": sorted(family_filter),
            "collect_guess_states": bool(args.collect_guess_states),
            "collect_guess_safe_left_min": int(args.collect_guess_safe_left_min),
            "collect_guess_safe_left_max": None
            if args.collect_guess_safe_left_max is None
            else int(args.collect_guess_safe_left_max),
            "collect_guess_max_per_game": int(args.collect_guess_max_per_game),
            "collect_guess_prefilter": args.collect_guess_prefilter,
            "collect_guess_policy_margin_max": (
                None
                if args.collect_guess_policy_margin_max is None
                else float(args.collect_guess_policy_margin_max)
            ),
            "collect_guess_open_candidates_min": int(args.collect_guess_open_candidates_min),
            "collect_guess_open_candidates_max": (
                None
                if args.collect_guess_open_candidates_max is None
                else int(args.collect_guess_open_candidates_max)
            ),
            "collect_guess_min_behavior_regret": float(args.collect_guess_min_behavior_regret),
            "collect_guess_behavior_mine_only": bool(args.collect_guess_behavior_mine_only),
            "collect_guess_region": args.collect_guess_region,
            "safe_left_threshold": int(args.safe_left_threshold),
            "specialist_weight": float(args.specialist_weight),
            "blend_scope": args.blend_scope,
            "specialist_margin": float(args.specialist_margin),
            "specialist_region_gate": args.specialist_region_gate,
            "gate_calibrator": str(args.gate_calibrator) if args.gate_calibrator else None,
            "gate_threshold": None if gate_calibrator is None else float(gate_calibrator.threshold),
            "inference_flips": not args.no_inference_flips,
            "inference_ensemble": args.inference_ensemble,
            "mining_metrics": mining_metrics,
            "source_families": dict(sorted(source_families.items())),
            "selected_families": dict(sorted(selected_counts.items())),
            "selected_records": len(selected),
            "elapsed_seconds": time.time() - started_at,
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "records": len(selected),
                "source_families": dict(sorted(source_families.items())),
                "selected_families": dict(sorted(selected_counts.items())),
                "mining_metrics": mining_metrics,
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def load_eval_trainer(path: Path, args: argparse.Namespace, *, risk_head_weight: float | None = None):
    trainer = load_checkpoint(path, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = "full"
    trainer.config.exact_limit = args.exact_limit
    trainer.config.guess_supervision_topk = max(1, int(args.guess_topk))
    trainer.config.inference_augment_flips = not args.no_inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = float(args.risk_head_weight if risk_head_weight is None else risk_head_weight)
    trainer.model.eval()
    return trainer


def mine_gated_failures(
    *,
    label_trainer,
    bases,
    base_weights,
    specialists,
    specialist_weights,
    args: argparse.Namespace,
    gate_calibrator: GateCalibrator | None = None,
):
    transitions = []
    wins = 0
    losses = 0
    wrong_flag_losses = 0
    forced_terminal_losses = 0
    endgame_losses = 0
    tail_decisions = 0
    specialist_applied_decisions = 0
    live_guess_label_attempts = 0
    live_guess_transitions = 0
    live_guess_skipped_per_game_limit = 0
    live_guess_wrong_flag_skips = 0
    live_guess_forced_skips = 0
    live_guess_unplaced_skips = 0
    live_guess_policy_margin_skips = 0
    live_guess_open_candidate_skips = 0
    live_guess_region_skips = 0
    live_guess_behavior_skips = 0
    selected_like_transitions = 0
    reached_record_cap = False
    total_safe = label_trainer.config.rows * label_trainer.config.cols - label_trainer.config.mines
    solver = label_trainer.solver
    started_at = time.time()
    last_partial_records = 0
    last_progress_at = started_at
    final_save_done = {"value": False}

    def current_metrics() -> dict[str, object]:
        return {
            "seed": int(args.seed),
            "games": int(args.games),
            "wins": wins,
            "losses": losses,
            "win_rate": wins / max(1, int(args.games)),
            "transitions": len(transitions),
            "wrong_flag_losses": wrong_flag_losses,
            "forced_terminal_losses": forced_terminal_losses,
            "endgame_losses": endgame_losses,
            "tail_decisions": tail_decisions,
            "specialist_applied_decisions": specialist_applied_decisions,
            "live_guess_label_attempts": live_guess_label_attempts,
            "live_guess_transitions": live_guess_transitions,
            "live_guess_skipped_per_game_limit": live_guess_skipped_per_game_limit,
            "live_guess_wrong_flag_skips": live_guess_wrong_flag_skips,
            "live_guess_forced_skips": live_guess_forced_skips,
            "live_guess_unplaced_skips": live_guess_unplaced_skips,
            "live_guess_policy_margin_skips": live_guess_policy_margin_skips,
            "live_guess_open_candidate_skips": live_guess_open_candidate_skips,
            "live_guess_region_skips": live_guess_region_skips,
            "live_guess_behavior_skips": live_guess_behavior_skips,
            "selected_like_transitions": selected_like_transitions,
            "reached_record_cap": bool(reached_record_cap),
            "elapsed_seconds": time.time() - started_at,
        }

    def maybe_report_progress(*, force: bool = False) -> None:
        nonlocal last_progress_at
        interval = max(0.0, float(args.progress_every_seconds))
        now = time.time()
        if not force and (interval <= 0.0 or now - last_progress_at < interval):
            return
        metrics = current_metrics()
        print(
            json.dumps(
                {
                    "status": "sampling",
                    "games_finished": int(wins + losses),
                    "selected_like_transitions": int(selected_like_transitions),
                    "live_guess_transitions": int(live_guess_transitions),
                    "live_guess_label_attempts": int(live_guess_label_attempts),
                    "live_guess_forced_skips": int(live_guess_forced_skips),
                    "live_guess_open_candidate_skips": int(live_guess_open_candidate_skips),
                    "live_guess_behavior_skips": int(live_guess_behavior_skips),
                    "elapsed_seconds": metrics["elapsed_seconds"],
                },
                ensure_ascii=False,
            ),
            flush=True,
        )
        last_progress_at = now

    def maybe_save_partial(*, reason: str = "periodic", force: bool = False) -> None:
        nonlocal last_partial_records
        save_every = max(0, int(args.save_every_records))
        if not force and save_every <= 0:
            return
        selected = _selected_output_transitions(transitions, args)
        selected_count = len(selected)
        if selected_count <= 0:
            return
        if (
            not force
            and selected_count // save_every <= int(last_partial_records) // save_every
        ):
            return
        source_families = Counter(transition.extreme_family or "unclassified" for transition in transitions)
        selected_counts = Counter(transition.extreme_family or "unclassified" for transition in selected)
        manifest = save_extreme_dataset(
            _partial_output_path(args),
            selected,
            metadata=_partial_metadata(
                args,
                metrics=current_metrics(),
                source_families=source_families,
                selected_counts=selected_counts,
                reason=reason,
            ),
        )
        last_partial_records = selected_count
        print(
            json.dumps(
                {
                    "status": "partial_saved",
                    "reason": reason,
                    "output": str(_partial_output_path(args)),
                    "records": selected_count,
                    "manifest": manifest.get("manifest"),
                },
                ensure_ascii=False,
            ),
            flush=True,
        )

    def save_partial_on_exit() -> None:
        if final_save_done["value"]:
            return
        maybe_save_partial(reason="exit", force=True)

    atexit.register(save_partial_on_exit)
    label_args = SimpleNamespace(
        endgame_safe_left=int(args.endgame_safe_left),
        include_plain_tail=False,
        endgame_weight=1,
        terminal_weight=1,
        guess_weight=1,
        edge_weight=1,
        corner_weight=1,
        wrong_flag_weight=1,
        counterfactual_labels=bool(args.counterfactual_labels),
        counterfactual_topk=max(1, int(args.counterfactual_topk)),
        counterfactual_model_topk=hard_loss_refine._counterfactual_model_topk(args),
        counterfactual_all_open=bool(args.counterfactual_all_open),
    )

    with torch.inference_mode():
        for batch_start in range(0, int(args.games), int(args.batch_size)):
            if reached_record_cap:
                break
            current_batch = min(int(args.batch_size), int(args.games) - batch_start)
            games = [make_game(label_trainer.config, seed=int(args.seed) + batch_start + offset) for offset in range(current_batch)]
            tails = [[] for _ in range(current_batch)]
            live_guess_counts = [0 for _ in range(current_batch)]
            steps = [0 for _ in range(current_batch)]
            active = set(range(current_batch))

            while active:
                if reached_record_cap:
                    break
                ready = []
                boards = []
                global_features_batch = []
                action_masks = []
                tail_mask = []
                safe_left_batch = []
                for index in list(active):
                    game = games[index]
                    if game.done or steps[index] >= label_trainer.config.max_steps:
                        active.remove(index)
                        continue
                    board, global_features, action_mask = hard_loss_refine.encode_state(game)
                    action_mask = label_trainer._decision_action_mask(action_mask)
                    if not action_mask.any():
                        active.remove(index)
                        continue
                    progress = float(np.clip(global_features[1], 0.0, 1.0))
                    safe_left = int(round((1.0 - progress) * total_safe))
                    ready.append(index)
                    boards.append(board)
                    global_features_batch.append(global_features)
                    action_masks.append(action_mask)
                    tail_mask.append(safe_left <= int(args.safe_left_threshold))
                    safe_left_batch.append(safe_left)

                if not ready:
                    continue

                board_batch = np.stack(boards)
                global_batch = np.stack(global_features_batch)
                mask_batch = np.stack(action_masks)
                flat_mask = torch.tensor(mask_batch.reshape(mask_batch.shape[0], -1), dtype=torch.bool, device=label_trainer.device)
                base_scores = gated.combine_model_scores(
                    [
                        trainer._predict_policy_scores_batch(
                            boards=board_batch,
                            global_features_batch=global_batch,
                            action_masks=mask_batch,
                            use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                        )
                        for trainer in bases
                    ],
                    base_weights,
                    reduction=args.model_ensemble_reduction,
                    flat_mask=flat_mask,
                )
                specialist_scores = gated.combine_model_scores(
                    [
                        trainer._predict_policy_scores_batch(
                            boards=board_batch,
                            global_features_batch=global_batch,
                            action_masks=mask_batch,
                            use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                        )
                        for trainer in specialists
                    ],
                    specialist_weights,
                    reduction=args.model_ensemble_reduction,
                    flat_mask=flat_mask,
                )
                scores_are_probabilities = bool(
                    label_trainer.config.inference_augment_flips
                    and label_trainer.config.inference_ensemble == "probs"
                )
                gate = gated._specialist_gate(
                    tail_mask=tail_mask,
                    base_scores=base_scores,
                    specialist_scores=specialist_scores,
                    blend_scope=args.blend_scope,
                    specialist_margin=float(args.specialist_margin),
                    specialist_region_gate=args.specialist_region_gate,
                    rows=label_trainer.config.rows,
                    cols=label_trainer.config.cols,
                    gate_calibrator=gate_calibrator,
                    board_batch=board_batch,
                    global_batch=global_batch,
                    mask_batch=mask_batch,
                    scores_are_probabilities=scores_are_probabilities,
                    device=label_trainer.device,
                )
                blended_scores = gated._blend_scores(
                    base_scores=base_scores,
                    specialist_scores=specialist_scores,
                    specialist_weight=float(args.specialist_weight),
                    blend_scope=args.blend_scope,
                    rows=label_trainer.config.rows,
                    cols=label_trainer.config.cols,
                )
                scores = torch.where(gate, blended_scores, base_scores)
                action_indices = scores.argmax(dim=1).detach().cpu().numpy()
                tail_decisions += int(sum(tail_mask))
                specialist_applied_decisions += int(gate.sum().item())

                for local_index, game_index in enumerate(ready):
                    game = games[game_index]
                    action_index = int(action_indices[local_index])
                    action = hard_loss_refine.decode_action_index(action_index, game.rows, game.cols)
                    model_open_candidates = hard_loss_refine._top_open_candidate_indices(
                        scores[local_index].detach().cpu().numpy(),
                        mask_batch[local_index],
                        rows=game.rows,
                        cols=game.cols,
                        topk=hard_loss_refine._counterfactual_model_topk(args),
                    )
                    state = hard_loss_refine.HardState(
                        hard_loss_refine.clone_game(game),
                        action_index,
                        action,
                        model_open_candidates=model_open_candidates,
                    )
                    if bool(args.collect_guess_states) and action.kind == hard_loss_refine.ActionType.OPEN:
                        safe_left = int(safe_left_batch[local_index])
                        per_game_limit = int(args.collect_guess_max_per_game)
                        if per_game_limit > 0 and live_guess_counts[game_index] >= per_game_limit:
                            live_guess_skipped_per_game_limit += 1
                        elif _safe_left_in_live_guess_window(safe_left, args):
                            if not _collect_guess_region_allowed(action, game, args):
                                live_guess_region_skips += 1
                            elif not _collect_guess_policy_margin_allowed(
                                scores[local_index].detach().cpu().numpy(),
                                mask_batch[local_index],
                                game,
                                args,
                            ):
                                live_guess_policy_margin_skips += 1
                            elif not _collect_guess_open_candidates_allowed(mask_batch[local_index], game, args):
                                live_guess_open_candidate_skips += 1
                            elif not game.mines_placed:
                                live_guess_label_attempts += 1
                                live_guess_unplaced_skips += 1
                            elif int((game.flagged & ~game.mines).sum()) > 0:
                                live_guess_label_attempts += 1
                                live_guess_wrong_flag_skips += 1
                            elif _live_guess_prefilter_has_forced(solver, game, args):
                                live_guess_label_attempts += 1
                                live_guess_forced_skips += 1
                            else:
                                live_guess_label_attempts += 1
                                transition, _tags = hard_loss_refine.label_hard_state(
                                    label_trainer,
                                    solver,
                                    state,
                                    label_args,
                                    terminal=False,
                                )
                                if (
                                    transition is not None
                                    and transition.expert_is_guess
                                    and (transition.extreme_family or "") in GUESS_FAMILIES
                                ):
                                    if not _collect_guess_behavior_allowed(transition, args):
                                        live_guess_behavior_skips += 1
                                    else:
                                        transitions.append(transition)
                                        if _transition_matches_output_filter(transition, args):
                                            selected_like_transitions += 1
                                        live_guess_counts[game_index] += 1
                                        live_guess_transitions += 1
                                        if selected_like_transitions >= int(args.max_records):
                                            reached_record_cap = True
                                            active.clear()
                                            break
                                        maybe_save_partial()
                    tails[game_index].append(state)
                    if len(tails[game_index]) > int(args.tail_states):
                        del tails[game_index][0]
                    game.step(action)
                    steps[game_index] += 1
                    if game.done or steps[game_index] >= label_trainer.config.max_steps:
                        active.discard(game_index)

            if reached_record_cap:
                break

            for game, tail in zip(games, tails):
                if game.won:
                    wins += 1
                    continue
                if not game.lost:
                    continue

                losses += 1
                terminal_wrong_flags = False
                terminal_forced = False
                terminal_endgame = False
                for offset, state in enumerate(tail):
                    transition, tags = hard_loss_refine.label_hard_state(
                        label_trainer,
                        solver,
                        state,
                        label_args,
                        terminal=(offset == len(tail) - 1),
                    )
                    if transition is None:
                        continue
                    transitions.append(transition)
                    if _transition_matches_output_filter(transition, args):
                        selected_like_transitions += 1
                    terminal_wrong_flags = terminal_wrong_flags or bool(tags["wrong_flags"])
                    terminal_forced = terminal_forced or bool(tags["forced_available"] and tags["terminal"])
                    terminal_endgame = terminal_endgame or bool(tags["endgame"] and tags["terminal"])
                    if selected_like_transitions >= int(args.max_records):
                        reached_record_cap = True
                        break
                    maybe_save_partial()
                wrong_flag_losses += int(terminal_wrong_flags)
                forced_terminal_losses += int(terminal_forced)
                endgame_losses += int(terminal_endgame)
                if reached_record_cap:
                    break
            maybe_report_progress()

    final_save_done["value"] = True
    maybe_report_progress(force=True)
    return transitions, current_metrics()


def _partial_output_path(args: argparse.Namespace) -> Path:
    if args.partial_output is not None:
        return Path(args.partial_output)
    output = Path(args.output)
    suffix = output.suffix or ".npz"
    return output.with_name(f"{output.stem}.partial{suffix}")


def _selected_output_transitions(transitions, args: argparse.Namespace):
    selected = []
    for transition in transitions:
        if _transition_matches_output_filter(transition, args):
            selected.append(transition)
        if len(selected) >= int(args.max_records):
            break
    return selected


def _partial_metadata(
    args: argparse.Namespace,
    *,
    metrics: dict[str, object],
    source_families: Counter[str],
    selected_counts: Counter[str],
    reason: str,
) -> dict[str, object]:
    return {
        "source": "gated_policy_partial_sampler",
        "partial": True,
        "save_reason": reason,
        "base_checkpoints": [
            str(path)
            for path in [args.base_checkpoint, *args.base_ensemble_checkpoints]
        ],
        "specialist_checkpoints": [
            str(path)
            for path in [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
        ],
        "games_requested": int(args.games),
        "seed": int(args.seed),
        "max_records": int(args.max_records),
        "save_every_records": int(args.save_every_records),
        "sample_family": args.sample_family,
        "family_filter": list(args.family_filter or []),
        "collect_guess_states": bool(args.collect_guess_states),
        "collect_guess_safe_left_min": int(args.collect_guess_safe_left_min),
        "collect_guess_safe_left_max": None
        if args.collect_guess_safe_left_max is None
        else int(args.collect_guess_safe_left_max),
        "collect_guess_open_candidates_min": int(args.collect_guess_open_candidates_min),
        "collect_guess_open_candidates_max": None
        if args.collect_guess_open_candidates_max is None
        else int(args.collect_guess_open_candidates_max),
        "collect_guess_min_behavior_regret": float(args.collect_guess_min_behavior_regret),
        "collect_guess_behavior_mine_only": bool(args.collect_guess_behavior_mine_only),
        "collect_guess_region": args.collect_guess_region,
        "specialist_region_gate": getattr(args, "specialist_region_gate", "any"),
        "gate_calibrator": str(getattr(args, "gate_calibrator", None)) if getattr(args, "gate_calibrator", None) else None,
        "gate_threshold": None
        if getattr(args, "gate_calibrator", None) is None
        else getattr(args, "gate_threshold", None),
        "counterfactual_labels": bool(args.counterfactual_labels),
        "counterfactual_all_open": bool(args.counterfactual_all_open),
        "counterfactual_topk": int(args.counterfactual_topk),
        "mining_metrics": metrics,
        "source_families": dict(sorted(source_families.items())),
        "selected_families": dict(sorted(selected_counts.items())),
        "selected_records": int(sum(selected_counts.values())),
    }


def _transition_matches_output_filter(transition, args: argparse.Namespace) -> bool:
    family = transition.extreme_family or "unclassified"
    if args.sample_family == "guess" and (not transition.expert_is_guess or family not in GUESS_FAMILIES):
        return False
    family_filter = set(args.family_filter or [])
    if family_filter and family not in family_filter:
        return False
    return True


def _collect_guess_region_allowed(action, game, args: argparse.Namespace) -> bool:
    region = str(getattr(args, "collect_guess_region", "any"))
    if region == "any":
        return True
    corner = bool(action.row in {0, game.rows - 1} and action.col in {0, game.cols - 1})
    edge = bool(action.row in {0, game.rows - 1} or action.col in {0, game.cols - 1})
    if region == "current-edge-or-corner":
        return edge
    if region == "current-edge":
        return edge and not corner
    if region == "current-corner":
        return corner
    if region == "current-interior":
        return not edge
    raise ValueError(f"unknown collect_guess_region {region!r}")


def _collect_guess_policy_margin_allowed(
    scores: np.ndarray,
    action_mask: np.ndarray,
    game,
    args: argparse.Namespace,
) -> bool:
    max_margin = getattr(args, "collect_guess_policy_margin_max", None)
    if max_margin is None:
        return True
    margin = _open_policy_top2_margin(scores, action_mask, rows=game.rows, cols=game.cols)
    return margin is not None and margin <= float(max_margin)


def _collect_guess_open_candidates_allowed(
    action_mask: np.ndarray,
    game,
    args: argparse.Namespace,
) -> bool:
    count = _legal_open_candidate_count(action_mask, rows=game.rows, cols=game.cols)
    min_count = int(getattr(args, "collect_guess_open_candidates_min", 0))
    max_count = getattr(args, "collect_guess_open_candidates_max", None)
    if count < min_count:
        return False
    if max_count is not None and count > int(max_count):
        return False
    return True


def _legal_open_candidate_count(action_mask: np.ndarray, *, rows: int, cols: int) -> int:
    del rows, cols
    open_channel = hard_loss_refine.action_channel(hard_loss_refine.ActionType.OPEN)
    mask = np.asarray(action_mask[open_channel], dtype=bool)
    return int(mask.sum())


def _collect_guess_behavior_allowed(transition, args: argparse.Namespace) -> bool:
    min_regret = float(getattr(args, "collect_guess_min_behavior_regret", 0.0))
    mine_only = bool(getattr(args, "collect_guess_behavior_mine_only", False))
    values = transition.counterfactual_open_values
    if values is None:
        return not mine_only and min_regret <= 0.0

    values_array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values_array)
    if not bool(finite.any()):
        return False

    rows, cols = values_array.shape
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != hard_loss_refine.action_channel(hard_loss_refine.ActionType.OPEN):
        return False
    row, col = divmod(cell_index, cols)
    behavior_value = float(values_array[row, col])
    if not np.isfinite(behavior_value):
        return False

    mine = False
    if transition.mine_mask is not None:
        mine_mask = np.asarray(transition.mine_mask, dtype=bool)
        if mine_mask.shape == values_array.shape:
            mine = bool(mine_mask[row, col])

    regret = float(np.max(values_array[finite]) - behavior_value)
    if mine_only and not mine:
        return False
    if min_regret > 0.0 and regret < min_regret:
        return False
    return True


def _open_policy_top2_margin(
    scores: np.ndarray,
    action_mask: np.ndarray,
    *,
    rows: int,
    cols: int,
) -> float | None:
    cells = int(rows) * int(cols)
    open_channel = hard_loss_refine.action_channel(hard_loss_refine.ActionType.OPEN)
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    open_start = open_channel * cells
    open_stop = open_start + cells
    if flat_scores.size < open_stop:
        return None
    legal = np.asarray(action_mask[open_channel], dtype=bool).reshape(-1)
    open_scores = flat_scores[open_start:open_stop]
    valid_scores = open_scores[legal & np.isfinite(open_scores)]
    if valid_scores.size < 2:
        return None
    top2 = np.partition(valid_scores, -2)[-2:]
    return float(top2.max() - top2.min())


def _has_forced_moves_fast(solver: MinesweeperSolver, game) -> bool:
    if not game.mines_placed:
        return False

    constraints = solver._build_constraints(game, game.flagged)
    safe, mines = solver._basic_deductions(constraints)
    new_safe = {idx for idx in safe if solver._is_hidden_unflagged(game, idx, game.flagged)}
    new_mines = {idx for idx in mines if solver._is_hidden_unflagged(game, idx, game.flagged)}
    if new_safe or new_mines:
        return True

    exact_safe, exact_mines, _ = solver._exact_component_probabilities(game, constraints, game.flagged)
    exact_safe = {idx for idx in exact_safe if solver._is_hidden_unflagged(game, idx, game.flagged)}
    exact_mines = {idx for idx in exact_mines if solver._is_hidden_unflagged(game, idx, game.flagged)}
    return bool(exact_safe or exact_mines)


def _live_guess_prefilter_has_forced(solver: MinesweeperSolver, game, args: argparse.Namespace) -> bool:
    mode = str(getattr(args, "collect_guess_prefilter", "exact"))
    if mode == "none":
        return False
    if mode == "basic":
        return _has_basic_forced_moves_fast(solver, game)
    return _has_forced_moves_fast(solver, game)


def _has_basic_forced_moves_fast(solver: MinesweeperSolver, game) -> bool:
    if not game.mines_placed:
        return False
    constraints = solver._build_constraints(game, game.flagged)
    safe, mines = solver._basic_deductions(constraints)
    if any(solver._is_hidden_unflagged(game, idx, game.flagged) for idx in safe):
        return True
    return any(solver._is_hidden_unflagged(game, idx, game.flagged) for idx in mines)


def _safe_left_in_live_guess_window(safe_left: int, args: argparse.Namespace) -> bool:
    min_safe_left = int(getattr(args, "collect_guess_safe_left_min", 0))
    max_safe_left = getattr(args, "collect_guess_safe_left_max", None)
    if int(safe_left) < min_safe_left:
        return False
    if max_safe_left is not None and int(safe_left) > int(max_safe_left):
        return False
    return True


if __name__ == "__main__":
    main()
