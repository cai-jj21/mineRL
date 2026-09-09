from __future__ import annotations

import argparse
import atexit
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import save_extreme_dataset
from minesweeper_rl.features import action_channel, action_to_index, decode_action_index, encode_state
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import load_checkpoint, make_game, transition_extreme_profile
from minesweeper_rl.types import Action, ActionType, EpisodeTransition


DEFAULT_OUTPUT = Path("artifacts/report_assets/extreme_training_dataset/sim_guess_transitions.npz")
HARD_REFINE_PATH = ROOT / "scripts" / "hard_loss_refine.py"
SPEC = importlib.util.spec_from_file_location("hard_loss_refine_for_sim_guess_dataset", HARD_REFINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {HARD_REFINE_PATH}")
hard_loss_refine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hard_loss_refine
SPEC.loader.exec_module(hard_loss_refine)

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
        description="Generate solver-labelled guess states for a pure-RL guess specialist."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--games", type=int, default=128)
    parser.add_argument("--seed", type=int, default=120000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--exact-limit", type=int, default=16)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument("--per-family-cap", type=int, default=2048)
    parser.add_argument("--max-guesses-per-game", type=int, default=0)
    parser.add_argument(
        "--trajectory-mode",
        choices=["solver", "policy"],
        default="solver",
        help="Collect guess states from the solver trajectory or the checkpoint's own policy trajectory.",
    )
    parser.add_argument(
        "--min-behavior-regret",
        type=float,
        default=0.0,
        help="Keep only labelled states whose selected OPEN action has at least this counterfactual regret.",
    )
    parser.add_argument(
        "--behavior-mine-only",
        action="store_true",
        help="Keep only states where the selected OPEN action is an actual mine in the offline layout.",
    )
    parser.add_argument(
        "--behavior-selection",
        choices=["and", "mine-only", "regret-only", "mine-or-regret"],
        default="and",
        help="How to combine mine and regret gates when selecting labelled guess states.",
    )
    parser.add_argument(
        "--save-every-records",
        type=int,
        default=0,
        help="Write a partial dataset after every N selected records. Use 0 to disable.",
    )
    parser.add_argument(
        "--partial-output",
        type=Path,
        default=None,
        help="Optional path for periodic partial saves. Defaults to <output>.partial.npz.",
    )
    parser.add_argument("--safe-left-threshold", type=int, default=140)
    parser.add_argument("--collect-safe-left-min", type=int, default=0)
    parser.add_argument("--collect-safe-left-max", type=int, default=None)
    parser.add_argument("--family-filter", action="append", choices=sorted(GUESS_FAMILIES), default=[])
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
        help="Number of policy top OPEN candidates to force into offline counterfactual labels.",
    )
    parser.add_argument(
        "--policy-candidate-checkpoint",
        action="append",
        type=Path,
        default=[],
        help="Optional policy checkpoint(s) whose top OPEN candidates should be included in labels.",
    )
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    args = parser.parse_args()

    trainer = load_checkpoint(args.checkpoint, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = "full"
    trainer.config.exact_limit = max(1, int(args.exact_limit))
    trainer.solver = MinesweeperSolver(exact_limit=trainer.config.exact_limit)
    trainer.config.guess_supervision_topk = max(1, int(args.guess_topk))
    trainer.config.inference_augment_flips = bool(args.inference_flips)
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = float(args.risk_head_weight)
    trainer.model.eval()
    if str(args.behavior_selection) in {"regret-only", "mine-or-regret"} and float(args.min_behavior_regret) <= 0.0:
        raise ValueError("--behavior-selection regret-only/mine-or-regret requires --min-behavior-regret > 0")
    if float(args.min_behavior_regret) > 0.0 and not args.counterfactual_labels:
        raise ValueError("--min-behavior-regret requires --counterfactual-labels")
    if args.behavior_mine_only and not args.counterfactual_labels:
        raise ValueError("--behavior-mine-only requires --counterfactual-labels")
    policy_candidate_trainers = _load_policy_candidate_trainers(args)

    transitions: list[EpisodeTransition] = []
    candidate_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    rejected_non_guess = 0
    skipped_safe_left = 0
    skipped_family = 0
    skipped_behavior = 0
    stopped_after_game_cap = 0
    scanned_guess_states = 0
    wins = 0
    losses = 0
    started_at = time.time()
    family_filter = set(args.family_filter or [])
    last_partial_save_records = 0
    final_save_done = {"value": False}

    def save_partial_on_exit() -> None:
        if final_save_done["value"] or not transitions:
            return
        _save_partial_dataset(
            args,
            transitions,
            family_filter=family_filter,
            candidate_counts=candidate_counts,
            selected_counts=selected_counts,
            rejected_non_guess=rejected_non_guess,
            skipped_safe_left=skipped_safe_left,
            skipped_family=skipped_family,
            skipped_behavior=skipped_behavior,
            stopped_after_game_cap=stopped_after_game_cap,
            scanned_guess_states=scanned_guess_states,
            wins=wins,
            losses=losses,
            started_at=started_at,
            reason="exit",
        )

    atexit.register(save_partial_on_exit)

    for game_offset in range(max(0, int(args.games))):
        game = make_game(trainer.config, seed=args.seed + game_offset)
        game.open_cell(game.rows // 2, game.cols // 2)
        game_guess_records = 0

        for _ in range(max(0, int(args.max_steps) - 1)):
            if game.done:
                break
            if int(args.max_guesses_per_game) > 0 and game_guess_records >= int(args.max_guesses_per_game):
                stopped_after_game_cap += 1
                break

            board, global_features, action_mask = encode_state(game)
            action_mask = trainer._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            snapshot = trainer.solver.analyze(game)
            expert_action_mask = trainer._solver_expert_action_mask(game, action_mask, snapshot)
            if expert_action_mask is None or not expert_action_mask.any():
                break

            if args.trajectory_mode == "solver":
                if snapshot.has_forced_moves:
                    action_index = _select_forced_action(game, snapshot, action_mask)
                else:
                    open_mask = action_mask[action_channel(ActionType.OPEN)]
                    if not open_mask.any():
                        break
                    action_index = trainer._select_solver_guess(snapshot, open_mask)
                    action_index = action_channel(ActionType.OPEN) * game.rows * game.cols + action_index
            else:
                with torch.inference_mode():
                    scores = trainer._predict_policy_scores_batch(
                        boards=board[None, ...],
                        global_features_batch=global_features[None, ...],
                        action_masks=action_mask[None, ...],
                        use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                    )
                action_index = int(scores.argmax(dim=1).item())

            action = decode_action_index(action_index, game.rows, game.cols)
            expert_is_guess = bool(not snapshot.has_forced_moves and action.kind == ActionType.OPEN)
            transition = None
            if expert_is_guess:
                scanned_guess_states += 1
                mine_mask = game.mines.copy()
                risk_map = snapshot.risk_map.copy()
                counterfactual_open_values = None
                if args.counterfactual_labels:
                    candidate_indices = _counterfactual_candidate_indices(
                        action_index=action_index,
                        game=game,
                        board=board,
                        global_features=global_features,
                        action_mask=action_mask,
                        policy_candidate_trainers=policy_candidate_trainers,
                        args=args,
                    )
                    counterfactual_open_values = hard_loss_refine.build_counterfactual_open_values(
                        game,
                        action_mask,
                        snapshot,
                        trainer.solver,
                        topk=max(1, int(args.counterfactual_topk)),
                        candidate_indices=candidate_indices,
                        include_all_open=bool(args.counterfactual_all_open),
                    )
                transition = EpisodeTransition(
                    board=board,
                    global_features=global_features,
                    action_mask=action_mask,
                    action_index=action_index,
                    expert_action_index=action_index,
                    reward=0.0,
                    done=False,
                    expert_action_mask=expert_action_mask,
                    mine_mask=mine_mask,
                    risk_map=risk_map,
                    counterfactual_open_values=counterfactual_open_values,
                    expert_is_guess=True,
                    source_quality=1.35,
                )
                profile = transition_extreme_profile(
                    transition,
                    mines=trainer.config.mines,
                    safe_left_threshold=args.safe_left_threshold,
                )
                safe_left = int(profile["safe_left"])
                if not bool(profile["guess"]):
                    rejected_non_guess += 1
                    transition = None
                elif str(profile["family"]) not in GUESS_FAMILIES:
                    rejected_non_guess += 1
                    transition = None
                elif not _safe_left_allowed(safe_left, args):
                    candidate_counts[str(profile["family"])] += 1
                    skipped_safe_left += 1
                    transition = None
                else:
                    family = str(profile["family"])
                    candidate_counts[family] += 1
                    if family_filter and family not in family_filter:
                        skipped_family += 1
                        transition = None
                    elif not _behavior_selection_allowed(
                        transition,
                        min_regret=float(args.min_behavior_regret),
                        mine_only=bool(args.behavior_mine_only),
                        behavior_selection=str(args.behavior_selection),
                    ):
                        skipped_behavior += 1
                        transition = None
                    else:
                        transition.extreme_score = float(profile["score"])
                        transition.extreme_family = family

            _, _, _, _ = game.step(action)
            if transition is None:
                continue

            if selected_counts[family] >= args.per_family_cap:
                continue
            if len(transitions) >= args.max_records:
                break
            transitions.append(transition)
            selected_counts[family] += 1
            game_guess_records += 1
            last_partial_save_records = _maybe_save_partial_dataset(
                args,
                transitions,
                family_filter=family_filter,
                candidate_counts=candidate_counts,
                selected_counts=selected_counts,
                rejected_non_guess=rejected_non_guess,
                skipped_safe_left=skipped_safe_left,
                skipped_family=skipped_family,
                skipped_behavior=skipped_behavior,
                stopped_after_game_cap=stopped_after_game_cap,
                scanned_guess_states=scanned_guess_states,
                wins=wins,
                losses=losses,
                started_at=started_at,
                last_saved_records=last_partial_save_records,
            )

        if len(transitions) >= args.max_records:
            break
        if game.won:
            wins += 1
        elif game.lost:
            losses += 1

    if not transitions:
        raise RuntimeError("no guess simulation transitions were generated")

    manifest = save_extreme_dataset(
        args.output,
        transitions,
        metadata=_dataset_metadata(
            args,
            family_filter=family_filter,
            candidate_counts=candidate_counts,
            selected_counts=selected_counts,
            rejected_non_guess=rejected_non_guess,
            skipped_safe_left=skipped_safe_left,
            skipped_family=skipped_family,
            skipped_behavior=skipped_behavior,
            stopped_after_game_cap=stopped_after_game_cap,
            scanned_guess_states=scanned_guess_states,
            wins=wins,
            losses=losses,
            started_at=started_at,
            partial=False,
            save_reason="final",
        ),
    )
    final_save_done["value"] = True
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "records": len(transitions),
                "candidate_families": dict(sorted(candidate_counts.items())),
                "selected_families": dict(sorted(selected_counts.items())),
                "scanned_guess_states": scanned_guess_states,
                "rejected_non_guess": rejected_non_guess,
                "skipped_safe_left": skipped_safe_left,
                "skipped_family": skipped_family,
                "skipped_behavior": skipped_behavior,
                "stopped_after_game_cap": stopped_after_game_cap,
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def _safe_left_allowed(safe_left: int, args: argparse.Namespace) -> bool:
    if int(safe_left) < int(args.collect_safe_left_min):
        return False
    if args.collect_safe_left_max is not None and int(safe_left) > int(args.collect_safe_left_max):
        return False
    return True


def _behavior_selection_allowed(
    transition: EpisodeTransition,
    *,
    min_regret: float,
    mine_only: bool,
    behavior_selection: str = "and",
) -> bool:
    values = transition.counterfactual_open_values
    if values is None:
        if behavior_selection == "mine-only":
            if transition.mine_mask is None:
                return False
            mine_mask = np.asarray(transition.mine_mask, dtype=bool)
            rows, cols = mine_mask.shape
            cells = rows * cols
            kind_index, cell_index = divmod(int(transition.action_index), cells)
            if kind_index != action_channel(ActionType.OPEN):
                return False
            row, col = divmod(cell_index, cols)
            return bool(mine_mask[row, col])
        if behavior_selection in {"regret-only", "mine-or-regret"}:
            return False
        return not mine_only and float(min_regret) <= 0.0

    values_array = np.asarray(values, dtype=np.float32)
    finite = np.isfinite(values_array)
    if not bool(finite.any()):
        return False

    rows, cols = values_array.shape
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != action_channel(ActionType.OPEN):
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
    if behavior_selection == "mine-only":
        return mine
    if behavior_selection == "regret-only":
        return regret >= float(min_regret)
    if behavior_selection == "mine-or-regret":
        return mine or regret >= float(min_regret)
    if mine_only and not mine:
        return False
    if float(min_regret) > 0.0 and regret < float(min_regret):
        return False
    return True


def _maybe_save_partial_dataset(
    args: argparse.Namespace,
    transitions: list[EpisodeTransition],
    *,
    family_filter: set[str],
    candidate_counts: Counter[str],
    selected_counts: Counter[str],
    rejected_non_guess: int,
    skipped_safe_left: int,
    skipped_family: int,
    skipped_behavior: int,
    stopped_after_game_cap: int,
    scanned_guess_states: int,
    wins: int,
    losses: int,
    started_at: float,
    last_saved_records: int,
) -> int:
    save_every = max(0, int(args.save_every_records))
    if save_every <= 0 or not transitions:
        return int(last_saved_records)
    if len(transitions) // save_every <= int(last_saved_records) // save_every:
        return int(last_saved_records)

    _save_partial_dataset(
        args,
        transitions,
        family_filter=family_filter,
        candidate_counts=candidate_counts,
        selected_counts=selected_counts,
        rejected_non_guess=rejected_non_guess,
        skipped_safe_left=skipped_safe_left,
        skipped_family=skipped_family,
        skipped_behavior=skipped_behavior,
        stopped_after_game_cap=stopped_after_game_cap,
        scanned_guess_states=scanned_guess_states,
        wins=wins,
        losses=losses,
        started_at=started_at,
        reason="periodic",
    )
    return len(transitions)


def _save_partial_dataset(
    args: argparse.Namespace,
    transitions: list[EpisodeTransition],
    *,
    family_filter: set[str],
    candidate_counts: Counter[str],
    selected_counts: Counter[str],
    rejected_non_guess: int,
    skipped_safe_left: int,
    skipped_family: int,
    skipped_behavior: int,
    stopped_after_game_cap: int,
    scanned_guess_states: int,
    wins: int,
    losses: int,
    started_at: float,
    reason: str,
) -> None:
    if not transitions:
        return
    path = _partial_output_path(args)
    manifest = save_extreme_dataset(
        path,
        transitions,
        metadata=_dataset_metadata(
            args,
            family_filter=family_filter,
            candidate_counts=candidate_counts,
            selected_counts=selected_counts,
            rejected_non_guess=rejected_non_guess,
            skipped_safe_left=skipped_safe_left,
            skipped_family=skipped_family,
            skipped_behavior=skipped_behavior,
            stopped_after_game_cap=stopped_after_game_cap,
            scanned_guess_states=scanned_guess_states,
            wins=wins,
            losses=losses,
            started_at=started_at,
            partial=True,
            save_reason=reason,
        ),
    )
    print(
        json.dumps(
            {
                "status": "partial_saved",
                "reason": reason,
                "output": str(path),
                "records": len(transitions),
                "manifest": manifest.get("manifest"),
            },
            ensure_ascii=False,
        ),
        flush=True,
    )


def _partial_output_path(args: argparse.Namespace) -> Path:
    if args.partial_output is not None:
        return Path(args.partial_output)
    output = Path(args.output)
    suffix = output.suffix or ".npz"
    return output.with_name(f"{output.stem}.partial{suffix}")


def _dataset_metadata(
    args: argparse.Namespace,
    *,
    family_filter: set[str],
    candidate_counts: Counter[str],
    selected_counts: Counter[str],
    rejected_non_guess: int,
    skipped_safe_left: int,
    skipped_family: int,
    skipped_behavior: int,
    stopped_after_game_cap: int,
    scanned_guess_states: int,
    wins: int,
    losses: int,
    started_at: float,
    partial: bool,
    save_reason: str,
) -> dict[str, Any]:
    selected_total = int(sum(selected_counts.values()))
    scanned_guess_states = int(scanned_guess_states)
    return {
        "source": "internal_simulation_policy_or_solver_trajectory",
        "trajectory_mode": str(getattr(args, "trajectory_mode", "solver")),
        "checkpoint": str(args.checkpoint),
        "games_requested": int(args.games),
        "games_completed": wins + losses,
        "wins": int(wins),
        "losses": int(losses),
        "seed": int(args.seed),
        "max_steps": int(args.max_steps),
        "exact_limit": int(args.exact_limit),
        "max_guesses_per_game": int(args.max_guesses_per_game),
        "save_every_records": int(args.save_every_records),
        "partial_output": str(_partial_output_path(args)),
        "partial": bool(partial),
        "save_reason": str(save_reason),
        "safe_left_threshold": int(args.safe_left_threshold),
        "collect_safe_left_min": int(args.collect_safe_left_min),
        "collect_safe_left_max": None
        if args.collect_safe_left_max is None
        else int(args.collect_safe_left_max),
        "family_filter": sorted(family_filter),
        "guess_topk": int(args.guess_topk),
        "counterfactual_labels": bool(args.counterfactual_labels),
        "counterfactual_topk": int(args.counterfactual_topk),
        "counterfactual_all_open": bool(args.counterfactual_all_open),
        "counterfactual_model_topk": _counterfactual_model_topk(args),
        "policy_candidate_checkpoints": [str(path) for path in args.policy_candidate_checkpoint],
        "risk_head_weight": float(args.risk_head_weight),
        "inference_flips": bool(args.inference_flips),
        "inference_ensemble": args.inference_ensemble,
        "behavior_selection": str(getattr(args, "behavior_selection", "and")),
        "per_family_cap": int(args.per_family_cap),
        "candidate_families": dict(sorted(candidate_counts.items())),
        "selected_families": dict(sorted(selected_counts.items())),
        "selected_records": selected_total,
        "scanned_guess_states": scanned_guess_states,
        "selected_to_scanned_guess_rate": (
            None if scanned_guess_states <= 0 else float(selected_total / scanned_guess_states)
        ),
        "rejected_non_guess": int(rejected_non_guess),
        "skipped_safe_left": int(skipped_safe_left),
        "skipped_family": int(skipped_family),
        "skipped_behavior": int(skipped_behavior),
        "min_behavior_regret": float(getattr(args, "min_behavior_regret", 0.0)),
        "behavior_mine_only": bool(getattr(args, "behavior_mine_only", False)),
        "stopped_after_game_cap": int(stopped_after_game_cap),
        "elapsed_seconds": time.time() - started_at,
    }


def _load_policy_candidate_trainers(args: argparse.Namespace) -> list[Any]:
    trainers = []
    for path in args.policy_candidate_checkpoint:
        trainer = load_checkpoint(path, device=args.device)
        trainer.config.rows = 16
        trainer.config.cols = 30
        trainer.config.mines = 99
        trainer.config.safe_radius = 1
        trainer.config.max_steps = args.max_steps
        trainer.config.decision_actions = "full"
        trainer.config.inference_augment_flips = bool(args.inference_flips)
        trainer.config.inference_ensemble = args.inference_ensemble
        trainer.model.eval()
        trainers.append(trainer)
    return trainers


def _counterfactual_candidate_indices(
    *,
    action_index: int,
    game: Any,
    board: np.ndarray,
    global_features: np.ndarray,
    action_mask: np.ndarray,
    policy_candidate_trainers: list[Any],
    args: argparse.Namespace,
) -> np.ndarray:
    candidate_parts = [
        hard_loss_refine._open_candidate_indices(
            action_index,
            rows=game.rows,
            cols=game.cols,
        )
    ]
    model_topk = _counterfactual_model_topk(args)
    if model_topk > 0:
        for trainer in policy_candidate_trainers:
            with torch.inference_mode():
                scores = trainer._predict_policy_scores_batch(
                    boards=board[None, ...],
                    global_features_batch=global_features[None, ...],
                    action_masks=action_mask[None, ...],
                    use_flip_ensemble=bool(trainer.config.inference_augment_flips),
                )[0].detach().cpu().numpy()
            candidate_parts.append(
                hard_loss_refine._top_open_candidate_indices(
                    scores,
                    action_mask,
                    rows=game.rows,
                    cols=game.cols,
                    topk=model_topk,
                )
            )
    if not candidate_parts:
        return np.empty(0, dtype=np.int64)
    return np.unique(np.concatenate(candidate_parts)).astype(np.int64, copy=False)


def _counterfactual_model_topk(args: argparse.Namespace) -> int:
    configured = args.counterfactual_model_topk
    if configured is None:
        return 0
    return max(0, int(configured))


def _select_forced_action(game: Any, snapshot: Any, action_mask: np.ndarray) -> int:
    mine_mask = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
    if mine_mask.any():
        row, col = next(zip(*np.where(mine_mask)))
        return action_to_index(Action(ActionType.FLAG, int(row), int(col)), game.rows, game.cols)

    safe_mask = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
    if safe_mask.any():
        row, col = next(zip(*np.where(safe_mask)))
        return action_to_index(Action(ActionType.OPEN, int(row), int(col)), game.rows, game.cols)
    raise RuntimeError("forced solver snapshot has no legal forced action")


if __name__ == "__main__":
    main()
