from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import save_extreme_dataset
from minesweeper_rl.features import action_channel, action_to_index, decode_action_index, encode_state
from minesweeper_rl.trainer import load_checkpoint, make_game, transition_extreme_profile
from minesweeper_rl.types import Action, ActionType, EpisodeTransition


DEFAULT_OUTPUT = Path("artifacts/report_assets/extreme_training_dataset/sim_extreme_transitions.npz")
HARD_REFINE_PATH = ROOT / "scripts" / "hard_loss_refine.py"
SPEC = importlib.util.spec_from_file_location("hard_loss_refine_for_sim_dataset", HARD_REFINE_PATH)
if SPEC is None or SPEC.loader is None:
    raise RuntimeError(f"could not load {HARD_REFINE_PATH}")
hard_loss_refine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hard_loss_refine
SPEC.loader.exec_module(hard_loss_refine)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate solver-labelled hard states from the internal Minesweeper environment."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--games", type=int, default=2048)
    parser.add_argument("--seed", type=int, default=100000)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-records", type=int, default=32768)
    parser.add_argument("--per-family-cap", type=int, default=8192)
    parser.add_argument("--safe-left-threshold", type=int, default=100)
    parser.add_argument("--guess-topk", type=int, default=8)
    parser.add_argument("--counterfactual-labels", action="store_true")
    parser.add_argument("--counterfactual-topk", type=int, default=64)
    parser.add_argument(
        "--counterfactual-all-open",
        action="store_true",
        help="Label every legal OPEN candidate in each guess state.",
    )
    args = parser.parse_args()

    trainer = load_checkpoint(args.checkpoint, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = "full"
    trainer.config.guess_supervision_topk = max(1, int(args.guess_topk))

    transitions: list[EpisodeTransition] = []
    candidate_counts: Counter[str] = Counter()
    selected_counts: Counter[str] = Counter()
    rejected_unclassified = 0
    wins = 0
    losses = 0
    started_at = time.time()

    for game_offset in range(max(0, int(args.games))):
        game = make_game(trainer.config, seed=args.seed + game_offset)
        game.open_cell(game.rows // 2, game.cols // 2)

        for _ in range(max(0, int(args.max_steps) - 1)):
            if game.done:
                break

            board, global_features, action_mask = encode_state(game)
            action_mask = trainer._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            snapshot = trainer.solver.analyze(game)
            expert_action_mask = trainer._solver_expert_action_mask(game, action_mask, snapshot)
            if expert_action_mask is None or not expert_action_mask.any():
                break

            if snapshot.has_forced_moves:
                action_index = _select_forced_action(game, snapshot, action_mask)
            else:
                open_mask = action_mask[action_channel(ActionType.OPEN)]
                if not open_mask.any():
                    break
                action_index = trainer._select_solver_guess(snapshot, open_mask)
                action_index = action_channel(ActionType.OPEN) * game.rows * game.cols + action_index

            action = decode_action_index(action_index, game.rows, game.cols)
            mine_mask = game.mines.copy()
            risk_map = snapshot.risk_map.copy()
            expert_is_guess = bool(not snapshot.has_forced_moves and action.kind == ActionType.OPEN)
            counterfactual_open_values = None
            if args.counterfactual_labels and expert_is_guess:
                counterfactual_open_values = hard_loss_refine.build_counterfactual_open_values(
                    game,
                    action_mask,
                    snapshot,
                    trainer.solver,
                    topk=max(1, int(args.counterfactual_topk)),
                    include_all_open=bool(args.counterfactual_all_open),
                )
            _, reward, done, _ = game.step(action)
            transition = EpisodeTransition(
                board=board,
                global_features=global_features,
                action_mask=action_mask,
                action_index=action_index,
                expert_action_index=action_index,
                reward=float(reward),
                done=bool(done),
                expert_action_mask=expert_action_mask,
                mine_mask=mine_mask,
                risk_map=risk_map,
                counterfactual_open_values=counterfactual_open_values,
                expert_is_guess=expert_is_guess,
                source_quality=1.3,
            )
            profile = transition_extreme_profile(
                transition,
                mines=trainer.config.mines,
                safe_left_threshold=args.safe_left_threshold,
            )
            if not bool(profile["extreme"]):
                continue

            family = str(profile["family"])
            if family == "ordinary":
                rejected_unclassified += 1
                continue
            transition.extreme_score = float(profile["score"])
            transition.extreme_family = family
            candidate_counts[family] += 1

            if selected_counts[family] >= args.per_family_cap:
                continue
            if len(transitions) >= args.max_records:
                break
            transitions.append(transition)
            selected_counts[family] += 1

        if len(transitions) >= args.max_records:
            break
        if game.won:
            wins += 1
        elif game.lost:
            losses += 1

    if not transitions:
        raise RuntimeError("no extreme simulation transitions were generated")

    manifest = save_extreme_dataset(
        args.output,
        transitions,
        metadata={
            "source": "internal_simulation_solver_trajectory",
            "checkpoint": str(args.checkpoint),
            "games_requested": int(args.games),
            "games_completed": wins + losses,
            "wins": wins,
            "losses": losses,
            "seed": int(args.seed),
            "max_steps": int(args.max_steps),
            "safe_left_threshold": int(args.safe_left_threshold),
            "guess_topk": int(args.guess_topk),
            "counterfactual_labels": bool(args.counterfactual_labels),
            "counterfactual_topk": int(args.counterfactual_topk),
            "counterfactual_all_open": bool(args.counterfactual_all_open),
            "per_family_cap": int(args.per_family_cap),
            "candidate_families": dict(sorted(candidate_counts.items())),
            "selected_families": dict(sorted(selected_counts.items())),
            "rejected_unclassified": rejected_unclassified,
            "elapsed_seconds": time.time() - started_at,
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "records": len(transitions),
                "candidate_families": dict(sorted(candidate_counts.items())),
                "selected_families": dict(sorted(selected_counts.items())),
                "rejected_unclassified": rejected_unclassified,
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


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
