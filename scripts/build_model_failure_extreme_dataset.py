from __future__ import annotations

import argparse
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
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import load_checkpoint


HARD_REFINE_PATH = ROOT / "scripts" / "hard_loss_refine.py"
SPEC = importlib.util.spec_from_file_location("hard_loss_refine_for_dataset", HARD_REFINE_PATH)
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
        description="Extract solver-labelled extreme states from the model's own failed games."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--mine-ensemble-checkpoint",
        dest="mine_ensemble_checkpoints",
        action="append",
        type=Path,
        default=[],
        help="Additional checkpoints whose policy scores are averaged while mining failures.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=130000)
    parser.add_argument("--mine-batch-size", type=int, default=32)
    parser.add_argument("--tail-states", type=int, default=16)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument("--endgame-safe-left", type=int, default=100)
    parser.add_argument("--exact-limit", type=int, default=32)
    parser.add_argument("--guess-topk", type=int, default=8)
    parser.add_argument("--counterfactual-labels", action="store_true")
    parser.add_argument("--counterfactual-topk", type=int, default=64)
    parser.add_argument(
        "--counterfactual-model-topk",
        type=int,
        default=None,
        help="Number of ensemble top OPEN candidates to force into offline labels.",
    )
    parser.add_argument(
        "--family-filter",
        action="append",
        choices=sorted(GUESS_FAMILIES),
        default=[],
        help="Optionally keep only specific extreme families. Repeat to include multiple families.",
    )
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--no-inference-flips", action="store_true")
    args = parser.parse_args()

    label_trainer = load_checkpoint(args.checkpoint, device=args.device)
    mine_trainers = [label_trainer, *[load_checkpoint(path, device=args.device) for path in args.mine_ensemble_checkpoints]]
    for trainer in mine_trainers:
        _configure_trainer_for_mining(trainer, args)
    label_trainer.solver = MinesweeperSolver(exact_limit=args.exact_limit)

    mining_args = SimpleNamespace(
        mine_games=max(0, int(args.games)),
        mine_batch_size=max(1, int(args.mine_batch_size)),
        tail_states=max(1, int(args.tail_states)),
        max_steps=max(1, int(args.max_steps)),
        endgame_safe_left=max(0, int(args.endgame_safe_left)),
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
    )

    started_at = time.time()
    all_transitions, mining_metrics = mine_hard_transitions_ensemble(
        label_trainer,
        mine_trainers,
        mining_args,
        seed=int(args.seed),
    )

    family_filter = set(args.family_filter or [])
    selected = []
    source_families: Counter[str] = Counter()
    for transition in all_transitions:
        family = transition.extreme_family or "unclassified"
        source_families[family] += 1
        if not transition.expert_is_guess or family not in GUESS_FAMILIES:
            continue
        if family_filter and family not in family_filter:
            continue
        selected.append(transition)
        if len(selected) >= args.max_records:
            break

    if not selected:
        raise RuntimeError(
            "no model-failure guess transitions were found; increase --games or --tail-states"
        )

    selected_counts = Counter(transition.extreme_family for transition in selected)
    manifest = save_extreme_dataset(
        args.output,
        selected,
        metadata={
            "source": "model_failure_tail_solver_labels",
            "checkpoint": str(args.checkpoint),
            "mine_ensemble_checkpoints": [str(path) for path in args.mine_ensemble_checkpoints],
            "games_requested": int(args.games),
            "seed": int(args.seed),
            "tail_states": int(args.tail_states),
            "max_steps": int(args.max_steps),
            "endgame_safe_left": int(args.endgame_safe_left),
            "exact_limit": int(args.exact_limit),
            "guess_topk": int(args.guess_topk),
            "counterfactual_labels": bool(args.counterfactual_labels),
            "counterfactual_topk": int(args.counterfactual_topk),
            "counterfactual_model_topk": hard_loss_refine._counterfactual_model_topk(args),
            "family_filter": sorted(family_filter),
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


def _configure_trainer_for_mining(trainer, args: argparse.Namespace) -> None:
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
    trainer.model.eval()


def mine_hard_transitions_ensemble(
    label_trainer,
    mine_trainers,
    args: SimpleNamespace,
    seed: int,
):
    transitions = []
    wins = 0
    losses = 0
    wrong_flag_losses = 0
    forced_terminal_losses = 0
    endgame_losses = 0
    solver = label_trainer.solver
    started_at = time.time()

    for batch_start in range(0, args.mine_games, args.mine_batch_size):
        current_batch = min(args.mine_batch_size, args.mine_games - batch_start)
        games = [hard_loss_refine.make_game(label_trainer.config, seed=seed + batch_start + offset) for offset in range(current_batch)]
        tails = [[] for _ in range(current_batch)]
        steps = [0 for _ in range(current_batch)]
        active = set(range(current_batch))

        while active:
            ready = []
            boards = []
            global_features_batch = []
            action_masks = []

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
                ready.append(index)
                boards.append(board)
                global_features_batch.append(global_features)
                action_masks.append(action_mask)

            if not ready:
                continue

            scores = _average_policy_scores(
                mine_trainers,
                np.stack(boards),
                np.stack(global_features_batch),
                np.stack(action_masks),
            )
            action_indices = scores.argmax(dim=1).detach().cpu().numpy()

            for local_index, game_index in enumerate(ready):
                game = games[game_index]
                action_index = int(action_indices[local_index])
                action = hard_loss_refine.decode_action_index(action_index, game.rows, game.cols)
                model_open_candidates = hard_loss_refine._top_open_candidate_indices(
                    scores[local_index].detach().cpu().numpy(),
                    action_masks[local_index],
                    rows=game.rows,
                    cols=game.cols,
                    topk=hard_loss_refine._counterfactual_model_topk(args),
                )
                tails[game_index].append(
                    hard_loss_refine.HardState(
                        hard_loss_refine.clone_game(game),
                        action_index,
                        action,
                        model_open_candidates=model_open_candidates,
                    )
                )
                if len(tails[game_index]) > args.tail_states:
                    del tails[game_index][0]
                game.step(action)
                steps[game_index] += 1
                if game.done or steps[game_index] >= label_trainer.config.max_steps:
                    active.discard(game_index)

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
                    args,
                    terminal=(offset == len(tail) - 1),
                )
                if transition is None:
                    continue
                weight = hard_loss_refine.transition_weight(tags, args)
                transitions.extend([transition] * weight)
                terminal_wrong_flags = terminal_wrong_flags or bool(tags["wrong_flags"])
                terminal_forced = terminal_forced or bool(tags["forced_available"] and tags["terminal"])
                terminal_endgame = terminal_endgame or bool(tags["endgame"] and tags["terminal"])
            wrong_flag_losses += int(terminal_wrong_flags)
            forced_terminal_losses += int(terminal_forced)
            endgame_losses += int(terminal_endgame)

    return transitions, {
        "seed": seed,
        "games": args.mine_games,
        "wins": wins,
        "losses": losses,
        "win_rate": wins / max(1, args.mine_games),
        "transitions": len(transitions),
        "wrong_flag_losses": wrong_flag_losses,
        "forced_terminal_losses": forced_terminal_losses,
        "endgame_losses": endgame_losses,
        "elapsed_seconds": time.time() - started_at,
    }


def _average_policy_scores(
    trainers,
    boards: np.ndarray,
    global_features_batch: np.ndarray,
    action_masks: np.ndarray,
):
    combined = None
    for trainer in trainers:
        part = trainer._predict_policy_scores_batch(
            boards=boards,
            global_features_batch=global_features_batch,
            action_masks=action_masks,
            use_flip_ensemble=bool(trainer.config.inference_augment_flips),
        )
        combined = part if combined is None else combined + part
    assert combined is not None
    return combined / float(len(trainers))


if __name__ == "__main__":
    main()
