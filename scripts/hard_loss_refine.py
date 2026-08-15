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

from minesweeper_rl.features import decode_action_index, encode_state
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint, make_game, save_checkpoint
from minesweeper_rl.types import Action, EpisodeTransition


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a policy on its own losing/endgame hard states.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/full_rlmix_hard_refine.pt"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/hard_loss_refine"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--seed", type=int, default=50000)
    parser.add_argument("--rounds", type=int, default=4)
    parser.add_argument("--mine-games", type=int, default=192)
    parser.add_argument("--mine-batch-size", type=int, default=64)
    parser.add_argument("--tail-states", type=int, default=12)
    parser.add_argument("--replay-size", type=int, default=8192)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--imitation-updates", type=int, default=80)
    parser.add_argument("--rl-updates", type=int, default=16)
    parser.add_argument("--eval-games", type=int, default=200)
    parser.add_argument("--eval-seed", type=int, default=0)
    parser.add_argument("--exact-limit", type=int, default=32)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--lr", type=float, default=8e-5)
    parser.add_argument("--pretrain-imitation-coef", type=float, default=1.0)
    parser.add_argument("--solver-imitation-coef", type=float, default=0.08)
    parser.add_argument("--mine-aux-coef", type=float, default=0.04)
    parser.add_argument("--risk-supervision-coef", type=float, default=0.08)
    parser.add_argument("--risk-head-coef", type=float, default=0.02)
    parser.add_argument("--risk-temperature", type=float, default=0.06)
    parser.add_argument("--guess-supervision-topk", type=int, default=8)
    parser.add_argument("--guess-imitation-weight", type=float, default=1.4)
    parser.add_argument("--endgame-safe-left", type=int, default=60)
    parser.add_argument("--endgame-weight", type=int, default=2)
    parser.add_argument("--wrong-flag-weight", type=int, default=4)
    parser.add_argument("--terminal-weight", type=int, default=3)
    parser.add_argument("--include-plain-tail", action="store_true")
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = load_checkpoint(args.checkpoint, device=args.device)
    configure_trainer(trainer, args)
    trainer.model.train()

    rng = np.random.default_rng(args.seed)
    replay: list[EpisodeTransition] = []
    best_metrics = evaluate(trainer, args)
    best_win_rate = float(best_metrics["win_rate"])
    history: list[dict[str, Any]] = [make_history_item(0, best_metrics, None, None, len(replay))]
    write_json(args.output_dir / "round_000.json", history[-1])
    save_checkpoint(
        args.save_path,
        trainer,
        episode=0,
        metrics={
            **best_metrics,
            "hard_refine_round": 0,
            "source_checkpoint": str(args.checkpoint),
            "hard_replay_size": 0,
        },
    )
    print(json.dumps(history[-1], ensure_ascii=False, separators=(",", ":")))

    for round_index in range(1, args.rounds + 1):
        mine_seed = int(args.seed + round_index * 100000)
        mined, mine_metrics = mine_hard_transitions(trainer, args, seed=mine_seed)
        replay.extend(mined)
        if len(replay) > args.replay_size:
            del replay[: len(replay) - args.replay_size]

        losses = run_refinement_updates(trainer, args, rng, replay, round_seed=mine_seed + 17)
        metrics = evaluate(trainer, args)
        item = make_history_item(round_index, metrics, mine_metrics, losses, len(replay))
        history.append(item)
        write_json(args.output_dir / f"round_{round_index:03d}.json", item)
        print(json.dumps(item, ensure_ascii=False, separators=(",", ":")))

        win_rate = float(metrics["win_rate"])
        if win_rate >= best_win_rate:
            best_win_rate = win_rate
            save_checkpoint(
                args.save_path,
                trainer,
                episode=round_index,
                metrics={
                    **metrics,
                    "hard_refine_round": round_index,
                    "source_checkpoint": str(args.checkpoint),
                    "hard_replay_size": len(replay),
                },
            )

    write_json(
        args.output_dir / "summary.json",
        {
            "config": vars_for_json(args),
            "best_win_rate": best_win_rate,
            "history": history,
            "save_path": str(args.save_path),
        },
    )


def configure_trainer(trainer: MinesweeperTrainer, args: argparse.Namespace) -> None:
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = "full"
    trainer.config.exact_limit = args.exact_limit
    trainer.config.lr = args.lr
    trainer.config.pretrain_imitation_coef = args.pretrain_imitation_coef
    trainer.config.solver_imitation_coef = args.solver_imitation_coef
    trainer.config.mine_aux_coef = args.mine_aux_coef
    trainer.config.risk_supervision_coef = args.risk_supervision_coef
    trainer.config.risk_head_coef = args.risk_head_coef
    trainer.config.risk_temperature = args.risk_temperature
    trainer.config.guess_supervision_topk = args.guess_supervision_topk
    trainer.config.guess_imitation_weight = args.guess_imitation_weight
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = 0.0
    trainer.solver = MinesweeperSolver(exact_limit=args.exact_limit)
    for group in trainer.optimizer.param_groups:
        group["lr"] = args.lr


def mine_hard_transitions(
    trainer: MinesweeperTrainer,
    args: argparse.Namespace,
    seed: int,
) -> tuple[list[EpisodeTransition], dict[str, Any]]:
    transitions: list[EpisodeTransition] = []
    wins = 0
    losses = 0
    wrong_flag_losses = 0
    forced_terminal_losses = 0
    endgame_losses = 0
    solver = trainer.solver
    started_at = time.time()

    for batch_start in range(0, args.mine_games, args.mine_batch_size):
        current_batch = min(args.mine_batch_size, args.mine_games - batch_start)
        games = [make_game(trainer.config, seed=seed + batch_start + offset) for offset in range(current_batch)]
        tails: list[list[HardState]] = [[] for _ in range(current_batch)]
        steps = [0 for _ in range(current_batch)]
        active = set(range(current_batch))

        while active:
            ready: list[int] = []
            boards: list[np.ndarray] = []
            global_features_batch: list[np.ndarray] = []
            action_masks: list[np.ndarray] = []

            for index in list(active):
                game = games[index]
                if game.done or steps[index] >= trainer.config.max_steps:
                    active.remove(index)
                    continue
                board, global_features, action_mask = encode_state(game)
                action_mask = trainer._decision_action_mask(action_mask)
                if not action_mask.any():
                    active.remove(index)
                    continue
                ready.append(index)
                boards.append(board)
                global_features_batch.append(global_features)
                action_masks.append(action_mask)

            if not ready:
                continue

            scores = trainer._predict_policy_scores_batch(
                boards=np.stack(boards),
                global_features_batch=np.stack(global_features_batch),
                action_masks=np.stack(action_masks),
                use_flip_ensemble=trainer.config.inference_augment_flips,
            )
            action_indices = scores.argmax(dim=1).detach().cpu().numpy()

            for local_index, game_index in enumerate(ready):
                game = games[game_index]
                action_index = int(action_indices[local_index])
                action = decode_action_index(action_index, game.rows, game.cols)
                tails[game_index].append(HardState(clone_game(game), action_index, action))
                if len(tails[game_index]) > args.tail_states:
                    del tails[game_index][0]
                game.step(action)
                steps[game_index] += 1
                if game.done or steps[game_index] >= trainer.config.max_steps:
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
                transition, tags = label_hard_state(trainer, solver, state, args, terminal=(offset == len(tail) - 1))
                if transition is None:
                    continue
                weight = transition_weight(tags, args)
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


def label_hard_state(
    trainer: MinesweeperTrainer,
    solver: MinesweeperSolver,
    state: "HardState",
    args: argparse.Namespace,
    terminal: bool,
) -> tuple[EpisodeTransition | None, dict[str, Any]]:
    game = state.game
    board, global_features, action_mask = encode_state(game)
    action_mask = trainer._decision_action_mask(action_mask)
    if not action_mask.any():
        return None, {}

    snapshot = solver.analyze(game) if game.mines_placed else None
    expert_mask = trainer._solver_expert_action_mask(game, action_mask, snapshot)
    if expert_mask is None or not expert_mask.any():
        return None, {}

    wrong_flags = int((game.flagged & ~game.mines).sum()) if game.mines_placed else 0
    revealed_safe = int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum())
    safe_left = int(game.total_safe_cells - revealed_safe)
    forced_available = bool(snapshot is not None and snapshot.has_forced_moves)
    expert_is_guess = bool(snapshot is not None and not forced_available and wrong_flags == 0)
    expert_index = int(np.flatnonzero(expert_mask.reshape(-1))[0])

    transition = EpisodeTransition(
        board=board,
        global_features=global_features,
        action_mask=action_mask,
        action_index=state.action_index,
        expert_action_index=expert_index,
        reward=0.0,
        done=False,
        expert_action_mask=expert_mask,
        mine_mask=game.mines.copy() if game.mines_placed else None,
        risk_map=snapshot.risk_map if snapshot is not None else None,
        expert_is_guess=expert_is_guess,
    )
    tags = {
        "terminal": terminal,
        "wrong_flags": wrong_flags,
        "forced_available": forced_available,
        "endgame": safe_left <= args.endgame_safe_left,
        "safe_left": safe_left,
    }
    is_hard = bool(tags["terminal"] or tags["wrong_flags"] or tags["forced_available"] or tags["endgame"])
    if not args.include_plain_tail and not is_hard:
        return None, tags
    return transition, tags


def transition_weight(tags: dict[str, Any], args: argparse.Namespace) -> int:
    weight = 1
    if tags.get("endgame"):
        weight = max(weight, int(args.endgame_weight))
    if tags.get("terminal"):
        weight = max(weight, int(args.terminal_weight))
    if int(tags.get("wrong_flags", 0)) > 0:
        weight = max(weight, int(args.wrong_flag_weight))
    return max(1, weight)


def run_refinement_updates(
    trainer: MinesweeperTrainer,
    args: argparse.Namespace,
    rng: np.random.Generator,
    replay: list[EpisodeTransition],
    round_seed: int,
) -> dict[str, Any]:
    losses: list[dict[str, float]] = []
    if replay:
        for _ in range(args.imitation_updates):
            batch = sample_transitions(replay, args.batch_size, rng)
            losses.append(trainer.update_imitation(batch))

    rl_losses: list[dict[str, float]] = []
    for update_index in range(args.rl_updates):
        transitions, _ = trainer.collect_episode(
            seed=round_seed + update_index,
            deterministic=False,
            mode="rl",
            learn_from_solver=True,
        )
        if transitions:
            rl_losses.append(trainer.update(transitions))

    return {
        "imitation": average_losses(losses),
        "rl": average_losses(rl_losses),
        "imitation_updates": len(losses),
        "rl_updates": len(rl_losses),
    }


def evaluate(trainer: MinesweeperTrainer, args: argparse.Namespace) -> dict[str, float | int | bool | None]:
    from minesweeper_rl.trainer import evaluate_policy_batched

    trainer.model.eval()
    metrics = evaluate_policy_batched(
        trainer,
        games=args.eval_games,
        seed=args.eval_seed,
        mode="rl",
        risk_weight=0.0,
        batch_size=64,
    )
    trainer.model.train()
    return metrics


def sample_transitions(
    transitions: list[EpisodeTransition],
    batch_size: int,
    rng: np.random.Generator,
) -> list[EpisodeTransition]:
    if len(transitions) <= batch_size:
        return list(transitions)
    indices = rng.choice(len(transitions), size=batch_size, replace=False)
    return [transitions[int(index)] for index in indices]


def average_losses(losses: list[dict[str, float]]) -> dict[str, float]:
    if not losses:
        return {}
    keys = sorted({key for loss in losses for key in loss})
    return {key: float(np.mean([loss.get(key, 0.0) for loss in losses])) for key in keys}


def clone_game(game: MinesweeperGame) -> MinesweeperGame:
    clone = MinesweeperGame(
        rows=game.rows,
        cols=game.cols,
        mines=game.mine_count,
        safe_radius=game.config.safe_radius,
        seed=game.seed,
    )
    clone.mines = game.mines.copy()
    clone.adjacent = game.adjacent.copy()
    clone.revealed = game.revealed.copy()
    clone.flagged = game.flagged.copy()
    clone.mines_placed = bool(game.mines_placed)
    clone.done = bool(game.done)
    clone.won = bool(game.won)
    clone.lost = bool(game.lost)
    clone.step_count = int(game.step_count)
    return clone


class HardState:
    def __init__(self, game: MinesweeperGame, action_index: int, action: Action) -> None:
        self.game = game
        self.action_index = action_index
        self.action = action


def make_history_item(
    round_index: int,
    metrics: dict[str, float | int | bool | None],
    mine_metrics: dict[str, Any] | None,
    losses: dict[str, Any] | None,
    replay_size: int,
) -> dict[str, Any]:
    return {
        "round": round_index,
        "metrics": metrics,
        "mine_metrics": mine_metrics,
        "losses": losses,
        "replay_size": replay_size,
    }


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def vars_for_json(args: argparse.Namespace) -> dict[str, Any]:
    result = vars(args).copy()
    for key, value in list(result.items()):
        if isinstance(value, Path):
            result[key] = str(value)
    return result


if __name__ == "__main__":
    main()
