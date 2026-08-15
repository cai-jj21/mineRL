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
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint, make_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a pure-RL checkpoint ensemble.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--ensemble-checkpoint", dest="ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--output", type=Path, default=None)
    args = parser.parse_args()

    checkpoints = [args.checkpoint, *args.ensemble_checkpoints]
    trainers = [load_eval_trainer(path, args) for path in checkpoints]
    started_at = time.time()
    metrics = evaluate_ensemble(
        trainers=trainers,
        games=args.games,
        seed=args.seed,
        batch_size=args.batch_size,
    )
    result = {
        "checkpoints": [str(path) for path in checkpoints],
        "device": str(trainers[0].device),
        "inference_augment_flips": args.inference_flips,
        "inference_ensemble": args.inference_ensemble,
        "solver_decision": False,
        "elapsed_seconds": time.time() - started_at,
        **metrics,
    }
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_eval_trainer(path: Path, args: argparse.Namespace) -> MinesweeperTrainer:
    trainer = load_checkpoint(path, device=args.device)
    trainer.config.rows = 16
    trainer.config.cols = 30
    trainer.config.mines = 99
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = "full"
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = 0.0
    trainer.model.eval()
    return trainer


def evaluate_ensemble(
    trainers: list[MinesweeperTrainer],
    games: int,
    seed: int,
    batch_size: int,
) -> dict[str, Any]:
    if not trainers:
        raise ValueError("at least one trainer is required")
    base = trainers[0]
    batch_size = max(1, int(batch_size))
    wins: list[bool] = []
    rewards: list[float] = []
    agent_steps: list[int] = []
    revealed_safe_cells: list[int] = []

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

            for index in list(active):
                game = game_batch[index]
                if game.done or batch_steps[index] >= base.config.max_steps:
                    active.remove(index)
                    continue

                board, global_feature, action_mask = encode_state(game)
                action_mask = base._decision_action_mask(action_mask)
                if not action_mask.any():
                    active.remove(index)
                    continue

                ready.append(index)
                boards.append(board)
                global_features.append(global_feature)
                action_masks.append(action_mask)

            if not ready:
                continue

            scores = average_policy_scores(
                trainers=trainers,
                boards=np.stack(boards),
                global_features_batch=np.stack(global_features),
                action_masks=np.stack(action_masks),
            )
            action_indices = scores.argmax(dim=1).detach().cpu().numpy()

            for local_index, game_index in enumerate(ready):
                game = game_batch[game_index]
                action = decode_action_index(int(action_indices[local_index]), base.config.rows, base.config.cols)
                _, reward, _, _ = game.step(action)
                batch_rewards[game_index] += float(reward)
                batch_steps[game_index] += 1
                if game.done or batch_steps[game_index] >= base.config.max_steps:
                    active.discard(game_index)

        for index, game in enumerate(game_batch):
            wins.append(bool(game.won))
            rewards.append(batch_rewards[index])
            agent_steps.append(batch_steps[index])
            revealed_safe_cells.append(
                int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum())
            )

    return {
        "games": float(games),
        "seed": seed,
        "win_rate": float(np.mean(wins)) if wins else 0.0,
        "wins": int(sum(wins)),
        "longest_streak": longest_streak(wins),
        "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
        "avg_agent_steps": float(np.mean(agent_steps)) if agent_steps else 0.0,
        "avg_revealed_safe_cells": float(np.mean(revealed_safe_cells)) if revealed_safe_cells else 0.0,
    }


def average_policy_scores(
    trainers: list[MinesweeperTrainer],
    boards: np.ndarray,
    global_features_batch: np.ndarray,
    action_masks: np.ndarray,
) -> torch.Tensor:
    combined = None
    for trainer in trainers:
        scores = trainer._predict_policy_scores_batch(
            boards=boards,
            global_features_batch=global_features_batch,
            action_masks=action_masks,
            use_flip_ensemble=bool(trainer.config.inference_augment_flips),
        )
        combined = scores if combined is None else combined + scores
    assert combined is not None
    return combined / float(len(trainers))


def longest_streak(wins: list[bool]) -> int:
    best = 0
    current = 0
    for won in wins:
        if won:
            current += 1
            best = max(best, current)
        else:
            current = 0
    return best


if __name__ == "__main__":
    main()
