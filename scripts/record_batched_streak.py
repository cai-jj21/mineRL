from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import asdict
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.features import decode_action_index, encode_state
from minesweeper_rl.replay import _make_frame, _model_decision, make_trace, save_trace
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint, make_game


def main() -> None:
    parser = argparse.ArgumentParser(description="Record a streak in the same batched inference context that found it.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--context-start-seed", type=int, required=True)
    parser.add_argument("--context-batch-size", type=int, default=256)
    parser.add_argument("--streak-start-seed", type=int, required=True)
    parser.add_argument("--streak-length", type=int, default=10)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--mode", choices=["rl", "policy"], default="rl")
    parser.add_argument("--risk-weight", type=float, default=0.0)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument("--safe-radius", type=int, default=1)
    parser.add_argument("--exact-limit", type=int, default=24)
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--decision-actions", choices=["open", "full"], default="full")
    parser.add_argument("--inference-flips", action="store_true")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    args = parser.parse_args()

    output_dir = args.output_dir or default_output_dir(args.streak_start_seed)
    output_dir.mkdir(parents=True, exist_ok=True)

    target_start_index = args.streak_start_seed - args.context_start_seed
    target_end_index = target_start_index + args.streak_length - 1
    if target_start_index < 0 or target_end_index >= args.context_batch_size:
        raise ValueError("streak seeds must be inside the context batch")

    trainer = load_record_trainer(args)
    started_at = time.time()
    traces = record_context_batch(
        trainer=trainer,
        context_start_seed=args.context_start_seed,
        context_batch_size=args.context_batch_size,
        target_indices=set(range(target_start_index, target_end_index + 1)),
        mode=args.mode,
        risk_weight=args.risk_weight,
    )

    records: list[dict[str, Any]] = []
    for index, trace in enumerate(sorted(traces, key=lambda item: item["seed"]), start=1):
        path = output_dir / f"game_{index:02d}_seed_{trace['seed']}.json"
        save_trace(path, trace)
        records.append(
            {
                "seed": trace["seed"],
                "path": str(path),
                "summary": trace["summary"],
                "frames": len(trace["frames"]),
            }
        )

    all_won = len(records) == args.streak_length and all(record["summary"]["won"] for record in records)
    manifest = {
        "status": "found" if all_won else "not_reproduced",
        "checkpoint": str(args.checkpoint),
        "context_start_seed": args.context_start_seed,
        "context_end_seed": args.context_start_seed + args.context_batch_size - 1,
        "context_batch_size": args.context_batch_size,
        "streak_start_seed": args.streak_start_seed if all_won else None,
        "streak_end_seed": args.streak_start_seed + args.streak_length - 1 if all_won else None,
        "streak_length": args.streak_length,
        "recording_note": "Recorded under the same batched inference context used by the streak search.",
        "final_decision_mode": args.mode,
        "solver_allowed_during_final_decision": False,
        "model_flip_ensemble": trainer.config.inference_augment_flips,
        "model_ensemble_method": trainer.config.inference_ensemble,
        "game_config": asdict(trainer.config.game_config),
        "device": str(trainer.device),
        "traces": records,
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def default_output_dir(streak_start_seed: int) -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "streak_runs" / f"batched_streak_{streak_start_seed}_{stamp}"


def load_record_trainer(args: argparse.Namespace) -> MinesweeperTrainer:
    trainer = load_checkpoint(args.checkpoint, device=args.device)
    trainer.config.rows = args.rows
    trainer.config.cols = args.cols
    trainer.config.mines = args.mines
    trainer.config.safe_radius = args.safe_radius
    trainer.config.exact_limit = args.exact_limit
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = args.decision_actions
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.solver = MinesweeperSolver(exact_limit=trainer.config.exact_limit)
    if trainer.device.type == "cuda":
        torch.backends.cudnn.benchmark = True
    return trainer


def record_context_batch(
    trainer: MinesweeperTrainer,
    context_start_seed: int,
    context_batch_size: int,
    target_indices: set[int],
    mode: str,
    risk_weight: float,
) -> list[dict[str, Any]]:
    game_batch = [make_game(trainer.config, seed=context_start_seed + offset) for offset in range(context_batch_size)]
    rewards = [0.0 for _ in range(context_batch_size)]
    agent_steps = [0 for _ in range(context_batch_size)]
    frame_indices = [0 for _ in range(context_batch_size)]
    frames: dict[int, list[dict[str, Any]]] = {index: [] for index in target_indices}
    active = list(range(context_batch_size))

    while active and any(not game_batch[index].done and agent_steps[index] < trainer.config.max_steps for index in target_indices):
        ready: list[int] = []
        boards: list[np.ndarray] = []
        global_features: list[np.ndarray] = []
        action_masks: list[np.ndarray] = []
        next_active: list[int] = []

        for index in active:
            game = game_batch[index]
            if game.done or agent_steps[index] >= trainer.config.max_steps:
                continue

            board, global_feature, action_mask = encode_state(game)
            action_mask = trainer._decision_action_mask(action_mask)
            if not action_mask.any():
                if index in target_indices:
                    frames[index].append(
                        _make_frame(
                            game=game,
                            snapshot=None,
                            frame_index=frame_indices[index],
                            event="stalled",
                            action=None,
                            decision={"type": "stalled", "mode": mode, "risk": None, "note": "no legal actions"},
                            reward=0.0,
                            cumulative_reward=rewards[index],
                            forced_steps=0,
                            guess_steps=agent_steps[index],
                            info={"valid": False, "reason": "no_legal_actions"},
                        )
                    )
                    frame_indices[index] += 1
                continue

            ready.append(index)
            boards.append(board)
            global_features.append(global_feature)
            action_masks.append(action_mask)
            next_active.append(index)

        active = next_active
        if not ready:
            continue

        scores = trainer._predict_policy_scores_batch(
            boards=np.stack(boards),
            global_features_batch=np.stack(global_features),
            action_masks=np.stack(action_masks),
            use_flip_ensemble=trainer.config.inference_augment_flips,
        )
        action_indices = scores.argmax(dim=1).detach().cpu().numpy()

        still_active: list[int] = []
        for local_index, game_index in enumerate(ready):
            game = game_batch[game_index]
            action_index = int(action_indices[local_index])
            action = decode_action_index(action_index, trainer.config.rows, trainer.config.cols)
            _, reward, _, info = game.step(action)
            rewards[game_index] += float(reward)
            agent_steps[game_index] += 1

            if game_index in target_indices:
                decision = _model_decision(
                    action=action,
                    action_index=action_index,
                    mode=mode,
                    risk_weight=risk_weight,
                    legal_actions=int(action_masks[local_index].sum()),
                )
                frames[game_index].append(
                    _make_frame(
                        game=game,
                        snapshot=None,
                        frame_index=frame_indices[game_index],
                        event=decision["type"],
                        action=action,
                        decision=decision,
                        reward=reward,
                        cumulative_reward=rewards[game_index],
                        forced_steps=0,
                        guess_steps=agent_steps[game_index],
                        info=info,
                    )
                )
                frame_indices[game_index] += 1

            if not game.done and agent_steps[game_index] < trainer.config.max_steps:
                still_active.append(game_index)
        active = still_active

    traces: list[dict[str, Any]] = []
    for index in sorted(target_indices):
        traces.append(
            make_trace(
                trainer=trainer,
                seed=context_start_seed + index,
                mode=mode,
                risk_weight=risk_weight,
                frames=frames[index],
            )
        )
    return traces


if __name__ == "__main__":
    main()
