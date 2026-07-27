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
from minesweeper_rl.replay import run_episode_trace, save_trace
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import MinesweeperTrainer, load_checkpoint, make_game
from minesweeper_rl.types import EpisodeSummary


def main() -> None:
    parser = argparse.ArgumentParser(description="Find and record a pure-RL winning streak.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--start-seed", type=int, default=100)
    parser.add_argument("--max-games", type=int, default=50000)
    parser.add_argument("--streak-length", type=int, default=10)
    parser.add_argument("--batch-size", type=int, default=256)
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

    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    progress_path = output_dir / "progress.jsonl"
    latest_path = output_dir / "latest.json"

    trainer = load_search_trainer(args)
    config_record = {
        "checkpoint": str(args.checkpoint),
        "start_seed": args.start_seed,
        "max_games": args.max_games,
        "streak_length": args.streak_length,
        "batch_size": args.batch_size,
        "mode": args.mode,
        "risk_weight": args.risk_weight,
        "final_decision_mode": args.mode,
        "solver_allowed_during_final_decision": False,
        "model_flip_ensemble": trainer.config.inference_augment_flips,
        "model_ensemble_method": trainer.config.inference_ensemble,
        "game_config": asdict(trainer.config.game_config),
        "device": str(trainer.device),
    }
    (output_dir / "search_config.json").write_text(json.dumps(config_record, indent=2), encoding="utf-8")

    started_at = time.time()
    current_streak: list[EpisodeSummary] = []
    total_games = 0
    total_wins = 0
    longest_streak: list[EpisodeSummary] = []
    verified_failures: list[dict[str, Any]] = []

    with progress_path.open("a", encoding="utf-8") as progress:
        for batch_start in range(0, args.max_games, args.batch_size):
            current_batch = min(args.batch_size, args.max_games - batch_start)
            seed = args.start_seed + batch_start
            summaries = run_batched_summaries(
                trainer=trainer,
                seed=seed,
                games=current_batch,
                batch_size=args.batch_size,
                mode=args.mode,
                risk_weight=args.risk_weight,
            )

            for summary in summaries:
                total_games += 1
                if summary.won:
                    total_wins += 1
                    current_streak.append(summary)
                    if len(current_streak) > len(longest_streak):
                        longest_streak = list(current_streak)
                else:
                    current_streak.clear()

                progress.write(json.dumps(summary.as_dict(), separators=(",", ":")),)
                progress.write("\n")

                if len(current_streak) >= args.streak_length:
                    candidate = current_streak[-args.streak_length :]
                    traces = verify_and_record_streak(
                        trainer=trainer,
                        summaries=candidate,
                        output_dir=output_dir,
                        mode=args.mode,
                        risk_weight=args.risk_weight,
                    )
                    if traces is not None:
                        manifest = make_manifest(
                            status="found",
                            config=config_record,
                            total_games=total_games,
                            total_wins=total_wins,
                            longest_streak=longest_streak,
                            traces=traces,
                            started_at=started_at,
                            verified_failures=verified_failures,
                        )
                        write_json(output_dir / "manifest.json", manifest)
                        write_json(latest_path, manifest)
                        print(json.dumps(manifest, indent=2))
                        return

                    verified_failures.append(
                        {
                            "candidate_start_seed": candidate[0].seed,
                            "candidate_end_seed": candidate[-1].seed,
                            "reason": "sequential_trace_verification_failed",
                        }
                    )
                    current_streak.clear()

            progress.flush()
            snapshot = make_manifest(
                status="searching",
                config=config_record,
                total_games=total_games,
                total_wins=total_wins,
                longest_streak=longest_streak,
                traces=[],
                started_at=started_at,
                verified_failures=verified_failures,
            )
            write_json(latest_path, snapshot)
            print(
                json.dumps(
                    {
                        "status": "searching",
                        "searched_games": total_games,
                        "next_seed": args.start_seed + total_games,
                        "win_rate_so_far": total_wins / total_games if total_games else 0.0,
                        "longest_streak_so_far": len(longest_streak),
                        "longest_streak_start_seed": longest_streak[0].seed if longest_streak else None,
                        "longest_streak_end_seed": longest_streak[-1].seed if longest_streak else None,
                    },
                    separators=(",", ":"),
                ),
                flush=True,
            )

    manifest = make_manifest(
        status="not_found",
        config=config_record,
        total_games=total_games,
        total_wins=total_wins,
        longest_streak=longest_streak,
        traces=[],
        started_at=started_at,
        verified_failures=verified_failures,
    )
    write_json(output_dir / "manifest.json", manifest)
    write_json(latest_path, manifest)
    print(json.dumps(manifest, indent=2))


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "streak_runs" / f"run_{stamp}"


def load_search_trainer(args: argparse.Namespace) -> MinesweeperTrainer:
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


def run_batched_summaries(
    trainer: MinesweeperTrainer,
    seed: int,
    games: int,
    batch_size: int,
    mode: str,
    risk_weight: float,
) -> list[EpisodeSummary]:
    summaries: list[EpisodeSummary] = []
    batch_size = max(1, int(batch_size))

    for start in range(0, games, batch_size):
        current_batch = min(batch_size, games - start)
        game_batch = [make_game(trainer.config, seed=seed + start + offset) for offset in range(current_batch)]
        rewards = [0.0 for _ in range(current_batch)]
        agent_steps = [0 for _ in range(current_batch)]
        active = list(range(current_batch))

        while active:
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
                action = decode_action_index(int(action_indices[local_index]), trainer.config.rows, trainer.config.cols)
                _, reward, _, _ = game.step(action)
                rewards[game_index] += float(reward)
                agent_steps[game_index] += 1
                if not game.done and agent_steps[game_index] < trainer.config.max_steps:
                    still_active.append(game_index)
            active = still_active

        for offset, game in enumerate(game_batch):
            summaries.append(
                trainer._episode_summary(
                    game=game,
                    seed=seed + start + offset,
                    total_reward=rewards[offset],
                    guess_steps=agent_steps[offset],
                    forced_steps=0,
                )
            )

    return summaries


def verify_and_record_streak(
    trainer: MinesweeperTrainer,
    summaries: list[EpisodeSummary],
    output_dir: Path,
    mode: str,
    risk_weight: float,
) -> list[dict[str, Any]] | None:
    traces: list[tuple[Path, dict[str, Any]]] = []
    for index, summary in enumerate(summaries, start=1):
        trace = run_episode_trace(
            trainer=trainer,
            seed=int(summary.seed),
            mode=mode,
            risk_weight=risk_weight,
            deterministic=True,
        )
        if not trace["summary"]["won"]:
            return None
        path = output_dir / f"game_{index:02d}_seed_{summary.seed}.json"
        traces.append((path, trace))

    records: list[dict[str, Any]] = []
    for path, trace in traces:
        save_trace(path, trace)
        records.append(
            {
                "seed": trace["seed"],
                "path": str(path),
                "summary": trace["summary"],
                "frames": len(trace["frames"]),
            }
        )
    return records


def make_manifest(
    status: str,
    config: dict[str, Any],
    total_games: int,
    total_wins: int,
    longest_streak: list[EpisodeSummary],
    traces: list[dict[str, Any]],
    started_at: float,
    verified_failures: list[dict[str, Any]],
) -> dict[str, Any]:
    return {
        "status": status,
        "checkpoint": config["checkpoint"],
        "config": config,
        "searched_games": total_games,
        "searched_seed_start": config["start_seed"],
        "searched_seed_end": None if total_games <= 0 else config["start_seed"] + total_games - 1,
        "win_rate_so_far": total_wins / total_games if total_games else 0.0,
        "longest_streak_so_far": len(longest_streak),
        "longest_streak_start_seed": longest_streak[0].seed if longest_streak else None,
        "longest_streak_end_seed": longest_streak[-1].seed if longest_streak else None,
        "streak_start_seed": traces[0]["seed"] if traces else None,
        "streak_end_seed": traces[-1]["seed"] if traces else None,
        "traces": traces,
        "verified_failures": verified_failures,
        "elapsed_seconds": time.time() - started_at,
    }


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
