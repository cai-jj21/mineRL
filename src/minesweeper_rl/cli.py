from __future__ import annotations

import argparse
import json
from pathlib import Path

from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import MinesweeperTrainer, TrainingConfig, evaluate_policy, evaluate_policy_batched, load_checkpoint


def main() -> None:
    parser = argparse.ArgumentParser(description="Minesweeper RL")
    subparsers = parser.add_subparsers(dest="command", required=True)

    train_parser = subparsers.add_parser("train", help="train the pure RL policy network")
    _add_common_args(train_parser)
    train_parser.add_argument("--episodes", type=int, default=1000)
    train_parser.add_argument("--lr", type=float, default=3e-4)
    train_parser.add_argument("--entropy-coef", type=float, default=0.01)
    train_parser.add_argument("--value-coef", type=float, default=0.5)
    train_parser.add_argument("--solver-imitation-coef", type=float, default=0.2)
    train_parser.add_argument("--mine-aux-coef", type=float, default=0.0)
    train_parser.add_argument("--risk-supervision-coef", type=float, default=0.0)
    train_parser.add_argument("--risk-temperature", type=float, default=0.08)
    train_parser.add_argument("--pretrain-episodes", type=int, default=0)
    train_parser.add_argument("--pretrain-imitation-coef", type=float, default=1.0)
    train_parser.add_argument("--pretrain-epochs-per-episode", type=int, default=1)
    train_parser.add_argument("--pretrain-batch-size", type=int, default=256)
    train_parser.add_argument("--expert-replay-size", type=int, default=8192)
    train_parser.add_argument("--dagger-replay-size", type=int, default=8192)
    train_parser.add_argument("--dagger-batch-size", type=int, default=256)
    train_parser.add_argument("--dagger-updates-per-episode", type=int, default=1)
    train_parser.add_argument("--guess-supervision-topk", type=int, default=5)
    train_parser.add_argument("--guess-imitation-weight", type=float, default=1.0)
    train_parser.add_argument("--hidden-channels", type=int, default=None)
    train_parser.add_argument("--residual-blocks", type=int, default=None)
    train_parser.add_argument("--global-policy-context", action="store_true")
    train_parser.add_argument("--long-range-context", action="store_true")
    train_parser.add_argument("--risk-head-coef", type=float, default=0.0)
    train_parser.add_argument("--exploration-temperature", type=float, default=1.0)
    train_parser.add_argument("--exploration-topk", type=int, default=0)
    train_parser.add_argument("--no-augment-flips", dest="augment_flips", action="store_false")
    train_parser.add_argument("--eval-every", type=int, default=100)
    train_parser.add_argument("--eval-games", type=int, default=50)
    train_parser.add_argument("--max-steps", type=int, default=2000)
    train_parser.add_argument("--checkpoint", type=Path, default=None)
    train_parser.add_argument("--save-path", type=Path, default=Path("checkpoints/rl.pt"))

    eval_parser = subparsers.add_parser("evaluate", help="evaluate solver, rl, or hybrid policy")
    _add_common_args(eval_parser)
    eval_parser.add_argument("--checkpoint", type=Path, default=None)
    eval_parser.add_argument("--games", type=int, default=100)
    eval_parser.add_argument("--mode", choices=["rl", "policy", "solver", "hybrid"], default="rl")
    eval_parser.add_argument("--risk-weight", type=float, default=1.0)
    eval_parser.add_argument("--batch-size", type=int, default=1)

    target_parser = subparsers.add_parser("target-check", help="check the expert-board target metrics")
    _add_common_args(target_parser)
    target_parser.add_argument("--checkpoint", type=Path, default=None)
    target_parser.add_argument("--games", type=int, default=3000)
    target_parser.add_argument("--required-win-rate", type=float, default=0.4)
    target_parser.add_argument("--required-streak", type=int, default=10)
    target_parser.add_argument("--batch-size", type=int, default=1)
    target_parser.set_defaults(seed=100, decision_actions="full")

    record_parser = subparsers.add_parser("record", help="save one replay JSON")
    _add_common_args(record_parser)
    record_parser.add_argument("--checkpoint", type=Path, default=None)
    record_parser.add_argument("--mode", choices=["rl", "policy", "solver", "hybrid"], default="rl")
    record_parser.add_argument("--risk-weight", type=float, default=1.0)
    record_parser.add_argument("--output", type=Path, default=Path("replays/latest.json"))
    record_parser.add_argument("--find-win", action="store_true")
    record_parser.add_argument("--max-attempts", type=int, default=500)
    record_parser.set_defaults(seed=100)

    watch_parser = subparsers.add_parser("watch", help="open the live/replay visualizer")
    _add_common_args(watch_parser)
    watch_parser.add_argument("--checkpoint", type=Path, default=None)
    watch_parser.add_argument("--mode", choices=["rl", "policy", "solver", "hybrid"], default="rl")
    watch_parser.add_argument("--risk-weight", type=float, default=1.0)
    watch_parser.add_argument("--replay", type=Path, default=None)
    watch_parser.add_argument("--save-replay", type=Path, default=None)
    watch_parser.add_argument("--speed-ms", type=int, default=350)
    watch_parser.add_argument("--paused", action="store_true")
    watch_parser.add_argument("--find-win", action="store_true")
    watch_parser.add_argument("--max-attempts", type=int, default=500)
    watch_parser.set_defaults(seed=100)

    args = parser.parse_args()
    if args.command == "train":
        default_config = TrainingConfig()
        hidden_channels = args.hidden_channels if args.hidden_channels is not None else default_config.hidden_channels
        residual_blocks = args.residual_blocks if args.residual_blocks is not None else default_config.residual_blocks
        config = TrainingConfig(
            rows=args.rows,
            cols=args.cols,
            mines=args.mines,
            safe_radius=args.safe_radius,
            exact_limit=args.exact_limit,
            seed=args.seed,
            episodes=args.episodes,
            lr=args.lr,
            entropy_coef=args.entropy_coef,
            value_coef=args.value_coef,
            solver_imitation_coef=args.solver_imitation_coef,
            mine_aux_coef=args.mine_aux_coef,
            risk_supervision_coef=args.risk_supervision_coef,
            risk_temperature=args.risk_temperature,
            hidden_channels=hidden_channels,
            residual_blocks=residual_blocks,
            global_policy_context=args.global_policy_context,
            long_range_context=args.long_range_context,
            pretrain_episodes=args.pretrain_episodes,
            pretrain_imitation_coef=args.pretrain_imitation_coef,
            pretrain_epochs_per_episode=args.pretrain_epochs_per_episode,
            pretrain_batch_size=args.pretrain_batch_size,
            expert_replay_size=args.expert_replay_size,
            dagger_replay_size=args.dagger_replay_size,
            dagger_batch_size=args.dagger_batch_size,
            dagger_updates_per_episode=args.dagger_updates_per_episode,
            guess_supervision_topk=args.guess_supervision_topk,
            guess_imitation_weight=args.guess_imitation_weight,
            exploration_temperature=args.exploration_temperature,
            exploration_topk=args.exploration_topk,
            inference_augment_flips=args.inference_flips,
            inference_ensemble=args.inference_ensemble,
            augment_flips=args.augment_flips,
            decision_actions=args.decision_actions,
            risk_head_coef=args.risk_head_coef,
            risk_head_weight=args.risk_head_weight,
            eval_every=args.eval_every,
            eval_games=args.eval_games,
            max_steps=args.max_steps,
            device=args.device,
        )
        if args.checkpoint is not None:
            config_overrides = {}
            if args.hidden_channels is not None:
                config_overrides["hidden_channels"] = args.hidden_channels
            if args.residual_blocks is not None:
                config_overrides["residual_blocks"] = args.residual_blocks
            if args.global_policy_context:
                config_overrides["global_policy_context"] = True
            if args.long_range_context:
                config_overrides["long_range_context"] = True
            trainer = load_checkpoint(args.checkpoint, device=args.device, config_overrides=config_overrides or None)
            trainer.config.rows = config.rows
            trainer.config.cols = config.cols
            trainer.config.mines = config.mines
            trainer.config.safe_radius = config.safe_radius
            trainer.config.exact_limit = config.exact_limit
            trainer.config.seed = config.seed
            trainer.config.episodes = config.episodes
            trainer.config.lr = config.lr
            trainer.config.entropy_coef = config.entropy_coef
            trainer.config.value_coef = config.value_coef
            trainer.config.solver_imitation_coef = config.solver_imitation_coef
            trainer.config.mine_aux_coef = config.mine_aux_coef
            trainer.config.risk_supervision_coef = config.risk_supervision_coef
            trainer.config.risk_temperature = config.risk_temperature
            trainer.config.hidden_channels = config.hidden_channels
            trainer.config.residual_blocks = config.residual_blocks
            trainer.config.global_policy_context = config.global_policy_context
            trainer.config.long_range_context = config.long_range_context
            trainer.config.pretrain_episodes = config.pretrain_episodes
            trainer.config.pretrain_imitation_coef = config.pretrain_imitation_coef
            trainer.config.pretrain_epochs_per_episode = config.pretrain_epochs_per_episode
            trainer.config.pretrain_batch_size = config.pretrain_batch_size
            trainer.config.expert_replay_size = config.expert_replay_size
            trainer.config.dagger_replay_size = config.dagger_replay_size
            trainer.config.dagger_batch_size = config.dagger_batch_size
            trainer.config.dagger_updates_per_episode = config.dagger_updates_per_episode
            trainer.config.guess_supervision_topk = config.guess_supervision_topk
            trainer.config.guess_imitation_weight = config.guess_imitation_weight
            trainer.config.exploration_temperature = config.exploration_temperature
            trainer.config.exploration_topk = config.exploration_topk
            trainer.config.inference_augment_flips = config.inference_augment_flips
            trainer.config.inference_ensemble = config.inference_ensemble
            trainer.config.augment_flips = config.augment_flips
            trainer.config.decision_actions = config.decision_actions
            trainer.config.risk_head_coef = config.risk_head_coef
            trainer.config.risk_head_weight = config.risk_head_weight
            trainer.config.eval_every = config.eval_every
            trainer.config.eval_games = config.eval_games
            trainer.config.max_steps = config.max_steps
            trainer.config.device = config.device
            for group in trainer.optimizer.param_groups:
                group["lr"] = config.lr
            trainer.solver = MinesweeperSolver(exact_limit=trainer.config.exact_limit)
        else:
            trainer = MinesweeperTrainer(config)
        metrics = trainer.train(save_path=args.save_path)
        print(json.dumps(metrics, indent=2))
    elif args.command == "evaluate":
        trainer = _load_or_create_trainer(args)
        evaluator = evaluate_policy_batched if args.batch_size > 1 else evaluate_policy
        metrics = evaluator(
            trainer,
            games=args.games,
            seed=args.seed,
            mode=args.mode,
            risk_weight=args.risk_weight,
            **({"batch_size": args.batch_size} if args.batch_size > 1 else {}),
        )
        print(json.dumps(metrics, indent=2))
    elif args.command == "record":
        from minesweeper_rl.replay import find_winning_trace, run_episode_trace, save_trace

        trainer = _load_or_create_trainer(args)
        if args.find_win:
            trace = find_winning_trace(
                trainer=trainer,
                start_seed=args.seed,
                max_attempts=args.max_attempts,
                mode=args.mode,
                risk_weight=args.risk_weight,
            )
        else:
            trace = run_episode_trace(
                trainer=trainer,
                seed=args.seed,
                mode=args.mode,
                risk_weight=args.risk_weight,
            )
        save_trace(args.output, trace)
        print(json.dumps({"output": str(args.output), "summary": trace["summary"]}, indent=2))
    elif args.command == "watch":
        from minesweeper_rl.replay import find_winning_trace, load_trace, save_trace
        from minesweeper_rl.visualizer import show_live_episode, show_trace

        if args.replay is not None:
            show_trace(load_trace(args.replay), speed_ms=args.speed_ms, paused=args.paused)
        elif args.find_win:
            trainer = _load_or_create_trainer(args)
            trace = find_winning_trace(
                trainer=trainer,
                start_seed=args.seed,
                max_attempts=args.max_attempts,
                mode=args.mode,
                risk_weight=args.risk_weight,
            )
            if args.save_replay is not None:
                save_trace(args.save_replay, trace)
            show_trace(trace, speed_ms=args.speed_ms, paused=args.paused)
        else:
            trainer = _load_or_create_trainer(args)
            show_live_episode(
                trainer=trainer,
                seed=args.seed,
                mode=args.mode,
                risk_weight=args.risk_weight,
                speed_ms=args.speed_ms,
                save_path=args.save_replay,
                paused=args.paused,
            )
    elif args.command == "target-check":
        trainer = _load_or_create_trainer(args)
        evaluator = evaluate_policy_batched if args.batch_size > 1 else evaluate_policy
        metrics = evaluator(
            trainer,
            games=args.games,
            seed=args.seed,
            mode="rl",
            risk_weight=0.0,
            **({"batch_size": args.batch_size} if args.batch_size > 1 else {}),
        )
        target_passed = (
            float(metrics["win_rate"]) >= args.required_win_rate
            and float(metrics["longest_streak"]) >= args.required_streak
        )
        metrics.update(
            {
                "required_win_rate": args.required_win_rate,
                "required_streak": args.required_streak,
                "final_decision_mode": "rl",
                "solver_allowed_during_final_decision": False,
                "model_flip_ensemble": trainer.config.inference_augment_flips,
                "model_ensemble_method": trainer.config.inference_ensemble,
                "target_passed": target_passed,
            }
        )
        print(json.dumps(metrics, indent=2))


def _load_or_create_trainer(args: argparse.Namespace) -> MinesweeperTrainer:
    if args.checkpoint is not None:
        trainer = load_checkpoint(args.checkpoint, device=args.device)
        trainer.config.rows = args.rows
        trainer.config.cols = args.cols
        trainer.config.mines = args.mines
        trainer.config.safe_radius = args.safe_radius
        trainer.config.exact_limit = args.exact_limit
        trainer.config.decision_actions = args.decision_actions
        trainer.config.risk_head_weight = args.risk_head_weight
        trainer.config.inference_augment_flips = args.inference_flips
        trainer.config.inference_ensemble = args.inference_ensemble
        trainer.solver = MinesweeperSolver(exact_limit=trainer.config.exact_limit)
        return trainer

    return MinesweeperTrainer(
        TrainingConfig(
            rows=args.rows,
            cols=args.cols,
            mines=args.mines,
            safe_radius=args.safe_radius,
            exact_limit=args.exact_limit,
            seed=args.seed,
            device=args.device,
            decision_actions=args.decision_actions,
            risk_head_weight=args.risk_head_weight,
            inference_augment_flips=args.inference_flips,
            inference_ensemble=args.inference_ensemble,
            eval_games=getattr(args, "games", 1),
            max_steps=getattr(args, "max_steps", 2000),
        )
    )


def _add_common_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--cols", type=int, default=30)
    parser.add_argument("--mines", type=int, default=99)
    parser.add_argument("--safe-radius", type=int, default=1)
    parser.add_argument("--exact-limit", type=int, default=24)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--decision-actions", choices=["open", "full"], default="open")
    parser.add_argument("--inference-flips", action="store_true")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="logits")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)


if __name__ == "__main__":
    main()
