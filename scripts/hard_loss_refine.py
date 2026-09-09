from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.distributions import Categorical
from torch.nn import functional as F

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.features import action_channel, decode_action_index, encode_state
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.extreme_replay import load_extreme_dataset
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import (
    MinesweeperTrainer,
    load_checkpoint,
    make_game,
    resolve_forced_moves,
    save_checkpoint,
    transition_extreme_profile,
    transition_sample_weight,
)
from minesweeper_rl.types import Action, ActionType, EpisodeTransition
from minesweeper_rl.windows_replay import load_windows_replay_transitions


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune a policy on its own losing/endgame hard states.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/full_rlmix_hard_refine.pt"))
    parser.add_argument("--save-last-path", type=Path)
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
    parser.add_argument("--counterfactual-coef", type=float, default=0.0)
    parser.add_argument("--counterfactual-margin", type=float, default=0.1)
    parser.add_argument("--counterfactual-temperature", type=float, default=0.08)
    parser.add_argument("--counterfactual-policy-coef", type=float, default=0.0)
    parser.add_argument("--counterfactual-policy-temperature", type=float, default=0.25)
    parser.add_argument("--counterfactual-policy-topk", type=int, default=16)
    parser.add_argument("--counterfactual-policy-min-gap", type=float, default=0.0)
    parser.add_argument(
        "--counterfactual-policy-full-action",
        action="store_true",
        help="Normalize counterfactual guess targets over all legal actions so OPEN keeps mass against FLAG/UNFLAG.",
    )
    parser.add_argument(
        "--counterfactual-only-guess",
        action="store_true",
        help="Exclude counterfactual-labelled guess rows from solver imitation; learn them from counterfactual policy loss.",
    )
    parser.add_argument("--counterfactual-value-head-coef", type=float, default=0.0)
    parser.add_argument("--counterfactual-value-head-clip", type=float, default=4.0)
    parser.add_argument("--counterfactual-value-rank-coef", type=float, default=0.0)
    parser.add_argument(
        "--counterfactual-value-target-mode",
        choices=["raw", "minmax", "zscore", "rank"],
        default="raw",
        help="Normalize finite counterfactual labels per state before value-head training.",
    )
    parser.add_argument("--counterfactual-labels", action="store_true")
    parser.add_argument("--counterfactual-topk", type=int, default=64)
    parser.add_argument(
        "--counterfactual-all-open",
        action="store_true",
        help="Label every legal OPEN candidate instead of only the risk/model shortlist.",
    )
    parser.add_argument(
        "--counterfactual-model-topk",
        type=int,
        default=None,
        help="Number of the model's top OPEN candidates to force into offline counterfactual labels.",
    )
    parser.add_argument("--teacher-checkpoint", type=Path)
    parser.add_argument(
        "--teacher-ensemble-checkpoint",
        action="append",
        default=[],
        help="Additional teacher checkpoint. Repeat to match the production teacher ensemble.",
    )
    parser.add_argument(
        "--teacher-ensemble-weight",
        action="append",
        type=float,
        default=[],
        help="Optional normalized weights for --teacher-checkpoint followed by repeated ensemble checkpoints.",
    )
    parser.add_argument("--teacher-kl-coef", type=float, default=0.0)
    parser.add_argument("--teacher-temperature", type=float, default=1.0)
    parser.add_argument(
        "--teacher-prior-policy-coef",
        type=float,
        default=0.0,
        help="Weight for a teacher-prior counterfactual policy-improvement target.",
    )
    parser.add_argument(
        "--teacher-prior-strength",
        type=float,
        default=0.15,
        help="Fraction of labelled OPEN probability mass moved toward counterfactual outcomes.",
    )
    parser.add_argument(
        "--teacher-prior-temperature",
        type=float,
        default=2.0,
        help="Temperature for the counterfactual distribution in the teacher-prior target.",
    )
    parser.add_argument(
        "--teacher-prior-objective",
        choices=["target", "ppo"],
        default="target",
        help="Offline policy-improvement objective used with the teacher prior.",
    )
    parser.add_argument(
        "--teacher-prior-clip",
        type=float,
        default=0.2,
        help="PPO ratio clip for the teacher-prior objective.",
    )
    parser.add_argument(
        "--teacher-prior-policy-topk",
        type=int,
        default=0,
        help="If positive, only improve the teacher's top-K labelled OPEN cells.",
    )
    parser.add_argument("--mine-aux-coef", type=float, default=0.04)
    parser.add_argument("--risk-supervision-coef", type=float, default=0.08)
    parser.add_argument("--risk-head-coef", type=float, default=0.02)
    parser.add_argument("--risk-temperature", type=float, default=0.06)
    parser.add_argument("--counterfactual-risk-head-coef", type=float, default=0.0)
    parser.add_argument("--counterfactual-risk-temperature", type=float, default=1.0)
    parser.add_argument(
        "--counterfactual-risk-loss-mode",
        choices=["bce", "listwise"],
        default="bce",
        help="Train the counterfactual risk head with per-cell BCE or state-wise candidate ranking.",
    )
    parser.add_argument(
        "--guess-survival-coef",
        type=float,
        default=0.0,
        help="Policy-head loss on guess states using the offline mine mask: push real mines down and safe opens up.",
    )
    parser.add_argument(
        "--guess-survival-mine-weight",
        type=float,
        default=4.0,
        help="Relative weight for mined cells in --guess-survival-coef supervision.",
    )
    parser.add_argument(
        "--guess-survival-topk",
        type=int,
        default=32,
        help="Top-K safe and mine OPEN logits used by guess survival ranking; 0 means all legal cells.",
    )
    parser.add_argument(
        "--guess-survival-margin",
        type=float,
        default=0.05,
        help="Margin for ranking mined OPEN cells below safe OPEN cells.",
    )
    parser.add_argument(
        "--behavior-mine-demotion-coef",
        type=float,
        default=0.0,
        help="Narrow offline loss that demotes only the behavior OPEN cell when it was a mine.",
    )
    parser.add_argument(
        "--behavior-mine-demotion-topk",
        type=int,
        default=4,
        help="Number of better counterfactual OPEN alternatives used for behavior-mine demotion; 0 means all.",
    )
    parser.add_argument(
        "--behavior-mine-demotion-margin",
        type=float,
        default=0.1,
        help="Minimum counterfactual value advantage required before demoting a mined behavior action.",
    )
    parser.add_argument("--guess-supervision-topk", type=int, default=8)
    parser.add_argument("--guess-imitation-weight", type=float, default=1.4)
    parser.add_argument("--endgame-safe-left", type=int, default=60)
    parser.add_argument("--endgame-weight", type=int, default=2)
    parser.add_argument("--guess-weight", type=int, default=3)
    parser.add_argument("--edge-weight", type=int, default=2)
    parser.add_argument("--corner-weight", type=int, default=4)
    parser.add_argument("--wrong-flag-weight", type=int, default=4)
    parser.add_argument("--terminal-weight", type=int, default=3)
    parser.add_argument("--freeze-backbone", action="store_true")
    parser.add_argument("--freeze-long-range", action="store_true")
    parser.add_argument(
        "--train-policy-only",
        action="store_true",
        help="Freeze all model parameters except policy_global and policy_head.",
    )
    parser.add_argument(
        "--no-train-policy-only",
        dest="train_policy_only",
        action="store_false",
        help="Allow the full model to train instead of only policy_global and policy_head.",
    )
    parser.add_argument(
        "--train-counterfactual-value-only",
        action="store_true",
        help="Freeze all model parameters except the counterfactual value head.",
    )
    parser.add_argument(
        "--train-risk-head-only",
        action="store_true",
        help="Freeze all model parameters except the learned risk head.",
    )
    parser.add_argument("--reset-optimizer", action="store_true")
    parser.add_argument("--include-plain-tail", action="store_true")
    parser.add_argument("--feedback-plan", type=Path)
    parser.add_argument("--replay-log-dir", type=Path)
    parser.add_argument("--replay-log-limit", type=int, default=8192)
    parser.add_argument("--extreme-dataset", type=Path)
    parser.add_argument(
        "--extreme-family-filter",
        action="append",
        default=[],
        help="Keep only the requested extreme family from the bootstrap extreme dataset. Repeat to include multiple families.",
    )
    parser.add_argument(
        "--extreme-safe-left-min",
        type=int,
        default=None,
        help="Keep only extreme bootstrap states with at least this many safe cells left.",
    )
    parser.add_argument(
        "--extreme-safe-left-max",
        type=int,
        default=None,
        help="Keep only extreme bootstrap states with at most this many safe cells left.",
    )
    parser.add_argument(
        "--extreme-behavior-mine-only",
        action="store_true",
        help="Keep only extreme bootstrap states where the recorded behavior OPEN hit a mine.",
    )
    parser.add_argument(
        "--extreme-min-behavior-regret",
        type=float,
        default=None,
        help="Keep only extreme bootstrap states whose behavior OPEN is at least this far below the best labelled OPEN value.",
    )
    parser.add_argument("--include-noisy-replay-paths", action="store_true")
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    explicit_overrides = explicit_cli_destinations(sys.argv[1:])
    args = parser.parse_args()

    if args.feedback_plan:
        apply_feedback_plan(args, args.feedback_plan, explicit_overrides=explicit_overrides)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    trainer = load_checkpoint(args.checkpoint, device=args.device)
    configure_trainer(trainer, args)
    trainer.model.train()
    teacher_trainers: list[MinesweeperTrainer] = []
    teacher_paths: list[Path] = []
    if args.teacher_checkpoint is not None:
        teacher_paths.append(args.teacher_checkpoint)
    teacher_paths.extend(Path(path) for path in args.teacher_ensemble_checkpoint)
    if teacher_paths and args.teacher_kl_coef > 0.0:
        for teacher_path in teacher_paths:
            teacher_trainer = load_checkpoint(teacher_path, device=args.device)
            configure_trainer(teacher_trainer, args)
            teacher_trainer.model.eval()
            teacher_trainers.append(teacher_trainer)
    teacher_weights = normalize_teacher_weights(
        args.teacher_ensemble_weight,
        expected=len(teacher_trainers),
    )

    rng = np.random.default_rng(args.seed)
    bootstrap_replay: list[EpisodeTransition] = []
    bootstrap_report: dict[str, Any] | None = None
    extreme_replay: list[EpisodeTransition] = []
    extreme_report: dict[str, Any] | None = None
    if args.extreme_dataset:
        extreme_replay, extreme_report = load_extreme_dataset(args.extreme_dataset)
        extreme_replay = filter_extreme_replay(
            extreme_replay,
            trainer=trainer,
            family_filter=args.extreme_family_filter,
            safe_left_min=args.extreme_safe_left_min,
            safe_left_max=args.extreme_safe_left_max,
            behavior_mine_only=args.extreme_behavior_mine_only,
            min_behavior_regret=args.extreme_min_behavior_regret,
            safe_left_threshold=args.endgame_safe_left,
        )
        if extreme_report is not None:
            metadata = extreme_report.get("metadata", extreme_report)
            if isinstance(metadata, dict):
                metadata = dict(metadata)
                metadata.update(
                    {
                        "family_filter": list(args.extreme_family_filter or []),
                        "safe_left_min": args.extreme_safe_left_min,
                        "safe_left_max": args.extreme_safe_left_max,
                        "behavior_mine_only": bool(args.extreme_behavior_mine_only),
                        "min_behavior_regret": args.extreme_min_behavior_regret,
                        "selected_extreme_records": len(extreme_replay),
                    }
                )
                extreme_report["metadata"] = metadata
    if args.replay_log_dir:
        bootstrap_replay, bootstrap_report = load_windows_replay_transitions(
            trainer,
            args.replay_log_dir,
            limit=args.replay_log_limit,
            exclude_noisy_paths=not args.include_noisy_replay_paths,
        )
    if extreme_replay:
        bootstrap_replay = extreme_replay + bootstrap_replay
    replay: list[EpisodeTransition] = []
    best_metrics = evaluate(trainer, args)
    best_win_rate = float(best_metrics["win_rate"])
    history: list[dict[str, Any]] = [
        make_history_item(
            0,
            best_metrics,
            None,
            None,
            len(replay),
            len(bootstrap_replay),
            bootstrap_report,
            extreme_report,
        )
    ]
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
            "bootstrap_replay_size": len(bootstrap_replay),
            "extreme_replay_size": len(extreme_replay),
        },
    )
    print(json.dumps(history[-1], ensure_ascii=False, separators=(",", ":")))

    for round_index in range(1, args.rounds + 1):
        mine_seed = int(args.seed + round_index * 100000)
        mined, mine_metrics = mine_hard_transitions(trainer, args, seed=mine_seed)
        replay.extend(mined)
        if len(replay) > args.replay_size:
            del replay[: len(replay) - args.replay_size]

        losses = run_refinement_updates(
            trainer,
            args,
            rng,
            replay,
            round_seed=mine_seed + 17,
            bootstrap_replay=bootstrap_replay,
            teacher_trainers=teacher_trainers,
            teacher_weights=teacher_weights,
        )
        metrics = evaluate(trainer, args)
        item = make_history_item(
            round_index,
            metrics,
            mine_metrics,
            losses,
            len(replay),
            len(bootstrap_replay),
            bootstrap_report,
            extreme_report,
        )
        history.append(item)
        write_json(args.output_dir / f"round_{round_index:03d}.json", item)
        print(json.dumps(item, ensure_ascii=False, separators=(",", ":")))
        if args.save_last_path is not None:
            save_checkpoint(
                args.save_last_path,
                trainer,
                episode=round_index,
                metrics={
                    **metrics,
                    "hard_refine_round": round_index,
                    "source_checkpoint": str(args.checkpoint),
                    "hard_replay_size": len(replay),
                    "bootstrap_replay_size": len(bootstrap_replay),
                    "extreme_replay_size": len(extreme_replay),
                    "save_reason": "last_round",
                },
            )

        win_rate = float(metrics["win_rate"])
        if win_rate > best_win_rate:
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
                    "bootstrap_replay_size": len(bootstrap_replay),
                    "extreme_replay_size": len(extreme_replay),
                },
            )

    write_json(
        args.output_dir / "summary.json",
        {
            "config": vars_for_json(args),
            "best_win_rate": best_win_rate,
            "history": history,
            "save_path": str(args.save_path),
            "bootstrap_replay_report": bootstrap_report,
            "extreme_replay_report": extreme_report,
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
    trainer.config.counterfactual_coef = float(getattr(args, "counterfactual_coef", 0.0))
    trainer.config.counterfactual_margin = float(getattr(args, "counterfactual_margin", 0.1))
    trainer.config.counterfactual_temperature = float(getattr(args, "counterfactual_temperature", 0.08))
    trainer.config.counterfactual_policy_coef = float(getattr(args, "counterfactual_policy_coef", 0.0))
    trainer.config.counterfactual_policy_temperature = float(getattr(args, "counterfactual_policy_temperature", 0.25))
    trainer.config.counterfactual_policy_topk = int(getattr(args, "counterfactual_policy_topk", 16))
    trainer.config.counterfactual_policy_min_gap = float(getattr(args, "counterfactual_policy_min_gap", 0.0))
    trainer.config.counterfactual_policy_full_action = bool(
        getattr(args, "counterfactual_policy_full_action", False)
    )
    trainer.config.counterfactual_only_guess = bool(
        getattr(args, "counterfactual_only_guess", False)
    )
    trainer.config.counterfactual_value_head_coef = float(
        getattr(args, "counterfactual_value_head_coef", 0.0)
    )
    trainer.config.counterfactual_value_head_clip = float(
        getattr(args, "counterfactual_value_head_clip", 4.0)
    )
    trainer.config.counterfactual_value_rank_coef = float(
        getattr(args, "counterfactual_value_rank_coef", 0.0)
    )
    trainer.config.counterfactual_value_target_mode = str(
        getattr(args, "counterfactual_value_target_mode", "raw")
    )
    trainer.config.mine_aux_coef = args.mine_aux_coef
    trainer.config.risk_supervision_coef = args.risk_supervision_coef
    trainer.config.risk_head_coef = args.risk_head_coef
    trainer.config.risk_temperature = args.risk_temperature
    trainer.config.counterfactual_risk_head_coef = float(getattr(args, "counterfactual_risk_head_coef", 0.0))
    trainer.config.counterfactual_risk_temperature = float(getattr(args, "counterfactual_risk_temperature", 1.0))
    trainer.config.counterfactual_risk_loss_mode = str(
        getattr(args, "counterfactual_risk_loss_mode", "bce")
    )
    trainer.config.guess_survival_coef = float(getattr(args, "guess_survival_coef", 0.0))
    trainer.config.guess_survival_mine_weight = float(getattr(args, "guess_survival_mine_weight", 4.0))
    trainer.config.guess_survival_topk = int(getattr(args, "guess_survival_topk", 32))
    trainer.config.guess_survival_margin = float(getattr(args, "guess_survival_margin", 0.05))
    trainer.config.behavior_mine_demotion_coef = float(
        getattr(args, "behavior_mine_demotion_coef", 0.0)
    )
    trainer.config.behavior_mine_demotion_topk = int(
        getattr(args, "behavior_mine_demotion_topk", 4)
    )
    trainer.config.behavior_mine_demotion_margin = float(
        getattr(args, "behavior_mine_demotion_margin", 0.1)
    )
    trainer.config.guess_supervision_topk = args.guess_supervision_topk
    trainer.config.guess_imitation_weight = args.guess_imitation_weight
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.config.risk_head_weight = 0.0
    trainer.solver = MinesweeperSolver(exact_limit=args.exact_limit)
    for group in trainer.optimizer.param_groups:
        group["lr"] = args.lr
    if args.freeze_backbone:
        for param in trainer.model.backbone.parameters():
            param.requires_grad = False
    if args.freeze_long_range and hasattr(trainer.model, "long_range"):
        for param in trainer.model.long_range.parameters():
            param.requires_grad = False
    train_policy_only = bool(getattr(args, "train_policy_only", False))
    if train_policy_only:
        for name, param in trainer.model.named_parameters():
            param.requires_grad = name.startswith("policy_global.") or name.startswith("policy_head.")
    train_counterfactual_value_only = bool(getattr(args, "train_counterfactual_value_only", False))
    if train_counterfactual_value_only:
        for name, param in trainer.model.named_parameters():
            param.requires_grad = name.startswith("counterfactual_head.")
    train_risk_head_only = bool(getattr(args, "train_risk_head_only", False))
    if train_risk_head_only:
        for name, param in trainer.model.named_parameters():
            param.requires_grad = name.startswith("risk_head.") or name.startswith("risk_global.")
    if (
        args.reset_optimizer
        or args.freeze_backbone
        or args.freeze_long_range
        or train_policy_only
        or train_counterfactual_value_only
        or train_risk_head_only
    ):
        trainable = [param for param in trainer.model.parameters() if param.requires_grad]
        if not trainable:
            raise ValueError("no trainable parameters remain after applying freeze options")
        trainer.optimizer = torch.optim.Adam(
            trainable,
            lr=args.lr,
        )


def explicit_cli_destinations(argv: list[str]) -> set[str]:
    destinations: set[str] = set()
    for token in argv:
        if not token.startswith("--"):
            continue
        option = token[2:].split("=", 1)[0]
        if not option:
            continue
        destination = option.replace("-", "_")
        destinations.add(destination)
        if destination.startswith("no_"):
            destinations.add(destination[3:])
    return destinations


def apply_feedback_plan(
    args: argparse.Namespace,
    path: Path,
    *,
    explicit_overrides: set[str] | None = None,
) -> None:
    data = load_json(path)
    params = data.get("hard_loss_refine_args") or {}
    if not isinstance(params, dict):
        raise TypeError(f"feedback plan missing hard_loss_refine_args mapping: {path}")
    explicit_overrides = set(explicit_overrides or set())
    for name, value in params.items():
        if name in explicit_overrides:
            continue
        if hasattr(args, name):
            current = getattr(args, name)
            if isinstance(current, bool):
                setattr(args, name, bool(value))
            elif isinstance(current, int) and not isinstance(current, bool):
                setattr(args, name, int(value))
            elif isinstance(current, float):
                setattr(args, name, float(value))
            elif isinstance(current, Path):
                setattr(args, name, Path(value))
            else:
                setattr(args, name, value)


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
                model_open_candidates = _top_open_candidate_indices(
                    scores[local_index].detach().cpu().numpy(),
                    action_masks[local_index],
                    rows=game.rows,
                    cols=game.cols,
                    topk=_counterfactual_model_topk(args),
                )
                tails[game_index].append(
                    HardState(
                        clone_game(game),
                        action_index,
                        action,
                        model_open_candidates=model_open_candidates,
                    )
                )
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
    counterfactual_open_values = None
    if bool(getattr(args, "counterfactual_labels", False)) and expert_is_guess:
        behavior_candidates = _open_candidate_indices(
            state.action_index,
            rows=game.rows,
            cols=game.cols,
        )
        model_open_candidates = getattr(state, "model_open_candidates", np.empty(0, dtype=np.int64))
        candidate_indices = np.unique(
            np.concatenate(
                [
                    behavior_candidates,
                    np.asarray(model_open_candidates, dtype=np.int64).reshape(-1),
                ]
            )
        )
        counterfactual_open_values = build_counterfactual_open_values(
            game,
            action_mask,
            snapshot,
            solver,
            topk=max(1, int(getattr(args, "counterfactual_topk", 64))),
            candidate_indices=candidate_indices,
            include_all_open=bool(getattr(args, "counterfactual_all_open", False)),
        )
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
        counterfactual_open_values=counterfactual_open_values,
        expert_is_guess=expert_is_guess,
    )
    extreme_profile = transition_extreme_profile(
        transition,
        mines=trainer.config.mines,
        safe_left_threshold=args.endgame_safe_left,
    )
    transition.extreme_score = float(extreme_profile["score"])
    transition.extreme_family = str(extreme_profile["family"])
    tags = {
        "terminal": terminal,
        "wrong_flags": wrong_flags,
        "forced_available": forced_available,
        "endgame": safe_left <= args.endgame_safe_left,
        "edge": bool(extreme_profile["edge"]),
        "corner": bool(extreme_profile["corner"]),
        "guess": bool(extreme_profile["guess"]),
        "safe_left": safe_left,
    }
    is_hard = bool(tags["terminal"] or tags["wrong_flags"] or tags["forced_available"] or tags["endgame"] or tags["guess"])
    if not args.include_plain_tail and not is_hard:
        return None, tags
    return transition, tags


def filter_extreme_replay(
    transitions: list[EpisodeTransition],
    *,
    trainer: MinesweeperTrainer,
    family_filter: list[str] | tuple[str, ...] | None = None,
    safe_left_min: int | None = None,
    safe_left_max: int | None = None,
    behavior_mine_only: bool = False,
    min_behavior_regret: float | None = None,
    safe_left_threshold: int,
) -> list[EpisodeTransition]:
    families = set(family_filter or [])
    filtered: list[EpisodeTransition] = []
    for transition in transitions:
        profile = transition_extreme_profile(
            transition,
            mines=trainer.config.mines,
            safe_left_threshold=safe_left_threshold,
        )
        family = str(profile["family"])
        safe_left = int(profile["safe_left"])
        if families and family not in families:
            continue
        if safe_left_min is not None and safe_left < int(safe_left_min):
            continue
        if safe_left_max is not None and safe_left > int(safe_left_max):
            continue
        if behavior_mine_only and not _transition_behavior_is_mine(transition):
            continue
        if min_behavior_regret is not None:
            regret = _transition_behavior_regret(transition)
            if regret is None or regret < float(min_behavior_regret):
                continue
        filtered.append(transition)
    return filtered


def _transition_behavior_is_mine(transition: EpisodeTransition) -> bool:
    if transition.mine_mask is None or transition.action_mask.ndim != 3:
        return False
    rows = int(transition.mine_mask.shape[0])
    cols = int(transition.mine_mask.shape[1])
    kind_index, cell_index = divmod(int(transition.action_index), rows * cols)
    if kind_index != action_channel(ActionType.OPEN):
        return False
    row, col = divmod(cell_index, cols)
    if not (0 <= row < rows and 0 <= col < cols):
        return False
    return bool(np.asarray(transition.mine_mask, dtype=bool)[row, col])


def _transition_behavior_regret(transition: EpisodeTransition) -> float | None:
    values = transition.counterfactual_open_values
    if values is None or transition.action_mask.ndim != 3:
        return None
    rows, cols = np.asarray(values).shape
    kind_index, cell_index = divmod(int(transition.action_index), int(rows) * int(cols))
    if kind_index != action_channel(ActionType.OPEN):
        return None
    row, col = divmod(cell_index, int(cols))
    if not (0 <= row < int(rows) and 0 <= col < int(cols)):
        return None
    values_array = np.asarray(values, dtype=np.float32)
    behavior_value = float(values_array[row, col])
    if not np.isfinite(behavior_value):
        return None
    open_mask = np.asarray(transition.action_mask[action_channel(ActionType.OPEN)], dtype=bool)
    valid = open_mask & np.isfinite(values_array)
    if not bool(valid.any()):
        return None
    return float(np.max(values_array[valid]) - behavior_value)


def _open_candidate_indices(action_index: int, *, rows: int, cols: int) -> np.ndarray:
    """Return the cell index for a behavior action when it is an OPEN."""

    cells = int(rows) * int(cols)
    kind_index, cell_index = divmod(int(action_index), cells)
    if kind_index != action_channel(ActionType.OPEN):
        return np.empty(0, dtype=np.int64)
    return np.asarray([cell_index], dtype=np.int64)


def _top_open_candidate_indices(
    scores: np.ndarray,
    action_mask: np.ndarray,
    *,
    rows: int,
    cols: int,
    topk: int,
) -> np.ndarray:
    """Return the policy's current top OPEN cell indices for offline labelling."""

    cells = int(rows) * int(cols)
    open_channel = action_channel(ActionType.OPEN)
    flat_scores = np.asarray(scores, dtype=np.float32).reshape(-1)
    if flat_scores.size < (open_channel + 1) * cells:
        return np.empty(0, dtype=np.int64)
    legal = np.asarray(action_mask[open_channel], dtype=bool).reshape(-1)
    if not bool(legal.any()):
        return np.empty(0, dtype=np.int64)
    open_scores = flat_scores[open_channel * cells : (open_channel + 1) * cells].copy()
    valid = legal & np.isfinite(open_scores)
    if not bool(valid.any()):
        return np.empty(0, dtype=np.int64)
    valid_indices = np.flatnonzero(valid)
    count = min(max(1, int(topk)), int(valid_indices.size))
    order = np.argsort(open_scores[valid_indices], kind="stable")[::-1][:count]
    return valid_indices[order].astype(np.int64, copy=False)


def _counterfactual_model_topk(args: Any) -> int:
    configured = getattr(args, "counterfactual_model_topk", None)
    if configured is None:
        configured = getattr(args, "counterfactual_topk", 64)
    return max(1, int(configured))


def transition_weight(tags: dict[str, Any], args: argparse.Namespace) -> int:
    weight = 1
    if tags.get("endgame"):
        weight = max(weight, int(args.endgame_weight))
    if tags.get("terminal"):
        weight = max(weight, int(args.terminal_weight))
    if tags.get("guess"):
        weight = max(weight, int(args.guess_weight))
    if tags.get("edge"):
        weight = max(weight, int(args.edge_weight))
    if tags.get("corner"):
        weight = max(weight, int(args.corner_weight))
    if int(tags.get("wrong_flags", 0)) > 0:
        weight = max(weight, int(args.wrong_flag_weight))
    return max(1, weight)


def build_counterfactual_open_values(
    game: MinesweeperGame,
    action_mask: np.ndarray,
    snapshot: Any,
    solver: MinesweeperSolver,
    *,
    topk: int,
    candidate_indices: np.ndarray | list[int] | None = None,
    include_all_open: bool = False,
) -> np.ndarray:
    """Score observable OPEN candidates by replaying them on the known board.

    This is an offline training label only. At inference the model receives no
    mine layout or counterfactual map.
    """
    open_channel = action_channel(ActionType.OPEN)
    legal = np.asarray(action_mask[open_channel], dtype=bool)
    values = np.full((game.rows, game.cols), np.nan, dtype=np.float32)
    candidates = np.flatnonzero(legal.reshape(-1))
    risk_candidates = np.empty(0, dtype=np.int64)
    if snapshot is not None and snapshot.risk_map is not None and candidates.size > topk:
        risks = np.asarray(snapshot.risk_map, dtype=np.float32).reshape(-1)[candidates]
        order = np.argsort(np.where(np.isfinite(risks), risks, np.inf), kind="stable")
        risk_candidates = candidates[order[:topk]]
    elif candidates.size > topk:
        risk_candidates = candidates[:topk]
    else:
        risk_candidates = candidates

    # Keep the behavior/model candidates in the offline label set. Without
    # this union, a failure action can be absent from the labels entirely,
    # so the policy never receives a direct signal to move away from it.
    requested = np.asarray(candidate_indices if candidate_indices is not None else [], dtype=np.int64).reshape(-1)
    legal_flat = legal.reshape(-1)
    requested = requested[(requested >= 0) & (requested < legal_flat.size)]
    requested = requested[legal_flat[requested]] if requested.size else requested
    full_expand_candidates = np.unique(
        np.concatenate([risk_candidates, requested])
    ).astype(np.int64, copy=False)
    if include_all_open:
        candidates = np.flatnonzero(legal_flat).astype(np.int64, copy=False)
    else:
        candidates = full_expand_candidates
    full_expand_set = set(int(index) for index in full_expand_candidates.tolist())

    base_safe = int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum())
    for flat_index in candidates:
        row, col = divmod(int(flat_index), game.cols)
        candidate = clone_game(game)
        _, _, _, info = candidate.open_cell(row, col)
        if bool(info.get("hit_mine", False)):
            value = -1.0
        elif candidate.won:
            value = 2.0
        else:
            final_safe = int((candidate.revealed & ~candidate.mines).sum()) if candidate.mines_placed else int(candidate.revealed.sum())
            delta_safe = max(0, final_safe - base_safe)
            if include_all_open and int(flat_index) not in full_expand_set:
                # Tier-2 labels keep the complete legal OPEN action set without
                # paying for exact solver expansion on every ordinary candidate.
                # The visible risk prior is intentionally small and only breaks
                # ties between candidates with similar immediate reveals.
                visible_risk = (
                    float(snapshot.risk_map[row, col])
                    if snapshot is not None and snapshot.risk_map is not None
                    else 0.5
                )
                value = 0.035 * float(delta_safe) + 0.08 * float(
                    np.clip(1.0 - visible_risk, 0.0, 1.0)
                )
                values[row, col] = np.float32(value)
                continue

            snapshot_after, forced_reward, forced_steps = resolve_forced_moves(candidate, solver)
            future_bonus = 0.0
            if not candidate.done and snapshot_after.best_guess_risk is not None:
                future_bonus = float(np.clip(1.0 - float(snapshot_after.best_guess_risk), 0.0, 1.0))
            value = (
                0.035 * float(delta_safe)
                + 0.02 * float(forced_steps)
                + 0.5 * max(0.0, float(forced_reward))
                + 0.25 * future_bonus
            )
        values[row, col] = np.float32(value)
    return values


def run_refinement_updates(
    trainer: MinesweeperTrainer,
    args: argparse.Namespace,
    rng: np.random.Generator,
    replay: list[EpisodeTransition],
    round_seed: int,
    bootstrap_replay: list[EpisodeTransition] | None = None,
    teacher_trainers: list[MinesweeperTrainer] | None = None,
    teacher_weights: list[float] | None = None,
) -> dict[str, Any]:
    losses: list[dict[str, float]] = []
    imitation_pool = list(bootstrap_replay or [])
    imitation_pool.extend(replay)
    if imitation_pool:
        for _ in range(args.imitation_updates):
            batch = sample_transitions(imitation_pool, args.batch_size, rng)
            if teacher_trainers and float(args.teacher_kl_coef) > 0.0:
                losses.append(
                    update_imitation_with_teacher_kl(
                        trainer,
                        teacher_trainers,
                        batch,
                        args,
                        teacher_weights=teacher_weights,
                    )
                )
            else:
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


def update_imitation_with_teacher_kl(
    trainer: MinesweeperTrainer,
    teacher_trainers: list[MinesweeperTrainer],
    transitions: list[EpisodeTransition],
    args: argparse.Namespace,
    *,
    teacher_weights: list[float] | None = None,
) -> dict[str, float]:
    transitions = trainer._augment_transitions(transitions)
    boards = torch.tensor(np.stack([t.board for t in transitions]), dtype=torch.float32, device=trainer.device)
    global_features = torch.tensor(
        np.stack([t.global_features for t in transitions]),
        dtype=torch.float32,
        device=trainer.device,
    )
    masks = torch.tensor(np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=trainer.device)
    logits, values, risk_logits, counterfactual_values = trainer.model.forward_with_aux(
        boards,
        global_features,
    )
    del values
    flat_logits = logits.view(logits.shape[0], -1)
    flat_masks = masks.view(masks.shape[0], -1)
    masked_logits = flat_logits.masked_fill(~flat_masks, -1e9)

    imitation_loss = trainer._expert_policy_loss(masked_logits, transitions, allow_action_fallback=True)
    counterfactual_loss = trainer._counterfactual_preference_loss(masked_logits, transitions)
    counterfactual_policy_loss = trainer._counterfactual_policy_loss(logits, masks, transitions)
    counterfactual_value_head_loss = trainer._counterfactual_value_head_loss(
        counterfactual_values,
        masks,
        transitions,
    )
    mine_aux_loss = trainer._mine_auxiliary_loss(logits, masks, transitions)
    risk_supervision_loss = trainer._risk_supervision_loss(logits, masks, transitions)
    risk_head_loss = trainer._risk_head_loss(risk_logits, masks, transitions)
    counterfactual_risk_head_loss = trainer._counterfactual_risk_head_loss(risk_logits, masks, transitions)
    guess_survival_loss = trainer._guess_survival_loss(logits, masks, transitions)
    behavior_mine_demotion_loss = trainer._behavior_mine_demotion_loss(logits, masks, transitions)
    teacher_kl_loss = teacher_policy_kl_loss(
        student_logits=masked_logits,
        teacher_trainers=teacher_trainers,
        teacher_weights=teacher_weights,
        boards=boards,
        global_features=global_features,
        flat_masks=flat_masks,
        temperature=max(1e-4, float(args.teacher_temperature)),
    )
    teacher_prior_policy_loss = teacher_prior_counterfactual_policy_loss(
        student_logits=masked_logits,
        flat_masks=flat_masks,
        transitions=transitions,
        teacher_trainers=teacher_trainers,
        teacher_weights=teacher_weights,
        boards=boards,
        global_features=global_features,
        strength=float(args.teacher_prior_strength),
        temperature=max(1e-4, float(args.teacher_prior_temperature)),
        objective=str(args.teacher_prior_objective),
        clip_epsilon=float(args.teacher_prior_clip),
        teacher_topk=int(args.teacher_prior_policy_topk),
    )

    loss = imitation_loss * trainer.config.pretrain_imitation_coef
    loss = loss + trainer.config.counterfactual_coef * counterfactual_loss
    loss = loss + trainer.config.counterfactual_policy_coef * counterfactual_policy_loss
    loss = loss + float(args.teacher_prior_policy_coef) * teacher_prior_policy_loss
    loss = loss + trainer.config.counterfactual_value_head_coef * counterfactual_value_head_loss
    loss = loss + trainer.config.mine_aux_coef * mine_aux_loss
    loss = loss + trainer.config.risk_supervision_coef * risk_supervision_loss
    loss = loss + trainer.config.risk_head_coef * risk_head_loss
    loss = loss + trainer.config.counterfactual_risk_head_coef * counterfactual_risk_head_loss
    loss = loss + trainer.config.guess_survival_coef * guess_survival_loss
    loss = loss + trainer.config.behavior_mine_demotion_coef * behavior_mine_demotion_loss
    loss = loss + float(args.teacher_kl_coef) * teacher_kl_loss
    entropy = Categorical(logits=masked_logits).entropy().mean()

    trainer.optimizer.zero_grad(set_to_none=True)
    loss.backward()
    torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), trainer.config.grad_clip)
    trainer.optimizer.step()

    return {
        "loss": float(loss.detach().cpu()),
        "policy_loss": 0.0,
        "value_loss": 0.0,
        "solver_imitation_loss": float(imitation_loss.detach().cpu()),
        "counterfactual_loss": float(counterfactual_loss.detach().cpu()),
        "counterfactual_policy_loss": float(counterfactual_policy_loss.detach().cpu()),
        "teacher_prior_policy_loss": float(teacher_prior_policy_loss.detach().cpu()),
        "counterfactual_value_head_loss": float(counterfactual_value_head_loss.detach().cpu()),
        "teacher_kl_loss": float(teacher_kl_loss.detach().cpu()),
        "mine_aux_loss": float(mine_aux_loss.detach().cpu()),
        "risk_supervision_loss": float(risk_supervision_loss.detach().cpu()),
        "risk_head_loss": float(risk_head_loss.detach().cpu()),
        "counterfactual_risk_head_loss": float(counterfactual_risk_head_loss.detach().cpu()),
        "guess_survival_loss": float(guess_survival_loss.detach().cpu()),
        "behavior_mine_demotion_loss": float(behavior_mine_demotion_loss.detach().cpu()),
        "entropy": float(entropy.detach().cpu()),
    }


def teacher_policy_kl_loss(
    *,
    student_logits: torch.Tensor,
    teacher_trainer: MinesweeperTrainer | None = None,
    teacher_trainers: list[MinesweeperTrainer] | None = None,
    teacher_weights: list[float] | None = None,
    teacher_probs: torch.Tensor | None = None,
    boards: torch.Tensor,
    global_features: torch.Tensor,
    flat_masks: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    with torch.no_grad():
        if teacher_probs is None:
            teacher_probs = build_teacher_policy_probabilities(
                teacher_trainer=teacher_trainer,
                teacher_trainers=teacher_trainers,
                teacher_weights=teacher_weights,
                boards=boards,
                global_features=global_features,
                flat_masks=flat_masks,
                temperature=temperature,
            )
        else:
            teacher_probs = teacher_probs.to(device=student_logits.device, dtype=student_logits.dtype)
    masked_student_logits = student_logits.masked_fill(~flat_masks, -1e9)
    student_log_probs = F.log_softmax(masked_student_logits / temperature, dim=1)
    return F.kl_div(student_log_probs, teacher_probs, reduction="batchmean") * (temperature * temperature)


def build_teacher_policy_probabilities(
    *,
    teacher_trainer: MinesweeperTrainer | None = None,
    teacher_trainers: list[MinesweeperTrainer] | None = None,
    teacher_weights: list[float] | None = None,
    boards: torch.Tensor,
    global_features: torch.Tensor,
    flat_masks: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    trainers = list(teacher_trainers or [])
    if teacher_trainer is not None:
        trainers.insert(0, teacher_trainer)
    if not trainers:
        raise ValueError("at least one teacher trainer is required")

    rows, cols = trainers[0].config.rows, trainers[0].config.cols
    channels = flat_masks.shape[1] // (rows * cols)
    if channels <= 0 or channels * rows * cols != flat_masks.shape[1]:
        raise ValueError("flat_masks shape is incompatible with the teacher board shape")
    masks = flat_masks.detach().cpu().numpy().reshape(flat_masks.shape[0], channels, rows, cols)
    board_array = boards.detach().cpu().numpy()
    global_array = global_features.detach().cpu().numpy()
    weights = np.ones(len(trainers), dtype=np.float32) if teacher_weights is None else np.asarray(
        teacher_weights,
        dtype=np.float32,
    )
    if weights.size != len(trainers):
        raise ValueError("teacher_weights must match the number of teacher trainers")
    weights = np.clip(weights, 0.0, np.inf)
    if not bool(weights.sum() > 0.0):
        weights.fill(1.0)
    weights /= weights.sum()

    teacher_parts: list[torch.Tensor] = []
    with torch.no_grad():
        for teacher in trainers:
            scores = teacher._predict_policy_scores_batch(
                boards=board_array,
                global_features_batch=global_array,
                action_masks=masks,
                use_flip_ensemble=bool(teacher.config.inference_augment_flips),
            )
            scores = scores.to(device=flat_masks.device, dtype=flat_masks.dtype if flat_masks.is_floating_point() else torch.float32)
            if (
                teacher.config.inference_augment_flips
                and teacher.config.inference_ensemble == "probs"
            ):
                teacher_logits = torch.where(
                    flat_masks,
                    torch.log(scores.clamp_min(1e-8)),
                    torch.full_like(scores, -1e9),
                )
            else:
                teacher_logits = scores.masked_fill(~flat_masks, -1e9)
            teacher_parts.append(F.softmax(teacher_logits / max(1e-4, float(temperature)), dim=1))

        teacher_probs = sum(
            (part * float(weight) for part, weight in zip(teacher_parts, weights, strict=True)),
            torch.zeros_like(teacher_parts[0]),
        )
        teacher_probs = teacher_probs.masked_fill(~flat_masks, 0.0)
        teacher_probs = teacher_probs / teacher_probs.sum(dim=1, keepdim=True).clamp_min(1e-8)
    return teacher_probs


def teacher_prior_counterfactual_policy_loss(
    *,
    student_logits: torch.Tensor,
    flat_masks: torch.Tensor,
    transitions: list[EpisodeTransition],
    teacher_trainer: MinesweeperTrainer | None = None,
    teacher_trainers: list[MinesweeperTrainer] | None = None,
    teacher_weights: list[float] | None = None,
    boards: torch.Tensor,
    global_features: torch.Tensor,
    strength: float,
    temperature: float,
    objective: str = "target",
    clip_epsilon: float = 0.2,
    teacher_topk: int = 0,
) -> torch.Tensor:
    if strength <= 0.0:
        return student_logits.new_tensor(0.0)
    strength = float(np.clip(strength, 0.0, 1.0))
    trainers = list(teacher_trainers or [])
    if teacher_trainer is not None:
        trainers.insert(0, teacher_trainer)
    if not trainers:
        raise ValueError("at least one teacher trainer is required")
    if objective not in {"target", "ppo"}:
        raise ValueError(f"unknown teacher-prior objective {objective!r}")
    if not 0.0 <= float(clip_epsilon) < 1.0:
        raise ValueError("teacher-prior clip must be in [0, 1)")
    teacher_probs = build_teacher_policy_probabilities(
        teacher_trainer=teacher_trainer,
        teacher_trainers=teacher_trainers,
        teacher_weights=teacher_weights,
        boards=boards,
        global_features=global_features,
        flat_masks=flat_masks,
        temperature=1.0,
    )
    rows, cols = trainers[0].config.rows, trainers[0].config.cols
    cells = rows * cols
    open_start = action_channel(ActionType.OPEN) * cells
    student_log_probs = F.log_softmax(student_logits.masked_fill(~flat_masks, -1e9), dim=1)
    losses: list[torch.Tensor] = []
    row_weights: list[float] = []
    for row_index, transition in enumerate(transitions):
        values = transition.counterfactual_open_values
        if values is None:
            continue
        flat_values = np.asarray(values, dtype=np.float32).reshape(-1)
        valid = transition.action_mask[action_channel(ActionType.OPEN)].reshape(-1) & np.isfinite(flat_values)
        valid_indices = np.flatnonzero(valid)
        if valid_indices.size < 2:
            continue

        indices = torch.tensor(open_start + valid_indices, dtype=torch.long, device=student_logits.device)
        teacher_open = teacher_probs[row_index, indices]
        if teacher_topk > 0 and valid_indices.size > int(teacher_topk):
            keep = torch.argsort(teacher_open, descending=True)[: int(teacher_topk)]
            indices = indices[keep]
            teacher_open = teacher_open[keep]
            selected_mask = keep.detach().cpu().numpy()
            valid_indices = valid_indices[selected_mask]
        if valid_indices.size < 2:
            continue
        teacher_mass = teacher_open.sum()
        if float(teacher_mass.detach().cpu()) <= 1e-8:
            continue

        selected_values = torch.tensor(
            flat_values[valid_indices],
            dtype=student_logits.dtype,
            device=student_logits.device,
        )
        teacher_open = teacher_open / teacher_mass.clamp_min(1e-8)
        if objective == "ppo":
            student_open_probs = F.softmax(student_logits[row_index, indices], dim=0)
            advantage = selected_values - (teacher_open.detach() * selected_values).sum()
            advantage = advantage / advantage.std(unbiased=False).clamp_min(1e-3)
            advantage = advantage.clamp(-3.0, 3.0)
            ratio = student_open_probs / teacher_open.detach().clamp_min(1e-4)
            clipped_ratio = ratio.clamp(
                1.0 - float(clip_epsilon),
                1.0 + float(clip_epsilon),
            )
            surrogate = torch.minimum(ratio * advantage, clipped_ratio * advantage)
            losses.append(-float(strength) * (teacher_open.detach() * surrogate).sum())
        else:
            target_open = F.softmax(selected_values / max(1e-4, float(temperature)), dim=0)
            mixed_open = (1.0 - strength) * teacher_open + strength * target_open
            target = teacher_probs[row_index].clone()
            target[indices] = teacher_mass * mixed_open
            target = target / target.sum().clamp_min(1e-8)
            losses.append(-(target * student_log_probs[row_index]).sum())
        row_weights.append(
            float(transition.source_quality)
            * (1.0 + min(2.0, float(transition.extreme_score)))
        )

    if not losses:
        return student_logits.new_tensor(0.0)
    weights = torch.tensor(row_weights, dtype=student_logits.dtype, device=student_logits.device).clamp_min(0.05)
    return (torch.stack(losses) * weights).sum() / weights.sum().clamp_min(1.0)


def normalize_teacher_weights(
    configured: list[float] | None,
    *,
    expected: int,
) -> list[float]:
    if expected <= 0:
        return []
    if not configured:
        return [1.0 / expected] * expected
    weights = np.asarray(configured, dtype=np.float32)
    if weights.size != expected:
        raise ValueError(
            "teacher-ensemble-weight must contain one value for each teacher "
            f"(expected {expected}, got {weights.size})"
        )
    if not np.isfinite(weights).all() or (weights < 0.0).any() or float(weights.sum()) <= 0.0:
        raise ValueError("teacher-ensemble-weight must be finite, non-negative, and sum to a positive value")
    weights = weights / weights.sum()
    return weights.astype(float).tolist()


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
    extreme_transitions = [transition for transition in transitions if transition.extreme_family]
    ordinary_transitions = [transition for transition in transitions if not transition.extreme_family]
    if extreme_transitions and ordinary_transitions:
        extreme_quota = min(len(extreme_transitions), max(1, int(round(batch_size * 0.75))))
        ordinary_quota = max(0, batch_size - extreme_quota)
        chosen = _sample_family_balanced_transitions(extreme_transitions, extreme_quota, rng)
        if ordinary_quota > 0:
            chosen.extend(_sample_weighted_transitions(ordinary_transitions, ordinary_quota, rng))
        rng.shuffle(chosen)
        return chosen
    if extreme_transitions:
        return _sample_family_balanced_transitions(extreme_transitions, batch_size, rng)
    return _sample_weighted_transitions(transitions, batch_size, rng)


def _sample_family_balanced_transitions(
    transitions: list[EpisodeTransition],
    batch_size: int,
    rng: np.random.Generator,
) -> list[EpisodeTransition]:
    if len(transitions) <= batch_size:
        return list(transitions)

    family_weights = {
        "corner_guess_tail": 6.0,
        "edge_guess_tail": 5.0,
        "guess_tail": 4.0,
        "corner_tail": 4.0,
        "edge_tail": 3.0,
        "high_risk": 1.5,
        "tail": 2.0,
        "ordinary": 1.0,
    }
    family_groups: dict[str, list[EpisodeTransition]] = {}
    for transition in transitions:
        family_groups.setdefault(transition.extreme_family or "ordinary", []).append(transition)

    active_families = [
        (family, family_groups[family], family_weights.get(family, 1.0))
        for family in sorted(family_groups)
        if family_groups[family]
    ]
    if not active_families:
        return _sample_weighted_transitions(transitions, batch_size, rng)

    total_weight = float(sum(weight for _, _, weight in active_families))
    if total_weight <= 0.0 or not np.isfinite(total_weight):
        return _sample_weighted_transitions(transitions, batch_size, rng)

    quotas: list[tuple[str, list[EpisodeTransition], int]] = []
    remainders: list[tuple[float, int]] = []
    assigned = 0
    for index, (family, family_transitions, weight) in enumerate(active_families):
        exact = batch_size * weight / total_weight
        quota = int(np.floor(exact))
        quotas.append((family, family_transitions, quota))
        assigned += quota
        remainders.append((exact - quota, index))

    remaining = batch_size - assigned
    for _, index in sorted(remainders, reverse=True):
        if remaining <= 0:
            break
        family, family_transitions, quota = quotas[index]
        quotas[index] = (family, family_transitions, quota + 1)
        remaining -= 1

    chosen: list[EpisodeTransition] = []
    for family, family_transitions, quota in quotas:
        if quota <= 0:
            continue
        chosen.extend(
            _sample_weighted_transitions(
                family_transitions,
                quota,
                rng,
                replace=quota > len(family_transitions),
            )
        )

    if len(chosen) < batch_size:
        chosen.extend(
            _sample_weighted_transitions(
                transitions,
                batch_size - len(chosen),
                rng,
                replace=batch_size - len(chosen) > len(transitions),
            )
        )
    if len(chosen) > batch_size:
        chosen = chosen[:batch_size]
    return chosen


def _sample_weighted_transitions(
    transitions: list[EpisodeTransition],
    batch_size: int,
    rng: np.random.Generator,
    *,
    replace: bool = False,
) -> list[EpisodeTransition]:
    if len(transitions) <= batch_size:
        if not replace:
            return list(transitions)
    weights = np.asarray([transition_sample_weight(transition) for transition in transitions], dtype=np.float64)
    total = float(weights.sum())
    if not np.isfinite(total) or total <= 0.0:
        indices = rng.choice(len(transitions), size=batch_size, replace=replace)
    else:
        indices = rng.choice(len(transitions), size=batch_size, replace=replace, p=weights / total)
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
    def __init__(
        self,
        game: MinesweeperGame,
        action_index: int,
        action: Action,
        *,
        model_open_candidates: np.ndarray | None = None,
    ) -> None:
        self.game = game
        self.action_index = action_index
        self.action = action
        self.model_open_candidates = (
            np.empty(0, dtype=np.int64)
            if model_open_candidates is None
            else np.asarray(model_open_candidates, dtype=np.int64).reshape(-1)
        )


def make_history_item(
    round_index: int,
    metrics: dict[str, float | int | bool | None],
    mine_metrics: dict[str, Any] | None,
    losses: dict[str, Any] | None,
    replay_size: int,
    bootstrap_replay_size: int,
    bootstrap_replay_report: dict[str, Any] | None,
    extreme_replay_report: dict[str, Any] | None,
) -> dict[str, Any]:
    return {
        "round": round_index,
        "metrics": metrics,
        "mine_metrics": mine_metrics,
        "losses": losses,
        "replay_size": replay_size,
        "bootstrap_replay_size": bootstrap_replay_size,
        "bootstrap_replay_report": bootstrap_replay_report,
        "extreme_replay_report": extreme_replay_report,
    }


def load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


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
