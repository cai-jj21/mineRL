from __future__ import annotations

import argparse
import copy
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.features import action_channel, decode_action_index, encode_state
from minesweeper_rl.gate_calibration import FEATURE_NAMES, build_gate_features
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import load_checkpoint, make_game, resolve_forced_moves
from minesweeper_rl.types import ActionType


GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_for_gate_dataset", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build solver-labelled disagreement states for a pure-model specialist gate."
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
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=256)
    parser.add_argument("--seed", type=int, default=140000)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--safe-left-threshold", type=int, default=100)
    parser.add_argument("--exact-limit", type=int, default=32)
    parser.add_argument("--label-margin", type=float, default=0.05)
    parser.add_argument("--label-mode", choices=["fast", "solver"], default="fast")
    parser.add_argument("--max-labels-per-game", type=int, default=8)
    parser.add_argument("--max-records", type=int, default=8192)
    parser.add_argument("--no-inference-flips", action="store_true")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    args = parser.parse_args()

    base_paths = [args.base_checkpoint, *args.base_ensemble_checkpoints]
    specialist_paths = [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
    base_weights = gated.normalize_ensemble_weights(
        args.base_model_weights,
        expected=len(base_paths),
        label="base-model-weight",
    )
    specialist_weights = gated.normalize_ensemble_weights(
        args.specialist_model_weights,
        expected=len(specialist_paths),
        label="specialist-model-weight",
    )
    eval_args = SimpleNamespace(
        device=args.device,
        max_steps=args.max_steps,
        decision_actions="full",
        inference_flips=not args.no_inference_flips,
        inference_ensemble=args.inference_ensemble,
        risk_head_weight=float(args.risk_head_weight),
    )
    base_risk_head_weight = (
        args.risk_head_weight if args.base_risk_head_weight is None else args.base_risk_head_weight
    )
    specialist_risk_head_weight = (
        args.risk_head_weight
        if args.specialist_risk_head_weight is None
        else args.specialist_risk_head_weight
    )
    bases = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=base_risk_head_weight)
        for path in base_paths
    ]
    specialists = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=specialist_risk_head_weight)
        for path in specialist_paths
    ]
    solver = MinesweeperSolver(exact_limit=args.exact_limit)

    started_at = time.time()
    payload = collect_disagreements(
        bases=bases,
        base_weights=base_weights,
        specialists=specialists,
        specialist_weights=specialist_weights,
        solver=solver,
        args=args,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        features=payload.pop("features"),
        labels=payload.pop("labels"),
        value_delta=payload.pop("value_delta"),
        base_value=payload.pop("base_value"),
        specialist_value=payload.pop("specialist_value"),
        game_index=payload.pop("game_index"),
        step=payload.pop("step"),
        safe_left=payload.pop("safe_left"),
        base_action=payload.pop("base_action"),
        specialist_action=payload.pop("specialist_action"),
    )
    report = {
        "ok": True,
        "dataset": str(args.output),
        "feature_names": list(FEATURE_NAMES),
        "records": int(len(payload["record_count"])),
        "positive_records": int(payload["positive_records"]),
        "negative_records": int(payload["negative_records"]),
        "positive_rate": float(payload["positive_records"] / max(1, len(payload["record_count"]))),
        "metadata": {
            "source": "base_specialist_disagreement_counterfactual_labels",
            "base_checkpoints": [str(path) for path in base_paths],
            "base_model_weights": base_weights,
            "specialist_checkpoints": [str(path) for path in specialist_paths],
            "specialist_model_weights": specialist_weights,
            "model_ensemble_reduction": args.model_ensemble_reduction,
            "risk_head_weight": float(args.risk_head_weight),
            "base_risk_head_weight": float(base_risk_head_weight),
            "specialist_risk_head_weight": float(specialist_risk_head_weight),
            "games": int(args.games),
            "seed": int(args.seed),
            "safe_left_threshold": int(args.safe_left_threshold),
            "exact_limit": int(args.exact_limit),
            "label_margin": float(args.label_margin),
            "label_mode": args.label_mode,
            "max_labels_per_game": int(args.max_labels_per_game),
            "inference_flips": not args.no_inference_flips,
            "inference_ensemble": args.inference_ensemble,
            "tail_states": int(payload["tail_states"]),
            "disagreements": int(payload["disagreements"]),
            "games_won": int(payload["games_won"]),
            "games_lost": int(payload["games_lost"]),
            "elapsed_seconds": time.time() - started_at,
        },
    }
    manifest_path = args.output.with_suffix(".json")
    manifest_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


def collect_disagreements(
    *,
    bases,
    base_weights: list[float],
    specialists,
    specialist_weights: list[float],
    solver: MinesweeperSolver,
    args: argparse.Namespace,
) -> dict[str, object]:
    base = bases[0]
    rows, cols = base.config.rows, base.config.cols
    total_safe = rows * cols - base.config.mines
    scores_are_probabilities = args.inference_ensemble == "probs"
    features: list[np.ndarray] = []
    labels: list[int] = []
    value_delta: list[float] = []
    base_value: list[float] = []
    specialist_value: list[float] = []
    game_index: list[int] = []
    steps: list[int] = []
    safe_left_values: list[int] = []
    base_actions: list[int] = []
    specialist_actions: list[int] = []
    tail_states = 0
    disagreements = 0
    games_won = 0
    games_lost = 0
    labels_per_game: dict[int, int] = {}
    games = int(args.games)
    batch_size = max(1, int(args.batch_size))

    with torch.inference_mode():
        for batch_start in range(0, games, batch_size):
            current_batch = min(batch_size, games - batch_start)
            game_batch = [
                make_game(base.config, seed=int(args.seed) + batch_start + offset)
                for offset in range(current_batch)
            ]
            game_steps = [0] * current_batch
            active = set(range(current_batch))
            while active and len(features) < int(args.max_records):
                ready: list[int] = []
                boards: list[np.ndarray] = []
                globals_batch: list[np.ndarray] = []
                masks: list[np.ndarray] = []
                for index in list(active):
                    game = game_batch[index]
                    if game.done or game_steps[index] >= args.max_steps:
                        active.remove(index)
                        continue
                    board, global_features, action_mask = encode_state(game)
                    action_mask = base._decision_action_mask(action_mask)
                    if not action_mask.any():
                        active.remove(index)
                        continue
                    ready.append(index)
                    boards.append(board)
                    globals_batch.append(global_features)
                    masks.append(action_mask)
                if not ready:
                    continue

                board_batch = np.stack(boards)
                global_batch = np.stack(globals_batch)
                mask_batch = np.stack(masks)
                flat_mask = torch.tensor(
                    mask_batch.reshape(len(ready), -1),
                    dtype=torch.bool,
                    device=base.device,
                )
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
                base_np = base_scores.detach().cpu().numpy()
                specialist_np = specialist_scores.detach().cpu().numpy()
                gate_features = build_gate_features(
                    base_scores=base_np,
                    specialist_scores=specialist_np,
                    global_features=global_batch,
                    boards=board_batch,
                    action_masks=mask_batch,
                    scores_are_probabilities=scores_are_probabilities,
                )
                open_cells = rows * cols
                base_open = base_np[:, :open_cells]
                specialist_open = specialist_np[:, :open_cells]
                open_mask = mask_batch[:, action_channel(ActionType.OPEN)].reshape(len(ready), -1)
                base_open_choice = np.argmax(np.where(open_mask, base_open, -np.inf), axis=1)
                specialist_open_choice = np.argmax(np.where(open_mask, specialist_open, -np.inf), axis=1)
                action_indices = base_np.argmax(axis=1)

                for local_index, game_index_value in enumerate(ready):
                    game = game_batch[game_index_value]
                    progress = float(np.clip(global_batch[local_index, 1], 0.0, 1.0))
                    safe_left = int(round((1.0 - progress) * total_safe))
                    is_tail = safe_left <= int(args.safe_left_threshold)
                    if is_tail:
                        tail_states += 1
                    base_open_index = int(base_open_choice[local_index])
                    specialist_open_index = int(specialist_open_choice[local_index])
                    disagree = base_open_index != specialist_open_index
                    game_key = int(args.seed) + batch_start + game_index_value
                    can_label = labels_per_game.get(game_key, 0) < max(1, int(args.max_labels_per_game))
                    if is_tail and disagree and can_label and len(features) < int(args.max_records):
                        disagreements += 1
                        br, bc = divmod(base_open_index, cols)
                        sr, sc = divmod(specialist_open_index, cols)
                        b_value = counterfactual_open_value(
                            game,
                            br,
                            bc,
                            solver,
                            mode=args.label_mode,
                        )
                        s_value = counterfactual_open_value(
                            game,
                            sr,
                            sc,
                            solver,
                            mode=args.label_mode,
                        )
                        delta = float(s_value - b_value)
                        features.append(gate_features[local_index])
                        labels.append(int(delta > float(args.label_margin)))
                        value_delta.append(delta)
                        base_value.append(float(b_value))
                        specialist_value.append(float(s_value))
                        game_index.append(game_key)
                        steps.append(int(game_steps[game_index_value]))
                        safe_left_values.append(safe_left)
                        base_actions.append(base_open_index)
                        specialist_actions.append(specialist_open_index)
                        labels_per_game[game_key] = labels_per_game.get(game_key, 0) + 1

                    action = decode_action_index(
                        int(action_indices[local_index]),
                        rows,
                        cols,
                    )
                    game.step(action)
                    game_steps[game_index_value] += 1
                    if game.done or game_steps[game_index_value] >= args.max_steps:
                        active.discard(game_index_value)

            for game in game_batch:
                games_won += int(game.won)
                games_lost += int(game.lost)

    feature_array = np.asarray(features, dtype=np.float32)
    if feature_array.ndim == 1:
        feature_array = feature_array.reshape(0, len(FEATURE_NAMES))
    record_count = np.arange(len(features), dtype=np.int64)
    return {
        "features": feature_array,
        "labels": np.asarray(labels, dtype=np.int64),
        "value_delta": np.asarray(value_delta, dtype=np.float32),
        "base_value": np.asarray(base_value, dtype=np.float32),
        "specialist_value": np.asarray(specialist_value, dtype=np.float32),
        "game_index": np.asarray(game_index, dtype=np.int64),
        "step": np.asarray(steps, dtype=np.int64),
        "safe_left": np.asarray(safe_left_values, dtype=np.int64),
        "base_action": np.asarray(base_actions, dtype=np.int64),
        "specialist_action": np.asarray(specialist_actions, dtype=np.int64),
        "record_count": record_count,
        "positive_records": int(sum(labels)),
        "negative_records": int(len(labels) - sum(labels)),
        "tail_states": tail_states,
        "disagreements": disagreements,
        "games_won": games_won,
        "games_lost": games_lost,
    }


def counterfactual_open_value(
    game,
    row: int,
    col: int,
    solver: MinesweeperSolver,
    *,
    mode: str,
) -> float:
    candidate = copy.deepcopy(game)
    _, _, _, info = candidate.open_cell(row, col)
    if bool(info.get("hit_mine", False)):
        return -1.0
    if candidate.won:
        return 2.0
    before = int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum())
    after = int((candidate.revealed & ~candidate.mines).sum()) if candidate.mines_placed else int(candidate.revealed.sum())
    delta_safe = max(0, after - before)
    if mode == "fast":
        return float(0.1 + 0.035 * delta_safe)
    snapshot, forced_reward, forced_steps = resolve_forced_moves(candidate, solver)
    future_bonus = 0.0
    if not candidate.done and snapshot.best_guess_risk is not None:
        future_bonus = float(np.clip(1.0 - float(snapshot.best_guess_risk), 0.0, 1.0))
    return float(
        0.035 * delta_safe
        + 0.02 * forced_steps
        + 0.5 * max(0.0, float(forced_reward))
        + 0.25 * future_bonus
    )


if __name__ == "__main__":
    main()
