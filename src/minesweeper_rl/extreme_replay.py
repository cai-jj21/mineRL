from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from minesweeper_rl.types import EpisodeTransition


def relabel_counterfactual_values_risk_aware(
    transitions: Iterable[EpisodeTransition],
    *,
    risk_weight: float = 1.0,
    outcome_weight: float = 0.05,
    mine_penalty: float = 1.0,
) -> list[EpisodeTransition]:
    """Build risk-first offline labels without exposing hidden layout at inference.

    Solver risk is the primary signal because it is derived from visible
    constraints. The known-layout counterfactual outcome is retained only as a
    small secondary tie-breaker, while actual mines become hard negatives.
    """

    if risk_weight < 0.0 or outcome_weight < 0.0 or mine_penalty <= 0.0:
        raise ValueError("risk_weight and outcome_weight must be non-negative; mine_penalty must be positive")

    relabelled: list[EpisodeTransition] = []
    for transition in transitions:
        values = transition.counterfactual_open_values
        risk_map = transition.risk_map
        if values is None or risk_map is None:
            relabelled.append(transition)
            continue

        values_array = np.asarray(values, dtype=np.float32)
        risk_array = np.asarray(risk_map, dtype=np.float32)
        finite = np.isfinite(values_array) & np.isfinite(risk_array)
        if not bool(finite.any()):
            relabelled.append(transition)
            continue

        outcome = np.clip(np.nan_to_num(values_array, nan=0.0), 0.0, 8.0)
        transformed = (
            float(risk_weight) * (1.0 - np.clip(risk_array, 0.0, 1.0))
            + float(outcome_weight) * np.log1p(outcome)
        ).astype(np.float32)
        transformed[~finite] = np.nan

        if transition.mine_mask is not None:
            mines = np.asarray(transition.mine_mask, dtype=bool)
            transformed[finite & mines] = -abs(float(mine_penalty))

        relabelled.append(
            EpisodeTransition(
                board=transition.board.copy(),
                global_features=transition.global_features.copy(),
                action_mask=transition.action_mask.copy(),
                action_index=transition.action_index,
                expert_action_index=transition.expert_action_index,
                reward=transition.reward,
                done=transition.done,
                expert_action_mask=None
                if transition.expert_action_mask is None
                else transition.expert_action_mask.copy(),
                mine_mask=None if transition.mine_mask is None else transition.mine_mask.copy(),
                risk_map=transition.risk_map.copy(),
                counterfactual_open_values=transformed,
                expert_is_guess=transition.expert_is_guess,
                source_quality=transition.source_quality,
                extreme_score=transition.extreme_score,
                extreme_family=transition.extreme_family,
            )
        )
    return relabelled


def save_extreme_dataset(
    path: Path | str,
    transitions: Iterable[EpisodeTransition],
    *,
    metadata: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(transitions)
    if not rows:
        raise ValueError("cannot save an empty extreme dataset")

    board = np.stack([row.board for row in rows]).astype(np.float32)
    global_features = np.stack([row.global_features for row in rows]).astype(np.float32)
    action_mask = np.stack([row.action_mask for row in rows]).astype(bool)
    action_index = np.asarray([row.action_index for row in rows], dtype=np.int64)
    expert_action_index = np.asarray(
        [-1 if row.expert_action_index is None else row.expert_action_index for row in rows],
        dtype=np.int64,
    )
    reward = np.asarray([row.reward for row in rows], dtype=np.float32)
    done = np.asarray([row.done for row in rows], dtype=bool)
    has_expert_mask = np.asarray([row.expert_action_mask is not None for row in rows], dtype=bool)
    expert_action_mask = np.zeros_like(action_mask, dtype=bool)
    for index, row in enumerate(rows):
        if row.expert_action_mask is not None:
            expert_action_mask[index] = row.expert_action_mask
    has_mine_mask = np.asarray([row.mine_mask is not None for row in rows], dtype=bool)
    mine_mask = np.zeros((len(rows), board.shape[-2], board.shape[-1]), dtype=bool)
    for index, row in enumerate(rows):
        if row.mine_mask is not None:
            mine_mask[index] = row.mine_mask
    has_risk_map = np.asarray([row.risk_map is not None for row in rows], dtype=bool)
    risk_map = np.zeros((len(rows), board.shape[-2], board.shape[-1]), dtype=np.float32)
    for index, row in enumerate(rows):
        if row.risk_map is not None:
            risk_map[index] = row.risk_map
    has_counterfactual = np.asarray(
        [row.counterfactual_open_values is not None for row in rows],
        dtype=bool,
    )
    counterfactual_open_values = np.full(
        (len(rows), board.shape[-2], board.shape[-1]),
        np.nan,
        dtype=np.float32,
    )
    for index, row in enumerate(rows):
        if row.counterfactual_open_values is not None:
            values = np.asarray(row.counterfactual_open_values, dtype=np.float32)
            if values.shape != counterfactual_open_values[index].shape:
                raise ValueError(
                    "counterfactual_open_values shape mismatch: "
                    f"expected {counterfactual_open_values[index].shape}, got {values.shape}"
                )
            counterfactual_open_values[index] = values
    expert_is_guess = np.asarray([row.expert_is_guess for row in rows], dtype=bool)
    source_quality = np.asarray([row.source_quality for row in rows], dtype=np.float32)
    extreme_score = np.asarray([row.extreme_score for row in rows], dtype=np.float32)
    extreme_family = np.asarray([row.extreme_family or "" for row in rows])

    np.savez_compressed(
        path,
        board=board,
        global_features=global_features,
        action_mask=action_mask,
        action_index=action_index,
        expert_action_index=expert_action_index,
        reward=reward,
        done=done,
        has_expert_mask=has_expert_mask,
        expert_action_mask=expert_action_mask,
        has_mine_mask=has_mine_mask,
        mine_mask=mine_mask,
        has_risk_map=has_risk_map,
        risk_map=risk_map,
        has_counterfactual=has_counterfactual,
        counterfactual_open_values=counterfactual_open_values,
        expert_is_guess=expert_is_guess,
        source_quality=source_quality,
        extreme_score=extreme_score,
        extreme_family=extreme_family,
    )

    families = Counter(row.extreme_family or "unclassified" for row in rows)
    label_summary = summarize_transition_labels(rows)
    manifest = {
        "ok": True,
        "dataset": str(path),
        "records": len(rows),
        "shape": {
            "board": list(board.shape),
            "global_features": list(global_features.shape),
            "action_mask": list(action_mask.shape),
        },
        "families": dict(sorted(families.items())),
        "label_summary": label_summary,
        "metadata": metadata or {},
    }
    manifest_path = path.with_suffix(".json")
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest["manifest"] = str(manifest_path)
    return manifest


def load_extreme_dataset(path: Path | str) -> tuple[list[EpisodeTransition], dict[str, Any]]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(path)

    with np.load(path, allow_pickle=False) as data:
        board = np.asarray(data["board"], dtype=np.float32)
        global_features = np.asarray(data["global_features"], dtype=np.float32)
        action_mask = np.asarray(data["action_mask"], dtype=bool)
        action_index = np.asarray(data["action_index"], dtype=np.int64)
        expert_action_index = np.asarray(data["expert_action_index"], dtype=np.int64)
        reward = np.asarray(data["reward"], dtype=np.float32)
        done = np.asarray(data["done"], dtype=bool)
        has_expert_mask = np.asarray(data["has_expert_mask"], dtype=bool)
        expert_action_mask = np.asarray(data["expert_action_mask"], dtype=bool)
        has_mine_mask = np.asarray(data["has_mine_mask"], dtype=bool)
        mine_mask = np.asarray(data["mine_mask"], dtype=bool)
        has_risk_map = np.asarray(data["has_risk_map"], dtype=bool)
        risk_map = np.asarray(data["risk_map"], dtype=np.float32)
        if "has_counterfactual" in data.files:
            has_counterfactual = np.asarray(data["has_counterfactual"], dtype=bool)
            counterfactual_open_values = np.asarray(data["counterfactual_open_values"], dtype=np.float32)
        else:
            has_counterfactual = np.zeros(board.shape[0], dtype=bool)
            counterfactual_open_values = np.full(
                (board.shape[0], board.shape[-2], board.shape[-1]),
                np.nan,
                dtype=np.float32,
            )
        expert_is_guess = np.asarray(data["expert_is_guess"], dtype=bool)
        source_quality = np.asarray(data["source_quality"], dtype=np.float32)
        extreme_score = np.asarray(data["extreme_score"], dtype=np.float32)
        extreme_family = np.asarray(data["extreme_family"])

    count = int(board.shape[0])
    arrays = (
        global_features,
        action_mask,
        action_index,
        expert_action_index,
        reward,
        done,
        has_expert_mask,
        expert_action_mask,
        has_mine_mask,
        mine_mask,
        has_risk_map,
        risk_map,
        has_counterfactual,
        counterfactual_open_values,
        expert_is_guess,
        source_quality,
        extreme_score,
        extreme_family,
    )
    if any(int(array.shape[0]) != count for array in arrays):
        raise ValueError(f"inconsistent extreme dataset record counts in {path}")

    manifest_path = path.with_suffix(".json")
    metadata: dict[str, Any] = {}
    if manifest_path.exists():
        try:
            metadata = json.loads(manifest_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            metadata = {}

    transitions: list[EpisodeTransition] = []
    for index in range(count):
        transitions.append(
            EpisodeTransition(
                board=np.ascontiguousarray(board[index]),
                global_features=np.ascontiguousarray(global_features[index]),
                action_mask=np.ascontiguousarray(action_mask[index]),
                action_index=int(action_index[index]),
                expert_action_index=(
                    None if int(expert_action_index[index]) < 0 else int(expert_action_index[index])
                ),
                reward=float(reward[index]),
                done=bool(done[index]),
                expert_action_mask=(
                    np.ascontiguousarray(expert_action_mask[index]) if has_expert_mask[index] else None
                ),
                mine_mask=np.ascontiguousarray(mine_mask[index]) if has_mine_mask[index] else None,
                risk_map=np.ascontiguousarray(risk_map[index]) if has_risk_map[index] else None,
                counterfactual_open_values=(
                    np.ascontiguousarray(counterfactual_open_values[index])
                    if has_counterfactual[index]
                    else None
                ),
                expert_is_guess=bool(expert_is_guess[index]),
                source_quality=float(source_quality[index]),
                extreme_score=float(extreme_score[index]),
                extreme_family=str(extreme_family[index]),
            )
        )
    return transitions, metadata


def summarize_transition_labels(transitions: Iterable[EpisodeTransition]) -> dict[str, Any]:
    rows = list(transitions)
    counterfactual_records = 0
    counterfactual_candidate_labels = 0
    negative_counterfactual_candidates = 0
    positive_counterfactual_candidates = 0
    best_counterfactual_values: list[float] = []
    behavior_counterfactual_values: list[float] = []
    behavior_regrets: list[float] = []

    for transition in rows:
        values = transition.counterfactual_open_values
        if values is None:
            continue
        value_array = np.asarray(values, dtype=np.float32)
        finite = np.isfinite(value_array)
        if not bool(finite.any()):
            continue

        counterfactual_records += 1
        counterfactual_candidate_labels += int(finite.sum())
        negative_counterfactual_candidates += int((value_array[finite] < 0.0).sum())
        positive_counterfactual_candidates += int((value_array[finite] > 0.0).sum())
        best_value = float(np.max(value_array[finite]))
        best_counterfactual_values.append(best_value)

        behavior_value = _behavior_counterfactual_value(transition, value_array)
        if behavior_value is not None:
            behavior_counterfactual_values.append(behavior_value)
            behavior_regrets.append(best_value - behavior_value)

    return {
        "guess_records": int(sum(transition.expert_is_guess for transition in rows)),
        "counterfactual_records": counterfactual_records,
        "counterfactual_candidate_labels": counterfactual_candidate_labels,
        "negative_counterfactual_candidates": negative_counterfactual_candidates,
        "positive_counterfactual_candidates": positive_counterfactual_candidates,
        "negative_counterfactual_rate": _safe_rate(
            negative_counterfactual_candidates,
            counterfactual_candidate_labels,
        ),
        "positive_counterfactual_rate": _safe_rate(
            positive_counterfactual_candidates,
            counterfactual_candidate_labels,
        ),
        "avg_best_counterfactual_value": _mean_or_none(best_counterfactual_values),
        "avg_behavior_counterfactual_value": _mean_or_none(behavior_counterfactual_values),
        "avg_behavior_regret": _mean_or_none(behavior_regrets),
    }


def _behavior_counterfactual_value(
    transition: EpisodeTransition,
    values: np.ndarray,
) -> float | None:
    if transition.action_mask.ndim != 3 or values.ndim != 2:
        return None
    rows, cols = values.shape
    cells = rows * cols
    kind_index, cell_index = divmod(int(transition.action_index), cells)
    if kind_index != 0:
        return None
    row, col = divmod(cell_index, cols)
    if not (0 <= row < rows and 0 <= col < cols):
        return None
    value = float(values[row, col])
    return value if np.isfinite(value) else None


def _mean_or_none(values: list[float]) -> float | None:
    return float(np.mean(values)) if values else None


def _safe_rate(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return float(numerator / denominator)
