from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "progress",
    "safe_left_ratio",
    "flag_ratio",
    "hidden_ratio",
    "remaining_mines_ratio",
    "covered_ratio",
    "base_open_entropy",
    "specialist_open_entropy",
    "base_open_gap",
    "specialist_open_gap",
    "base_top_probability",
    "specialist_top_probability",
    "base_probability_on_specialist",
    "specialist_probability_on_base",
    "specialist_probability_advantage",
    "base_rank_of_specialist",
    "specialist_rank_of_base",
    "base_edge_distance",
    "specialist_edge_distance",
    "base_is_edge",
    "specialist_is_edge",
    "base_is_corner",
    "specialist_is_corner",
    "action_distance",
    "base_frontier",
    "specialist_frontier",
    "base_hidden_neighbors",
    "specialist_hidden_neighbors",
    "base_flag_neighbors",
    "specialist_flag_neighbors",
)


@dataclass
class GateCalibrator:
    """Small offline-trained selector for base vs specialist policy scores."""

    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float
    threshold: float = 0.5
    feature_names: tuple[str, ...] = FEATURE_NAMES

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("gate calibrator parameters must be one-dimensional")
        if not (len(self.means) == len(self.scales) == len(self.weights) == len(self.feature_names)):
            raise ValueError("gate calibrator feature and parameter counts must match")
        if not np.isfinite(self.means).all() or not np.isfinite(self.scales).all():
            raise ValueError("gate calibrator normalization parameters must be finite")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("gate calibrator model parameters must be finite")
        self.scales = np.maximum(self.scales, 1e-6)
        self.threshold = float(np.clip(self.threshold, 0.0, 1.0))

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2 or features.shape[1] != len(self.weights):
            raise ValueError(
                f"expected gate features with shape [batch, {len(self.weights)}], got {features.shape}"
            )
        normalized = (features - self.means[None, :]) / self.scales[None, :]
        logits = normalized @ self.weights + float(self.bias)
        logits = np.clip(logits, -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def should_apply_features(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba_features(features) >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "minesweeper_rl_gate_calibrator_v1",
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "threshold": float(self.threshold),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "GateCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if feature_names != FEATURE_NAMES:
            raise ValueError("gate calibrator feature schema does not match this code version")
        return cls(
            means=np.asarray(payload["means"], dtype=np.float32),
            scales=np.asarray(payload["scales"], dtype=np.float32),
            weights=np.asarray(payload["weights"], dtype=np.float32),
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
            feature_names=feature_names,
        )


def build_gate_features(
    *,
    base_scores: np.ndarray,
    specialist_scores: np.ndarray,
    global_features: np.ndarray,
    boards: np.ndarray,
    action_masks: np.ndarray,
    scores_are_probabilities: bool,
) -> np.ndarray:
    """Build selector features from visible state and model scores only."""

    base_scores = np.asarray(base_scores, dtype=np.float32)
    specialist_scores = np.asarray(specialist_scores, dtype=np.float32)
    global_features = np.asarray(global_features, dtype=np.float32)
    boards = np.asarray(boards, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    if base_scores.ndim != 2 or specialist_scores.shape != base_scores.shape:
        raise ValueError("base and specialist scores must have matching [batch, actions] shapes")
    if global_features.ndim != 2 or global_features.shape[0] != base_scores.shape[0]:
        raise ValueError("global_features must have one row per score row")
    if boards.ndim != 4 or boards.shape[0] != base_scores.shape[0]:
        raise ValueError("boards must have one row per score row")
    if action_masks.ndim != 4 or action_masks.shape[0] != base_scores.shape[0]:
        raise ValueError("action_masks must have one row per score row")

    batch, _, rows, cols = action_masks.shape
    cells = rows * cols
    if base_scores.shape[1] != action_masks.shape[1] * cells:
        raise ValueError("score and action-mask dimensions do not match")
    open_mask = action_masks[:, 0].reshape(batch, cells)
    base_open = base_scores[:, :cells]
    specialist_open = specialist_scores[:, :cells]
    base_probs = _open_distributions(base_open, open_mask, scores_are_probabilities)
    specialist_probs = _open_distributions(specialist_open, open_mask, scores_are_probabilities)

    base_choice = np.argmax(np.where(open_mask, base_probs, -np.inf), axis=1)
    specialist_choice = np.argmax(np.where(open_mask, specialist_probs, -np.inf), axis=1)
    base_order = np.argsort(np.where(open_mask, base_probs, -np.inf), axis=1)[:, ::-1]
    specialist_order = np.argsort(np.where(open_mask, specialist_probs, -np.inf), axis=1)[:, ::-1]
    valid_counts = np.maximum(open_mask.sum(axis=1), 1)

    base_top, base_second = _top_two(base_probs, open_mask)
    specialist_top, specialist_second = _top_two(specialist_probs, open_mask)
    base_rank_of_specialist = _normalized_rank(base_order, specialist_choice, valid_counts)
    specialist_rank_of_base = _normalized_rank(specialist_order, base_choice, valid_counts)

    base_rows, base_cols = np.divmod(base_choice, cols)
    specialist_rows, specialist_cols = np.divmod(specialist_choice, cols)
    base_edge_distance, base_is_edge, base_is_corner = _geometry(base_rows, base_cols, rows, cols)
    specialist_edge_distance, specialist_is_edge, specialist_is_corner = _geometry(
        specialist_rows,
        specialist_cols,
        rows,
        cols,
    )
    base_frontier, base_hidden_neighbors, base_flag_neighbors = _board_values(
        boards,
        base_rows,
        base_cols,
    )
    specialist_frontier, specialist_hidden_neighbors, specialist_flag_neighbors = _board_values(
        boards,
        specialist_rows,
        specialist_cols,
    )

    progress = _global_column(global_features, 1)
    features = np.stack(
        [
            _global_column(global_features, 0),
            1.0 - progress,
            _global_column(global_features, 2),
            _global_column(global_features, 3),
            _global_column(global_features, 4),
            _global_column(global_features, 5),
            _entropy(base_probs, open_mask),
            _entropy(specialist_probs, open_mask),
            base_top - base_second,
            specialist_top - specialist_second,
            base_top,
            specialist_top,
            base_probs[np.arange(batch), specialist_choice],
            specialist_probs[np.arange(batch), base_choice],
            specialist_probs[np.arange(batch), specialist_choice]
            - base_probs[np.arange(batch), specialist_choice]
            + base_probs[np.arange(batch), base_choice]
            - specialist_probs[np.arange(batch), base_choice],
            base_rank_of_specialist,
            specialist_rank_of_base,
            base_edge_distance,
            specialist_edge_distance,
            base_is_edge.astype(np.float32),
            specialist_is_edge.astype(np.float32),
            base_is_corner.astype(np.float32),
            specialist_is_corner.astype(np.float32),
            (np.abs(base_rows - specialist_rows) + np.abs(base_cols - specialist_cols)).astype(np.float32)
            / max(1.0, float(rows + cols - 2)),
            base_frontier,
            specialist_frontier,
            base_hidden_neighbors,
            specialist_hidden_neighbors,
            base_flag_neighbors,
            specialist_flag_neighbors,
        ],
        axis=1,
    )
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def _global_column(global_features: np.ndarray, index: int) -> np.ndarray:
    if global_features.shape[1] <= index:
        return np.zeros(global_features.shape[0], dtype=np.float32)
    return global_features[:, index]


def _open_distributions(scores: np.ndarray, mask: np.ndarray, probabilities: bool) -> np.ndarray:
    if probabilities:
        clipped = np.where(mask, np.maximum(scores, 0.0), 0.0)
        totals = clipped.sum(axis=1, keepdims=True)
        fallback = mask.astype(np.float32) / np.maximum(mask.sum(axis=1, keepdims=True), 1)
        return np.where(totals > 1e-8, clipped / np.maximum(totals, 1e-8), fallback)
    masked = np.where(mask, scores, -1e9)
    shifted = masked - masked.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 0.0)) * mask
    return exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-8)


def _top_two(probabilities: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    ordered = np.sort(np.where(mask, probabilities, -np.inf), axis=1)[:, ::-1]
    top = np.nan_to_num(ordered[:, 0], nan=0.0, neginf=0.0)
    second = np.nan_to_num(ordered[:, 1], nan=0.0, neginf=0.0) if ordered.shape[1] > 1 else np.zeros_like(top)
    return top, second


def _entropy(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    terms = np.where(mask, probabilities * np.log(np.maximum(probabilities, 1e-8)), 0.0)
    raw = -terms.sum(axis=1)
    normalizer = np.log(np.maximum(mask.sum(axis=1), 2))
    return np.divide(raw, normalizer, out=np.zeros_like(raw), where=normalizer > 0.0)


def _normalized_rank(order: np.ndarray, target: np.ndarray, valid_counts: np.ndarray) -> np.ndarray:
    ranks = np.zeros(target.shape[0], dtype=np.float32)
    for index in range(target.shape[0]):
        position = int(np.flatnonzero(order[index] == target[index])[0])
        ranks[index] = position / max(1.0, float(valid_counts[index] - 1))
    return ranks


def _geometry(
    rows: np.ndarray,
    cols: np.ndarray,
    row_count: int,
    col_count: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    edge_distance = np.minimum.reduce(
        [
            rows.astype(np.float32),
            (row_count - 1 - rows).astype(np.float32),
            cols.astype(np.float32),
            (col_count - 1 - cols).astype(np.float32),
        ]
    )
    is_edge = (edge_distance == 0.0)
    is_corner = ((rows == 0) | (rows == row_count - 1)) & ((cols == 0) | (cols == col_count - 1))
    return edge_distance / max(1.0, float(min(row_count, col_count))), is_edge, is_corner


def _board_values(
    boards: np.ndarray,
    rows: np.ndarray,
    cols: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    batch = np.arange(boards.shape[0])
    frontier = boards[batch, 11, rows, cols] if boards.shape[1] > 11 else np.zeros(batch.shape[0])
    hidden_neighbors = boards[batch, 13, rows, cols] if boards.shape[1] > 13 else np.zeros(batch.shape[0])
    flag_neighbors = boards[batch, 12, rows, cols] if boards.shape[1] > 12 else np.zeros(batch.shape[0])
    return frontier, hidden_neighbors, flag_neighbors
