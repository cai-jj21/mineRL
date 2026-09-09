from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


FEATURE_NAMES = (
    "safe_left_ratio",
    "flag_ratio",
    "hidden_ratio",
    "remaining_mines_ratio",
    "covered_ratio",
    "base_open_entropy",
    "candidate_open_entropy",
    "base_open_gap",
    "candidate_open_gap",
    "base_top_probability",
    "candidate_top_probability",
    "candidate_probability_advantage",
    "candidate_choice_disagree",
    "candidate_edge_distance",
    "base_edge_distance",
    "candidate_is_edge",
    "candidate_is_corner",
    "base_is_edge",
    "base_is_corner",
    "candidate_risk",
    "base_risk",
    "risk_advantage",
    "candidate_safe_probability",
    "base_safe_probability",
    "base_rank_of_candidate",
    "candidate_rank_of_base",
    "choice_distance",
)


@dataclass
class CandidateGateCalibrator:
    """Visible-state gate deciding whether a candidate override is worthwhile."""

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
            raise ValueError("candidate gate parameters must be one-dimensional")
        if not (len(self.means) == len(self.scales) == len(self.weights) == len(self.feature_names)):
            raise ValueError("candidate gate feature and parameter counts must match")
        if not np.isfinite(self.means).all() or not np.isfinite(self.scales).all():
            raise ValueError("candidate gate normalization parameters must be finite")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("candidate gate model parameters must be finite")
        self.scales = np.maximum(self.scales, 1e-6)
        self.threshold = float(np.clip(self.threshold, 0.0, 1.0))

    def predict_proba_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.ndim != 2 or features.shape[1] != len(self.weights):
            raise ValueError(
                f"expected candidate gate features with shape [batch, {len(self.weights)}], "
                f"got {features.shape}"
            )
        normalized = (features - self.means[None, :]) / self.scales[None, :]
        logits = np.clip(normalized @ self.weights + float(self.bias), -40.0, 40.0)
        return (1.0 / (1.0 + np.exp(-logits))).astype(np.float32)

    def should_apply_features(self, features: np.ndarray) -> np.ndarray:
        return self.predict_proba_features(features) >= self.threshold

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "minesweeper_rl_candidate_gate_calibrator_v1",
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
    def load(cls, path: Path | str) -> "CandidateGateCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(payload.get("feature_names", FEATURE_NAMES))
        if feature_names != FEATURE_NAMES:
            raise ValueError("candidate gate feature schema does not match this code version")
        return cls(
            means=np.asarray(payload["means"], dtype=np.float32),
            scales=np.asarray(payload["scales"], dtype=np.float32),
            weights=np.asarray(payload["weights"], dtype=np.float32),
            bias=float(payload["bias"]),
            threshold=float(payload.get("threshold", 0.5)),
            feature_names=feature_names,
        )


def build_candidate_gate_features(
    *,
    base_scores: np.ndarray,
    candidate_probabilities: np.ndarray,
    global_features: np.ndarray,
    action_masks: np.ndarray,
    risk_maps: np.ndarray | None = None,
    scores_are_probabilities: bool,
) -> np.ndarray:
    """Build state-level gate features from visible policy/candidate signals."""

    base_scores = np.asarray(base_scores, dtype=np.float32)
    candidate_probabilities = np.asarray(candidate_probabilities, dtype=np.float32)
    global_features = np.asarray(global_features, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    if action_masks.ndim != 4:
        raise ValueError("action_masks must have shape [batch, actions, rows, cols]")
    batch, _, rows, cols = action_masks.shape
    cells = rows * cols
    if base_scores.shape != (batch, action_masks.shape[1] * cells):
        raise ValueError("base_scores shape does not match action_masks")
    if candidate_probabilities.shape != (batch, cells):
        raise ValueError("candidate_probabilities must have shape [batch, rows * cols]")

    open_mask = action_masks[:, 0].reshape(batch, cells)
    base_probs = _open_distributions(base_scores[:, :cells], open_mask, scores_are_probabilities)
    candidate_probs = np.where(open_mask, np.maximum(candidate_probabilities, 0.0), 0.0)
    candidate_probs /= np.maximum(candidate_probs.sum(axis=1, keepdims=True), 1e-8)

    base_order = np.argsort(np.where(open_mask, base_probs, -np.inf), axis=1)[:, ::-1]
    candidate_order = np.argsort(np.where(open_mask, candidate_probs, -np.inf), axis=1)[:, ::-1]
    base_choice = np.argmax(np.where(open_mask, base_probs, -np.inf), axis=1)
    candidate_choice = np.argmax(np.where(open_mask, candidate_probs, -np.inf), axis=1)
    valid_counts = np.maximum(open_mask.sum(axis=1), 1)

    base_top, base_second = _top_two(base_probs, open_mask)
    candidate_top, candidate_second = _top_two(candidate_probs, open_mask)
    base_rows, base_cols = np.divmod(base_choice, cols)
    candidate_rows, candidate_cols = np.divmod(candidate_choice, cols)
    base_edge, base_is_edge, base_is_corner = _geometry(base_rows, base_cols, rows, cols)
    candidate_edge, candidate_is_edge, candidate_is_corner = _geometry(
        candidate_rows,
        candidate_cols,
        rows,
        cols,
    )

    if risk_maps is None:
        base_risk = np.zeros(batch, dtype=np.float32)
        candidate_risk = np.zeros(batch, dtype=np.float32)
    else:
        risk_maps = np.asarray(risk_maps, dtype=np.float32)
        if risk_maps.shape != (batch, rows, cols):
            raise ValueError("risk_maps must have shape [batch, rows, cols]")
        risk_flat = np.nan_to_num(risk_maps.reshape(batch, cells), nan=0.0, posinf=1.0, neginf=0.0)
        base_risk = risk_flat[np.arange(batch), base_choice]
        candidate_risk = risk_flat[np.arange(batch), candidate_choice]

    features = np.stack(
        [
            1.0 - _global_column(global_features, 1),
            _global_column(global_features, 2),
            _global_column(global_features, 3),
            _global_column(global_features, 4),
            _global_column(global_features, 5),
            _entropy(base_probs, open_mask),
            _entropy(candidate_probs, open_mask),
            base_top - base_second,
            candidate_top - candidate_second,
            base_top,
            candidate_top,
            candidate_top - base_top,
            (candidate_choice != base_choice).astype(np.float32),
            candidate_edge,
            base_edge,
            candidate_is_edge.astype(np.float32),
            candidate_is_corner.astype(np.float32),
            base_is_edge.astype(np.float32),
            base_is_corner.astype(np.float32),
            candidate_risk,
            base_risk,
            base_risk - candidate_risk,
            1.0 - candidate_risk,
            1.0 - base_risk,
            _rank_of_target(base_order, candidate_choice, valid_counts),
            _rank_of_target(candidate_order, base_choice, valid_counts),
            (np.abs(candidate_rows - base_rows) + np.abs(candidate_cols - base_cols)).astype(np.float32)
            / max(1.0, float(rows + cols - 2)),
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
    second = (
        np.nan_to_num(ordered[:, 1], nan=0.0, neginf=0.0)
        if ordered.shape[1] > 1
        else np.zeros_like(top)
    )
    return top, second


def _entropy(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    terms = np.where(mask, probabilities * np.log(np.maximum(probabilities, 1e-8)), 0.0)
    raw = -terms.sum(axis=1)
    normalizer = np.log(np.maximum(mask.sum(axis=1), 2))
    return np.divide(raw, normalizer, out=np.zeros_like(raw), where=normalizer > 0.0)


def _rank_of_target(order: np.ndarray, target: np.ndarray, valid_counts: np.ndarray) -> np.ndarray:
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
    is_edge = edge_distance == 0.0
    is_corner = ((rows == 0) | (rows == row_count - 1)) & ((cols == 0) | (cols == col_count - 1))
    return (
        edge_distance / max(1.0, float(min(row_count, col_count))),
        is_edge,
        is_corner,
    )
