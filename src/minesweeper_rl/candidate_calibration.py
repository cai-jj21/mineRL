from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


CANDIDATE_FEATURE_NAMES = (
    "step_ratio",
    "progress",
    "flag_ratio",
    "hidden_ratio",
    "remaining_mines_ratio",
    "covered_ratio",
    "base_probability",
    "specialist_probability",
    "probability_advantage",
    "base_rank",
    "specialist_rank",
    "row_norm",
    "col_norm",
    "edge_distance",
    "center_prior",
    "is_edge",
    "is_corner",
    "frontier",
    "flag_neighbors",
    "hidden_neighbors",
    "revealed_neighbors",
    "neighbor_clue_count",
    "neighbor_clue_sum",
    "neighbor_clue_mean",
    "neighbor_clue_max",
)
SOLVER_RISK_FEATURE_NAMES = (
    "solver_risk",
    "solver_safe_probability",
    "solver_risk_rank",
)
RISK_AWARE_CANDIDATE_FEATURE_NAMES = CANDIDATE_FEATURE_NAMES + SOLVER_RISK_FEATURE_NAMES
BOARD_CONTEXT_CHANNEL_COUNT = 20


@dataclass
class CandidateValueCalibrator:
    """Visible-feature open-candidate value model trained from offline labels."""

    means: np.ndarray
    scales: np.ndarray
    weights: np.ndarray
    bias: float
    temperature: float = 0.35
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        self.weights = np.asarray(self.weights, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1 or self.weights.ndim != 1:
            raise ValueError("candidate calibrator parameters must be one-dimensional")
        if not (len(self.means) == len(self.scales) == len(self.weights) == len(self.feature_names)):
            raise ValueError("candidate calibrator feature and parameter counts must match")
        if not np.isfinite(self.weights).all() or not np.isfinite(self.bias):
            raise ValueError("candidate calibrator weights must be finite")
        self.scales = np.maximum(self.scales, 1e-6)
        self.temperature = max(float(self.temperature), 1e-6)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[-1] != len(self.weights):
            raise ValueError(f"expected {len(self.weights)} candidate feature columns, got {features.shape}")
        normalized = (features - self.means) / self.scales
        return (_linear_dense(normalized, self.weights) + float(self.bias)).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "minesweeper_rl_candidate_value_calibrator_v1",
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "weights": self.weights.tolist(),
            "bias": float(self.bias),
            "temperature": float(self.temperature),
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "CandidateValueCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(payload.get("feature_names", CANDIDATE_FEATURE_NAMES))
        validate_candidate_feature_names(feature_names)
        return cls(
            means=np.asarray(payload["means"], dtype=np.float32),
            scales=np.asarray(payload["scales"], dtype=np.float32),
            weights=np.asarray(payload["weights"], dtype=np.float32),
            bias=float(payload["bias"]),
            temperature=float(payload.get("temperature", 0.35)),
            feature_names=feature_names,
        )


@dataclass
class NeuralCandidateValueCalibrator:
    """Small MLP candidate value model trained from visible OPEN features."""

    means: np.ndarray
    scales: np.ndarray
    layers: list[dict[str, object]]
    temperature: float = 0.35
    feature_names: tuple[str, ...] = CANDIDATE_FEATURE_NAMES

    def __post_init__(self) -> None:
        self.means = np.asarray(self.means, dtype=np.float32)
        self.scales = np.asarray(self.scales, dtype=np.float32)
        if self.means.ndim != 1 or self.scales.ndim != 1:
            raise ValueError("neural candidate calibrator normalization must be one-dimensional")
        if len(self.means) != len(self.scales) or len(self.means) != len(self.feature_names):
            raise ValueError("neural candidate calibrator feature and normalization counts must match")
        if not self.layers:
            raise ValueError("neural candidate calibrator requires at least one layer")

        expected_in = len(self.feature_names)
        normalized_layers: list[dict[str, object]] = []
        for layer in self.layers:
            weights = np.asarray(layer["weights"], dtype=np.float32)
            bias = np.asarray(layer["bias"], dtype=np.float32)
            activation = str(layer.get("activation", "identity"))
            if weights.ndim != 2 or bias.ndim != 1:
                raise ValueError("neural candidate layer weights must be 2D and bias must be 1D")
            if weights.shape[0] != expected_in or weights.shape[1] != bias.shape[0]:
                raise ValueError("neural candidate layer shape mismatch")
            if activation not in {"relu", "silu", "tanh", "identity"}:
                raise ValueError(f"unsupported neural candidate activation: {activation}")
            normalized_layers.append(
                {
                    "weights": weights,
                    "bias": bias,
                    "activation": activation,
                }
            )
            expected_in = int(weights.shape[1])
        if expected_in != 1:
            raise ValueError("neural candidate calibrator must output one value per candidate")
        self.layers = normalized_layers
        self.scales = np.maximum(self.scales, 1e-6)
        self.temperature = max(float(self.temperature), 1e-6)

    def predict_features(self, features: np.ndarray) -> np.ndarray:
        features = np.asarray(features, dtype=np.float32)
        if features.ndim == 1:
            features = features[None, :]
        if features.shape[-1] != len(self.feature_names):
            raise ValueError(f"expected {len(self.feature_names)} candidate feature columns, got {features.shape}")
        x = (features - self.means) / self.scales
        for layer in self.layers:
            x = _matrix_dense(x, np.asarray(layer["weights"], dtype=np.float32)) + np.asarray(
                layer["bias"],
                dtype=np.float32,
            )
            activation = str(layer["activation"])
            if activation == "relu":
                x = np.maximum(x, 0.0)
            elif activation == "silu":
                x = x / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))
            elif activation == "tanh":
                x = np.tanh(x)
        return x.reshape(-1).astype(np.float32)

    def to_dict(self) -> dict[str, object]:
        return {
            "format": "minesweeper_rl_neural_candidate_value_calibrator_v1",
            "feature_names": list(self.feature_names),
            "means": self.means.tolist(),
            "scales": self.scales.tolist(),
            "temperature": float(self.temperature),
            "layers": [
                {
                    "weights": np.asarray(layer["weights"], dtype=np.float32).tolist(),
                    "bias": np.asarray(layer["bias"], dtype=np.float32).tolist(),
                    "activation": str(layer["activation"]),
                }
                for layer in self.layers
            ],
        }

    def save(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> "NeuralCandidateValueCalibrator":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        feature_names = tuple(payload.get("feature_names", CANDIDATE_FEATURE_NAMES))
        validate_candidate_feature_names(feature_names)
        return cls(
            means=np.asarray(payload["means"], dtype=np.float32),
            scales=np.asarray(payload["scales"], dtype=np.float32),
            layers=list(payload["layers"]),
            temperature=float(payload.get("temperature", 0.35)),
            feature_names=feature_names,
        )


CandidateCalibrator = CandidateValueCalibrator | NeuralCandidateValueCalibrator


def load_candidate_calibrator(path: Path | str) -> CandidateCalibrator:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    calibrator_format = payload.get("format")
    if calibrator_format == "minesweeper_rl_candidate_value_calibrator_v1":
        return CandidateValueCalibrator.load(path)
    if calibrator_format == "minesweeper_rl_neural_candidate_value_calibrator_v1":
        return NeuralCandidateValueCalibrator.load(path)
    raise ValueError(f"unsupported candidate calibrator format: {calibrator_format}")


def _linear_dense(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (features.astype(np.float32, copy=False) * weights.astype(np.float32, copy=False)).sum(axis=1)


def _matrix_dense(features: np.ndarray, weights: np.ndarray) -> np.ndarray:
    return (
        features.astype(np.float32, copy=False)[:, :, None]
        * weights.astype(np.float32, copy=False)[None, :, :]
    ).sum(axis=1)


def build_candidate_features(
    *,
    base_scores: np.ndarray,
    specialist_scores: np.ndarray,
    global_features: np.ndarray,
    boards: np.ndarray,
    action_masks: np.ndarray,
    scores_are_probabilities: bool,
    risk_maps: np.ndarray | None = None,
    local_context_radius: int = 0,
) -> np.ndarray:
    base_scores = np.asarray(base_scores, dtype=np.float32)
    specialist_scores = np.asarray(specialist_scores, dtype=np.float32)
    global_features = np.asarray(global_features, dtype=np.float32)
    boards = np.asarray(boards, dtype=np.float32)
    action_masks = np.asarray(action_masks, dtype=bool)
    if action_masks.ndim != 4:
        raise ValueError("action_masks must have shape [batch, actions, rows, cols]")
    batch, _, rows, cols = action_masks.shape
    cells = rows * cols
    if base_scores.shape != specialist_scores.shape or base_scores.shape != (batch, action_masks.shape[1] * cells):
        raise ValueError("score shape does not match action mask shape")

    open_mask = action_masks[:, 0].reshape(batch, cells)
    if risk_maps is not None:
        risk_maps = np.asarray(risk_maps, dtype=np.float32)
        if risk_maps.shape != (batch, rows, cols):
            raise ValueError(
                "risk_maps must have shape [batch, rows, cols], "
                f"got {risk_maps.shape}, expected {(batch, rows, cols)}"
            )
    local_context_radius = max(0, int(local_context_radius))
    base_prob = _open_distributions(base_scores[:, :cells], open_mask, scores_are_probabilities)
    specialist_prob = _open_distributions(specialist_scores[:, :cells], open_mask, scores_are_probabilities)
    base_rank = _normalized_ranks(base_prob, open_mask)
    specialist_rank = _normalized_ranks(specialist_prob, open_mask)

    row_grid = np.repeat(np.arange(rows, dtype=np.float32), cols)
    col_grid = np.tile(np.arange(cols, dtype=np.float32), rows)
    row_norm = row_grid / max(1.0, float(rows - 1))
    col_norm = col_grid / max(1.0, float(cols - 1))
    edge_distance = np.minimum.reduce([row_grid, rows - 1 - row_grid, col_grid, cols - 1 - col_grid])
    edge_distance = edge_distance / max(1.0, float(min(rows, cols)))
    is_edge = edge_distance == 0.0
    is_corner = ((row_grid == 0.0) | (row_grid == rows - 1)) & ((col_grid == 0.0) | (col_grid == cols - 1))
    center_prior = _board_channel(boards, 19).reshape(batch, cells)
    frontier = _board_channel(boards, 11).reshape(batch, cells)
    flag_neighbors = _board_channel(boards, 12).reshape(batch, cells)
    hidden_neighbors = _board_channel(boards, 13).reshape(batch, cells)
    revealed_neighbors = _board_channel(boards, 14).reshape(batch, cells)
    clue_count, clue_sum, clue_mean, clue_max = _neighbor_clue_features(boards, rows, cols)
    repeated_globals = [
        _global_column(global_features, index)[:, None].repeat(cells, axis=1)
        for index in range(6)
    ]

    feature_parts = [
        *repeated_globals,
        base_prob,
        specialist_prob,
        specialist_prob - base_prob,
        base_rank,
        specialist_rank,
        np.broadcast_to(row_norm, (batch, cells)),
        np.broadcast_to(col_norm, (batch, cells)),
        np.broadcast_to(edge_distance, (batch, cells)),
        center_prior,
        np.broadcast_to(is_edge.astype(np.float32), (batch, cells)),
        np.broadcast_to(is_corner.astype(np.float32), (batch, cells)),
        frontier,
        flag_neighbors,
        hidden_neighbors,
        revealed_neighbors,
        clue_count,
        clue_sum,
        clue_mean,
        clue_max,
    ]
    if risk_maps is not None:
        solver_risk = np.clip(risk_maps.reshape(batch, cells), 0.0, 1.0)
        solver_safe_probability = 1.0 - solver_risk
        solver_risk_rank = _normalized_ranks(
            solver_safe_probability,
            open_mask,
        )
        feature_parts.extend(
            [
                solver_risk,
                solver_safe_probability,
                solver_risk_rank,
            ]
        )
    if local_context_radius > 0:
        padded = np.pad(
            boards[:, :BOARD_CONTEXT_CHANNEL_COUNT],
            (
                (0, 0),
                (0, BOARD_CONTEXT_CHANNEL_COUNT - min(boards.shape[1], BOARD_CONTEXT_CHANNEL_COUNT)),
                (local_context_radius, local_context_radius),
                (local_context_radius, local_context_radius),
            ),
            mode="constant",
        )
        for channel in range(BOARD_CONTEXT_CHANNEL_COUNT):
            for dr in range(-local_context_radius, local_context_radius + 1):
                for dc in range(-local_context_radius, local_context_radius + 1):
                    shifted = padded[
                        :,
                        channel,
                        local_context_radius + dr : local_context_radius + dr + rows,
                        local_context_radius + dc : local_context_radius + dc + cols,
                    ]
                    feature_parts.append(shifted.reshape(batch, cells))
    features = np.stack(feature_parts, axis=2)
    return np.nan_to_num(features.astype(np.float32), nan=0.0, posinf=0.0, neginf=0.0)


def candidate_policy_probabilities(
    calibrator: CandidateCalibrator,
    *,
    base_scores: np.ndarray,
    specialist_scores: np.ndarray,
    global_features: np.ndarray,
    boards: np.ndarray,
    action_masks: np.ndarray,
    scores_are_probabilities: bool,
    risk_maps: np.ndarray | None = None,
    local_context_radius: int | None = None,
) -> np.ndarray:
    feature_options = candidate_feature_options(calibrator.feature_names)
    if local_context_radius is None:
        local_context_radius = int(feature_options["local_context_radius"])
    features = build_candidate_features(
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        global_features=global_features,
        boards=boards,
        action_masks=action_masks,
        scores_are_probabilities=scores_are_probabilities,
        risk_maps=risk_maps,
        local_context_radius=int(local_context_radius),
    )
    batch, cells, feature_count = features.shape
    values = calibrator.predict_features(features.reshape(batch * cells, feature_count)).reshape(batch, cells)
    open_mask = action_masks[:, 0].reshape(batch, cells)
    masked = np.where(open_mask, values / calibrator.temperature, -1e9)
    shifted = masked - masked.max(axis=1, keepdims=True)
    exponentials = np.exp(np.clip(shifted, -80.0, 0.0)) * open_mask
    return (exponentials / np.maximum(exponentials.sum(axis=1, keepdims=True), 1e-8)).astype(np.float32)


def build_candidate_feature_names(
    *,
    solver_risk_features: bool = False,
    local_context_radius: int = 0,
) -> tuple[str, ...]:
    radius = max(0, int(local_context_radius))
    names = list(CANDIDATE_FEATURE_NAMES)
    if solver_risk_features:
        names.extend(SOLVER_RISK_FEATURE_NAMES)
    if radius > 0:
        for channel in range(BOARD_CONTEXT_CHANNEL_COUNT):
            for dr in range(-radius, radius + 1):
                for dc in range(-radius, radius + 1):
                    names.append(f"context_c{channel}_dr{dr}_dc{dc}")
    return tuple(names)


def candidate_feature_options(feature_names: tuple[str, ...] | list[str]) -> dict[str, object]:
    names = tuple(feature_names)
    if names[: len(CANDIDATE_FEATURE_NAMES)] != CANDIDATE_FEATURE_NAMES:
        raise ValueError("candidate calibrator feature schema does not match this code version")
    cursor = len(CANDIDATE_FEATURE_NAMES)
    solver_risk_features = names[cursor : cursor + len(SOLVER_RISK_FEATURE_NAMES)] == SOLVER_RISK_FEATURE_NAMES
    if solver_risk_features:
        cursor += len(SOLVER_RISK_FEATURE_NAMES)
    context_names = names[cursor:]
    if not context_names:
        local_context_radius = 0
    else:
        parsed: list[tuple[int, int, int]] = []
        for name in context_names:
            parts = name.split("_")
            if len(parts) != 4 or parts[0] != "context":
                raise ValueError("candidate calibrator feature schema does not match this code version")
            try:
                channel = int(parts[1][1:])
                dr = int(parts[2][2:])
                dc = int(parts[3][2:])
            except (TypeError, ValueError, IndexError):
                raise ValueError("candidate calibrator feature schema does not match this code version") from None
            if channel < 0 or channel >= BOARD_CONTEXT_CHANNEL_COUNT:
                raise ValueError("candidate calibrator feature schema does not match this code version")
            parsed.append((channel, dr, dc))
        max_radius = max(max(abs(dr), abs(dc)) for _, dr, dc in parsed)
        expected = build_candidate_feature_names(
            solver_risk_features=solver_risk_features,
            local_context_radius=max_radius,
        )
        if names != expected:
            raise ValueError("candidate calibrator feature schema does not match this code version")
        local_context_radius = max_radius
    expected = build_candidate_feature_names(
        solver_risk_features=solver_risk_features,
        local_context_radius=local_context_radius,
    )
    if names != expected:
        raise ValueError("candidate calibrator feature schema does not match this code version")
    return {
        "solver_risk_features": solver_risk_features,
        "local_context_radius": local_context_radius,
    }


def validate_candidate_feature_names(feature_names: tuple[str, ...] | list[str]) -> None:
    candidate_feature_options(tuple(feature_names))


def _global_column(global_features: np.ndarray, index: int) -> np.ndarray:
    if global_features.shape[1] <= index:
        return np.zeros(global_features.shape[0], dtype=np.float32)
    return global_features[:, index]


def _board_channel(boards: np.ndarray, index: int) -> np.ndarray:
    if boards.shape[1] <= index:
        return np.zeros((boards.shape[0], boards.shape[2], boards.shape[3]), dtype=np.float32)
    return boards[:, index]


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


def _normalized_ranks(probabilities: np.ndarray, mask: np.ndarray) -> np.ndarray:
    order = np.argsort(np.where(mask, probabilities, -np.inf), axis=1)[:, ::-1]
    ranks = np.zeros_like(probabilities, dtype=np.float32)
    counts = np.maximum(mask.sum(axis=1), 1)
    for row in range(probabilities.shape[0]):
        ranks[row, order[row]] = np.arange(probabilities.shape[1], dtype=np.float32)
        ranks[row, ~mask[row]] = float(counts[row])
        ranks[row] = ranks[row] / max(1.0, float(counts[row] - 1))
    return ranks


def _neighbor_clue_features(
    boards: np.ndarray,
    rows: int,
    cols: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    batch = boards.shape[0]
    clue_count = np.zeros((batch, rows, cols), dtype=np.float32)
    clue_sum = np.zeros((batch, rows, cols), dtype=np.float32)
    clue_max = np.zeros((batch, rows, cols), dtype=np.float32)
    for clue in range(1, 9):
        channel = 2 + clue
        if boards.shape[1] <= channel:
            continue
        layer = boards[:, channel]
        summed = _neighbor_sum(layer)
        clue_count += summed
        clue_sum += summed * float(clue)
        clue_max = np.maximum(clue_max, _neighbor_max(layer) * float(clue))
    clue_mean = np.divide(clue_sum, np.maximum(clue_count, 1.0), out=np.zeros_like(clue_sum), where=clue_count > 0.0)
    return (
        clue_count.reshape(batch, rows * cols) / 8.0,
        clue_sum.reshape(batch, rows * cols) / 36.0,
        clue_mean.reshape(batch, rows * cols) / 8.0,
        clue_max.reshape(batch, rows * cols) / 8.0,
    )


def _neighbor_sum(layer: np.ndarray) -> np.ndarray:
    batch, rows, cols = layer.shape
    padded = np.pad(layer, ((0, 0), (1, 1), (1, 1)), mode="constant")
    result = np.zeros((batch, rows, cols), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            result += padded[:, dr : dr + rows, dc : dc + cols]
    return result


def _neighbor_max(layer: np.ndarray) -> np.ndarray:
    batch, rows, cols = layer.shape
    padded = np.pad(layer, ((0, 0), (1, 1), (1, 1)), mode="constant")
    result = np.zeros((batch, rows, cols), dtype=np.float32)
    for dr in range(3):
        for dc in range(3):
            if dr == 1 and dc == 1:
                continue
            result = np.maximum(result, padded[:, dr : dr + rows, dc : dc + cols])
    return result
