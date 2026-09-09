from __future__ import annotations

import argparse
import os
import json
import sys
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.gate_calibration import FEATURE_NAMES, GateCalibrator


def main() -> None:
    parser = argparse.ArgumentParser(description="Train an offline logistic gate calibrator.")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--val-fraction", type=float, default=0.25)
    parser.add_argument("--epochs", type=int, default=1200)
    parser.add_argument("--lr", type=float, default=0.05)
    parser.add_argument("--l2", type=float, default=0.01)
    parser.add_argument("--threshold-min", type=float, default=0.35)
    parser.add_argument("--threshold-max", type=float, default=0.9)
    parser.add_argument("--threshold-steps", type=int, default=56)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    with np.load(args.dataset, allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float32)
        labels = np.asarray(data["labels"], dtype=np.float32)
        value_delta = np.asarray(data["value_delta"], dtype=np.float32)
        game_index = np.asarray(data["game_index"], dtype=np.int64)
    if features.ndim != 2 or features.shape[1] != len(FEATURE_NAMES):
        raise ValueError(f"expected [N, {len(FEATURE_NAMES)}] features, got {features.shape}")
    if len(features) < 8 or len(labels) != len(features):
        raise ValueError("dataset is too small or inconsistent")
    if len(np.unique(labels)) < 2:
        raise ValueError("gate dataset must contain both positive and negative labels")

    train_mask, val_mask = split_by_game(game_index, float(args.val_fraction))
    means = features[train_mask].mean(axis=0)
    scales = features[train_mask].std(axis=0)
    scales = np.maximum(scales, 1e-5)
    x_train = (features[train_mask] - means) / scales
    y_train = labels[train_mask]
    y_val = labels[val_mask]
    class_weight = balanced_class_weights(y_train)
    weights = np.zeros(features.shape[1], dtype=np.float64)
    bias = 0.0

    for _ in range(max(1, int(args.epochs))):
        logits = np.clip(x_train @ weights + bias, -40.0, 40.0)
        probabilities = 1.0 / (1.0 + np.exp(-logits))
        sample_weights = np.where(y_train > 0.5, class_weight[1], class_weight[0])
        residual = (probabilities - y_train) * sample_weights
        grad_w = (x_train.T @ residual) / max(1, len(x_train)) + float(args.l2) * weights
        grad_b = float(residual.mean())
        weights -= float(args.lr) * grad_w
        bias -= float(args.lr) * grad_b

    calibrator = GateCalibrator(
        means=means,
        scales=scales,
        weights=weights.astype(np.float32),
        bias=float(bias),
        threshold=0.5,
    )
    val_prob = calibrator.predict_proba_features(features[val_mask])
    threshold, threshold_metrics = choose_threshold(
        val_prob,
        y_val,
        value_delta[val_mask],
        threshold_min=float(args.threshold_min),
        threshold_max=float(args.threshold_max),
        threshold_steps=int(args.threshold_steps),
    )
    calibrator.threshold = threshold
    calibrator.save(args.output)
    report = {
        "ok": True,
        "dataset": str(args.dataset),
        "output": str(args.output),
        "records": int(len(features)),
        "train_records": int(train_mask.sum()),
        "validation_records": int(val_mask.sum()),
        "train_positive_rate": float(y_train.mean()),
        "validation_positive_rate": float(y_val.mean()),
        "threshold": float(threshold),
        "validation": threshold_metrics,
        "feature_weights": {
            name: float(weight)
            for name, weight in sorted(zip(FEATURE_NAMES, weights, strict=True), key=lambda item: abs(item[1]), reverse=True)
        },
    }
    report_path = args.output.with_name(args.output.name + ".report.json")
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    report["report"] = str(report_path)
    print(json.dumps(report, ensure_ascii=False, indent=2))


def split_by_game(game_index: np.ndarray, val_fraction: float) -> tuple[np.ndarray, np.ndarray]:
    unique_games = np.unique(game_index)
    if len(unique_games) < 2:
        order = np.arange(len(game_index))
        split = max(1, int(round(len(order) * (1.0 - val_fraction))))
        train_mask = np.zeros(len(order), dtype=bool)
        train_mask[order[:split]] = True
        return train_mask, ~train_mask
    cutoff = max(1, int(round(len(unique_games) * (1.0 - val_fraction))))
    train_games = set(unique_games[:cutoff].tolist())
    train_mask = np.asarray([game in train_games for game in game_index], dtype=bool)
    return train_mask, ~train_mask


def balanced_class_weights(labels: np.ndarray) -> dict[int, float]:
    positives = max(1, int((labels > 0.5).sum()))
    negatives = max(1, int((labels <= 0.5).sum()))
    total = positives + negatives
    return {0: total / (2.0 * negatives), 1: total / (2.0 * positives)}


def choose_threshold(
    probabilities: np.ndarray,
    labels: np.ndarray,
    deltas: np.ndarray,
    *,
    threshold_min: float,
    threshold_max: float,
    threshold_steps: int,
) -> tuple[float, dict[str, float | int]]:
    candidates = np.linspace(threshold_min, threshold_max, max(2, threshold_steps))
    best: tuple[float, tuple[float, float, float], dict[str, float | int]] | None = None
    for threshold in candidates:
        applied = probabilities >= threshold
        # The first term is the offline action-quality gain. The other terms
        # prefer a useful but selective gate when several thresholds tie.
        gain = float(np.where(applied, deltas, 0.0).mean())
        precision = float(labels[applied].mean()) if applied.any() else 0.0
        coverage = float(applied.mean())
        score = (gain, precision, -coverage)
        metrics = {
            "threshold": float(threshold),
            "offline_mean_gain": gain,
            "positive_precision": precision,
            "coverage": coverage,
            "applied_records": int(applied.sum()),
        }
        if best is None or score > best[1]:
            best = (float(threshold), score, metrics)
    assert best is not None
    return best[0], best[2]


if __name__ == "__main__":
    main()
