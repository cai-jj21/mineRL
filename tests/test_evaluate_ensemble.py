from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_ensemble.py"
SPEC = importlib.util.spec_from_file_location("evaluate_ensemble_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
evaluate_ensemble = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = evaluate_ensemble
SPEC.loader.exec_module(evaluate_ensemble)


class FixedScoreTrainer:
    def __init__(self, value: float) -> None:
        self.value = float(value)
        self.config = type("Config", (), {"inference_augment_flips": False})()

    def _predict_policy_scores_batch(
        self,
        *,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        action_masks: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        del global_features_batch, action_masks, use_flip_ensemble
        return torch.full((boards.shape[0], 4), self.value, dtype=torch.float32)


def test_normalize_model_weights_defaults_to_equal() -> None:
    assert evaluate_ensemble.normalize_model_weights([], expected=3) == pytest.approx(
        [1 / 3, 1 / 3, 1 / 3]
    )


def test_average_policy_scores_uses_model_weights() -> None:
    scores = evaluate_ensemble.average_policy_scores(
        trainers=[FixedScoreTrainer(1.0), FixedScoreTrainer(3.0)],
        model_weights=[0.25, 0.75],
        boards=np.zeros((2, 1, 2, 2), dtype=np.float32),
        global_features_batch=np.zeros((2, 6), dtype=np.float32),
        action_masks=np.ones((2, 4, 1, 1), dtype=bool),
    )

    assert torch.allclose(scores, torch.full((2, 4), 2.5))
