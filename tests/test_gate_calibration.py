import numpy as np

from minesweeper_rl.gate_calibration import FEATURE_NAMES, GateCalibrator, build_gate_features


def test_gate_features_have_stable_schema_and_finite_values() -> None:
    boards = np.zeros((2, 20, 4, 5), dtype=np.float32)
    boards[:, 0] = 1.0
    masks = np.zeros((2, 4, 4, 5), dtype=bool)
    masks[:, 0, 1, 1] = True
    masks[:, 0, 1, 2] = True
    base = np.full((2, 4 * 4 * 5), -1.0, dtype=np.float32)
    specialist = base.copy()
    base[:, 1 * 5 + 1] = 0.8
    base[:, 1 * 5 + 2] = 0.2
    specialist[:, 1 * 5 + 1] = 0.1
    specialist[:, 1 * 5 + 2] = 0.9
    features = build_gate_features(
        base_scores=base,
        specialist_scores=specialist,
        global_features=np.zeros((2, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
    )
    assert features.shape == (2, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, FEATURE_NAMES.index("specialist_probability_advantage")] > 0.0


def test_gate_calibrator_round_trip(tmp_path) -> None:
    calibrator = GateCalibrator(
        means=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        scales=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        weights=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        bias=0.0,
        threshold=0.6,
    )
    path = tmp_path / "gate.json"
    calibrator.save(path)
    loaded = GateCalibrator.load(path)
    values = loaded.predict_proba_features(np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))
    assert values.shape == (2,)
    assert np.allclose(values, 0.5)
