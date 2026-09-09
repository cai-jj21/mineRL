import numpy as np

from minesweeper_rl.candidate_gate import (
    FEATURE_NAMES,
    CandidateGateCalibrator,
    build_candidate_gate_features,
)


def test_candidate_gate_features_are_finite_and_follow_open_mask() -> None:
    masks = np.zeros((1, 4, 2, 3), dtype=bool)
    masks[:, 0, 0, 1:] = True
    scores = np.full((1, 24), -1.0, dtype=np.float32)
    scores[0, 1] = 0.8
    scores[0, 2] = 0.2
    candidate = np.zeros((1, 6), dtype=np.float32)
    candidate[0, 2] = 1.0

    features = build_candidate_gate_features(
        base_scores=scores,
        candidate_probabilities=candidate,
        global_features=np.zeros((1, 6), dtype=np.float32),
        action_masks=masks,
        scores_are_probabilities=True,
    )

    assert features.shape == (1, len(FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert features[0, FEATURE_NAMES.index("candidate_choice_disagree")] == 1.0


def test_candidate_gate_round_trips_and_applies_threshold(tmp_path) -> None:
    gate = CandidateGateCalibrator(
        means=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        scales=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        weights=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        bias=1.0,
        threshold=0.7,
    )
    path = tmp_path / "candidate_gate.json"
    gate.save(path)

    loaded = CandidateGateCalibrator.load(path)
    accepted = loaded.should_apply_features(np.zeros((2, len(FEATURE_NAMES)), dtype=np.float32))

    assert accepted.tolist() == [True, True]
