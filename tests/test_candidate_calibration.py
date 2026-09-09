import numpy as np
import importlib.util
import sys
from pathlib import Path

import torch

from minesweeper_rl.candidate_calibration import (
    CANDIDATE_FEATURE_NAMES,
    RISK_AWARE_CANDIDATE_FEATURE_NAMES,
    CandidateValueCalibrator,
    NeuralCandidateValueCalibrator,
    build_candidate_feature_names,
    build_candidate_features,
    candidate_feature_options,
    candidate_policy_probabilities,
    load_candidate_calibrator,
)
from minesweeper_rl.types import EpisodeTransition


ROOT = Path(__file__).resolve().parents[1]
TRAIN_CALIBRATOR_PATH = ROOT / "scripts" / "train_candidate_calibrator.py"
TRAIN_CALIBRATOR_SPEC = importlib.util.spec_from_file_location(
    "train_candidate_calibrator_test",
    TRAIN_CALIBRATOR_PATH,
)
assert TRAIN_CALIBRATOR_SPEC is not None and TRAIN_CALIBRATOR_SPEC.loader is not None
train_candidate_calibrator = importlib.util.module_from_spec(TRAIN_CALIBRATOR_SPEC)
sys.modules[TRAIN_CALIBRATOR_SPEC.name] = train_candidate_calibrator
TRAIN_CALIBRATOR_SPEC.loader.exec_module(train_candidate_calibrator)


def test_candidate_features_and_probabilities_are_finite() -> None:
    boards = np.zeros((1, 20, 4, 5), dtype=np.float32)
    boards[:, 0] = 1.0
    masks = np.zeros((1, 4, 4, 5), dtype=bool)
    masks[:, 0, 1, 1] = True
    masks[:, 0, 1, 2] = True
    base = np.full((1, 4 * 4 * 5), -1.0, dtype=np.float32)
    specialist = base.copy()
    base[:, 1 * 5 + 1] = 0.8
    base[:, 1 * 5 + 2] = 0.2
    specialist[:, 1 * 5 + 1] = 0.2
    specialist[:, 1 * 5 + 2] = 0.8

    features = build_candidate_features(
        base_scores=base,
        specialist_scores=specialist,
        global_features=np.zeros((1, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
    )
    calibrator = CandidateValueCalibrator(
        means=np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        scales=np.ones(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        weights=np.zeros(len(CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        bias=0.0,
    )
    probs = candidate_policy_probabilities(
        calibrator,
        base_scores=base,
        specialist_scores=specialist,
        global_features=np.zeros((1, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
    )

    assert features.shape == (1, 20, len(CANDIDATE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.isclose(probs.sum(), 1.0)
    assert probs[0, 1 * 5 + 1] > 0.0


def test_solver_risk_features_are_available_to_risk_aware_calibrator() -> None:
    boards = np.zeros((1, 20, 2, 3), dtype=np.float32)
    masks = np.zeros((1, 4, 2, 3), dtype=bool)
    masks[:, 0, 0, 1:] = True
    base = np.zeros((1, 24), dtype=np.float32)
    specialist = np.zeros((1, 24), dtype=np.float32)
    risk_maps = np.asarray([[[0.0, 0.2, 0.8], [0.0, 0.1, 0.7]]], dtype=np.float32)
    features = build_candidate_features(
        base_scores=base,
        specialist_scores=specialist,
        global_features=np.zeros((1, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
        risk_maps=risk_maps,
    )
    calibrator = CandidateValueCalibrator(
        means=np.zeros(len(RISK_AWARE_CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        scales=np.ones(len(RISK_AWARE_CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        weights=np.zeros(len(RISK_AWARE_CANDIDATE_FEATURE_NAMES), dtype=np.float32),
        bias=0.0,
        feature_names=RISK_AWARE_CANDIDATE_FEATURE_NAMES,
    )
    probs = candidate_policy_probabilities(
        calibrator,
        base_scores=base,
        specialist_scores=specialist,
        global_features=np.zeros((1, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
        risk_maps=risk_maps,
    )

    assert features.shape == (1, 6, len(RISK_AWARE_CANDIDATE_FEATURE_NAMES))
    assert np.isfinite(features).all()
    assert np.isclose(probs.sum(), 1.0)


def test_local_context_features_round_trip_and_capture_neighbors() -> None:
    names = build_candidate_feature_names(local_context_radius=1)
    options = candidate_feature_options(names)
    assert options == {"solver_risk_features": False, "local_context_radius": 1}

    boards = np.zeros((1, 20, 2, 3), dtype=np.float32)
    boards[0, 0, 0, 2] = 1.0
    masks = np.zeros((1, 4, 2, 3), dtype=bool)
    masks[:, 0, :, :] = True
    scores = np.zeros((1, 24), dtype=np.float32)
    features = build_candidate_features(
        base_scores=scores,
        specialist_scores=scores,
        global_features=np.zeros((1, 6), dtype=np.float32),
        boards=boards,
        action_masks=masks,
        scores_are_probabilities=True,
        local_context_radius=1,
    )

    context_index = names.index("context_c0_dr0_dc1")
    assert features.shape == (1, 6, len(names))
    assert features[0, 0 * 3 + 1, context_index] == 1.0


def test_neural_candidate_calibrator_round_trips(tmp_path) -> None:
    feature_count = len(CANDIDATE_FEATURE_NAMES)
    calibrator = NeuralCandidateValueCalibrator(
        means=np.zeros(feature_count, dtype=np.float32),
        scales=np.ones(feature_count, dtype=np.float32),
        layers=[
            {
                "weights": np.ones((feature_count, 3), dtype=np.float32) * 0.1,
                "bias": np.zeros(3, dtype=np.float32),
                "activation": "silu",
            },
            {
                "weights": np.ones((3, 1), dtype=np.float32),
                "bias": np.zeros(1, dtype=np.float32),
                "activation": "identity",
            },
        ],
    )
    path = tmp_path / "candidate_mlp.json"
    calibrator.save(path)

    loaded = load_candidate_calibrator(path)
    values = loaded.predict_features(np.ones((2, feature_count), dtype=np.float32))

    assert values.shape == (2,)
    assert np.isfinite(values).all()


def test_candidate_training_filter_uses_family_and_safe_left() -> None:
    def transition(family: str, progress: float) -> EpisodeTransition:
        return EpisodeTransition(
            board=np.zeros((20, 16, 30), dtype=np.float32),
            global_features=np.asarray([0.0, progress, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            action_mask=np.zeros((4, 16, 30), dtype=bool),
            action_index=0,
            expert_action_index=None,
            reward=0.0,
            done=False,
            counterfactual_open_values=np.zeros((16, 30), dtype=np.float32),
            expert_is_guess=True,
            extreme_family=family,
        )

    rows = [
        transition("guess", 0.50),
        transition("guess_tail", 0.90),
        transition("edge_guess_tail", 0.90),
    ]

    filtered = train_candidate_calibrator.filter_training_transitions(
        rows,
        family_filter=["guess_tail"],
        safe_left_min=30,
        safe_left_max=50,
    )

    assert [row.extreme_family for row in filtered] == ["guess_tail"]


def test_dataset_weights_are_validated_and_applied() -> None:
    transition = EpisodeTransition(
        board=np.zeros((20, 16, 30), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=np.zeros((4, 16, 30), dtype=bool),
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        source_quality=1.5,
    )

    weights = train_candidate_calibrator.normalize_dataset_weights([1.0, 2.5], expected=2)
    weighted = train_candidate_calibrator.weighted_transitions([transition], dataset_weight=3.0)

    assert weights == [1.0, 2.5]
    assert weighted[0].source_quality == 4.5
    assert transition.source_quality == 1.5


def test_dataset_weights_reject_bad_count() -> None:
    try:
        train_candidate_calibrator.normalize_dataset_weights([1.0], expected=2)
    except ValueError as exc:
        assert "count must match" in str(exc)
    else:
        raise AssertionError("expected invalid dataset-weight count to fail")


def test_transition_rank_weight_uses_source_quality() -> None:
    action_mask = np.zeros((4, 2, 2), dtype=bool)
    action_mask[0] = True
    transition = EpisodeTransition(
        board=np.zeros((20, 2, 2), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        source_quality=2.0,
    )
    values = np.asarray([0.0, 4.0, 1.0, 2.0], dtype=np.float32)
    valid = np.ones(4, dtype=bool)

    weight = train_candidate_calibrator._transition_rank_weight(
        transition=transition,
        values=values,
        valid=valid,
        target_clip=4.0,
    )

    assert weight == 4.0


def test_pairwise_policy_topk_masks_match_base_specialist_shortlist() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 3.0
    specialist[0, 4] = 3.0
    action_mask = np.zeros((1, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    shortlist = train_candidate_calibrator._policy_topk_open_masks(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=2,
        cols=3,
        topk=1,
    )

    assert shortlist[0].tolist() == [True, False, False, False, True, False]


def test_negative_probability_mass_penalty_tracks_negative_mass() -> None:
    log_probs = torch.log(
        torch.tensor(
            [
                [0.9, 0.1],
                [0.1, 0.9],
            ],
            dtype=torch.float32,
        )
    )
    negative_mask = torch.tensor(
        [
            [False, True],
            [False, True],
        ]
    )

    penalties = train_candidate_calibrator._negative_probability_mass_penalty(
        log_probs=log_probs,
        negative_mask=negative_mask,
    )

    assert penalties[1] > penalties[0]


def test_weighted_softmax_rank_loss_can_penalize_negative_mass() -> None:
    log_probs = torch.log(torch.tensor([[0.2, 0.8]], dtype=torch.float32))
    targets = torch.tensor([[1.0, 0.0]], dtype=torch.float32)
    weights = torch.ones(1, dtype=torch.float32)
    negative_mask = torch.tensor([[False, True]])

    base = train_candidate_calibrator._weighted_softmax_rank_loss(
        log_probs=log_probs,
        targets=targets,
        negative_mask=negative_mask,
        weights=weights,
        negative_mass_coef=0.0,
    )
    penalized = train_candidate_calibrator._weighted_softmax_rank_loss(
        log_probs=log_probs,
        targets=targets,
        negative_mask=negative_mask,
        weights=weights,
        negative_mass_coef=1.0,
    )

    assert penalized > base


def test_qvalue_regression_uses_huber_loss() -> None:
    features = np.zeros((6, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
    features[:, 0] = np.arange(6, dtype=np.float32)
    features[:, 1] = np.arange(6, dtype=np.float32)
    targets = np.asarray([0.0, 1.0, 2.0, 3.0, 4.0, -1.0], dtype=np.float32)

    calibrator, metrics = train_candidate_calibrator.fit_mlp(
        features,
        targets,
        hidden_layers=1,
        hidden_units=8,
        epochs=3,
        learning_rate=1e-3,
        weight_decay=0.0,
        validation_fraction=0.2,
        negative_weight=2.0,
        positive_weight=1.0,
        seed=7,
        batch_size=4,
        temperature=0.35,
        loss_kind="huber",
        huber_beta=1.0,
    )

    assert calibrator.feature_names == CANDIDATE_FEATURE_NAMES
    assert metrics["loss_kind"] == "huber"
    assert metrics["huber_beta"] == 1.0


def test_pairwise_ranker_learns_best_candidate_on_toy_states() -> None:
    transitions = 6
    cells = 4
    features = np.zeros((transitions, cells, len(CANDIDATE_FEATURE_NAMES)), dtype=np.float32)
    valid = np.ones((transitions, cells), dtype=bool)
    values = np.tile(np.asarray([0.0, 3.0, -1.0, 1.0], dtype=np.float32), (transitions, 1))
    weights = np.ones(transitions, dtype=np.float32)
    for index in range(cells):
        features[:, index, 0] = float(index)
        features[:, index, 1] = values[:, index]

    calibrator, metrics = train_candidate_calibrator.fit_pairwise(
        transition_features=features,
        transition_valid_masks=valid,
        transition_values=values,
        transition_weights=weights,
        hidden_layers=0,
        hidden_units=8,
        epochs=25,
        learning_rate=5e-2,
        weight_decay=0.0,
        validation_fraction=0.2,
        seed=11,
        batch_size=3,
        temperature=0.35,
        negative_count=2,
        min_gap=0.1,
        pairwise_temperature=0.5,
        bad_negative_fraction=0.5,
        score_l2=1e-4,
    )

    predictions = calibrator.predict_features(features.reshape(transitions * cells, -1)).reshape(transitions, cells)

    assert metrics["train_pair_count"] > 0
    assert int(np.argmax(predictions[0])) == 1
