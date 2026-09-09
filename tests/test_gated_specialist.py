from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch
import numpy as np

from minesweeper_rl.gate_calibration import FEATURE_NAMES, GateCalibrator

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
SPEC = importlib.util.spec_from_file_location("evaluate_gated_specialist_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
gated = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gated
SPEC.loader.exec_module(gated)


def test_open_only_blend_preserves_non_open_channels() -> None:
    base = torch.zeros((1, 4 * 2 * 3), dtype=torch.float32)
    specialist = torch.ones_like(base)

    blended = gated._blend_scores(
        base_scores=base,
        specialist_scores=specialist,
        specialist_weight=0.5,
        blend_scope="open",
        rows=2,
        cols=3,
    )

    assert torch.equal(blended[:, :6], torch.full((1, 6), 0.5))
    assert torch.equal(blended[:, 6:], torch.zeros((1, 18)))


def test_blend_weight_is_clamped() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.ones_like(base)

    blended = gated._blend_scores(
        base_scores=base,
        specialist_scores=specialist,
        specialist_weight=2.0,
        blend_scope="all",
        rows=2,
        cols=3,
    )

    assert torch.equal(blended, specialist)


def test_normalize_ensemble_weights_defaults_to_equal() -> None:
    weights = gated.normalize_ensemble_weights([], expected=2, label="base")

    assert weights == [0.5, 0.5]


def test_normalize_ensemble_weights_rejects_bad_count() -> None:
    with pytest.raises(ValueError, match="count must match"):
        gated.normalize_ensemble_weights([1.0], expected=2, label="base")


def test_geomean_model_scores_prefers_consensus() -> None:
    base = torch.tensor([[0.9, 0.1]], dtype=torch.float32)
    specialist = torch.tensor([[0.8, 0.2]], dtype=torch.float32)
    mask = torch.ones_like(base, dtype=torch.bool)

    combined = gated.combine_model_scores(
        [base, specialist],
        [0.5, 0.5],
        reduction="geomean",
        flat_mask=mask,
    )

    assert combined.argmax(dim=1).item() == 0
    assert combined[0, 0] > combined[0, 1]


def test_open_disagree_margin_gate_requires_disagreement_and_margin() -> None:
    base = torch.zeros((3, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 1.0
    specialist[0, 1] = 2.0
    specialist[0, 2] = 1.0
    base[1, 0] = 1.0
    specialist[1, 1] = 2.0
    specialist[1, 2] = 1.9
    base[2, 0] = 1.0
    specialist[2, 0] = 2.0
    specialist[2, 1] = 1.0

    gate = gated._open_disagree_margin_gate(
        base_scores=base,
        specialist_scores=specialist,
        margin=0.5,
        rows=2,
        cols=3,
    )

    assert gate[:, 0].tolist() == [True, False, False]


def test_adjustment_gate_batch_defaults_to_safe_left() -> None:
    gate = gated._adjustment_gate_batch(
        "safe-left",
        safe_left=np.asarray([10, 80, 120]),
        threshold=80,
        solver_guess=np.asarray([False, True, True]),
    )

    assert gate.tolist() == [True, True, False]


def test_adjustment_gate_batch_can_require_solver_guess() -> None:
    gate = gated._adjustment_gate_batch(
        "safe-left-and-solver-guess",
        safe_left=np.asarray([10, 80, 120]),
        threshold=80,
        solver_guess=np.asarray([False, True, True]),
    )

    assert gate.tolist() == [False, True, False]


def test_candidate_gate_can_be_independent_from_specialist_gate() -> None:
    gate = gated._adjustment_gate_batch(
        "solver-guess",
        safe_left=np.asarray([10, 80, 120]),
        threshold=80,
        solver_guess=np.asarray([False, True, False]),
    )

    assert gate.tolist() == [False, True, False]


def test_evaluate_gated_supports_open_only_decisions() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    gate = gated._open_disagree_margin_gate(
        base_scores=base,
        specialist_scores=specialist,
        margin=0.1,
        rows=2,
        cols=3,
    )

    assert gate.shape == (1, 1)


def test_calibrated_gate_still_requires_margin() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 1.0
    specialist[0, 1] = 2.0
    specialist[0, 2] = 1.95
    calibrator = GateCalibrator(
        means=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        scales=np.ones(len(FEATURE_NAMES), dtype=np.float32),
        weights=np.zeros(len(FEATURE_NAMES), dtype=np.float32),
        bias=10.0,
        threshold=0.5,
    )

    gate = gated._specialist_gate(
        tail_mask=[True],
        base_scores=base,
        specialist_scores=specialist,
        blend_scope="open-disagree-margin",
        specialist_margin=0.1,
        rows=2,
        cols=3,
        gate_calibrator=calibrator,
        board_batch=np.zeros((1, 20, 2, 3), dtype=np.float32),
        global_batch=np.zeros((1, 6), dtype=np.float32),
        mask_batch=np.ones((1, 4, 2, 3), dtype=bool),
        scores_are_probabilities=False,
    )

    assert gate[:, 0].tolist() == [False]


def test_counterfactual_value_blend_only_changes_active_open_shortlist() -> None:
    scores = torch.zeros((2, 24), dtype=torch.float32)
    scores[0, 0] = 0.4
    scores[0, 1] = 0.1
    scores[1, 0] = 0.4
    scores[1, 1] = 0.1
    scores[:, 6:] = 0.05
    values = torch.zeros((2, 6), dtype=torch.float32)
    values[:, 1] = 3.0
    value_mask = torch.zeros((2, 6), dtype=torch.bool)
    value_mask[:, :2] = True

    adjusted = gated._blend_counterfactual_value_open_scores(
        scores=scores,
        counterfactual_values=values,
        value_tail=torch.tensor([True, False]),
        value_mask=value_mask,
        weight=1.0,
        temperature=0.1,
        scores_are_probabilities=True,
        rows=2,
        cols=3,
    )

    assert adjusted[0, 1] > adjusted[0, 0]
    assert torch.equal(adjusted[1, :6], scores[1, :6])
    assert torch.equal(adjusted[:, 6:], scores[:, 6:])


def test_candidate_blend_only_changes_active_open_shortlist() -> None:
    scores = torch.zeros((2, 24), dtype=torch.float32)
    scores[0, 0] = 0.4
    scores[0, 1] = 0.1
    scores[1, 0] = 0.4
    scores[1, 1] = 0.1
    scores[:, 6:] = 0.05
    candidate_probs = torch.zeros((2, 6), dtype=torch.float32)
    candidate_probs[:, 1] = 1.0
    candidate_mask = torch.zeros((2, 6), dtype=torch.bool)
    candidate_mask[:, :2] = True

    adjusted = gated._blend_candidate_open_scores(
        scores=scores,
        candidate_probabilities=candidate_probs,
        candidate_tail=torch.tensor([True, False]),
        candidate_mask=candidate_mask,
        weight=1.0,
        rows=2,
        cols=3,
    )

    assert adjusted[0, 1] > adjusted[0, 0]
    assert torch.equal(adjusted[1, :6], scores[1, :6])
    assert torch.equal(adjusted[:, 6:], scores[:, 6:])


def test_candidate_probability_blend_preserves_open_score_scale() -> None:
    scores = torch.zeros((1, 24), dtype=torch.float32)
    scores[0, :6] = torch.tensor([0.20, 0.10, 0.05, 0.03, 0.02, 0.01])
    candidate_probs = torch.zeros((1, 6), dtype=torch.float32)
    candidate_probs[0, 1] = 1.0
    candidate_mask = torch.zeros((1, 6), dtype=torch.bool)
    candidate_mask[0, :2] = True

    adjusted = gated._blend_candidate_open_scores(
        scores=scores,
        candidate_probabilities=candidate_probs,
        candidate_tail=torch.tensor([True]),
        candidate_mask=candidate_mask,
        weight=1.0,
        rows=2,
        cols=3,
        scores_are_probabilities=True,
    )

    assert adjusted[0, 1].item() == pytest.approx(scores[0, :6].sum().item())
    assert adjusted[0, 0].item() == pytest.approx(0.0)
    assert adjusted[0, 2].item() == pytest.approx(scores[0, 2].item())


def test_candidate_ranker_margin_gate_requires_decisive_legal_ranking() -> None:
    candidate_probabilities = np.asarray(
        [
            [0.40, 0.35, 0.25, 0.0],
            [0.70, 0.20, 0.10, 0.0],
        ],
        dtype=np.float32,
    )
    action_masks = np.zeros((2, 4, 2, 2), dtype=bool)
    action_masks[:, 0, 0, :3] = True

    gate = gated._candidate_ranker_margin_gate(
        candidate_probabilities=candidate_probabilities,
        action_masks=action_masks,
        min_margin=0.1,
    )

    assert gate.tolist() == [False, True]


def test_candidate_logit_blend_is_centered_on_active_shortlist() -> None:
    scores = torch.zeros((1, 24), dtype=torch.float32)
    scores[0, :6] = torch.tensor([2.0, 1.0, 0.5, 0.0, -0.5, -1.0])
    candidate_probs = torch.zeros((1, 6), dtype=torch.float32)
    candidate_probs[0, 1] = 0.75
    candidate_probs[0, 0] = 0.25
    candidate_mask = torch.zeros((1, 6), dtype=torch.bool)
    candidate_mask[0, :2] = True

    adjusted = gated._blend_candidate_open_scores(
        scores=scores,
        candidate_probabilities=candidate_probs,
        candidate_tail=torch.tensor([True]),
        candidate_mask=candidate_mask,
        weight=1.0,
        rows=2,
        cols=3,
        scores_are_probabilities=False,
    )

    assert adjusted[0, 1] > adjusted[0, 0]
    assert torch.equal(adjusted[0, 2:], scores[0, 2:])


def test_candidate_risk_safety_filter_rejects_higher_risk_choice() -> None:
    before = torch.zeros((1, 24), dtype=torch.float32)
    after = before.clone()
    before[0, 0] = 0.8
    before[0, 1] = 0.7
    after[0, 1] = 0.9
    action_mask = np.zeros((1, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True
    risk_maps = np.asarray([[[0.1, 0.6, 0.4], [0.3, 0.2, 0.5]]], dtype=np.float32)

    violations = gated._candidate_risk_safety_violations(
        before_scores=before,
        after_scores=after,
        action_masks=action_mask,
        risk_maps=risk_maps,
        active=torch.tensor([True]),
        rows=2,
        cols=3,
        mode="not-higher",
    )

    assert violations.tolist() == [True]


def test_candidate_shortlist_includes_ranker_top_candidates() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 4.0
    specialist[0, 1] = 4.0
    candidate_probabilities = np.zeros((1, 6), dtype=np.float32)
    candidate_probabilities[0, 4] = 1.0
    action_mask = np.zeros((1, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    shortlist = gated._candidate_shortlist_mask(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=2,
        cols=3,
        topk=1,
        candidate_probabilities=candidate_probabilities,
    )

    assert shortlist[0, :6].tolist() == [True, True, False, False, True, False]


def test_policy_only_candidate_shortlist_excludes_ranker_only_candidate() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 4.0
    specialist[0, 1] = 4.0
    action_mask = np.zeros((1, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    shortlist = gated._candidate_shortlist_mask(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=2,
        cols=3,
        topk=1,
        candidate_probabilities=None,
    )

    assert shortlist[0, :6].tolist() == [True, True, False, False, False, False]


def test_counterfactual_value_shortlist_includes_value_candidates() -> None:
    base = torch.zeros((1, 24), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    values = torch.zeros((1, 6), dtype=torch.float32)
    base[0, 0] = 3.0
    specialist[0, 1] = 3.0
    values[0, 2] = 3.0
    action_mask = np.zeros((1, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    shortlist = gated._counterfactual_value_shortlist_mask(
        base_scores=base,
        specialist_scores=specialist,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=2,
        cols=3,
        topk=1,
    )

    assert shortlist[0, :3].tolist() == [True, True, True]


def test_counterfactual_value_disagree_gate_requires_confident_disagreement() -> None:
    scores = torch.zeros((2, 24), dtype=torch.float32)
    scores[0, 0] = 3.0
    scores[1, 0] = 3.0
    values = torch.zeros((2, 6), dtype=torch.float32)
    values[0, 1] = 1.0
    values[0, 2] = 0.1
    values[1, 1] = 0.3
    values[1, 2] = 0.2
    action_mask = np.zeros((2, 4, 2, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    gate = gated._counterfactual_value_disagree_gate(
        scores=scores,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=2,
        cols=3,
        margin=0.5,
    )

    assert gate.tolist() == [True, False]


def test_counterfactual_value_policy_margin_gate_targets_uncertain_open_choices() -> None:
    scores = torch.zeros((3, 24), dtype=torch.float32)
    scores[0, 0] = 0.35
    scores[0, 1] = 0.32
    scores[1, 0] = 0.80
    scores[1, 1] = 0.10
    scores[2, 0] = 0.40
    action_mask = np.zeros((3, 4, 2, 3), dtype=bool)
    action_mask[0, 0, 0, :2] = True
    action_mask[1, 0, 0, :2] = True
    action_mask[2, 0, 0, 0] = True

    gate = gated._counterfactual_value_policy_margin_gate(
        scores=scores,
        action_masks=action_mask,
        rows=2,
        cols=3,
        max_margin=0.05,
    )

    assert gate.tolist() == [True, False, False]


def test_counterfactual_value_region_gate_can_target_boundary_choices() -> None:
    scores = torch.zeros((3, 36), dtype=torch.float32)
    scores[0, 0] = 3.0
    scores[1, 4] = 3.0
    scores[2, 4] = 3.0
    values = torch.zeros((3, 9), dtype=torch.float32)
    values[0, 4] = 3.0
    values[1, 0] = 3.0
    values[2, 4] = 3.0
    action_mask = np.zeros((3, 4, 3, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    current_gate = gated._counterfactual_value_region_gate(
        scores=scores,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="current-edge-or-corner",
    )
    value_gate = gated._counterfactual_value_region_gate(
        scores=scores,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="value-edge-or-corner",
    )
    either_gate = gated._counterfactual_value_region_gate(
        scores=scores,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="either-edge-or-corner",
    )
    interior_gate = gated._counterfactual_value_region_gate(
        scores=scores,
        counterfactual_values=values,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="both-interior",
    )

    assert current_gate.tolist() == [True, False, False]
    assert value_gate.tolist() == [False, True, False]
    assert either_gate.tolist() == [True, True, False]
    assert interior_gate.tolist() == [False, False, True]


def test_specialist_region_gate_can_target_base_and_specialist_geometry() -> None:
    base = torch.zeros((3, 36), dtype=torch.float32)
    specialist = torch.zeros_like(base)
    base[0, 0] = 3.0
    specialist[0, 4] = 3.0
    base[1, 4] = 3.0
    specialist[1, 0] = 3.0
    base[2, 4] = 3.0
    specialist[2, 4] = 3.0
    action_mask = np.zeros((3, 4, 3, 3), dtype=bool)
    action_mask[:, 0, :, :] = True

    current_gate = gated._specialist_region_gate(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="current-edge-or-corner",
    )
    specialist_gate = gated._specialist_region_gate(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="specialist-edge-or-corner",
    )
    either_gate = gated._specialist_region_gate(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="either-edge-or-corner",
    )
    interior_gate = gated._specialist_region_gate(
        base_scores=base,
        specialist_scores=specialist,
        action_masks=action_mask,
        rows=3,
        cols=3,
        region_gate="both-interior",
    )

    assert current_gate.tolist() == [True, False, False]
    assert specialist_gate.tolist() == [False, True, False]
    assert either_gate.tolist() == [True, True, False]
    assert interior_gate.tolist() == [False, False, True]
