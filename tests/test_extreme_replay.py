from __future__ import annotations

from pathlib import Path

import numpy as np

from minesweeper_rl.extreme_replay import (
    load_extreme_dataset,
    relabel_counterfactual_values_risk_aware,
    save_extreme_dataset,
)
from minesweeper_rl.features import action_to_index
from minesweeper_rl.types import Action, ActionType, EpisodeTransition


def make_transition() -> EpisodeTransition:
    rows, cols = 2, 3
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    expert_mask = np.zeros_like(action_mask)
    expert_mask[0, 0, 0] = True
    return EpisodeTransition(
        board=np.arange(rows * cols, dtype=np.float32).reshape(1, rows, cols),
        global_features=np.array([0.0, 0.9, 1.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=0,
        reward=-1.0,
        done=True,
        expert_action_mask=expert_mask,
        mine_mask=np.array([[False, True, False], [False, False, False]], dtype=bool),
        risk_map=np.full((rows, cols), 0.25, dtype=np.float32),
        counterfactual_open_values=np.array([[0.5, -1.0, 0.25], [0.25, 0.75, np.nan]], dtype=np.float32),
        expert_is_guess=True,
        source_quality=0.8,
        extreme_score=3.75,
        extreme_family="corner_guess_tail",
    )


def test_extreme_dataset_round_trip(tmp_path: Path) -> None:
    path = tmp_path / "extreme_transitions.npz"
    manifest = save_extreme_dataset(path, [make_transition()], metadata={"source": "test"})

    transitions, loaded_manifest = load_extreme_dataset(path)

    assert manifest["records"] == 1
    assert manifest["label_summary"]["guess_records"] == 1
    assert manifest["label_summary"]["counterfactual_records"] == 1
    assert manifest["label_summary"]["counterfactual_candidate_labels"] == 5
    assert manifest["label_summary"]["negative_counterfactual_candidates"] == 1
    assert np.isclose(manifest["label_summary"]["avg_behavior_regret"], 0.25)
    assert loaded_manifest["metadata"]["source"] == "test"
    assert len(transitions) == 1
    transition = transitions[0]
    assert np.array_equal(transition.board, make_transition().board)
    assert transition.expert_action_index == 0
    assert transition.expert_is_guess is True
    assert np.isclose(transition.source_quality, 0.8)
    assert np.isclose(transition.extreme_score, 3.75)
    assert transition.extreme_family == "corner_guess_tail"
    assert transition.expert_action_mask is not None
    assert transition.mine_mask is not None
    assert transition.risk_map is not None
    assert transition.counterfactual_open_values is not None
    assert np.isclose(transition.counterfactual_open_values[0, 0], 0.5)
    assert np.isnan(transition.counterfactual_open_values[1, 2])
    assert (path.with_suffix(".json")).exists()


def test_risk_aware_relabel_prioritizes_risk_and_mines() -> None:
    transition = make_transition()
    relabelled = relabel_counterfactual_values_risk_aware(
        [transition],
        risk_weight=1.0,
        outcome_weight=0.0,
        mine_penalty=1.0,
    )[0]

    assert relabelled.counterfactual_open_values is not None
    values = relabelled.counterfactual_open_values
    assert np.isclose(values[0, 1], -1.0)
    assert values[0, 0] > values[0, 1]
