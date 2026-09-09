from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np

from minesweeper_rl.features import action_channel, action_to_index
from minesweeper_rl.types import Action, ActionType, EpisodeTransition


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_extreme_gate_calibration_dataset.py"
SPEC = importlib.util.spec_from_file_location("extreme_gate_dataset_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
extreme_gate_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = extreme_gate_dataset
SPEC.loader.exec_module(extreme_gate_dataset)


def make_transition() -> EpisodeTransition:
    rows, cols = 2, 3
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    action_mask[0, 0, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = -1.0
    counterfactual[0, 1] = 2.0
    return EpisodeTransition(
        board=np.zeros((20, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
        expert_is_guess=True,
    )


def test_gate_label_marks_specialist_better_than_base() -> None:
    rows, cols = 2, 3
    transition = make_transition()
    base_scores = np.zeros(4 * rows * cols, dtype=np.float32)
    specialist_scores = np.zeros_like(base_scores)
    base_scores[0] = 2.0
    base_scores[1] = 1.0
    specialist_scores[0] = 1.0
    specialist_scores[1] = 2.0

    label = extreme_gate_dataset._gate_label_for_transition(
        transition=transition,
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        rows=rows,
        cols=cols,
        label_margin=0.05,
        include_agreements=False,
        require_base_open=True,
    )

    assert label is not None
    assert label["label"] == 1
    assert label["value_delta"] == 3.0
    assert label["base_open_action"] == 0
    assert label["specialist_open_action"] == 1


def test_gate_label_skips_agreements_by_default() -> None:
    rows, cols = 2, 3
    transition = make_transition()
    base_scores = np.zeros(4 * rows * cols, dtype=np.float32)
    specialist_scores = np.zeros_like(base_scores)
    base_scores[0] = 2.0
    specialist_scores[0] = 2.0

    label = extreme_gate_dataset._gate_label_for_transition(
        transition=transition,
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        rows=rows,
        cols=cols,
        label_margin=0.05,
        include_agreements=False,
        require_base_open=False,
    )

    assert label is None


def test_gate_label_can_require_base_full_open() -> None:
    rows, cols = 2, 3
    transition = make_transition()
    transition.action_mask[action_channel(ActionType.FLAG), 0, 0] = True
    base_scores = np.zeros(4 * rows * cols, dtype=np.float32)
    specialist_scores = np.zeros_like(base_scores)
    base_scores[0] = 2.0
    base_scores[1] = 1.0
    base_scores[rows * cols] = 3.0
    specialist_scores[1] = 2.0

    label = extreme_gate_dataset._gate_label_for_transition(
        transition=transition,
        base_scores=base_scores,
        specialist_scores=specialist_scores,
        rows=rows,
        cols=cols,
        label_margin=0.05,
        include_agreements=False,
        require_base_open=True,
    )

    assert label is None
