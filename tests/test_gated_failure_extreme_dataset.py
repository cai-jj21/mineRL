from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_gated_failure_extreme_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_gated_failure_extreme_dataset_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
gated_failure_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = gated_failure_dataset
SPEC.loader.exec_module(gated_failure_dataset)


def test_partial_output_defaults_to_sidecar_npz(tmp_path: Path) -> None:
    args = SimpleNamespace(output=tmp_path / "gated_guess.npz", partial_output=None)

    assert gated_failure_dataset._partial_output_path(args) == tmp_path / "gated_guess.partial.npz"


def test_selected_output_transitions_respects_guess_family_and_cap() -> None:
    transitions = [
        _transition("guess", expert_is_guess=True),
        _transition("edge_guess", expert_is_guess=True),
        _transition("corner_guess_tail", expert_is_guess=True),
        _transition("endgame_tail", expert_is_guess=True),
        _transition("guess", expert_is_guess=False),
    ]
    args = SimpleNamespace(sample_family="guess", family_filter=[], max_records=2)

    selected = gated_failure_dataset._selected_output_transitions(transitions, args)

    assert [transition.extreme_family for transition in selected] == ["guess", "edge_guess"]


def test_selected_output_transitions_applies_family_filter() -> None:
    transitions = [
        _transition("guess", expert_is_guess=True),
        _transition("edge_guess", expert_is_guess=True),
        _transition("corner_guess", expert_is_guess=True),
    ]
    args = SimpleNamespace(sample_family="guess", family_filter=["corner_guess"], max_records=8)

    selected = gated_failure_dataset._selected_output_transitions(transitions, args)

    assert [transition.extreme_family for transition in selected] == ["corner_guess"]


def test_partial_metadata_reports_selected_counts() -> None:
    args = SimpleNamespace(
        base_checkpoint=Path("base20.pt"),
        base_ensemble_checkpoints=[Path("base100.pt")],
        specialist_checkpoint=Path("specialist.pt"),
        specialist_ensemble_checkpoints=[],
        games=128,
        seed=987,
        max_records=64,
        save_every_records=16,
        sample_family="guess",
        family_filter=["guess"],
        collect_guess_states=True,
        collect_guess_safe_left_min=141,
        collect_guess_safe_left_max=391,
        collect_guess_open_candidates_min=20,
        collect_guess_open_candidates_max=120,
        collect_guess_min_behavior_regret=2.5,
        collect_guess_behavior_mine_only=False,
        collect_guess_region="current-interior",
        counterfactual_labels=True,
        counterfactual_all_open=True,
        counterfactual_topk=64,
    )

    metadata = gated_failure_dataset._partial_metadata(
        args,
        metrics={"wins": 3, "losses": 2},
        source_families=Counter({"guess": 4, "edge_guess": 1}),
        selected_counts=Counter({"guess": 2}),
        reason="periodic",
    )

    assert metadata["partial"] is True
    assert metadata["save_reason"] == "periodic"
    assert metadata["selected_records"] == 2
    assert metadata["selected_families"] == {"guess": 2}
    assert metadata["source_families"] == {"edge_guess": 1, "guess": 4}
    assert metadata["collect_guess_open_candidates_min"] == 20
    assert metadata["collect_guess_open_candidates_max"] == 120
    assert metadata["collect_guess_min_behavior_regret"] == 2.5


def _transition(family: str, *, expert_is_guess: bool) -> SimpleNamespace:
    return SimpleNamespace(extreme_family=family, expert_is_guess=expert_is_guess)


def test_collect_guess_region_can_split_edge_and_corner() -> None:
    game = SimpleNamespace(rows=16, cols=30)
    corner = SimpleNamespace(row=0, col=0)
    edge = SimpleNamespace(row=0, col=8)
    interior = SimpleNamespace(row=4, col=8)

    assert gated_failure_dataset._collect_guess_region_allowed(
        corner,
        game,
        SimpleNamespace(collect_guess_region="current-corner"),
    )
    assert not gated_failure_dataset._collect_guess_region_allowed(
        edge,
        game,
        SimpleNamespace(collect_guess_region="current-corner"),
    )
    assert gated_failure_dataset._collect_guess_region_allowed(
        edge,
        game,
        SimpleNamespace(collect_guess_region="current-edge"),
    )
    assert not gated_failure_dataset._collect_guess_region_allowed(
        corner,
        game,
        SimpleNamespace(collect_guess_region="current-edge"),
    )
    assert gated_failure_dataset._collect_guess_region_allowed(
        interior,
        game,
        SimpleNamespace(collect_guess_region="current-interior"),
    )


def test_collect_guess_open_candidate_count_window() -> None:
    action_mask = np.zeros((3, 16, 30), dtype=bool)
    action_mask[0, 0, :10] = True
    game = SimpleNamespace(rows=16, cols=30)

    assert gated_failure_dataset._collect_guess_open_candidates_allowed(
        action_mask,
        game,
        SimpleNamespace(collect_guess_open_candidates_min=5, collect_guess_open_candidates_max=12),
    )
    assert not gated_failure_dataset._collect_guess_open_candidates_allowed(
        action_mask,
        game,
        SimpleNamespace(collect_guess_open_candidates_min=11, collect_guess_open_candidates_max=None),
    )
    assert not gated_failure_dataset._collect_guess_open_candidates_allowed(
        action_mask,
        game,
        SimpleNamespace(collect_guess_open_candidates_min=0, collect_guess_open_candidates_max=9),
    )


def test_collect_guess_behavior_filter_keeps_high_regret_and_mine() -> None:
    values = np.full((16, 30), np.nan, dtype=np.float32)
    values[2, 3] = -1.0
    values[2, 4] = 4.0
    mine_mask = np.zeros((16, 30), dtype=bool)
    transition = SimpleNamespace(
        action_index=2 * 30 + 3,
        counterfactual_open_values=values,
        mine_mask=mine_mask,
    )

    assert gated_failure_dataset._collect_guess_behavior_allowed(
        transition,
        SimpleNamespace(collect_guess_min_behavior_regret=4.0, collect_guess_behavior_mine_only=False),
    )
    assert not gated_failure_dataset._collect_guess_behavior_allowed(
        transition,
        SimpleNamespace(collect_guess_min_behavior_regret=6.0, collect_guess_behavior_mine_only=False),
    )
    assert not gated_failure_dataset._collect_guess_behavior_allowed(
        transition,
        SimpleNamespace(collect_guess_min_behavior_regret=0.0, collect_guess_behavior_mine_only=True),
    )

    mine_mask[2, 3] = True
    assert gated_failure_dataset._collect_guess_behavior_allowed(
        transition,
        SimpleNamespace(collect_guess_min_behavior_regret=0.0, collect_guess_behavior_mine_only=True),
    )
