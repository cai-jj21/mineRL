from __future__ import annotations

import importlib.util
import sys
from collections import Counter
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "build_sim_guess_training_dataset.py"
SPEC = importlib.util.spec_from_file_location("build_sim_guess_training_dataset_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
sim_guess_dataset = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = sim_guess_dataset
SPEC.loader.exec_module(sim_guess_dataset)


def test_partial_output_defaults_to_sidecar_npz(tmp_path: Path) -> None:
    args = SimpleNamespace(output=tmp_path / "plain_guess.npz", partial_output=None)

    assert sim_guess_dataset._partial_output_path(args) == tmp_path / "plain_guess.partial.npz"


def test_dataset_metadata_reports_sampling_yield(tmp_path: Path) -> None:
    args = _args(tmp_path)

    metadata = sim_guess_dataset._dataset_metadata(
        args,
        family_filter={"guess"},
        candidate_counts=Counter({"guess": 9}),
        selected_counts=Counter({"guess": 3}),
        rejected_non_guess=2,
        skipped_safe_left=1,
        skipped_family=3,
        skipped_behavior=0,
        stopped_after_game_cap=4,
        scanned_guess_states=12,
        wins=5,
        losses=6,
        started_at=0.0,
        partial=True,
        save_reason="periodic",
    )

    assert metadata["partial"] is True
    assert metadata["save_reason"] == "periodic"
    assert metadata["selected_records"] == 3
    assert metadata["scanned_guess_states"] == 12
    assert metadata["selected_to_scanned_guess_rate"] == 0.25
    assert metadata["selected_families"] == {"guess": 3}
    assert metadata["family_filter"] == ["guess"]
    assert metadata["behavior_selection"] == "and"


def test_behavior_selection_modes_cover_shortboard_cases() -> None:
    transition = SimpleNamespace(
        action_index=0 * 16 * 30 + 2 * 30 + 3,
        counterfactual_open_values=np.full((16, 30), np.nan, dtype=np.float32),
        mine_mask=np.zeros((16, 30), dtype=bool),
    )
    transition.counterfactual_open_values[2, 3] = -1.0
    transition.counterfactual_open_values[2, 4] = 2.0

    assert sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=2.0,
        mine_only=False,
        behavior_selection="and",
    )
    assert not sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=4.0,
        mine_only=False,
        behavior_selection="and",
    )
    assert sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=2.0,
        mine_only=False,
        behavior_selection="regret-only",
    )
    assert not sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=4.0,
        mine_only=False,
        behavior_selection="regret-only",
    )
    assert not sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=4.0,
        mine_only=False,
        behavior_selection="mine-only",
    )
    assert sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=2.0,
        mine_only=False,
        behavior_selection="mine-or-regret",
    )

    transition.mine_mask[2, 3] = True
    assert sim_guess_dataset._behavior_selection_allowed(
        transition,
        min_regret=0.0,
        mine_only=True,
        behavior_selection="mine-only",
    )


def _args(tmp_path: Path) -> SimpleNamespace:
    return SimpleNamespace(
        checkpoint=Path("model.pt"),
        output=tmp_path / "plain_guess.npz",
        partial_output=None,
        games=32,
        seed=123,
        max_steps=600,
        exact_limit=8,
        max_records=512,
        per_family_cap=512,
        max_guesses_per_game=4,
        save_every_records=16,
        safe_left_threshold=140,
        collect_safe_left_min=141,
        collect_safe_left_max=381,
        guess_topk=8,
        counterfactual_labels=True,
        counterfactual_topk=64,
        counterfactual_all_open=True,
        counterfactual_model_topk=None,
        policy_candidate_checkpoint=[],
        risk_head_weight=0.0,
        inference_flips=True,
        inference_ensemble="probs",
        behavior_selection="and",
    )
