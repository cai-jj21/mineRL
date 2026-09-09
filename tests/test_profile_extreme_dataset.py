from __future__ import annotations

from pathlib import Path

import numpy as np

from minesweeper_rl.extreme_replay import save_extreme_dataset
from minesweeper_rl.features import action_to_index
from minesweeper_rl.types import Action, ActionType, EpisodeTransition
from scripts import profile_extreme_dataset


def test_profile_transition_reports_region_and_regret() -> None:
    transition = _transition(row=0, col=3, values={(0, 3): -1.0, (2, 2): 5.0})

    row = profile_extreme_dataset.profile_transition(
        transition,
        dataset="sample.npz",
        index=4,
        total_safe=391,
    )

    assert row["dataset"] == "sample.npz"
    assert row["index"] == 4
    assert row["region"] == "edge"
    assert row["safe_left"] == 196
    assert row["safe_left_bucket"] == "141_240"
    assert row["open_candidate_count"] == 120
    assert row["open_candidate_bucket"] == "081_200"
    assert row["counterfactual_label_count"] == 2
    assert row["negative_counterfactual_candidates"] == 1
    assert row["behavior_counterfactual_value"] == -1.0
    assert row["behavior_regret"] == 6.0
    assert row["behavior_regret_bucket"] == "4_8"
    assert row["behavior_mine"] is True


def test_profile_datasets_round_trips_json_ready_rows(tmp_path: Path) -> None:
    dataset = tmp_path / "profile_source.npz"
    transitions = [
        _transition(row=0, col=0, values={(0, 0): 1.0, (1, 1): 2.0}),
        _transition(row=5, col=5, values={(5, 5): -1.0, (5, 6): 8.5}),
    ]
    save_extreme_dataset(dataset, transitions, metadata={"source": "test"})

    report = profile_extreme_dataset.profile_datasets([dataset], total_safe=391)

    assert report["summary"]["record_count"] == 2
    assert report["summary"]["regions"] == {"corner": 1, "interior": 1}
    assert report["summary"]["behavior_regret_buckets"] == {"1_4": 1, "gte_8": 1}
    assert report["by_region"]["corner"]["record_count"] == 1


def test_write_record_csv(tmp_path: Path) -> None:
    path = tmp_path / "records.csv"
    profile_extreme_dataset.write_record_csv(
        path,
        [
            {
                "dataset": "sample.npz",
                "index": 0,
                "family": "guess",
                "expert_is_guess": True,
                "region": "interior",
                "safe_left": 200,
                "safe_left_bucket": "141_240",
                "open_candidate_count": 100,
                "open_candidate_bucket": "081_200",
                "counterfactual_label_count": 2,
                "negative_counterfactual_rate": 0.5,
                "best_counterfactual_value": 3.0,
                "behavior_counterfactual_value": 1.0,
                "behavior_regret": 2.0,
                "behavior_regret_bucket": "1_4",
                "behavior_mine": False,
                "action_row": 4,
                "action_col": 5,
            }
        ],
    )

    text = path.read_text(encoding="utf-8")

    assert "dataset,index,family" in text
    assert "sample.npz,0,guess" in text


def _transition(row: int, col: int, values: dict[tuple[int, int], float]) -> EpisodeTransition:
    rows, cols = 16, 30
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :4, :] = True
    counterfactual_values = np.full((rows, cols), np.nan, dtype=np.float32)
    for (value_row, value_col), value in values.items():
        counterfactual_values[value_row, value_col] = value
    mine_mask = np.zeros((rows, cols), dtype=bool)
    mine_mask[row, col] = counterfactual_values[row, col] < 0.0
    return EpisodeTransition(
        board=np.zeros((20, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, row, col), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        mine_mask=mine_mask,
        counterfactual_open_values=counterfactual_values,
        expert_is_guess=True,
        extreme_family="guess",
    )
