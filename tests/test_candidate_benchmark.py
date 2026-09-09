import importlib.util
import sys
from pathlib import Path

import numpy as np

from minesweeper_rl.types import EpisodeTransition


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "benchmark_candidate_calibrator.py"
SPEC = importlib.util.spec_from_file_location("benchmark_candidate_calibrator_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


def _transition() -> EpisodeTransition:
    action_mask = np.zeros((4, 2, 3), dtype=bool)
    action_mask[0] = True
    return EpisodeTransition(
        board=np.zeros((20, 2, 3), dtype=np.float32),
        global_features=np.asarray([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=np.asarray([[0.0, 2.0, 1.0], [0.0, -1.0, 0.5]], dtype=np.float32),
        extreme_family="guess",
    )


def test_score_transition_reports_candidate_improvement() -> None:
    row = benchmark.score_transition(
        transition=_transition(),
        base_open_scores=np.asarray([0.9, 0.2, 0.3, 0.1, 0.0, 0.4], dtype=np.float32),
        candidate_probabilities=np.asarray([0.0, 0.8, 0.1, 0.0, 0.0, 0.1], dtype=np.float32),
        near_best_margin=0.25,
    )

    assert row is not None
    assert row["base_regret"] == 2.0
    assert row["candidate_regret"] == 0.0
    assert row["candidate_improved"] == 1.0


def test_benchmark_summary_has_family_and_safe_left_buckets() -> None:
    row = benchmark.score_transition(
        transition=_transition(),
        base_open_scores=np.zeros(6, dtype=np.float32),
        candidate_probabilities=np.asarray([0.0, 0.8, 0.1, 0.0, 0.0, 0.1], dtype=np.float32),
        near_best_margin=0.25,
    )
    summary = benchmark.summarize_benchmark_rows([row])

    assert summary["records"] == 1
    assert "guess" in summary["families"]
    assert "141_240" in summary["safe_left_buckets"]
