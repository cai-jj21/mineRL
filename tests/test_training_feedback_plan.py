from __future__ import annotations

import argparse
import importlib.util
import json
import sqlite3
import sys
from pathlib import Path

import numpy as np

from minesweeper_rl.types import EpisodeTransition


ROOT = Path(__file__).resolve().parents[1]
PLAN_PATH = ROOT / "scripts" / "generate_training_feedback_plan.py"
REFINE_PATH = ROOT / "scripts" / "hard_loss_refine.py"

PLAN_SPEC = importlib.util.spec_from_file_location("training_feedback_plan_test", PLAN_PATH)
assert PLAN_SPEC is not None and PLAN_SPEC.loader is not None
training_feedback = importlib.util.module_from_spec(PLAN_SPEC)
sys.modules[PLAN_SPEC.name] = training_feedback
PLAN_SPEC.loader.exec_module(training_feedback)

REFINE_SPEC = importlib.util.spec_from_file_location("hard_loss_refine_feedback_test", REFINE_PATH)
assert REFINE_SPEC is not None and REFINE_SPEC.loader is not None
hard_loss_refine = importlib.util.module_from_spec(REFINE_SPEC)
sys.modules[REFINE_SPEC.name] = hard_loss_refine
REFINE_SPEC.loader.exec_module(hard_loss_refine)


def create_feedback_database(path: Path) -> None:
    conn = sqlite3.connect(path)
    conn.executescript(
        """
        CREATE TABLE v_dws_run_kpi (
            run_id TEXT,
            win_rate_completed REAL,
            avg_elapsed_seconds REAL,
            avg_actions_per_second REAL,
            total_reclicks INTEGER,
            total_unconfirmed_open_actions INTEGER,
            total_read_recoveries INTEGER
        );
        INSERT INTO v_dws_run_kpi VALUES ('desktop_run', 0.396, 27.2, 9.7, 0, 0, 0);

        CREATE TABLE v_ads_experiment_dashboard (
            name TEXT,
            games INTEGER,
            wins INTEGER,
            win_rate REAL
        );
        INSERT INTO v_ads_experiment_dashboard VALUES ('Internal RL ensemble', 1000, 430, 0.43);

        CREATE TABLE v_execution_anomalies (
            run_id TEXT,
            total_execution_anomalies INTEGER
        );
        INSERT INTO v_execution_anomalies VALUES ('desktop_run', 0);

        CREATE TABLE diagnostic_runs (
            diagnostic_run_id TEXT,
            primary_issue TEXT,
            next_focus TEXT,
            target_win_rate REAL,
            target_avg_seconds REAL,
            target_passed INTEGER
        );
        INSERT INTO diagnostic_runs VALUES ('diag', 'target_met_in_logs', 'longer_validation', 0.4, 60.0, 1);

        CREATE TABLE v_ads_failure_training_signal (
            failure_analysis_id TEXT,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            wrong_flag_loss_rate REAL,
            terminal_forced_rate REAL,
            terminal_target_known_mine_rate REAL,
            terminal_target_known_safe_rate REAL,
            avg_target_risk REAL,
            avg_best_guess_risk REAL,
            avg_risk_gap_to_best_guess REAL,
            selected_endgame_safe_left INTEGER,
            selected_endgame_loss_rate REAL,
            selected_endgame_loss_count INTEGER,
            recommended_exact_limit INTEGER
        );
        INSERT INTO v_ads_failure_training_signal
        VALUES ('failure_analysis_summary', 500, 209, 291, 0.418, 0.23, 0.316, 0.024, 0.22, 0.287, 0.226, 0.062, 100, 0.73, 213, 32);

        CREATE TABLE failure_endgame_bucket (
            bucket TEXT,
            safe_left_threshold INTEGER,
            loss_count INTEGER,
            loss_rate REAL,
            has_wrong_flags INTEGER,
            forced_available INTEGER,
            target_known_mine INTEGER,
            avg_target_risk REAL,
            avg_best_guess_risk REAL
        );
        INSERT INTO failure_endgame_bucket VALUES ('safe_left_le_60', 60, 189, 0.65, 34, 47, 6, 0.35, 0.29);
        INSERT INTO failure_endgame_bucket VALUES ('safe_left_le_100', 100, 213, 0.73, 42, 54, 6, 0.33, 0.27);

        CREATE TABLE failure_exact_limit (
            exact_limit INTEGER,
            terminal_states INTEGER,
            forced_available INTEGER,
            target_known_mine INTEGER,
            target_known_safe INTEGER,
            avg_target_risk REAL,
            avg_best_guess_risk REAL
        );
        INSERT INTO failure_exact_limit VALUES (8, 117, 35, 1, 30, 0.289, 0.258);
        INSERT INTO failure_exact_limit VALUES (32, 117, 45, 2, 30, 0.278, 0.217);

        CREATE TABLE failure_risk_bucket (
            bucket TEXT,
            bucket_count INTEGER
        );
        INSERT INTO failure_risk_bucket VALUES ('<0.20', 121);

        CREATE TABLE training_dataset_profile (
            profile_name TEXT,
            record_count INTEGER,
            counterfactual_records INTEGER,
            counterfactual_candidate_labels INTEGER,
            negative_counterfactual_candidates INTEGER,
            avg_behavior_regret REAL,
            high_regret_records INTEGER,
            known_behavior_mine_records INTEGER
        );
        INSERT INTO training_dataset_profile VALUES ('asset_profile', 10, 10, 100, 20, 4.0, 6, 2);

        CREATE TABLE training_dataset_record (
            family TEXT,
            region TEXT,
            safe_left_bucket TEXT,
            behavior_regret_bucket TEXT,
            counterfactual_label_count INTEGER,
            negative_counterfactual_candidates INTEGER,
            behavior_regret REAL,
            behavior_mine INTEGER
        );
        INSERT INTO training_dataset_record VALUES ('guess', 'interior', '241_391', 'gte_8', 10, 2, 8.0, 0);
        INSERT INTO training_dataset_record VALUES ('edge_guess_tail', 'edge', '000_060', '1_4', 5, 2, 2.0, 1);
        """
    )
    conn.commit()
    conn.close()


def test_build_training_feedback_plan_derives_weighted_refine_args(tmp_path: Path) -> None:
    database = tmp_path / "experiments.sqlite"
    create_feedback_database(database)

    report = training_feedback.build_training_feedback_plan(database=database, target_win_rate=0.47)

    args = report["hard_loss_refine_args"]
    assert args["endgame_safe_left"] == 100
    assert args["endgame_weight"] == 4
    assert args["wrong_flag_weight"] == 5
    assert args["terminal_weight"] == 4
    assert args["exact_limit"] == 32
    assert args["counterfactual_labels"] is True
    assert args["counterfactual_topk"] == 64
    assert args["counterfactual_coef"] > 0.0
    assert args["counterfactual_policy_coef"] > 0.0
    assert args["counterfactual_policy_full_action"] is True
    assert args["counterfactual_only_guess"] is True
    assert args["counterfactual_value_head_coef"] > 0.0
    assert args["counterfactual_value_rank_coef"] > 0.0
    assert args["counterfactual_risk_head_coef"] > 0.0
    assert args["train_policy_only"] is True
    assert args["extreme_family_filter"] == ["edge_guess_tail"]
    assert args["extreme_min_behavior_regret"] is None
    assert args["mine_games"] == 256
    assert args["imitation_updates"] == 64
    assert args["rl_updates"] == 16
    assert args["eval_games"] == 200
    assert args["lr"] == 5e-5
    assert args["reset_optimizer"] is True
    assert Path(args["replay_log_dir"]) == Path("artifacts/windows_agent")
    assert args["replay_log_limit"] == 8192
    assert args["extreme_dataset"] == str(
        Path(
            "artifacts/report_assets/extreme_training_dataset/"
            "guess_counterfactual_policy_v5_stratified_mineonly_merged.npz"
        )
    )
    assert "hard_loss_refine.py" in report["commands"]["hard_loss_refine"]
    assert "--replay-log-dir" in report["commands"]["hard_loss_refine"]
    assert "--extreme-dataset" in report["commands"]["hard_loss_refine"]
    assert "build_extreme_training_dataset.py" in report["commands"]["build_extreme_training_dataset"]
    assert "build_sim_guess_training_dataset.py" in report["commands"]["build_sim_guess_counterfactual_dataset"]
    assert "build_model_failure_extreme_dataset.py" in report["commands"]["build_model_failure_guess_counterfactual_dataset"]
    assert "merge_extreme_datasets.py" in report["commands"]["merge_extreme_datasets"]
    assert "analyze_extreme_policy_alignment.py" in report["commands"]["analyze_extreme_policy_alignment"]
    assert "export_training_feedback_dataset.py" in report["commands"]["export_training_feedback_dataset"]
    assert report["data_asset_pipeline"][-1] == "hard_loss_refine"
    assert report["signals"]["sample_family_profile"]["sample_count"] == 0
    assert any(item["name"] == "terminal_forced_move_imitation" for item in report["training_focus"])

    markdown = training_feedback.render_markdown(report)
    assert "Extreme Sample Mix" in markdown
    assert "Extreme Decision Mix" in markdown
    assert "数据反哺训练策略计划" in markdown
    assert "safe_left <= 100" in markdown


def test_summarize_training_dataset_assets_uses_persisted_profile_and_record_rows() -> None:
    summary = training_feedback.summarize_training_dataset_assets(
        [
            {
                "profile_name": "asset_a",
                "record_count": 10,
                "counterfactual_records": 10,
                "counterfactual_candidate_labels": 100,
                "negative_counterfactual_candidates": 20,
                "avg_behavior_regret": 4.0,
                "high_regret_records": 6,
                "known_behavior_mine_records": 2,
            },
            {
                "profile_name": "asset_b",
                "record_count": 5,
                "counterfactual_records": 5,
                "counterfactual_candidate_labels": 50,
                "negative_counterfactual_candidates": 5,
                "avg_behavior_regret": 2.0,
                "high_regret_records": 1,
                "known_behavior_mine_records": 1,
            },
        ],
        [
            {"family": "guess", "region": "interior", "safe_left_bucket": "241_391", "behavior_regret_bucket": "gte_8", "counterfactual_label_count": 10, "negative_counterfactual_candidates": 2, "behavior_regret": 8.0, "behavior_mine": 0},
            {"family": "edge_guess_tail", "region": "edge", "safe_left_bucket": "000_060", "behavior_regret_bucket": "1_4", "counterfactual_label_count": 5, "negative_counterfactual_candidates": 2, "behavior_regret": 2.0, "behavior_mine": 1},
        ],
    )

    assert summary["profile_count"] == 2
    assert summary["total_records"] == 15
    assert summary["counterfactual_candidate_labels"] == 150
    assert summary["negative_counterfactual_rate"] == 25 / 150
    assert summary["weighted_avg_behavior_regret"] == 50 / 15
    assert summary["known_behavior_mine_records"] == 1
    assert summary["behavior_mine_labeled_records"] == 2
    assert summary["behavior_mine_rate"] == 0.5
    assert summary["families"] == {"edge_guess_tail": 1, "guess": 1}
    assert summary["regions"] == {"edge": 1, "interior": 1}


def test_hard_loss_refine_applies_feedback_plan(tmp_path: Path) -> None:
    plan_path = tmp_path / "training_feedback_plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "hard_loss_refine_args": {
                    "exact_limit": 32,
                    "endgame_safe_left": 100,
                    "endgame_weight": 4,
                    "guess_weight": 4,
                    "edge_weight": 3,
                    "corner_weight": 4,
                    "wrong_flag_weight": 5,
                    "terminal_weight": 4,
                    "guess_imitation_weight": 1.68,
                    "counterfactual_policy_full_action": True,
                    "counterfactual_only_guess": True,
                    "reset_optimizer": True,
                    "extreme_dataset": "artifacts/report_assets/extreme_training_dataset/extreme_transitions.npz",
                    "include_plain_tail": False,
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        exact_limit=8,
        endgame_safe_left=60,
        endgame_weight=2,
        guess_weight=3,
        edge_weight=2,
        corner_weight=4,
        wrong_flag_weight=4,
        terminal_weight=3,
        guess_imitation_weight=1.4,
        counterfactual_policy_full_action=False,
        counterfactual_only_guess=False,
        reset_optimizer=False,
        extreme_dataset=Path("artifacts/report_assets/extreme_training_dataset/extreme_transitions.npz"),
        include_plain_tail=True,
    )

    hard_loss_refine.apply_feedback_plan(args, plan_path)

    assert args.exact_limit == 32
    assert args.endgame_safe_left == 100
    assert args.endgame_weight == 4
    assert args.guess_weight == 4
    assert args.edge_weight == 3
    assert args.corner_weight == 4
    assert args.wrong_flag_weight == 5
    assert args.terminal_weight == 4
    assert args.guess_imitation_weight == 1.68
    assert args.counterfactual_policy_full_action is True
    assert args.counterfactual_only_guess is True
    assert args.reset_optimizer is True
    assert args.extreme_dataset == Path("artifacts/report_assets/extreme_training_dataset/extreme_transitions.npz")
    assert args.include_plain_tail is False


def test_feedback_plan_preserves_explicit_cli_overrides(tmp_path: Path) -> None:
    plan_path = tmp_path / "plan.json"
    plan_path.write_text(
        json.dumps(
            {
                "hard_loss_refine_args": {
                    "rounds": 3,
                    "mine_games": 256,
                    "eval_games": 200,
                    "inference_flips": True,
                }
            }
        ),
        encoding="utf-8",
    )
    args = argparse.Namespace(
        rounds=1,
        mine_games=32,
        eval_games=50,
        inference_flips=False,
    )

    hard_loss_refine.apply_feedback_plan(
        args,
        plan_path,
        explicit_overrides={"rounds", "eval_games", "inference_flips"},
    )

    assert args.rounds == 1
    assert args.mine_games == 256
    assert args.eval_games == 50
    assert args.inference_flips is False


def test_explicit_cli_destinations_maps_no_flags_to_positive_destination() -> None:
    destinations = hard_loss_refine.explicit_cli_destinations(
        ["--rounds", "1", "--no-inference-flips", "--mine-games=32"]
    )

    assert "rounds" in destinations
    assert "mine_games" in destinations
    assert "inference_flips" in destinations


def test_derive_hard_loss_args_raises_weights_for_extreme_samples() -> None:
    profile = training_feedback.summarize_sample_family_profile(
        [
            {"sample_family": "endgame_edge_tail_replay"},
            {"sample_family": "edge_guess_ranking"},
            {"sample_family": "endgame_corner_tail_replay"},
            {"sample_family": "corner_guess_ranking"},
            {"sample_family": "guess_risk_ranking"},
        ]
    )

    args = training_feedback.derive_hard_loss_args(
        target_gap=0.05,
        wrong_flag_rate=0.12,
        terminal_forced_rate=0.28,
        risk_gap=0.06,
        selected_endgame_safe_left=100,
        selected_endgame_loss_rate=0.62,
        recommended_exact_limit=32,
        sample_family_profile=profile,
        decision_family_profile={
            "guess_decisions": 10,
            "edge_guess_decisions": 2,
            "corner_guess_decisions": 1,
            "high_risk_guess_decisions": 3,
        },
        replay_log_dir=Path("artifacts/windows_agent"),
        replay_log_limit=8192,
        extreme_dataset=Path("artifacts/report_assets/extreme_training_dataset/extreme_transitions.npz"),
        training_asset_summary={
            "known_behavior_mine_records": 4,
            "behavior_mine_labeled_records": 10,
            "behavior_mine_rate": 0.4,
        },
    )

    assert args["tail_states"] == 16
    assert args["guess_weight"] == 4
    assert args["edge_weight"] == 3
    assert args["corner_weight"] == 4
    assert args["guess_survival_coef"] == 0.0
    assert args["guess_survival_mine_weight"] == 4.0
    assert args["guess_survival_topk"] == 16
    assert args["guess_survival_margin"] == 0.08
    assert args["behavior_mine_demotion_coef"] > 0.0
    assert args["behavior_mine_demotion_topk"] == 4
    assert args["behavior_mine_demotion_margin"] == 0.08
    assert args["train_policy_only"] is True
    assert args["counterfactual_policy_full_action"] is True
    assert args["counterfactual_only_guess"] is True
    assert args["reset_optimizer"] is True


def test_hard_loss_refine_samples_high_value_transitions_more_often() -> None:
    rows, cols = 2, 2
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True

    low = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
    )
    high = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.95, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=-1.0,
        done=True,
        expert_is_guess=True,
    )

    class FakeRng:
        def __init__(self) -> None:
            self.probabilities = None

        def choice(self, a, size=None, replace=None, p=None):
            self.probabilities = p
            return np.array([1])

    rng = FakeRng()
    sampled = hard_loss_refine.sample_transitions([low, high], 1, rng)

    assert rng.probabilities is not None
    assert rng.probabilities[1] > rng.probabilities[0]
    assert sampled[0] is high
