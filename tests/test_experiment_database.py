from __future__ import annotations

import importlib.util
import json
import sqlite3
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
BUILD_PATH = ROOT / "scripts" / "build_experiment_database.py"
ANALYZE_PATH = ROOT / "scripts" / "analyze_experiment_database.py"

BUILD_SPEC = importlib.util.spec_from_file_location("build_experiment_database_test", BUILD_PATH)
assert BUILD_SPEC is not None and BUILD_SPEC.loader is not None
db_builder = importlib.util.module_from_spec(BUILD_SPEC)
sys.modules[BUILD_SPEC.name] = db_builder
BUILD_SPEC.loader.exec_module(db_builder)

ANALYZE_SPEC = importlib.util.spec_from_file_location("analyze_experiment_database_test", ANALYZE_PATH)
assert ANALYZE_SPEC is not None and ANALYZE_SPEC.loader is not None
db_analyzer = importlib.util.module_from_spec(ANALYZE_SPEC)
sys.modules[ANALYZE_SPEC.name] = db_analyzer
ANALYZE_SPEC.loader.exec_module(db_analyzer)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def create_fixture(root: Path) -> dict[str, Path]:
    run_dir = root / "windows_run"
    write_json(
        run_dir / "per_game_summary.json",
        {
            "file_count": 2,
            "completed_games": 2,
            "terminal_games": 2,
            "wins": 1,
            "losses": 1,
            "incomplete_games": 0,
            "win_rate_completed": 0.5,
            "longest_streak": 1,
            "longest_streak_start": 1,
            "longest_streak_end": 1,
            "avg_elapsed_seconds": 11.5,
            "avg_actions_per_second": 2.0,
            "avg_agent_steps": 2.0,
            "avg_physical_open_actions": 1.0,
            "total_reclicks": 1,
            "total_click_unready_actions": 0,
            "total_unconfirmed_open_actions": 0,
            "total_read_recoveries": 0,
        },
    )
    write_json(
        run_dir / "game_001.json",
        {
            "game_index": 1,
            "checkpoint": "ckpt.pt",
            "ensemble_checkpoints": ["ckpt2.pt"],
            "summary": {
                "won": True,
                "lost": False,
                "done": True,
                "agent_steps": 2,
                "physical_open_actions": 1,
                "click_issued_actions": 1,
                "flags": 1,
                "revealed_safe_cells": 380,
                "elapsed_seconds": 10.0,
                "actions_per_second": 3.0,
                "seconds_per_action": 0.3,
                "reclicks": 0,
                "click_unready_actions": 0,
                "unconfirmed_open_actions": 0,
                "read_recoveries": 0,
                "read_repairs": 0,
                "read_restores": 0,
                "open_zero_progress_actions": 0,
                "open_target_miss_with_progress": 0,
                "solver_assist_actions": 0,
                "solver_safety_filter_actions": 0,
                "basic_safety_filter_actions": 0,
                "quick_number_read_actions": 1,
                "quick_number_read_fallbacks": 0,
                "terminal_dialog": "娓告垙鑳滃埄",
                "first_open_action": {"action": {"kind": "open", "row": 8, "col": 15}, "revealed_delta": 5},
            },
            "actions": [
                {
                    "step": 0,
                    "action_index": 255,
                    "action": {"kind": "open", "row": 8, "col": 15},
                    "selection_elapsed": 0.01,
                    "screen_target": {"x": 100, "y": 200},
                    "before_target": {"revealed": False, "flagged": False, "adjacent": 0},
                    "click": {
                        "issued": True,
                        "method": "mouse_event",
                        "edge_click": False,
                        "cursor_cell_at_target": {"same_cell": True},
                        "elapsed_seconds": 0.14,
                    },
                    "solver_audit": {
                        "available": True,
                        "has_forced_moves": True,
                        "forced_safe_count": 3,
                        "forced_mine_count": 1,
                        "target": {"known_safe": True, "known_mine": False, "risk": 0.1},
                        "best_guess_risk": 0.2,
                    },
                    "after_target": {"revealed": True, "flagged": False, "adjacent": 0},
                    "changed": True,
                    "progress": True,
                    "revealed_delta": 5,
                    "target_revealed_after_open": True,
                    "confirmed_open_cells": 5,
                    "persistent_revealed_cells": 5,
                    "quick_number_read": {
                        "attempted": True,
                        "cell_number": 0,
                        "capture_seconds": 0.01,
                        "classification_seconds": 0.02,
                    },
                    "after_read_elapsed": 0.03,
                },
                {
                    "step": 1,
                    "action_index": 707,
                    "action": {"kind": "flag", "row": 7, "col": 17},
                    "virtual_only": True,
                    "virtual_change": "flag",
                },
            ],
        },
    )
    write_json(
        run_dir / "game_002.json",
        {
            "game_index": 2,
            "summary": {
                "won": False,
                "lost": True,
                "done": True,
                "agent_steps": 1,
                "physical_open_actions": 1,
                "click_issued_actions": 1,
                "flags": 0,
                "revealed_safe_cells": 40,
                "elapsed_seconds": 13.0,
                "actions_per_second": 1.0,
                "reclicks": 1,
                "click_unready_actions": 0,
                "unconfirmed_open_actions": 0,
                "read_recoveries": 0,
                "terminal_dialog": "娓告垙澶辫触",
            },
            "actions": [
                {
                    "step": 0,
                    "action_index": 1,
                    "action": {"kind": "open", "row": 0, "col": 1},
                    "revealed_delta": 1,
                }
            ],
        },
    )
    stats = root / "statistical_summary.json"
    write_json(
        stats,
        {
            "experiments": [
                {
                    "name": "Toy Windows",
                    "source": "windows_run/per_game_summary.json",
                    "wins": 1,
                    "games": 2,
                    "win_rate": 0.5,
                    "wilson_low": 0.1,
                    "wilson_high": 0.9,
                    "target_win_rate": 0.4,
                    "point_target_pass": True,
                    "wilson_lower_target_pass": False,
                    "note": "fixture",
                }
            ]
        },
    )
    evidence = root / "evidence_validation.json"
    claims = root / "claim_audit.json"
    write_json(evidence, {"checks": [{"id": "evidence.ok", "ok": True, "observed": 1, "expected": 1, "detail": "ok"}]})
    write_json(claims, {"checks": [{"id": "claims.ok", "ok": True, "observed": "x", "expected": "x", "detail": "ok"}]})
    manifest = root / "artifact_manifest.json"
    write_json(manifest, {"artifacts": [{"path": "a.json", "exists": True, "size_bytes": 3, "sha256": "abc"}]})
    internal = root / "internal_eval.json"
    write_json(
        internal,
        {
            "checkpoints": ["ckpt.pt", "ckpt2.pt"],
            "games": 2,
            "wins": 1,
            "win_rate": 0.5,
            "longest_streak": 1,
            "elapsed_seconds": 1.0,
            "avg_agent_steps": 2.0,
        },
    )
    analysis = root / "analysis.json"
    write_json(
        analysis,
        {
            "input": "windows_run",
            "games": 2,
            "wins": 1,
            "losses": 1,
            "win_rate": 0.5,
            "avg_elapsed_seconds": 11.5,
            "avg_revealed_safe_cells": 210.0,
            "signal_counts": {"late_loss": 1},
            "diagnosis": {
                "primary_issue": "sample",
                "next_focus": "more_data",
                "target_win_rate": 0.4,
                "target_avg_seconds": 60.0,
                "target_passed": True,
                "recommendations": ["keep monitoring"],
            },
            "games_detail": [
                {
                    "path": str(run_dir / "game_002.json"),
                    "won": False,
                    "lost": True,
                    "done": True,
                    "agent_steps": 1,
                    "elapsed_seconds": 13.0,
                    "revealed_safe_cells": 40,
                    "first_open_revealed_delta": 2,
                    "signals": ["late_loss"],
                }
            ],
        },
    )
    failure = root / "failure_analysis_summary.json"
    write_json(
        failure,
        {
            "loss_analysis": {
                "games": 10,
                "wins": 4,
                "losses": 6,
                "win_rate": 0.4,
                "wrong_flag_losses": 2,
                "wrong_flag_loss_rate": 2 / 6,
                "terminal_forced_available": 3,
                "terminal_forced_rate": 3 / 6,
                "terminal_target_known_mine": 1,
                "terminal_target_known_mine_rate": 1 / 6,
                "terminal_target_known_safe_visible_solver": 2,
                "terminal_target_known_safe_rate": 2 / 6,
                "avg_wrong_flags_on_loss": 0.5,
                "avg_flags_on_loss": 10.0,
                "avg_target_risk": 0.3,
                "avg_best_guess_risk": 0.2,
                "avg_risk_gap_to_best_guess": 0.1,
                "risk_buckets": [{"bucket": "<0.20", "count": 2}],
                "endgame": {
                    "safe_left_le_60": {
                        "count": 4,
                        "loss_rate": 0.66,
                        "has_wrong_flags": 2,
                        "forced_available": 3,
                        "target_known_mine": 1,
                        "avg_target_risk": 0.3,
                        "avg_best_guess_risk": 0.2,
                    }
                },
            },
            "exact_limit_compare": {
                "limits": [
                    {
                        "limit": 32,
                        "terminal_states": 5,
                        "forced_available": 3,
                        "target_known_mine": 1,
                        "target_known_safe": 2,
                        "avg_target_risk": 0.28,
                        "avg_best_guess_risk": 0.2,
                    }
                ]
            },
            "examples": [
                {
                    "seed": 7,
                    "steps_before_loss": 100,
                    "action_kind": "open",
                    "row": 1,
                    "col": 2,
                    "safe_left": 12,
                    "wrong_flags": 1,
                    "forced_safe": 0,
                    "forced_mines": 0,
                    "target_risk": 0.4,
                    "best_guess_risk": 0.2,
                    "target_known_mine": False,
                    "target_known_safe": False,
                }
            ],
        },
    )
    return {
        "run_dir": run_dir,
        "stats": stats,
        "evidence": evidence,
        "claims": claims,
        "manifest": manifest,
        "internal": internal,
        "analysis": analysis,
        "failure": failure,
    }


def build_fixture_database(tmp_path: Path, include_actions: str) -> Path:
    paths = create_fixture(tmp_path)
    database = tmp_path / "experiments.sqlite"
    db_builder.build_database(
        output=database,
        windows_run_dir=paths["run_dir"],
        internal_eval=paths["internal"],
        analysis_report=paths["analysis"],
        failure_analysis=paths["failure"],
        statistical_summary=paths["stats"],
        evidence_validation=paths["evidence"],
        claim_audit=paths["claims"],
        artifact_manifest=paths["manifest"],
        run_id="toy_run",
        streak_start=1,
        streak_end=1,
        include_actions=include_actions,
    )
    return database


def scalar(database: Path, sql: str) -> int | float | str | None:
    with sqlite3.connect(database) as conn:
        return conn.execute(sql).fetchone()[0]


def test_build_database_ingests_games_actions_and_quality(tmp_path: Path) -> None:
    database = build_fixture_database(tmp_path, include_actions="all")

    assert scalar(database, "SELECT COUNT(*) FROM game_summary") == 2
    assert scalar(database, "SELECT COUNT(*) FROM action_event") == 3
    assert scalar(database, "SELECT SUM(won) FROM game_summary") == 1
    assert scalar(database, "SELECT SUM(virtual_flag_actions) FROM game_summary") == 1
    assert scalar(database, "SELECT COUNT(*) FROM quality_checks") == 2
    assert scalar(database, "SELECT COUNT(*) FROM etl_batch") == 1
    assert scalar(database, "SELECT COUNT(*) FROM source_file") == 10
    assert scalar(database, "SELECT COUNT(*) FROM model_registry") == 2
    assert scalar(database, "SELECT COUNT(*) FROM experiment_run") == 2
    assert scalar(database, "SELECT COUNT(*) FROM experiment_model_member") == 4
    assert scalar(database, "SELECT COUNT(*) FROM promotion_decision") == 2
    assert scalar(database, "SELECT COUNT(*) FROM diagnostic_runs") == 1
    assert scalar(database, "SELECT COUNT(*) FROM diagnostic_game_detail") == 1
    assert scalar(database, "SELECT COUNT(*) FROM diagnostic_signal") == 1
    assert scalar(database, "SELECT COUNT(*) FROM failure_analysis_run") == 1
    assert scalar(database, "SELECT COUNT(*) FROM failure_endgame_bucket") == 1
    assert scalar(database, "SELECT COUNT(*) FROM failure_exact_limit") == 1
    assert scalar(database, "SELECT COUNT(*) FROM failure_attribution") == 1
    assert scalar(database, "SELECT COUNT(*) FROM decision_event") == 3
    assert scalar(database, "SELECT terminal_edge FROM failure_attribution") == 1
    assert scalar(database, "SELECT terminal_corner FROM failure_attribution") == 0
    assert scalar(database, "SELECT selected_endgame_safe_left FROM v_ads_failure_training_signal") == 60
    assert scalar(database, "SELECT COUNT(*) FROM v_ads_model_registry") == 2
    assert scalar(database, "SELECT COUNT(*) FROM v_ads_experiment_registry") == 2
    assert scalar(database, "SELECT primary_failure_type FROM v_ads_failure_attribution_summary") == "execution_click"
    assert scalar(database, "SELECT sample_family FROM v_ads_training_sample_candidates") == "policy_loss_review"
    assert scalar(database, "SELECT COUNT(*) FROM v_ads_extreme_decision_candidates") == 1
    assert scalar(database, "SELECT sample_family FROM v_ads_extreme_decision_candidates") == "edge_guess"
    assert scalar(database, "SELECT decision_source FROM v_dwd_decision_event WHERE game_index = 1 AND step = 0") == "rl_policy"
    assert scalar(database, "SELECT total_execution_anomalies FROM v_execution_anomalies WHERE run_id = 'toy_run'") == 1


def test_build_database_can_limit_actions_to_streak(tmp_path: Path) -> None:
    database = build_fixture_database(tmp_path, include_actions="streak")

    assert scalar(database, "SELECT COUNT(*) FROM game_summary") == 2
    assert scalar(database, "SELECT COUNT(*) FROM action_event") == 2
    assert scalar(database, "SELECT COUNT(*) FROM v_ten_streak_games") == 1


def test_build_database_ingests_training_feedback_profile(tmp_path: Path) -> None:
    paths = create_fixture(tmp_path)
    profile = tmp_path / "training_profile.json"
    write_json(
        profile,
        {
            "datasets": [{"path": "guess.npz", "records": 2}],
            "summary": {
                "record_count": 2,
                "counterfactual_records": 2,
                "counterfactual_candidate_labels": 500,
                "negative_counterfactual_candidates": 100,
                "negative_counterfactual_rate": 0.2,
                "avg_behavior_regret": 6.5,
                "median_behavior_regret": 5.0,
                "max_behavior_regret": 12.0,
                "high_regret_records": 1,
                "high_regret_rate": 0.5,
                "known_behavior_mine_records": 1,
                "behavior_mine_rate": 0.5,
            },
            "records": [
                {
                    "index": 0,
                    "dataset": "guess.npz",
                    "family": "guess",
                    "expert_is_guess": True,
                    "source_quality": 1.2,
                    "extreme_score": 2.0,
                    "action_row": 4,
                    "action_col": 5,
                    "region": "interior",
                    "safe_left": 200,
                    "safe_left_bucket": "141_240",
                    "open_candidate_count": 300,
                    "open_candidate_bucket": "201_480",
                    "counterfactual_label_count": 300,
                    "negative_counterfactual_candidates": 60,
                    "positive_counterfactual_candidates": 240,
                    "negative_counterfactual_rate": 0.2,
                    "best_counterfactual_value": 8.0,
                    "behavior_counterfactual_value": 1.0,
                    "behavior_regret": 7.0,
                    "behavior_regret_bucket": "4_8",
                    "behavior_mine": False,
                },
                {
                    "index": 1,
                    "dataset": "guess.npz",
                    "family": "guess",
                    "expert_is_guess": True,
                    "source_quality": 1.2,
                    "extreme_score": 3.0,
                    "action_row": 0,
                    "action_col": 0,
                    "region": "corner",
                    "safe_left": 30,
                    "safe_left_bucket": "000_060",
                    "open_candidate_count": 10,
                    "open_candidate_bucket": "000_020",
                    "counterfactual_label_count": 200,
                    "negative_counterfactual_candidates": 40,
                    "positive_counterfactual_candidates": 160,
                    "negative_counterfactual_rate": 0.2,
                    "best_counterfactual_value": 12.0,
                    "behavior_counterfactual_value": 0.0,
                    "behavior_regret": 12.0,
                    "behavior_regret_bucket": "gte_8",
                    "behavior_mine": True,
                },
            ],
        },
    )
    database = tmp_path / "experiments.sqlite"
    db_builder.build_database(
        output=database,
        windows_run_dir=paths["run_dir"],
        internal_eval=paths["internal"],
        analysis_report=paths["analysis"],
        failure_analysis=paths["failure"],
        statistical_summary=paths["stats"],
        evidence_validation=paths["evidence"],
        claim_audit=paths["claims"],
        artifact_manifest=paths["manifest"],
        training_profiles=[profile],
        run_id="toy_run",
        streak_start=1,
        streak_end=1,
        include_actions="none",
    )

    assert scalar(database, "SELECT COUNT(*) FROM training_dataset_profile") == 1
    assert scalar(database, "SELECT COUNT(*) FROM training_dataset_record") == 2
    assert scalar(database, "SELECT COUNT(*) FROM v_ads_training_feedback_slice") == 2
    assert scalar(database, "SELECT priority FROM v_ads_training_feedback_priority LIMIT 1") == "critical"
    assert scalar(database, "SELECT behavior_mine FROM training_dataset_record WHERE record_index = 1") == 1
    assert scalar(database, "SELECT COUNT(*) FROM source_file WHERE source_type = 'training_dataset_profile'") == 1


def test_board_location_helpers_distinguish_edges_and_corners() -> None:
    assert db_builder.is_edge_cell(0, 1)
    assert not db_builder.is_corner_cell(0, 1)
    assert db_builder.is_corner_cell(0, 0)
    assert not db_builder.is_edge_cell(None, 0)


def test_analyze_database_exports_query_sections(tmp_path: Path) -> None:
    database = build_fixture_database(tmp_path, include_actions="all")

    report = db_analyzer.analyze_database(database)
    markdown = db_analyzer.render_markdown(report)

    assert report["experiment_leaderboard"][0]["name"] == "Toy Windows"
    assert report["model_registry"][0]["training_role"] == "primary"
    assert report["experiment_registry"][0]["run_type"] == "internal_eval"
    assert report["etl_batches"][0]["status"] == "success"
    assert report["source_inventory"][0]["layer"] in {"audit", "diagnostic", "dwd", "manifest", "mart", "ods"}
    assert report["diagnostic_runs"][0]["primary_issue"] == "sample"
    assert report["diagnostic_signal_profile"][0]["signal_name"] == "late_loss"
    assert report["failure_training_signal"][0]["recommended_exact_limit"] == 32
    assert report["training_dataset_profiles"] == []
    assert report["training_feedback_slices"] == []
    assert report["training_feedback_priority"] == []
    assert report["failure_attribution_summary"][0]["primary_failure_type"] == "execution_click"
    assert report["training_sample_candidates"][0]["sample_family"] == "policy_loss_review"
    assert report["extreme_decision_candidates"][0]["sample_family"] == "edge_guess"
    assert report["decision_event_preview"][0]["decision_source"] == "rl_policy"
    assert report["quality_summary"][0]["failed"] == 0
    assert "## ETL 鎵规" in markdown
    assert "## ODS 鏉ユ簮娓呭崟" in markdown
    assert "## DWS Run KPI" in markdown
    assert "Model Registry" in markdown
    assert "Training Sample Candidates" in markdown
    assert "Extreme Decision Candidates" in markdown
    assert "## 璇婃柇淇″彿" in markdown
    assert "## 璁粌鍙嶅摵淇″彿" in markdown
    assert "## 瀹為獙鑳滅巼姒?" in markdown
    assert "toy_run" in markdown
    assert "Toy Windows" in markdown
