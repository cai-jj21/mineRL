from __future__ import annotations

import argparse
import hashlib
import json
import re
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable


DEFAULT_WINDOWS_RUN_DIR = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000")
DEFAULT_INTERNAL_EVAL = Path("artifacts/ensemble_20_100best_eval_1000_seed0.json")
DEFAULT_ANALYSIS = Path("artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/analysis.json")
DEFAULT_STATS = Path("artifacts/report_assets/statistical_summary.json")
DEFAULT_FAILURE_ANALYSIS = Path("artifacts/report_assets/failure_analysis_summary.json")
DEFAULT_EVIDENCE = Path("artifacts/report_assets/evidence_validation.json")
DEFAULT_CLAIMS = Path("artifacts/report_assets/claim_audit.json")
DEFAULT_MANIFEST = Path("artifacts/report_assets/artifact_manifest.json")
DEFAULT_OUTPUT = Path("artifacts/report_assets/minesweeper_experiments.sqlite")
DEFAULT_TRAINING_PROFILES = (
    Path("artifacts/report_assets/extreme_training_dataset/gated_live_guess_profile_with_edge_corner.json"),
)

GAME_RE = re.compile(r"game_(\d+)\.json$")
TOTAL_SAFE_CELLS = 16 * 30 - 99
BOARD_ROWS = 16
BOARD_COLS = 30


def is_edge_cell(row: int | None, col: int | None) -> bool:
    if row is None or col is None:
        return False
    return row in {0, BOARD_ROWS - 1} or col in {0, BOARD_COLS - 1}


def is_corner_cell(row: int | None, col: int | None) -> bool:
    if row is None or col is None:
        return False
    return row in {0, BOARD_ROWS - 1} and col in {0, BOARD_COLS - 1}


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Build a SQLite experiment database from Minesweeper RL artifacts.")
    parser.add_argument("--windows-run-dir", type=Path, default=DEFAULT_WINDOWS_RUN_DIR)
    parser.add_argument("--internal-eval", type=Path, default=DEFAULT_INTERNAL_EVAL)
    parser.add_argument("--analysis-report", type=Path, default=DEFAULT_ANALYSIS)
    parser.add_argument("--statistical-summary", type=Path, default=DEFAULT_STATS)
    parser.add_argument("--failure-analysis", type=Path, default=DEFAULT_FAILURE_ANALYSIS)
    parser.add_argument("--evidence-validation", type=Path, default=DEFAULT_EVIDENCE)
    parser.add_argument("--claim-audit", type=Path, default=DEFAULT_CLAIMS)
    parser.add_argument("--artifact-manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument(
        "--training-profile",
        dest="training_profiles",
        action="append",
        type=Path,
        default=None,
        help="Training feedback profile JSON. Repeat for multiple profiles; omit with --no-default-training-profile.",
    )
    parser.add_argument(
        "--no-default-training-profile",
        action="store_true",
        help="Do not load the default training feedback profile.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--run-id", default=None, help="defaults to the Windows run directory name")
    parser.add_argument("--streak-start", type=int, default=747)
    parser.add_argument("--streak-end", type=int, default=756)
    parser.add_argument(
        "--include-actions",
        choices=["none", "streak", "all"],
        default="streak",
        help="action_event ingestion scope; all is the fully detailed database",
    )
    args = parser.parse_args()

    if args.no_default_training_profile:
        training_profiles = []
    elif args.training_profiles is None:
        training_profiles = list(DEFAULT_TRAINING_PROFILES)
    else:
        training_profiles = list(args.training_profiles)
    report = build_database(
        output=args.output,
        windows_run_dir=args.windows_run_dir,
        internal_eval=args.internal_eval,
        analysis_report=args.analysis_report,
        statistical_summary=args.statistical_summary,
        failure_analysis=args.failure_analysis,
        evidence_validation=args.evidence_validation,
        claim_audit=args.claim_audit,
        artifact_manifest=args.artifact_manifest,
        training_profiles=training_profiles,
        run_id=args.run_id,
        streak_start=args.streak_start,
        streak_end=args.streak_end,
        include_actions=args.include_actions,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


def build_database(
    *,
    output: Path,
    windows_run_dir: Path = DEFAULT_WINDOWS_RUN_DIR,
    internal_eval: Path = DEFAULT_INTERNAL_EVAL,
    analysis_report: Path = DEFAULT_ANALYSIS,
    statistical_summary: Path = DEFAULT_STATS,
    failure_analysis: Path = DEFAULT_FAILURE_ANALYSIS,
    evidence_validation: Path = DEFAULT_EVIDENCE,
    claim_audit: Path = DEFAULT_CLAIMS,
    artifact_manifest: Path = DEFAULT_MANIFEST,
    training_profiles: Iterable[Path] = (),
    run_id: str | None = None,
    streak_start: int = 747,
    streak_end: int = 756,
    include_actions: str = "streak",
) -> dict[str, Any]:
    if include_actions not in {"none", "streak", "all"}:
        raise ValueError(f"unsupported include_actions: {include_actions}")

    training_profiles = [Path(path) for path in training_profiles]
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()

    windows_run_dir = windows_run_dir.resolve()
    run_id = run_id or windows_run_dir.name
    batch_id = build_batch_id()
    started_at = utc_now()

    conn = sqlite3.connect(output)
    try:
        conn.execute("PRAGMA foreign_keys = ON")
        create_schema(conn)
        write_metadata(
            conn,
            {
                "schema_version": "3",
                "primary_windows_run_id": run_id,
                "etl_batch_id": batch_id,
                "windows_run_dir": str(windows_run_dir),
                "internal_eval": str(internal_eval),
                "analysis_report": str(analysis_report),
                "statistical_summary": str(statistical_summary),
                "failure_analysis": str(failure_analysis),
                "include_actions": include_actions,
                "streak_start": str(streak_start),
                "streak_end": str(streak_end),
                "training_profiles": dump_json([str(path) for path in training_profiles]),
            },
        )
        conn.execute(
            """
            INSERT INTO etl_batch(
                batch_id, started_at, status, source_root, target_database, include_actions,
                streak_start, streak_end, source_file_count, total_row_count, raw_metadata_json
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                batch_id,
                started_at,
                "running",
                str(windows_run_dir),
                str(output),
                include_actions,
                streak_start,
                streak_end,
                0,
                0,
                dump_json(
                    {
                        "windows_run_dir": str(windows_run_dir),
                        "internal_eval": str(internal_eval),
                        "analysis_report": str(analysis_report),
                        "statistical_summary": str(statistical_summary),
                        "failure_analysis": str(failure_analysis),
                        "evidence_validation": str(evidence_validation),
                        "claim_audit": str(claim_audit),
                "artifact_manifest": str(artifact_manifest),
                "training_profiles": [str(path) for path in training_profiles],
            }
        ),
            ),
        )
        windows_counts = ingest_windows_run(
            conn,
            batch_id=batch_id,
            run_id=run_id,
            run_dir=windows_run_dir,
            streak_start=streak_start,
            streak_end=streak_end,
            include_actions=include_actions,
        )
        internal_count = ingest_internal_eval(conn, internal_eval)
        analysis_counts = ingest_diagnostic_analysis(conn, analysis_report)
        failure_counts = ingest_failure_analysis(conn, failure_analysis)
        metric_count = ingest_experiment_metrics(conn, statistical_summary)
        promotion_count = ingest_promotion_decisions(conn)
        evidence_count = ingest_quality_checks(conn, "evidence_validation", evidence_validation)
        claim_count = ingest_quality_checks(conn, "claim_audit", claim_audit)
        artifact_count = ingest_artifact_manifest(conn, artifact_manifest)
        training_profile_counts = ingest_training_dataset_profiles(conn, training_profiles)
        source_count, source_row_count = ingest_source_inventory(
            conn,
            batch_id=batch_id,
            windows_run_dir=windows_run_dir,
            internal_eval=internal_eval,
            analysis_report=analysis_report,
            failure_analysis=failure_analysis,
            statistical_summary=statistical_summary,
            evidence_validation=evidence_validation,
            claim_audit=claim_audit,
            artifact_manifest=artifact_manifest,
            training_profiles=training_profiles,
        )
        create_views(conn)
        model_count = count_rows(conn, "model_registry")
        experiment_run_count = count_rows(conn, "experiment_run")
        experiment_member_count = count_rows(conn, "experiment_model_member")
        conn.execute(
            """
            UPDATE etl_batch
            SET finished_at = ?, status = ?, source_file_count = ?, total_row_count = ?
            WHERE batch_id = ?
            """,
            (utc_now(), "success", source_count, source_row_count, batch_id),
        )
        conn.commit()
    finally:
        conn.close()

    return {
        "ok": True,
        "output": str(output),
        "batch_id": batch_id,
        "windows_run_dir": str(windows_run_dir),
        "include_actions": include_actions,
        "tables": {
            "runs": 1 + internal_count + analysis_counts["runs"],
            "game_summary": windows_counts["games"],
            "action_event": windows_counts["actions"],
            "decision_event": windows_counts["decisions"],
            "failure_attribution": windows_counts["failures"],
            "model_registry": model_count,
            "experiment_run": experiment_run_count,
            "experiment_model_member": experiment_member_count,
            "promotion_decision": promotion_count,
            "experiment_metrics": metric_count,
            "quality_checks": evidence_count + claim_count,
            "artifact_manifest": artifact_count,
            "training_dataset_profile": training_profile_counts["profiles"],
            "training_dataset_record": training_profile_counts["records"],
            "etl_batch": 1,
            "source_file": source_count,
            "diagnostic_runs": analysis_counts["runs"],
            "diagnostic_game_detail": analysis_counts["games"],
            "diagnostic_signal": analysis_counts["signals"],
            "diagnostic_recommendation": analysis_counts["recommendations"],
            "failure_analysis_run": failure_counts["runs"],
            "failure_endgame_bucket": failure_counts["endgame_buckets"],
            "failure_risk_bucket": failure_counts["risk_buckets"],
            "failure_exact_limit": failure_counts["exact_limits"],
            "failure_example": failure_counts["examples"],
        },
        "streak_range": {"start": streak_start, "end": streak_end},
        "training_profiles": [str(path) for path in training_profiles if path.exists()],
    }


def create_schema(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE TABLE database_metadata (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );

        CREATE TABLE etl_batch (
            batch_id TEXT PRIMARY KEY,
            started_at TEXT NOT NULL,
            finished_at TEXT,
            status TEXT NOT NULL,
            source_root TEXT NOT NULL,
            target_database TEXT NOT NULL,
            include_actions TEXT NOT NULL,
            streak_start INTEGER NOT NULL,
            streak_end INTEGER NOT NULL,
            source_file_count INTEGER NOT NULL,
            total_row_count INTEGER NOT NULL,
            raw_metadata_json TEXT NOT NULL
        );

        CREATE TABLE source_file (
            batch_id TEXT NOT NULL REFERENCES etl_batch(batch_id) ON DELETE CASCADE,
            layer TEXT NOT NULL,
            source_type TEXT NOT NULL,
            file_path TEXT NOT NULL,
            file_name TEXT NOT NULL,
            file_size_bytes INTEGER NOT NULL,
            sha256 TEXT NOT NULL,
            row_count INTEGER,
            loaded_at TEXT NOT NULL,
            PRIMARY KEY (batch_id, file_path)
        );

        CREATE TABLE runs (
            run_id TEXT PRIMARY KEY,
            run_type TEXT NOT NULL,
            source_path TEXT NOT NULL,
            strategy TEXT,
            checkpoint TEXT,
            ensemble_checkpoints_json TEXT,
            games_total INTEGER,
            completed_games INTEGER,
            terminal_games INTEGER,
            wins INTEGER,
            losses INTEGER,
            incomplete_games INTEGER,
            win_rate_completed REAL,
            longest_streak INTEGER,
            longest_streak_start INTEGER,
            longest_streak_end INTEGER,
            avg_elapsed_seconds REAL,
            avg_actions_per_second REAL,
            avg_agent_steps REAL,
            avg_physical_open_actions REAL,
            total_reclicks INTEGER,
            total_click_unready_actions INTEGER,
            total_unconfirmed_open_actions INTEGER,
            total_read_recoveries INTEGER,
            raw_json TEXT
        );

        CREATE TABLE model_registry (
            model_id TEXT PRIMARY KEY,
            checkpoint_path TEXT NOT NULL,
            checkpoint_name TEXT NOT NULL,
            checkpoint_exists INTEGER NOT NULL,
            checkpoint_sha256 TEXT,
            checkpoint_size_bytes INTEGER,
            training_role TEXT NOT NULL,
            registered_at TEXT NOT NULL,
            metadata_json TEXT NOT NULL
        );

        CREATE TABLE experiment_run (
            experiment_run_id TEXT PRIMARY KEY,
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            run_type TEXT NOT NULL,
            strategy TEXT,
            source_path TEXT NOT NULL,
            final_decision_mode TEXT,
            solver_allowed_during_final_decision INTEGER,
            solver_assist TEXT,
            solver_exact_limit INTEGER,
            inference_flips INTEGER,
            inference_ensemble TEXT,
            decision_action_mode TEXT,
            risk_head_weight REAL,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            avg_elapsed_seconds REAL,
            config_json TEXT NOT NULL
        );

        CREATE TABLE experiment_model_member (
            experiment_run_id TEXT NOT NULL REFERENCES experiment_run(experiment_run_id) ON DELETE CASCADE,
            model_id TEXT NOT NULL REFERENCES model_registry(model_id) ON DELETE CASCADE,
            member_index INTEGER NOT NULL,
            role TEXT NOT NULL,
            PRIMARY KEY (experiment_run_id, model_id, member_index)
        );

        CREATE TABLE promotion_decision (
            decision_id TEXT PRIMARY KEY,
            experiment_run_id TEXT NOT NULL REFERENCES experiment_run(experiment_run_id) ON DELETE CASCADE,
            model_id TEXT,
            target_win_rate REAL NOT NULL,
            target_avg_seconds REAL,
            passed INTEGER NOT NULL,
            promoted INTEGER NOT NULL,
            reason TEXT NOT NULL,
            decided_at TEXT NOT NULL,
            evidence_json TEXT NOT NULL,
            FOREIGN KEY (model_id) REFERENCES model_registry(model_id) ON DELETE SET NULL
        );

        CREATE TABLE diagnostic_runs (
            diagnostic_run_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            input TEXT NOT NULL,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            avg_elapsed_seconds REAL,
            avg_revealed_safe_cells REAL,
            primary_issue TEXT,
            next_focus TEXT,
            target_win_rate REAL,
            target_avg_seconds REAL,
            target_passed INTEGER,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE diagnostic_signal (
            diagnostic_run_id TEXT NOT NULL REFERENCES diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
            signal_name TEXT NOT NULL,
            signal_count INTEGER NOT NULL,
            PRIMARY KEY (diagnostic_run_id, signal_name)
        );

        CREATE TABLE diagnostic_game_detail (
            diagnostic_run_id TEXT NOT NULL REFERENCES diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
            game_index INTEGER NOT NULL,
            path TEXT NOT NULL,
            won INTEGER NOT NULL,
            lost INTEGER NOT NULL,
            done INTEGER NOT NULL,
            agent_steps INTEGER,
            elapsed_seconds REAL,
            revealed_safe_cells INTEGER,
            first_open_revealed_delta INTEGER,
            first_open_dirty_read INTEGER,
            no_progress_actions INTEGER,
            open_zero_progress_actions INTEGER,
            open_target_miss_with_progress INTEGER,
            open_target_miss_nearest_avg_manhattan REAL,
            open_target_miss_nearest_max_manhattan INTEGER,
            unconfirmed_open_actions INTEGER,
            open_confirm_passive_reads INTEGER,
            max_confirmed_open_cells INTEGER,
            confirmed_open_remembered_cells INTEGER,
            cleared_confirmed_virtual_flags INTEGER,
            cleared_confirmed_blocked_opens INTEGER,
            max_persistent_revealed_cells INTEGER,
            remembered_revealed_cells INTEGER,
            read_repair_actions INTEGER,
            read_restore_actions INTEGER,
            read_recovery_actions INTEGER,
            open_confirm_read_recoveries INTEGER,
            repeat_open_target_actions INTEGER,
            click_mismatch_actions INTEGER,
            sendinput_fallback_actions INTEGER,
            sendinput_error_actions INTEGER,
            click_methods_json TEXT,
            basic_audited_opens INTEGER,
            basic_known_mine_opens INTEGER,
            basic_known_safe_opens INTEGER,
            basic_safety_filter_actions INTEGER,
            basic_safety_blocked_opens INTEGER,
            solver_audited_opens INTEGER,
            solver_known_mine_opens INTEGER,
            solver_high_risk_opens INTEGER,
            terminal_action_json TEXT,
            signals_json TEXT,
            PRIMARY KEY (diagnostic_run_id, game_index)
        );

        CREATE TABLE diagnostic_recommendation (
            diagnostic_run_id TEXT NOT NULL REFERENCES diagnostic_runs(diagnostic_run_id) ON DELETE CASCADE,
            recommendation_index INTEGER NOT NULL,
            recommendation TEXT NOT NULL,
            PRIMARY KEY (diagnostic_run_id, recommendation_index)
        );

        CREATE TABLE failure_analysis_run (
            failure_analysis_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL,
            games INTEGER,
            wins INTEGER,
            losses INTEGER,
            win_rate REAL,
            wrong_flag_losses INTEGER,
            wrong_flag_loss_rate REAL,
            terminal_forced_available INTEGER,
            terminal_forced_rate REAL,
            terminal_target_known_mine INTEGER,
            terminal_target_known_mine_rate REAL,
            terminal_target_known_safe_visible_solver INTEGER,
            terminal_target_known_safe_rate REAL,
            avg_wrong_flags_on_loss REAL,
            avg_flags_on_loss REAL,
            avg_target_risk REAL,
            avg_best_guess_risk REAL,
            avg_risk_gap_to_best_guess REAL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE failure_endgame_bucket (
            failure_analysis_id TEXT NOT NULL REFERENCES failure_analysis_run(failure_analysis_id) ON DELETE CASCADE,
            bucket TEXT NOT NULL,
            safe_left_threshold INTEGER,
            loss_count INTEGER,
            loss_rate REAL,
            has_wrong_flags INTEGER,
            forced_available INTEGER,
            target_known_mine INTEGER,
            avg_target_risk REAL,
            avg_best_guess_risk REAL,
            PRIMARY KEY (failure_analysis_id, bucket)
        );

        CREATE TABLE failure_risk_bucket (
            failure_analysis_id TEXT NOT NULL REFERENCES failure_analysis_run(failure_analysis_id) ON DELETE CASCADE,
            bucket TEXT NOT NULL,
            bucket_count INTEGER NOT NULL,
            PRIMARY KEY (failure_analysis_id, bucket)
        );

        CREATE TABLE failure_exact_limit (
            failure_analysis_id TEXT NOT NULL REFERENCES failure_analysis_run(failure_analysis_id) ON DELETE CASCADE,
            exact_limit INTEGER NOT NULL,
            terminal_states INTEGER,
            forced_available INTEGER,
            target_known_mine INTEGER,
            target_known_safe INTEGER,
            avg_target_risk REAL,
            avg_best_guess_risk REAL,
            PRIMARY KEY (failure_analysis_id, exact_limit)
        );

        CREATE TABLE failure_example (
            failure_analysis_id TEXT NOT NULL REFERENCES failure_analysis_run(failure_analysis_id) ON DELETE CASCADE,
            example_index INTEGER NOT NULL,
            seed INTEGER,
            steps_before_loss INTEGER,
            action_kind TEXT,
            row INTEGER,
            col INTEGER,
            safe_left INTEGER,
            wrong_flags INTEGER,
            forced_safe INTEGER,
            forced_mines INTEGER,
            target_risk REAL,
            best_guess_risk REAL,
            target_known_mine INTEGER,
            target_known_safe INTEGER,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (failure_analysis_id, example_index)
        );

        CREATE TABLE game_summary (
            run_id TEXT NOT NULL REFERENCES runs(run_id) ON DELETE CASCADE,
            game_index INTEGER NOT NULL,
            path TEXT NOT NULL,
            won INTEGER NOT NULL,
            lost INTEGER NOT NULL,
            done INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            terminal_dialog TEXT,
            elapsed_seconds REAL,
            agent_steps INTEGER,
            physical_open_actions INTEGER,
            click_issued_actions INTEGER,
            virtual_flag_actions INTEGER,
            flags INTEGER,
            revealed_safe_cells INTEGER,
            actions_per_second REAL,
            seconds_per_action REAL,
            reclicks INTEGER,
            click_unready_actions INTEGER,
            unconfirmed_open_actions INTEGER,
            read_recoveries INTEGER,
            read_repairs INTEGER,
            read_restores INTEGER,
            open_zero_progress_actions INTEGER,
            open_target_miss_with_progress INTEGER,
            solver_assist_actions INTEGER,
            solver_safety_filter_actions INTEGER,
            basic_safety_filter_actions INTEGER,
            quick_number_read_actions INTEGER,
            quick_number_read_fallbacks INTEGER,
            first_open_row INTEGER,
            first_open_col INTEGER,
            first_open_revealed_delta INTEGER,
            raw_summary_json TEXT,
            PRIMARY KEY (run_id, game_index)
        );

        CREATE TABLE action_event (
            run_id TEXT NOT NULL,
            game_index INTEGER NOT NULL,
            step INTEGER NOT NULL,
            action_index INTEGER,
            kind TEXT NOT NULL,
            row INTEGER,
            col INTEGER,
            selection_elapsed REAL,
            virtual_only INTEGER,
            virtual_change TEXT,
            issued INTEGER,
            click_method TEXT,
            screen_x INTEGER,
            screen_y INTEGER,
            before_revealed INTEGER,
            before_flagged INTEGER,
            before_adjacent INTEGER,
            after_revealed INTEGER,
            after_flagged INTEGER,
            after_adjacent INTEGER,
            changed INTEGER,
            progress INTEGER,
            revealed_delta INTEGER,
            target_revealed_after_open INTEGER,
            confirmed_open_cells INTEGER,
            persistent_revealed_cells INTEGER,
            quick_number_attempted INTEGER,
            quick_number_value INTEGER,
            quick_capture_seconds REAL,
            quick_classification_seconds REAL,
            after_read_elapsed REAL,
            edge_click INTEGER,
            cursor_same_cell INTEGER,
            PRIMARY KEY (run_id, game_index, step),
            FOREIGN KEY (run_id, game_index) REFERENCES game_summary(run_id, game_index) ON DELETE CASCADE
        );

        CREATE TABLE decision_event (
            run_id TEXT NOT NULL,
            game_index INTEGER NOT NULL,
            step INTEGER NOT NULL,
            decision_source TEXT NOT NULL,
            action_kind TEXT NOT NULL,
            row INTEGER,
            col INTEGER,
            selected_by_solver_assist INTEGER NOT NULL,
            queued_from_solver_batch INTEGER NOT NULL,
            queued_from_chord INTEGER NOT NULL,
            solver_assist_decision TEXT,
            solver_audit_available INTEGER NOT NULL,
            solver_has_forced_moves INTEGER NOT NULL,
            solver_forced_safe_count INTEGER,
            solver_forced_mine_count INTEGER,
            target_known_safe INTEGER,
            target_known_mine INTEGER,
            target_risk REAL,
            best_guess_risk REAL,
            basic_known_safe INTEGER,
            basic_known_mine INTEGER,
            before_revealed INTEGER,
            after_revealed INTEGER,
            revealed_delta INTEGER,
            target_revealed_after_open INTEGER,
            click_issued INTEGER,
            click_method TEXT,
            click_elapsed_seconds REAL,
            after_read_elapsed REAL,
            terminal_dialog TEXT,
            raw_json TEXT NOT NULL,
            PRIMARY KEY (run_id, game_index, step),
            FOREIGN KEY (run_id, game_index, step) REFERENCES action_event(run_id, game_index, step) ON DELETE CASCADE
        );

        CREATE TABLE failure_attribution (
            run_id TEXT NOT NULL,
            game_index INTEGER NOT NULL,
            outcome TEXT NOT NULL,
            primary_failure_type TEXT NOT NULL,
            model_issue INTEGER NOT NULL,
            execution_issue INTEGER NOT NULL,
            read_issue INTEGER NOT NULL,
            click_issue INTEGER NOT NULL,
            endgame_issue INTEGER NOT NULL,
            wrong_flag_issue INTEGER NOT NULL,
            solver_forced_missed INTEGER NOT NULL,
            high_risk_terminal INTEGER NOT NULL,
            revealed_safe_cells INTEGER,
            safe_cells_left INTEGER,
            terminal_step INTEGER,
            terminal_action_kind TEXT,
            terminal_row INTEGER,
            terminal_col INTEGER,
            terminal_target_risk REAL,
            terminal_best_guess_risk REAL,
            terminal_known_safe INTEGER,
            terminal_known_mine INTEGER,
            terminal_edge INTEGER NOT NULL,
            terminal_corner INTEGER NOT NULL,
            evidence_json TEXT NOT NULL,
            PRIMARY KEY (run_id, game_index),
            FOREIGN KEY (run_id, game_index) REFERENCES game_summary(run_id, game_index) ON DELETE CASCADE
        );

        CREATE TABLE experiment_metrics (
            name TEXT PRIMARY KEY,
            source TEXT,
            games INTEGER,
            wins INTEGER,
            win_rate REAL,
            wilson_low REAL,
            wilson_high REAL,
            target_win_rate REAL,
            point_target_pass INTEGER,
            wilson_lower_target_pass INTEGER,
            note TEXT,
            raw_json TEXT
        );

        CREATE TABLE quality_checks (
            check_set TEXT NOT NULL,
            check_id TEXT NOT NULL,
            ok INTEGER NOT NULL,
            observed_json TEXT,
            expected_json TEXT,
            detail TEXT,
            PRIMARY KEY (check_set, check_id)
        );

        CREATE TABLE artifact_manifest (
            path TEXT PRIMARY KEY,
            exists_flag INTEGER NOT NULL,
            size_bytes INTEGER,
            sha256 TEXT
        );

        CREATE TABLE training_dataset_profile (
            profile_id TEXT PRIMARY KEY,
            source_path TEXT NOT NULL UNIQUE,
            profile_name TEXT NOT NULL,
            record_count INTEGER NOT NULL,
            counterfactual_records INTEGER,
            counterfactual_candidate_labels INTEGER,
            negative_counterfactual_candidates INTEGER,
            negative_counterfactual_rate REAL,
            avg_behavior_regret REAL,
            median_behavior_regret REAL,
            max_behavior_regret REAL,
            high_regret_records INTEGER,
            high_regret_rate REAL,
            known_behavior_mine_records INTEGER,
            behavior_mine_rate REAL,
            raw_json TEXT NOT NULL
        );

        CREATE TABLE training_dataset_record (
            profile_id TEXT NOT NULL REFERENCES training_dataset_profile(profile_id) ON DELETE CASCADE,
            record_index INTEGER NOT NULL,
            dataset TEXT,
            source_index INTEGER,
            family TEXT,
            expert_is_guess INTEGER,
            source_quality REAL,
            extreme_score REAL,
            action_row INTEGER,
            action_col INTEGER,
            region TEXT,
            safe_left INTEGER,
            safe_left_bucket TEXT,
            open_candidate_count INTEGER,
            open_candidate_bucket TEXT,
            counterfactual_label_count INTEGER,
            negative_counterfactual_candidates INTEGER,
            positive_counterfactual_candidates INTEGER,
            negative_counterfactual_rate REAL,
            best_counterfactual_value REAL,
            behavior_counterfactual_value REAL,
            behavior_regret REAL,
            behavior_regret_bucket TEXT,
            behavior_mine INTEGER,
            PRIMARY KEY (profile_id, record_index)
        );

        CREATE INDEX idx_training_dataset_record_family
            ON training_dataset_record(profile_id, family);
        CREATE INDEX idx_training_dataset_record_shortfall
            ON training_dataset_record(profile_id, region, safe_left_bucket, open_candidate_bucket);
        """
    )


def create_views(conn: sqlite3.Connection) -> None:
    conn.executescript(
        """
        CREATE VIEW v_experiment_leaderboard AS
        SELECT
            name,
            games,
            wins,
            win_rate,
            wilson_low,
            wilson_high,
            point_target_pass,
            wilson_lower_target_pass,
            source
        FROM experiment_metrics
        ORDER BY win_rate DESC, games DESC;

        CREATE VIEW v_windows_game_outcomes AS
        SELECT
            run_id,
            game_index,
            outcome,
            won,
            lost,
            done,
            elapsed_seconds,
            agent_steps,
            physical_open_actions,
            flags,
            revealed_safe_cells,
            actions_per_second,
            reclicks,
            unconfirmed_open_actions,
            read_recoveries
        FROM game_summary;

        CREATE VIEW v_execution_anomalies AS
        SELECT
            run_id,
            COUNT(*) AS games,
            SUM(COALESCE(reclicks, 0)) AS total_reclicks,
            SUM(COALESCE(click_unready_actions, 0)) AS total_click_unready_actions,
            SUM(COALESCE(unconfirmed_open_actions, 0)) AS total_unconfirmed_open_actions,
            SUM(COALESCE(read_recoveries, 0)) AS total_read_recoveries,
            SUM(COALESCE(open_target_miss_with_progress, 0)) AS total_open_target_miss_with_progress,
            SUM(
                COALESCE(reclicks, 0)
                + COALESCE(click_unready_actions, 0)
                + COALESCE(unconfirmed_open_actions, 0)
                + COALESCE(read_recoveries, 0)
                + COALESCE(open_target_miss_with_progress, 0)
            ) AS total_execution_anomalies
        FROM game_summary
        GROUP BY run_id;

        CREATE VIEW v_action_mix AS
        SELECT
            run_id,
            kind,
            COUNT(*) AS actions,
            SUM(CASE WHEN virtual_only THEN 1 ELSE 0 END) AS virtual_actions,
            AVG(COALESCE(revealed_delta, 0)) AS avg_revealed_delta
        FROM action_event
        GROUP BY run_id, kind;

        CREATE VIEW v_ads_model_registry AS
        SELECT
            m.model_id,
            m.checkpoint_name,
            m.checkpoint_path,
            m.checkpoint_exists,
            m.checkpoint_size_bytes,
            m.training_role,
            COUNT(DISTINCT emm.experiment_run_id) AS experiment_count,
            MAX(er.win_rate) AS best_observed_win_rate,
            MAX(er.avg_elapsed_seconds) AS best_observed_avg_elapsed_seconds
        FROM model_registry AS m
        LEFT JOIN experiment_model_member AS emm ON emm.model_id = m.model_id
        LEFT JOIN experiment_run AS er ON er.experiment_run_id = emm.experiment_run_id
        GROUP BY m.model_id;

        CREATE VIEW v_ads_experiment_registry AS
        SELECT
            er.experiment_run_id,
            er.run_type,
            er.strategy,
            er.final_decision_mode,
            er.solver_allowed_during_final_decision,
            er.solver_assist,
            er.solver_exact_limit,
            er.inference_flips,
            er.inference_ensemble,
            er.decision_action_mode,
            er.risk_head_weight,
            er.games,
            er.wins,
            er.losses,
            er.win_rate,
            er.avg_elapsed_seconds,
            GROUP_CONCAT(m.checkpoint_name, ', ') AS checkpoints
        FROM experiment_run AS er
        LEFT JOIN experiment_model_member AS emm ON emm.experiment_run_id = er.experiment_run_id
        LEFT JOIN model_registry AS m ON m.model_id = emm.model_id
        GROUP BY er.experiment_run_id;

        CREATE VIEW v_dwd_decision_event AS
        SELECT
            run_id,
            game_index,
            step,
            decision_source,
            action_kind,
            row,
            col,
            selected_by_solver_assist,
            solver_audit_available,
            solver_has_forced_moves,
            solver_forced_safe_count,
            solver_forced_mine_count,
            target_known_safe,
            target_known_mine,
            target_risk,
            best_guess_risk,
            revealed_delta,
            target_revealed_after_open,
            click_issued,
            click_method,
            click_elapsed_seconds,
            after_read_elapsed,
            terminal_dialog
        FROM decision_event;

        CREATE VIEW v_ads_extreme_decision_candidates AS
        SELECT
            d.run_id,
            d.game_index,
            d.step,
            d.decision_source,
            d.action_kind,
            d.row,
            d.col,
            (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) AS safe_cells_left,
            d.target_risk,
            d.best_guess_risk,
            (COALESCE(d.target_risk, 0.0) - COALESCE(d.best_guess_risk, COALESCE(d.target_risk, 0.0))) AS risk_gap_to_best_guess,
            CASE WHEN d.row IN (0, 15) OR d.col IN (0, 29) THEN 1 ELSE 0 END AS edge_candidate,
            CASE WHEN d.row IN (0, 15) AND d.col IN (0, 29) THEN 1 ELSE 0 END AS corner_candidate,
            CASE
                WHEN (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) <= 60
                     AND (d.row IN (0, 15) AND d.col IN (0, 29)) THEN 'corner_guess_tail'
                WHEN (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) <= 60
                     AND (d.row IN (0, 15) OR d.col IN (0, 29)) THEN 'edge_guess_tail'
                WHEN (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) <= 60 THEN 'guess_tail'
                WHEN d.row IN (0, 15) AND d.col IN (0, 29) THEN 'corner_guess'
                WHEN d.row IN (0, 15) OR d.col IN (0, 29) THEN 'edge_guess'
                WHEN COALESCE(d.target_risk, 0.0) - COALESCE(d.best_guess_risk, COALESCE(d.target_risk, 0.0)) >= 0.05 THEN 'risk_gap_guess'
                WHEN COALESCE(d.target_risk, 0.0) >= 0.33 THEN 'high_risk_guess'
                ELSE 'guess'
            END AS sample_family,
            CASE
                WHEN (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) <= 60
                     AND (d.row IN (0, 15) AND d.col IN (0, 29)) THEN 'high'
                WHEN (391 - COALESCE(a.persistent_revealed_cells, a.confirmed_open_cells, gs.revealed_safe_cells, 0)) <= 60
                     OR COALESCE(d.target_risk, 0.0) - COALESCE(d.best_guess_risk, COALESCE(d.target_risk, 0.0)) >= 0.05
                     OR (d.row IN (0, 15) OR d.col IN (0, 29)) THEN 'medium'
                ELSE 'low'
            END AS priority
        FROM decision_event AS d
        LEFT JOIN action_event AS a
          ON a.run_id = d.run_id AND a.game_index = d.game_index AND a.step = d.step
        LEFT JOIN game_summary AS gs
          ON gs.run_id = d.run_id AND gs.game_index = d.game_index
        WHERE d.decision_source = 'rl_policy'
          AND d.action_kind = 'open'
          AND d.solver_has_forced_moves = 0;

        CREATE VIEW v_ads_failure_attribution_summary AS
        SELECT
            run_id,
            primary_failure_type,
            COUNT(*) AS games,
            SUM(model_issue) AS model_issue_games,
            SUM(execution_issue) AS execution_issue_games,
            SUM(read_issue) AS read_issue_games,
            SUM(click_issue) AS click_issue_games,
            SUM(endgame_issue) AS endgame_issue_games,
            SUM(wrong_flag_issue) AS wrong_flag_issue_games,
            SUM(solver_forced_missed) AS solver_forced_missed_games,
            SUM(terminal_edge) AS terminal_edge_games,
            SUM(terminal_corner) AS terminal_corner_games,
            AVG(revealed_safe_cells) AS avg_revealed_safe_cells,
            AVG(safe_cells_left) AS avg_safe_cells_left,
            AVG(terminal_target_risk) AS avg_terminal_target_risk
        FROM failure_attribution
        GROUP BY run_id, primary_failure_type;

        CREATE VIEW v_ads_training_sample_candidates AS
        SELECT
            fa.run_id,
            fa.game_index,
            fa.primary_failure_type,
            fa.safe_cells_left,
            fa.terminal_step,
            fa.terminal_action_kind,
            fa.terminal_row,
            fa.terminal_col,
            fa.terminal_target_risk,
            fa.terminal_best_guess_risk,
            fa.terminal_edge,
            fa.terminal_corner,
            CASE
                WHEN fa.wrong_flag_issue THEN 'wrong_flag_suppression'
                WHEN fa.solver_forced_missed THEN 'terminal_forced_move_imitation'
                WHEN fa.endgame_issue AND fa.terminal_corner THEN 'endgame_corner_tail_replay'
                WHEN fa.endgame_issue AND fa.terminal_edge THEN 'endgame_edge_tail_replay'
                WHEN fa.endgame_issue THEN 'endgame_tail_replay'
                WHEN fa.high_risk_terminal AND fa.terminal_corner THEN 'corner_guess_ranking'
                WHEN fa.high_risk_terminal AND fa.terminal_edge THEN 'edge_guess_ranking'
                WHEN fa.high_risk_terminal THEN 'guess_risk_ranking'
                ELSE 'policy_loss_review'
            END AS sample_family,
            CASE
                WHEN fa.wrong_flag_issue OR fa.solver_forced_missed OR (fa.endgame_issue AND fa.terminal_corner) THEN 'high'
                WHEN fa.endgame_issue OR fa.high_risk_terminal OR fa.terminal_edge THEN 'medium'
                ELSE 'low'
            END AS priority
        FROM failure_attribution AS fa
        WHERE fa.outcome = 'loss';

        CREATE VIEW v_quality_summary AS
        SELECT
            check_set,
            COUNT(*) AS checks,
            SUM(CASE WHEN ok THEN 1 ELSE 0 END) AS passed,
            SUM(CASE WHEN ok THEN 0 ELSE 1 END) AS failed
        FROM quality_checks
        GROUP BY check_set;

        CREATE VIEW v_ten_streak_games AS
        SELECT g.*
        FROM game_summary AS g
        WHERE g.run_id = (SELECT value FROM database_metadata WHERE key = 'primary_windows_run_id')
          AND g.game_index BETWEEN
              CAST((SELECT value FROM database_metadata WHERE key = 'streak_start') AS INTEGER)
              AND
              CAST((SELECT value FROM database_metadata WHERE key = 'streak_end') AS INTEGER)
        ORDER BY g.game_index;

        CREATE VIEW v_ods_source_inventory AS
        SELECT
            layer,
            source_type,
            COUNT(*) AS file_count,
            SUM(file_size_bytes) AS total_size_bytes,
            SUM(COALESCE(row_count, 0)) AS total_row_count,
            MIN(loaded_at) AS first_loaded_at,
            MAX(loaded_at) AS last_loaded_at
        FROM source_file
        GROUP BY layer, source_type
        ORDER BY layer, source_type;

        CREATE VIEW v_dwd_game_session AS
        SELECT
            g.run_id,
            r.run_type,
            g.game_index,
            g.outcome,
            g.won,
            g.lost,
            g.done,
            g.elapsed_seconds,
            g.agent_steps,
            g.physical_open_actions,
            g.virtual_flag_actions,
            g.flags,
            g.revealed_safe_cells,
            g.actions_per_second,
            g.reclicks,
            g.click_unready_actions,
            g.unconfirmed_open_actions,
            g.read_recoveries,
            g.open_target_miss_with_progress
        FROM game_summary AS g
        JOIN runs AS r ON r.run_id = g.run_id;

        CREATE VIEW v_dwd_action_event AS
        SELECT
            run_id,
            game_index,
            step,
            kind,
            row,
            col,
            virtual_only,
            issued,
            click_method,
            revealed_delta,
            target_revealed_after_open,
            quick_number_attempted,
            quick_number_value,
            after_read_elapsed
        FROM action_event;

        CREATE VIEW v_dws_run_kpi AS
        SELECT
            run_id,
            COUNT(*) AS games,
            SUM(won) AS wins,
            SUM(lost) AS losses,
            SUM(CASE WHEN outcome = 'incomplete' THEN 1 ELSE 0 END) AS incomplete_games,
            CAST(SUM(won) AS REAL) / NULLIF(SUM(CASE WHEN done THEN 1 ELSE 0 END), 0) AS win_rate_completed,
            AVG(elapsed_seconds) AS avg_elapsed_seconds,
            AVG(actions_per_second) AS avg_actions_per_second,
            AVG(agent_steps) AS avg_agent_steps,
            SUM(COALESCE(reclicks, 0)) AS total_reclicks,
            SUM(COALESCE(click_unready_actions, 0)) AS total_click_unready_actions,
            SUM(COALESCE(unconfirmed_open_actions, 0)) AS total_unconfirmed_open_actions,
            SUM(COALESCE(read_recoveries, 0)) AS total_read_recoveries,
            SUM(COALESCE(open_target_miss_with_progress, 0)) AS total_open_target_miss_with_progress
        FROM game_summary
        GROUP BY run_id;

        CREATE VIEW v_ads_experiment_dashboard AS
        SELECT
            name,
            source,
            games,
            wins,
            win_rate,
            target_win_rate,
            win_rate - target_win_rate AS target_gap,
            wilson_low,
            wilson_high,
            point_target_pass,
            wilson_lower_target_pass,
            note
        FROM experiment_metrics
        ORDER BY win_rate DESC, games DESC;

        CREATE VIEW v_ads_diagnostic_signal_profile AS
        SELECT
            diagnostic_run_id,
            signal_name,
            signal_count
        FROM diagnostic_signal
        ORDER BY signal_count DESC, signal_name;

        CREATE VIEW v_ads_diagnostic_game_summary AS
        SELECT
            diagnostic_run_id,
            COUNT(*) AS sampled_games,
            SUM(won) AS wins,
            SUM(lost) AS losses,
            AVG(elapsed_seconds) AS avg_elapsed_seconds,
            AVG(revealed_safe_cells) AS avg_revealed_safe_cells,
            SUM(COALESCE(open_target_miss_with_progress, 0)) AS open_target_miss_with_progress,
            SUM(COALESCE(unconfirmed_open_actions, 0)) AS unconfirmed_open_actions,
            SUM(COALESCE(read_recovery_actions, 0)) AS read_recovery_actions
        FROM diagnostic_game_detail
        GROUP BY diagnostic_run_id;

        CREATE VIEW v_ads_diagnostic_recommendation AS
        SELECT
            diagnostic_run_id,
            recommendation_index,
            recommendation
        FROM diagnostic_recommendation
        ORDER BY diagnostic_run_id, recommendation_index;

        CREATE VIEW v_ads_failure_training_signal AS
        SELECT
            r.failure_analysis_id,
            r.games,
            r.wins,
            r.losses,
            r.win_rate,
            r.wrong_flag_loss_rate,
            r.terminal_forced_rate,
            r.terminal_target_known_mine_rate,
            r.terminal_target_known_safe_rate,
            r.avg_target_risk,
            r.avg_best_guess_risk,
            r.avg_risk_gap_to_best_guess,
            (
                SELECT b.safe_left_threshold
                FROM failure_endgame_bucket AS b
                WHERE b.failure_analysis_id = r.failure_analysis_id
                ORDER BY (COALESCE(b.loss_rate, 0.0) * COALESCE(b.loss_count, 0)) DESC,
                         b.safe_left_threshold DESC
                LIMIT 1
            ) AS selected_endgame_safe_left,
            (
                SELECT b.loss_rate
                FROM failure_endgame_bucket AS b
                WHERE b.failure_analysis_id = r.failure_analysis_id
                ORDER BY (COALESCE(b.loss_rate, 0.0) * COALESCE(b.loss_count, 0)) DESC,
                         b.safe_left_threshold DESC
                LIMIT 1
            ) AS selected_endgame_loss_rate,
            (
                SELECT b.loss_count
                FROM failure_endgame_bucket AS b
                WHERE b.failure_analysis_id = r.failure_analysis_id
                ORDER BY (COALESCE(b.loss_rate, 0.0) * COALESCE(b.loss_count, 0)) DESC,
                         b.safe_left_threshold DESC
                LIMIT 1
            ) AS selected_endgame_loss_count,
            (
                SELECT e.exact_limit
                FROM failure_exact_limit AS e
                WHERE e.failure_analysis_id = r.failure_analysis_id
                ORDER BY COALESCE(e.forced_available, 0) DESC,
                         COALESCE(e.target_known_safe, 0) DESC,
                         COALESCE(e.avg_best_guess_risk, 1.0) ASC,
                         e.exact_limit DESC
                LIMIT 1
            ) AS recommended_exact_limit
        FROM failure_analysis_run AS r;

        CREATE VIEW v_ads_training_dataset_profile AS
        SELECT
            profile_id,
            profile_name,
            source_path,
            record_count,
            counterfactual_records,
            counterfactual_candidate_labels,
            negative_counterfactual_rate,
            avg_behavior_regret,
            median_behavior_regret,
            max_behavior_regret,
            high_regret_records,
            high_regret_rate,
            known_behavior_mine_records,
            behavior_mine_rate
        FROM training_dataset_profile
        ORDER BY avg_behavior_regret DESC, record_count DESC;

        CREATE VIEW v_ads_training_feedback_slice AS
        SELECT
            profile_id,
            region,
            safe_left_bucket,
            open_candidate_bucket,
            behavior_regret_bucket,
            COUNT(*) AS records,
            SUM(CASE WHEN expert_is_guess THEN 1 ELSE 0 END) AS guess_records,
            SUM(CASE WHEN behavior_mine THEN 1 ELSE 0 END) AS behavior_mine_records,
            AVG(safe_left) AS avg_safe_left,
            AVG(open_candidate_count) AS avg_open_candidate_count,
            AVG(counterfactual_label_count) AS avg_counterfactual_label_count,
            AVG(negative_counterfactual_rate) AS avg_negative_counterfactual_rate,
            AVG(behavior_regret) AS avg_behavior_regret,
            MAX(behavior_regret) AS max_behavior_regret
        FROM training_dataset_record
        GROUP BY profile_id, region, safe_left_bucket, open_candidate_bucket, behavior_regret_bucket
        ORDER BY avg_behavior_regret DESC, records DESC;

        CREATE VIEW v_ads_training_feedback_priority AS
        SELECT
            profile_id,
            record_index,
            dataset,
            family,
            region,
            safe_left,
            open_candidate_count,
            behavior_regret,
            behavior_mine,
            CASE
                WHEN behavior_regret >= 8.0 AND behavior_mine THEN 'critical'
                WHEN behavior_regret >= 8.0 THEN 'high'
                WHEN behavior_regret >= 4.0 OR behavior_mine THEN 'medium'
                ELSE 'low'
            END AS priority
        FROM training_dataset_record
        ORDER BY
            CASE
                WHEN behavior_regret >= 8.0 AND behavior_mine THEN 0
                WHEN behavior_regret >= 8.0 THEN 1
                WHEN behavior_regret >= 4.0 OR behavior_mine THEN 2
                ELSE 3
            END,
            behavior_regret DESC;
        """
    )


def write_metadata(conn: sqlite3.Connection, metadata: dict[str, str]) -> None:
    conn.executemany(
        "INSERT INTO database_metadata(key, value) VALUES (?, ?)",
        sorted(metadata.items()),
    )


def register_models_for_payload(
    conn: sqlite3.Connection,
    payload: dict[str, Any],
    *,
    default_role: str = "primary",
) -> list[tuple[int, str, str]]:
    checkpoints = checkpoint_paths_from_payload(payload)
    members: list[tuple[int, str, str]] = []
    seen: set[str] = set()
    for index, checkpoint in enumerate(checkpoints):
        if checkpoint in seen:
            continue
        seen.add(checkpoint)
        role = default_role if index == 0 else "ensemble_member"
        model_id = register_model(conn, checkpoint, role=role, metadata={"member_index": index})
        members.append((index, model_id, role))
    return members


def checkpoint_paths_from_payload(payload: dict[str, Any]) -> list[str]:
    if isinstance(payload.get("checkpoints"), list):
        return [str(path) for path in payload.get("checkpoints", []) if path]
    paths: list[str] = []
    checkpoint = payload.get("checkpoint")
    if checkpoint:
        paths.append(str(checkpoint))
    for checkpoint in payload.get("ensemble_checkpoints", []) or []:
        if checkpoint:
            paths.append(str(checkpoint))
    return paths


def register_model(conn: sqlite3.Connection, checkpoint: str, *, role: str, metadata: dict[str, Any]) -> str:
    resolved = resolve_existing_path(Path(checkpoint))
    exists = resolved.exists()
    checkpoint_sha = sha256_path(resolved) if exists else None
    model_id_source = checkpoint_sha or normalize_path_text(checkpoint)
    model_id = "model_" + hashlib.sha256(model_id_source.encode("utf-8")).hexdigest()[:16]
    conn.execute(
        """
        INSERT OR IGNORE INTO model_registry(
            model_id, checkpoint_path, checkpoint_name, checkpoint_exists, checkpoint_sha256,
            checkpoint_size_bytes, training_role, registered_at, metadata_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            model_id,
            str(checkpoint),
            Path(checkpoint).name,
            as_int_bool(exists),
            checkpoint_sha,
            resolved.stat().st_size if exists else None,
            role,
            utc_now(),
            dump_json(metadata),
        ),
    )
    return model_id


def insert_experiment_run(
    conn: sqlite3.Connection,
    *,
    run_id: str,
    run_type: str,
    source_path: str,
    strategy: str | None,
    payload: dict[str, Any],
    summary: dict[str, Any],
    model_members: list[tuple[int, str, str]] | None = None,
) -> None:
    solver_assist = str(payload.get("solver_assist", "none") or "none")
    explicit_final_mode = payload.get("final_decision_mode")
    final_decision_mode = str(explicit_final_mode or ("solver_assisted_rl" if solver_assist != "none" else "rl"))
    solver_allowed = payload.get("solver_allowed_during_final_decision")
    if solver_allowed is None:
        solver_allowed = solver_assist != "none"
    win_rate = float_or_none(summary.get("win_rate_completed"))
    if win_rate is None:
        win_rate = float_or_none(payload.get("win_rate"))
    conn.execute(
        """
        INSERT INTO experiment_run(
            experiment_run_id, run_id, run_type, strategy, source_path, final_decision_mode,
            solver_allowed_during_final_decision, solver_assist, solver_exact_limit,
            inference_flips, inference_ensemble, decision_action_mode, risk_head_weight,
            games, wins, losses, win_rate, avg_elapsed_seconds, config_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            run_id,
            run_type,
            strategy,
            source_path,
            final_decision_mode,
            as_int_bool(solver_allowed),
            solver_assist,
            int_or_none(payload.get("solver_exact_limit")),
            as_int_bool(payload.get("inference_augment_flips", payload.get("inference_flips"))),
            payload.get("inference_ensemble"),
            payload.get("decision_action_mode") or payload.get("flag_mode"),
            float_or_none(payload.get("model_risk_head_weight", payload.get("risk_head_weight"))),
            int_or_none(summary.get("file_count", summary.get("games", payload.get("games")))),
            int_or_none(summary.get("wins", payload.get("wins"))),
            int_or_none(summary.get("losses", payload.get("losses"))),
            win_rate,
            float_or_none(summary.get("avg_elapsed_seconds", payload.get("elapsed_seconds"))),
            dump_json({"payload": payload, "summary": summary}),
        ),
    )
    for member_index, model_id, role in model_members or []:
        conn.execute(
            """
            INSERT INTO experiment_model_member(experiment_run_id, model_id, member_index, role)
            VALUES (?, ?, ?, ?)
            """,
            (run_id, model_id, member_index, role),
        )


def ingest_promotion_decisions(conn: sqlite3.Connection, *, default_target_win_rate: float = 0.40) -> int:
    rows = []
    decided_at = utc_now()
    query = """
        SELECT
            e.experiment_run_id,
            e.run_type,
            e.win_rate,
            e.avg_elapsed_seconds,
            e.games,
            e.wins,
            e.losses,
            m.model_id
        FROM experiment_run AS e
        LEFT JOIN experiment_model_member AS m
          ON m.experiment_run_id = e.experiment_run_id AND m.member_index = 0
        WHERE e.run_type IN ('windows_desktop', 'internal_eval')
        ORDER BY e.experiment_run_id
    """
    for row in conn.execute(query):
        experiment_run_id, run_type, win_rate, avg_seconds, games, wins, losses, model_id = row
        target_avg_seconds = 60.0 if run_type == "windows_desktop" else None
        win_passed = win_rate is not None and float(win_rate) >= default_target_win_rate
        speed_passed = target_avg_seconds is None or avg_seconds is None or float(avg_seconds) <= target_avg_seconds
        passed = bool(win_passed and speed_passed)
        reason = promotion_reason(
            run_type=str(run_type),
            win_rate=float_or_none(win_rate),
            target_win_rate=default_target_win_rate,
            avg_seconds=float_or_none(avg_seconds),
            target_avg_seconds=target_avg_seconds,
            passed=passed,
        )
        decision_id = f"promotion_{experiment_run_id}"
        rows.append(
            (
                decision_id,
                experiment_run_id,
                model_id,
                default_target_win_rate,
                target_avg_seconds,
                as_int_bool(passed),
                as_int_bool(passed),
                reason,
                decided_at,
                dump_json(
                    {
                        "run_type": run_type,
                        "games": games,
                        "wins": wins,
                        "losses": losses,
                        "win_rate": win_rate,
                        "avg_elapsed_seconds": avg_seconds,
                    }
                ),
            )
        )
    conn.executemany(
        """
        INSERT INTO promotion_decision(
            decision_id, experiment_run_id, model_id, target_win_rate, target_avg_seconds,
            passed, promoted, reason, decided_at, evidence_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def promotion_reason(
    *,
    run_type: str,
    win_rate: float | None,
    target_win_rate: float,
    avg_seconds: float | None,
    target_avg_seconds: float | None,
    passed: bool,
) -> str:
    if passed:
        return "meets win-rate target" + (" and desktop speed target" if target_avg_seconds is not None else "")
    if win_rate is None:
        return "missing win-rate evidence"
    if win_rate < target_win_rate:
        return f"win rate {win_rate:.3f} below target {target_win_rate:.3f}"
    if target_avg_seconds is not None and avg_seconds is not None and avg_seconds > target_avg_seconds:
        return f"desktop average seconds {avg_seconds:.2f} above target {target_avg_seconds:.2f}"
    return f"{run_type} did not meet promotion gate"


def ingest_windows_run(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    run_id: str,
    run_dir: Path,
    streak_start: int,
    streak_end: int,
    include_actions: str,
) -> dict[str, int]:
    summary_path = run_dir / "per_game_summary.json"
    run_summary = load_json(summary_path)
    game_paths = sorted(run_dir.glob("game_*.json"), key=game_sort_key)
    first_game = load_json(game_paths[0]) if game_paths else {}
    model_members = register_models_for_payload(conn, first_game)
    conn.execute(
        """
        INSERT INTO runs(
            run_id, run_type, source_path, strategy, checkpoint, ensemble_checkpoints_json,
            games_total, completed_games, terminal_games, wins, losses, incomplete_games,
            win_rate_completed, longest_streak, longest_streak_start, longest_streak_end,
            avg_elapsed_seconds, avg_actions_per_second, avg_agent_steps, avg_physical_open_actions,
            total_reclicks, total_click_unready_actions, total_unconfirmed_open_actions, total_read_recoveries,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "windows_desktop",
            str(run_dir),
            "pure_rl_ensemble",
            first_game.get("checkpoint"),
            dump_json(first_game.get("ensemble_checkpoints", [])),
            int_or_none(run_summary.get("file_count")) or len(game_paths),
            int_or_none(run_summary.get("completed_games")),
            int_or_none(run_summary.get("terminal_games")),
            int_or_none(run_summary.get("wins")),
            int_or_none(run_summary.get("losses")),
            int_or_none(run_summary.get("incomplete_games")),
            float_or_none(run_summary.get("win_rate_completed")),
            int_or_none(run_summary.get("longest_streak")),
            int_or_none(run_summary.get("longest_streak_start")),
            int_or_none(run_summary.get("longest_streak_end")),
            float_or_none(run_summary.get("avg_elapsed_seconds")),
            float_or_none(run_summary.get("avg_actions_per_second")),
            float_or_none(run_summary.get("avg_agent_steps")),
            float_or_none(run_summary.get("avg_physical_open_actions")),
            int_or_none(run_summary.get("total_reclicks")),
            int_or_none(run_summary.get("total_click_unready_actions")),
            int_or_none(run_summary.get("total_unconfirmed_open_actions")),
            int_or_none(run_summary.get("total_read_recoveries")),
            dump_json(run_summary),
        ),
    )
    insert_experiment_run(
        conn,
        run_id=run_id,
        run_type="windows_desktop",
        source_path=str(run_dir),
        strategy="pure_rl_ensemble",
        payload=run_config_from_game(first_game),
        summary=run_summary,
        model_members=model_members,
    )

    game_rows = []
    action_rows = []
    decision_rows = []
    failure_rows = []
    for path in game_paths:
        game = load_json(path)
        summary = game.get("summary", {})
        game_index = int_or_none(game.get("game_index")) or game_sort_key(path)
        actions = game.get("actions", [])
        virtual_flag_actions = count_virtual_flags(actions)
        first_open = summary.get("first_open_action") or {}
        first_open_action = first_open.get("action") or {}
        outcome = classify_outcome(summary)
        game_rows.append(
            (
                run_id,
                game_index,
                str(path),
                as_int_bool(summary.get("won")),
                as_int_bool(summary.get("lost")),
                as_int_bool(summary.get("done")),
                outcome,
                summary.get("terminal_dialog"),
                float_or_none(summary.get("elapsed_seconds")),
                int_or_none(summary.get("agent_steps")),
                int_or_none(summary.get("physical_open_actions")),
                int_or_none(summary.get("click_issued_actions")),
                virtual_flag_actions,
                int_or_none(summary.get("flags")),
                int_or_none(summary.get("revealed_safe_cells")),
                float_or_none(summary.get("actions_per_second")),
                float_or_none(summary.get("seconds_per_action")),
                int_or_none(summary.get("reclicks")),
                int_or_none(summary.get("click_unready_actions")),
                int_or_none(summary.get("unconfirmed_open_actions")),
                int_or_none(summary.get("read_recoveries")),
                int_or_none(summary.get("read_repairs")),
                int_or_none(summary.get("read_restores")),
                int_or_none(summary.get("open_zero_progress_actions")),
                int_or_none(summary.get("open_target_miss_with_progress")),
                int_or_none(summary.get("solver_assist_actions")),
                int_or_none(summary.get("solver_safety_filter_actions")),
                int_or_none(summary.get("basic_safety_filter_actions")),
                int_or_none(summary.get("quick_number_read_actions")),
                int_or_none(summary.get("quick_number_read_fallbacks")),
                int_or_none(first_open_action.get("row")),
                int_or_none(first_open_action.get("col")),
                int_or_none(first_open.get("revealed_delta")),
                dump_json(summary),
            )
        )
        if outcome != "win":
            failure_rows.append(failure_attribution_row(run_id, game_index, summary, actions))
        if should_ingest_actions(game_index, include_actions, streak_start, streak_end):
            for action in actions:
                action_rows.append(action_to_row(run_id, game_index, action))
                decision_rows.append(decision_event_to_row(run_id, game_index, action))

    conn.executemany(
        """
        INSERT INTO game_summary(
            run_id, game_index, path, won, lost, done, outcome, terminal_dialog,
            elapsed_seconds, agent_steps, physical_open_actions, click_issued_actions,
            virtual_flag_actions, flags, revealed_safe_cells, actions_per_second, seconds_per_action,
            reclicks, click_unready_actions, unconfirmed_open_actions, read_recoveries, read_repairs,
            read_restores, open_zero_progress_actions, open_target_miss_with_progress,
            solver_assist_actions, solver_safety_filter_actions, basic_safety_filter_actions,
            quick_number_read_actions, quick_number_read_fallbacks,
            first_open_row, first_open_col, first_open_revealed_delta, raw_summary_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        game_rows,
    )
    conn.executemany(
        """
        INSERT INTO failure_attribution(
            run_id, game_index, outcome, primary_failure_type, model_issue, execution_issue,
            read_issue, click_issue, endgame_issue, wrong_flag_issue, solver_forced_missed,
            high_risk_terminal, revealed_safe_cells, safe_cells_left, terminal_step,
            terminal_action_kind, terminal_row, terminal_col, terminal_target_risk,
            terminal_best_guess_risk, terminal_known_safe, terminal_known_mine,
            terminal_edge, terminal_corner, evidence_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        failure_rows,
    )
    conn.executemany(
        """
        INSERT INTO action_event(
            run_id, game_index, step, action_index, kind, row, col, selection_elapsed,
            virtual_only, virtual_change, issued, click_method, screen_x, screen_y,
            before_revealed, before_flagged, before_adjacent,
            after_revealed, after_flagged, after_adjacent,
            changed, progress, revealed_delta, target_revealed_after_open,
            confirmed_open_cells, persistent_revealed_cells,
            quick_number_attempted, quick_number_value, quick_capture_seconds,
            quick_classification_seconds, after_read_elapsed, edge_click, cursor_same_cell
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        action_rows,
    )
    conn.executemany(
        """
        INSERT INTO decision_event(
            run_id, game_index, step, decision_source, action_kind, row, col,
            selected_by_solver_assist, queued_from_solver_batch, queued_from_chord,
            solver_assist_decision, solver_audit_available, solver_has_forced_moves,
            solver_forced_safe_count, solver_forced_mine_count, target_known_safe,
            target_known_mine, target_risk, best_guess_risk, basic_known_safe,
            basic_known_mine, before_revealed, after_revealed, revealed_delta,
            target_revealed_after_open, click_issued, click_method, click_elapsed_seconds,
            after_read_elapsed, terminal_dialog, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        decision_rows,
    )
    return {"games": len(game_rows), "actions": len(action_rows), "decisions": len(decision_rows), "failures": len(failure_rows)}


def ingest_internal_eval(conn: sqlite3.Connection, path: Path) -> int:
    if not path.exists():
        return 0
    data = load_json(path)
    run_id = path.stem
    model_members = register_models_for_payload(conn, data)
    conn.execute(
        """
        INSERT INTO runs(
            run_id, run_type, source_path, strategy, checkpoint, ensemble_checkpoints_json,
            games_total, completed_games, terminal_games, wins, losses, incomplete_games,
            win_rate_completed, longest_streak, avg_elapsed_seconds, avg_agent_steps,
            raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            "internal_eval",
            str(path),
            "pure_rl_ensemble",
            first_or_none(data.get("checkpoints")),
            dump_json(data.get("checkpoints", [])),
            int_or_none(data.get("games")),
            int_or_none(data.get("games")),
            int_or_none(data.get("games")),
            int_or_none(data.get("wins")),
            None,
            0,
            float_or_none(data.get("win_rate")),
            int_or_none(data.get("longest_streak")),
            float_or_none(data.get("elapsed_seconds")),
            float_or_none(data.get("avg_agent_steps")),
            dump_json(data),
        ),
    )
    insert_experiment_run(
        conn,
        run_id=run_id,
        run_type="internal_eval",
        source_path=str(path),
        strategy="pure_rl_ensemble",
        payload=data,
        summary={
            "games": data.get("games"),
            "wins": data.get("wins"),
            "losses": None,
            "win_rate_completed": data.get("win_rate"),
            "avg_elapsed_seconds": None,
        },
        model_members=model_members,
    )
    return 1


def ingest_experiment_metrics(conn: sqlite3.Connection, path: Path) -> int:
    data = load_json(path)
    rows = []
    for row in data.get("experiments", []):
        rows.append(
            (
                row.get("name"),
                row.get("source"),
                int_or_none(row.get("games")),
                int_or_none(row.get("wins")),
                float_or_none(row.get("win_rate")),
                float_or_none(row.get("wilson_low")),
                float_or_none(row.get("wilson_high")),
                float_or_none(row.get("target_win_rate")),
                as_int_bool(row.get("point_target_pass")),
                as_int_bool(row.get("wilson_lower_target_pass")),
                row.get("note"),
                dump_json(row),
            )
        )
    conn.executemany(
        """
        INSERT INTO experiment_metrics(
            name, source, games, wins, win_rate, wilson_low, wilson_high,
            target_win_rate, point_target_pass, wilson_lower_target_pass, note, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def ingest_quality_checks(conn: sqlite3.Connection, check_set: str, path: Path) -> int:
    data = load_json(path)
    rows = []
    for row in data.get("checks", []):
        rows.append(
            (
                check_set,
                row.get("id"),
                as_int_bool(row.get("ok")),
                dump_json(row.get("observed")),
                dump_json(row.get("expected")),
                row.get("detail"),
            )
        )
    conn.executemany(
        """
        INSERT INTO quality_checks(check_set, check_id, ok, observed_json, expected_json, detail)
        VALUES (?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows)


def ingest_artifact_manifest(conn: sqlite3.Connection, path: Path) -> int:
    data = load_json(path)
    rows = [
        (
            row.get("path"),
            as_int_bool(row.get("exists")),
            int_or_none(row.get("size_bytes")),
            row.get("sha256"),
        )
        for row in data.get("artifacts", [])
    ]
    conn.executemany(
        "INSERT INTO artifact_manifest(path, exists_flag, size_bytes, sha256) VALUES (?, ?, ?, ?)",
        rows,
    )
    return len(rows)


def ingest_training_dataset_profiles(
    conn: sqlite3.Connection,
    paths: Iterable[Path],
) -> dict[str, int]:
    profile_rows: list[tuple[Any, ...]] = []
    record_rows: list[tuple[Any, ...]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            continue
        data = load_json(path)
        summary = data.get("summary", {}) or {}
        records = data.get("records", []) or []
        profile_id = training_profile_id(path)
        profile_rows.append(
            (
                profile_id,
                str(path),
                path.stem,
                int_or_none(summary.get("record_count")) or len(records),
                int_or_none(summary.get("counterfactual_records")),
                int_or_none(summary.get("counterfactual_candidate_labels")),
                int_or_none(summary.get("negative_counterfactual_candidates")),
                float_or_none(summary.get("negative_counterfactual_rate")),
                float_or_none(summary.get("avg_behavior_regret")),
                float_or_none(summary.get("median_behavior_regret")),
                float_or_none(summary.get("max_behavior_regret")),
                int_or_none(summary.get("high_regret_records")),
                float_or_none(summary.get("high_regret_rate")),
                int_or_none(summary.get("known_behavior_mine_records")),
                float_or_none(summary.get("behavior_mine_rate")),
                dump_json(
                    {
                        "datasets": data.get("datasets", []),
                        "summary": summary,
                        "by_family": data.get("by_family", {}),
                        "by_region": data.get("by_region", {}),
                        "by_safe_left_bucket": data.get("by_safe_left_bucket", {}),
                        "by_open_candidate_bucket": data.get("by_open_candidate_bucket", {}),
                        "by_behavior_regret_bucket": data.get("by_behavior_regret_bucket", {}),
                    }
                ),
            )
        )
        for index, record in enumerate(records):
            record_rows.append(
                (
                    profile_id,
                    index,
                    record.get("dataset"),
                    int_or_none(record.get("index")),
                    record.get("family"),
                    as_int_bool(record.get("expert_is_guess")),
                    float_or_none(record.get("source_quality")),
                    float_or_none(record.get("extreme_score")),
                    int_or_none(record.get("action_row")),
                    int_or_none(record.get("action_col")),
                    record.get("region"),
                    int_or_none(record.get("safe_left")),
                    record.get("safe_left_bucket"),
                    int_or_none(record.get("open_candidate_count")),
                    record.get("open_candidate_bucket"),
                    int_or_none(record.get("counterfactual_label_count")),
                    int_or_none(record.get("negative_counterfactual_candidates")),
                    int_or_none(record.get("positive_counterfactual_candidates")),
                    float_or_none(record.get("negative_counterfactual_rate")),
                    float_or_none(record.get("best_counterfactual_value")),
                    float_or_none(record.get("behavior_counterfactual_value")),
                    float_or_none(record.get("behavior_regret")),
                    record.get("behavior_regret_bucket"),
                    as_int_bool_or_none(record.get("behavior_mine")),
                )
            )

    conn.executemany(
        """
        INSERT INTO training_dataset_profile(
            profile_id, source_path, profile_name, record_count, counterfactual_records,
            counterfactual_candidate_labels, negative_counterfactual_candidates,
            negative_counterfactual_rate, avg_behavior_regret, median_behavior_regret,
            max_behavior_regret, high_regret_records, high_regret_rate,
            known_behavior_mine_records, behavior_mine_rate, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        profile_rows,
    )
    conn.executemany(
        """
        INSERT INTO training_dataset_record(
            profile_id, record_index, dataset, source_index, family, expert_is_guess,
            source_quality, extreme_score, action_row, action_col, region, safe_left,
            safe_left_bucket, open_candidate_count, open_candidate_bucket,
            counterfactual_label_count, negative_counterfactual_candidates,
            positive_counterfactual_candidates, negative_counterfactual_rate,
            best_counterfactual_value, behavior_counterfactual_value, behavior_regret,
            behavior_regret_bucket, behavior_mine
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        record_rows,
    )
    return {"profiles": len(profile_rows), "records": len(record_rows)}


def ingest_diagnostic_analysis(conn: sqlite3.Connection, path: Path) -> dict[str, int]:
    if not path.exists():
        return {"runs": 0, "games": 0, "signals": 0, "recommendations": 0}

    data = load_json(path)
    diagnosis = data.get("diagnosis", {})
    signal_counts = data.get("signal_counts", {})
    games_detail = data.get("games_detail", [])
    recommendations = diagnosis.get("recommendations", [])
    diagnostic_run_id = f"{path.parent.name}_{path.stem}"

    conn.execute(
        """
        INSERT INTO runs(
            run_id, run_type, source_path, strategy, games_total, completed_games,
            terminal_games, wins, losses, incomplete_games, win_rate_completed,
            avg_elapsed_seconds, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            diagnostic_run_id,
            "diagnostic_analysis",
            str(path),
            "diagnostic_analysis",
            int_or_none(data.get("games")),
            int_or_none(data.get("games")),
            int_or_none(data.get("games")),
            int_or_none(data.get("wins")),
            int_or_none(data.get("losses")),
            None,
            float_or_none(data.get("win_rate")),
            float_or_none(data.get("avg_elapsed_seconds")),
            dump_json(data),
        ),
    )
    conn.execute(
        """
        INSERT INTO diagnostic_runs(
            diagnostic_run_id, source_path, input, games, wins, losses, win_rate,
            avg_elapsed_seconds, avg_revealed_safe_cells, primary_issue, next_focus,
            target_win_rate, target_avg_seconds, target_passed, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            diagnostic_run_id,
            str(path),
            str(data.get("input", "")),
            int_or_none(data.get("games")),
            int_or_none(data.get("wins")),
            int_or_none(data.get("losses")),
            float_or_none(data.get("win_rate")),
            float_or_none(data.get("avg_elapsed_seconds")),
            float_or_none(data.get("avg_revealed_safe_cells")),
            diagnosis.get("primary_issue"),
            diagnosis.get("next_focus"),
            float_or_none(diagnosis.get("target_win_rate")),
            float_or_none(diagnosis.get("target_avg_seconds")),
            as_int_bool(diagnosis.get("target_passed")),
            dump_json(data),
        ),
    )

    signal_rows = [
        (diagnostic_run_id, str(signal_name), int_or_none(signal_count) or 0)
        for signal_name, signal_count in signal_counts.items()
    ]
    conn.executemany(
        "INSERT INTO diagnostic_signal(diagnostic_run_id, signal_name, signal_count) VALUES (?, ?, ?)",
        signal_rows,
    )

    game_rows = []
    for detail_index, detail in enumerate(games_detail, start=1):
        game_path = Path(str(detail.get("path", "")))
        game_index = game_index_from_path(game_path) or detail_index
        game_rows.append(
            (
                diagnostic_run_id,
                game_index,
                str(detail.get("path", "")),
                as_int_bool(detail.get("won")),
                as_int_bool(detail.get("lost")),
                as_int_bool(detail.get("done")),
                int_or_none(detail.get("agent_steps")),
                float_or_none(detail.get("elapsed_seconds")),
                int_or_none(detail.get("revealed_safe_cells")),
                int_or_none(detail.get("first_open_revealed_delta")),
                as_int_bool(detail.get("first_open_dirty_read")),
                int_or_none(detail.get("no_progress_actions")),
                int_or_none(detail.get("open_zero_progress_actions")),
                int_or_none(detail.get("open_target_miss_with_progress")),
                float_or_none(detail.get("open_target_miss_nearest_avg_manhattan")),
                int_or_none(detail.get("open_target_miss_nearest_max_manhattan")),
                int_or_none(detail.get("unconfirmed_open_actions")),
                int_or_none(detail.get("open_confirm_passive_reads")),
                int_or_none(detail.get("max_confirmed_open_cells")),
                int_or_none(detail.get("confirmed_open_remembered_cells")),
                int_or_none(detail.get("cleared_confirmed_virtual_flags")),
                int_or_none(detail.get("cleared_confirmed_blocked_opens")),
                int_or_none(detail.get("max_persistent_revealed_cells")),
                int_or_none(detail.get("remembered_revealed_cells")),
                int_or_none(detail.get("read_repair_actions")),
                int_or_none(detail.get("read_restore_actions")),
                int_or_none(detail.get("read_recovery_actions")),
                int_or_none(detail.get("open_confirm_read_recoveries")),
                int_or_none(detail.get("repeat_open_target_actions")),
                int_or_none(detail.get("click_mismatch_actions")),
                int_or_none(detail.get("sendinput_fallback_actions")),
                int_or_none(detail.get("sendinput_error_actions")),
                dump_json(detail.get("click_methods", {})),
                int_or_none(detail.get("basic_audited_opens")),
                int_or_none(detail.get("basic_known_mine_opens")),
                int_or_none(detail.get("basic_known_safe_opens")),
                int_or_none(detail.get("basic_safety_filter_actions")),
                int_or_none(detail.get("basic_safety_blocked_opens")),
                int_or_none(detail.get("solver_audited_opens")),
                int_or_none(detail.get("solver_known_mine_opens")),
                int_or_none(detail.get("solver_high_risk_opens")),
                dump_json(detail.get("terminal_action", {})),
                dump_json(detail.get("signals", [])),
            )
        )
    conn.executemany(
        """
        INSERT INTO diagnostic_game_detail(
            diagnostic_run_id, game_index, path, won, lost, done, agent_steps,
            elapsed_seconds, revealed_safe_cells, first_open_revealed_delta,
            first_open_dirty_read, no_progress_actions, open_zero_progress_actions,
            open_target_miss_with_progress, open_target_miss_nearest_avg_manhattan,
            open_target_miss_nearest_max_manhattan, unconfirmed_open_actions,
            open_confirm_passive_reads, max_confirmed_open_cells, confirmed_open_remembered_cells,
            cleared_confirmed_virtual_flags, cleared_confirmed_blocked_opens,
            max_persistent_revealed_cells, remembered_revealed_cells, read_repair_actions,
            read_restore_actions, read_recovery_actions, open_confirm_read_recoveries,
            repeat_open_target_actions, click_mismatch_actions, sendinput_fallback_actions,
            sendinput_error_actions, click_methods_json, basic_audited_opens,
            basic_known_mine_opens, basic_known_safe_opens, basic_safety_filter_actions,
            basic_safety_blocked_opens, solver_audited_opens, solver_known_mine_opens,
            solver_high_risk_opens, terminal_action_json, signals_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        game_rows,
    )

    recommendation_rows = [
        (diagnostic_run_id, index, str(recommendation))
        for index, recommendation in enumerate(recommendations, start=1)
    ]
    conn.executemany(
        """
        INSERT INTO diagnostic_recommendation(diagnostic_run_id, recommendation_index, recommendation)
        VALUES (?, ?, ?)
        """,
        recommendation_rows,
    )
    return {
        "runs": 1,
        "games": len(game_rows),
        "signals": len(signal_rows),
        "recommendations": len(recommendation_rows),
    }


def ingest_failure_analysis(conn: sqlite3.Connection, path: Path) -> dict[str, int]:
    if not path.exists():
        return {"runs": 0, "endgame_buckets": 0, "risk_buckets": 0, "exact_limits": 0, "examples": 0}

    data = load_json(path)
    loss = data.get("loss_analysis", {}) or {}
    exact = data.get("exact_limit_compare", {}) or {}
    failure_analysis_id = path.stem

    conn.execute(
        """
        INSERT INTO failure_analysis_run(
            failure_analysis_id, source_path, games, wins, losses, win_rate,
            wrong_flag_losses, wrong_flag_loss_rate, terminal_forced_available,
            terminal_forced_rate, terminal_target_known_mine, terminal_target_known_mine_rate,
            terminal_target_known_safe_visible_solver, terminal_target_known_safe_rate,
            avg_wrong_flags_on_loss, avg_flags_on_loss, avg_target_risk, avg_best_guess_risk,
            avg_risk_gap_to_best_guess, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            failure_analysis_id,
            str(path),
            int_or_none(loss.get("games")),
            int_or_none(loss.get("wins")),
            int_or_none(loss.get("losses")),
            float_or_none(loss.get("win_rate")),
            int_or_none(loss.get("wrong_flag_losses")),
            float_or_none(loss.get("wrong_flag_loss_rate")),
            int_or_none(loss.get("terminal_forced_available")),
            float_or_none(loss.get("terminal_forced_rate")),
            int_or_none(loss.get("terminal_target_known_mine")),
            float_or_none(loss.get("terminal_target_known_mine_rate")),
            int_or_none(loss.get("terminal_target_known_safe_visible_solver")),
            float_or_none(loss.get("terminal_target_known_safe_rate")),
            float_or_none(loss.get("avg_wrong_flags_on_loss")),
            float_or_none(loss.get("avg_flags_on_loss")),
            float_or_none(loss.get("avg_target_risk")),
            float_or_none(loss.get("avg_best_guess_risk")),
            float_or_none(loss.get("avg_risk_gap_to_best_guess")),
            dump_json(data),
        ),
    )

    endgame_rows = []
    for bucket, payload in sorted((loss.get("endgame", {}) or {}).items(), key=lambda item: item[0]):
        threshold = safe_left_threshold(bucket)
        endgame_rows.append(
            (
                failure_analysis_id,
                str(bucket),
                threshold,
                int_or_none(payload.get("count")),
                float_or_none(payload.get("loss_rate")),
                int_or_none(payload.get("has_wrong_flags")),
                int_or_none(payload.get("forced_available")),
                int_or_none(payload.get("target_known_mine")),
                float_or_none(payload.get("avg_target_risk")),
                float_or_none(payload.get("avg_best_guess_risk")),
            )
        )
    conn.executemany(
        """
        INSERT INTO failure_endgame_bucket(
            failure_analysis_id, bucket, safe_left_threshold, loss_count, loss_rate,
            has_wrong_flags, forced_available, target_known_mine, avg_target_risk, avg_best_guess_risk
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        endgame_rows,
    )

    risk_rows = [
        (failure_analysis_id, str(bucket.get("bucket")), int_or_none(bucket.get("count")) or 0)
        for bucket in loss.get("risk_buckets", [])
    ]
    conn.executemany(
        "INSERT INTO failure_risk_bucket(failure_analysis_id, bucket, bucket_count) VALUES (?, ?, ?)",
        risk_rows,
    )

    exact_rows = [
        (
            failure_analysis_id,
            int_or_none(row.get("limit")) or 0,
            int_or_none(row.get("terminal_states")),
            int_or_none(row.get("forced_available")),
            int_or_none(row.get("target_known_mine")),
            int_or_none(row.get("target_known_safe")),
            float_or_none(row.get("avg_target_risk")),
            float_or_none(row.get("avg_best_guess_risk")),
        )
        for row in exact.get("limits", [])
    ]
    conn.executemany(
        """
        INSERT INTO failure_exact_limit(
            failure_analysis_id, exact_limit, terminal_states, forced_available,
            target_known_mine, target_known_safe, avg_target_risk, avg_best_guess_risk
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """,
        exact_rows,
    )

    example_rows = []
    for index, example in enumerate(data.get("examples", []), start=1):
        example_rows.append(
            (
                failure_analysis_id,
                index,
                int_or_none(example.get("seed")),
                int_or_none(example.get("steps_before_loss")),
                example.get("action_kind"),
                int_or_none(example.get("row")),
                int_or_none(example.get("col")),
                int_or_none(example.get("safe_left")),
                int_or_none(example.get("wrong_flags")),
                int_or_none(example.get("forced_safe")),
                int_or_none(example.get("forced_mines")),
                float_or_none(example.get("target_risk")),
                float_or_none(example.get("best_guess_risk")),
                as_int_bool(example.get("target_known_mine")),
                as_int_bool(example.get("target_known_safe")),
                dump_json(example),
            )
        )
    conn.executemany(
        """
        INSERT INTO failure_example(
            failure_analysis_id, example_index, seed, steps_before_loss, action_kind,
            row, col, safe_left, wrong_flags, forced_safe, forced_mines,
            target_risk, best_guess_risk, target_known_mine, target_known_safe, raw_json
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        example_rows,
    )

    return {
        "runs": 1,
        "endgame_buckets": len(endgame_rows),
        "risk_buckets": len(risk_rows),
        "exact_limits": len(exact_rows),
        "examples": len(example_rows),
    }


def ingest_source_inventory(
    conn: sqlite3.Connection,
    *,
    batch_id: str,
    windows_run_dir: Path,
    internal_eval: Path,
    analysis_report: Path,
    failure_analysis: Path,
    statistical_summary: Path,
    evidence_validation: Path,
    claim_audit: Path,
    artifact_manifest: Path,
    training_profiles: Iterable[Path] = (),
) -> tuple[int, int]:
    rows: list[tuple[Any, ...]] = []
    loaded_at = utc_now()

    summary_path = windows_run_dir / "per_game_summary.json"
    add_source_file(rows, batch_id, "dwd", "windows_game_summary", summary_path, 1, loaded_at)
    for game_path in sorted(windows_run_dir.glob("game_*.json"), key=game_sort_key):
        game = load_json(game_path)
        add_source_file(rows, batch_id, "ods", "windows_game_log", game_path, 1 + len(game.get("actions", [])), loaded_at)

    add_source_file(rows, batch_id, "mart", "internal_eval", internal_eval, json_row_count(internal_eval, "one"), loaded_at)
    add_source_file(
        rows,
        batch_id,
        "mart",
        "statistical_summary",
        statistical_summary,
        json_row_count(statistical_summary, "experiments"),
        loaded_at,
    )
    add_source_file(
        rows,
        batch_id,
        "audit",
        "evidence_validation",
        evidence_validation,
        json_row_count(evidence_validation, "checks"),
        loaded_at,
    )
    add_source_file(rows, batch_id, "audit", "claim_audit", claim_audit, json_row_count(claim_audit, "checks"), loaded_at)
    add_source_file(
        rows,
        batch_id,
        "manifest",
        "artifact_manifest",
        artifact_manifest,
        json_row_count(artifact_manifest, "artifacts"),
        loaded_at,
    )
    if analysis_report.exists():
        analysis = load_json(analysis_report)
        row_count = (
            1
            + len(analysis.get("games_detail", []))
            + len(analysis.get("signal_counts", {}))
            + len((analysis.get("diagnosis", {}) or {}).get("recommendations", []))
        )
        add_source_file(rows, batch_id, "diagnostic", "diagnostic_analysis", analysis_report, row_count, loaded_at)
    if failure_analysis.exists():
        failure_data = load_json(failure_analysis)
        loss = failure_data.get("loss_analysis", {}) or {}
        exact = failure_data.get("exact_limit_compare", {}) or {}
        row_count = (
            1
            + len(loss.get("endgame", {}) or {})
            + len(loss.get("risk_buckets", []) or [])
            + len(exact.get("limits", []) or [])
            + len(failure_data.get("examples", []) or [])
        )
        add_source_file(rows, batch_id, "diagnostic", "failure_analysis", failure_analysis, row_count, loaded_at)
    for profile_path in training_profiles:
        profile_path = Path(profile_path)
        if not profile_path.exists():
            continue
        profile = load_json(profile_path)
        add_source_file(
            rows,
            batch_id,
            "training_feedback",
            "training_dataset_profile",
            profile_path,
            1 + len(profile.get("records", []) or []),
            loaded_at,
        )

    conn.executemany(
        """
        INSERT INTO source_file(
            batch_id, layer, source_type, file_path, file_name,
            file_size_bytes, sha256, row_count, loaded_at
        )
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        rows,
    )
    return len(rows), sum(int(row[7] or 0) for row in rows)


def run_config_from_game(game: dict[str, Any]) -> dict[str, Any]:
    timing = game.get("timing", {}) or {}
    return {
        "version": game.get("version"),
        "source": game.get("source"),
        "checkpoint": game.get("checkpoint"),
        "ensemble_checkpoints": game.get("ensemble_checkpoints", []),
        "click_method": game.get("click_method"),
        "resolved_click_method": game.get("resolved_click_method"),
        "persistent_reveals": game.get("persistent_reveals"),
        "audit_basic": game.get("audit_basic"),
        "basic_safety_filter": game.get("basic_safety_filter"),
        "solver_assist": game.get("solver_assist", "none"),
        "solver_exact_limit": game.get("solver_exact_limit"),
        "final_decision_mode": game.get("final_decision_mode", "rl"),
        "solver_allowed_during_final_decision": game.get("solver_allowed_during_final_decision", False),
        "decision_action_mode": game.get("decision_action_mode"),
        "inference_flips": game.get("inference_flips"),
        "inference_ensemble": game.get("inference_ensemble"),
        "model_risk_head_weight": game.get("model_risk_head_weight"),
        "timing": timing,
    }


def decision_event_to_row(run_id: str, game_index: int, event: dict[str, Any]) -> tuple[Any, ...]:
    action = event.get("action") or {}
    solver_assist = event.get("solver_assist") or {}
    solver_audit = event.get("solver_audit") or {}
    solver_target = solver_audit.get("target") or {}
    basic_audit = event.get("basic_audit") or {}
    before = event.get("before_target") or {}
    after = event.get("after_target") or {}
    click = event.get("click") or {}
    solver_assist_applied = bool(solver_assist.get("applied"))
    decision_source = classify_decision_source(event)
    return (
        run_id,
        game_index,
        int_or_none(event.get("step")),
        decision_source,
        action.get("kind", "unknown"),
        int_or_none(action.get("row")),
        int_or_none(action.get("col")),
        as_int_bool(solver_assist_applied and solver_assist.get("decision") == "forced_safe_open"),
        as_int_bool(event.get("queued_from_solver_batch")),
        as_int_bool(event.get("queued_from_chord")),
        solver_assist.get("decision"),
        as_int_bool(solver_audit.get("available")),
        as_int_bool(solver_audit.get("has_forced_moves")),
        int_or_none(solver_audit.get("forced_safe_count")),
        int_or_none(solver_audit.get("forced_mine_count")),
        as_int_bool(solver_target.get("known_safe")) if solver_target else None,
        as_int_bool(solver_target.get("known_mine")) if solver_target else None,
        float_or_none(solver_target.get("risk")) if solver_target else None,
        float_or_none(solver_audit.get("best_guess_risk")),
        as_int_bool(basic_audit.get("known_safe")) if basic_audit else None,
        as_int_bool(basic_audit.get("known_mine")) if basic_audit else None,
        as_int_bool(before.get("revealed")) if before else None,
        as_int_bool(after.get("revealed")) if after else None,
        int_or_none(event.get("revealed_delta")),
        as_int_bool(event.get("target_revealed_after_open")),
        as_int_bool(click.get("issued")),
        click.get("method"),
        float_or_none(click.get("elapsed_seconds")),
        float_or_none(event.get("after_read_elapsed")),
        event.get("terminal_dialog"),
        dump_json(event),
    )


def classify_decision_source(event: dict[str, Any]) -> str:
    if event.get("queued_from_solver_batch"):
        return "solver_batch_queue"
    if event.get("queued_from_chord"):
        return "chord_queue"
    solver_assist = event.get("solver_assist") or {}
    if solver_assist.get("applied") and solver_assist.get("decision") == "forced_safe_open":
        return "solver_forced_open"
    if solver_assist.get("applied") and solver_assist.get("decision") == "virtual_flag_batch":
        return "solver_virtual_flag"
    if event.get("virtual_only"):
        return "virtual_memory"
    return "rl_policy"


def failure_attribution_row(
    run_id: str,
    game_index: int,
    summary: dict[str, Any],
    actions: list[dict[str, Any]],
) -> tuple[Any, ...]:
    outcome = classify_outcome(summary)
    terminal_action = terminal_action_from_actions(actions)
    terminal_audit = (terminal_action.get("solver_audit") if terminal_action else None) or {}
    terminal_target = terminal_audit.get("target") or {}
    revealed_safe = int_or_none(summary.get("revealed_safe_cells"))
    safe_left = None if revealed_safe is None else max(0, TOTAL_SAFE_CELLS - revealed_safe)
    execution_issue = bool(
        int_or_none(summary.get("reclicks"))
        or int_or_none(summary.get("click_unready_actions"))
        or int_or_none(summary.get("unconfirmed_open_actions"))
        or int_or_none(summary.get("open_target_miss_with_progress"))
    )
    read_issue = bool(int_or_none(summary.get("read_recoveries")) or int_or_none(summary.get("read_repairs")))
    click_issue = bool(
        int_or_none(summary.get("reclicks"))
        or int_or_none(summary.get("click_unready_actions"))
        or int_or_none(summary.get("unconfirmed_open_actions"))
    )
    wrong_flag_issue = bool(terminal_action and int_or_none(terminal_action.get("wrong_flags")))
    solver_forced_missed = bool(terminal_audit.get("has_forced_moves") and not terminal_target.get("known_safe"))
    high_risk_terminal = float_or_none(terminal_target.get("risk")) is not None and float(terminal_target.get("risk")) >= 0.5
    endgame_issue = safe_left is not None and safe_left <= 60
    action = (terminal_action.get("action") if terminal_action else None) or {}
    terminal_row = int_or_none(action.get("row"))
    terminal_col = int_or_none(action.get("col"))
    terminal_edge = is_edge_cell(terminal_row, terminal_col)
    terminal_corner = is_corner_cell(terminal_row, terminal_col)
    model_issue = bool(
        outcome == "loss"
        and not execution_issue
        and (solver_forced_missed or high_risk_terminal or wrong_flag_issue or endgame_issue)
    )
    primary = primary_failure_type(
        outcome=outcome,
        execution_issue=execution_issue,
        read_issue=read_issue,
        click_issue=click_issue,
        wrong_flag_issue=wrong_flag_issue,
        solver_forced_missed=solver_forced_missed,
        high_risk_terminal=high_risk_terminal,
        endgame_issue=endgame_issue,
    )
    evidence = {
        "summary": {
            "reclicks": summary.get("reclicks"),
            "click_unready_actions": summary.get("click_unready_actions"),
            "unconfirmed_open_actions": summary.get("unconfirmed_open_actions"),
            "read_recoveries": summary.get("read_recoveries"),
            "read_repairs": summary.get("read_repairs"),
            "open_target_miss_with_progress": summary.get("open_target_miss_with_progress"),
        },
        "terminal_action": terminal_action,
        "terminal_edge": terminal_edge,
        "terminal_corner": terminal_corner,
    }
    return (
        run_id,
        game_index,
        outcome,
        primary,
        as_int_bool(model_issue),
        as_int_bool(execution_issue),
        as_int_bool(read_issue),
        as_int_bool(click_issue),
        as_int_bool(endgame_issue),
        as_int_bool(wrong_flag_issue),
        as_int_bool(solver_forced_missed),
        as_int_bool(high_risk_terminal),
        revealed_safe,
        safe_left,
        int_or_none(terminal_action.get("step")) if terminal_action else None,
        action.get("kind"),
        terminal_row,
        terminal_col,
        float_or_none(terminal_target.get("risk")) if terminal_target else None,
        float_or_none(terminal_audit.get("best_guess_risk")),
        as_int_bool(terminal_target.get("known_safe")) if terminal_target else None,
        as_int_bool(terminal_target.get("known_mine")) if terminal_target else None,
        as_int_bool(terminal_edge),
        as_int_bool(terminal_corner),
        dump_json(evidence),
    )


def terminal_action_from_actions(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for event in reversed(actions):
        if event.get("terminal_dialog") or event.get("terminal_detected_at"):
            return event
    return actions[-1] if actions else None


def primary_failure_type(
    *,
    outcome: str,
    execution_issue: bool,
    read_issue: bool,
    click_issue: bool,
    wrong_flag_issue: bool,
    solver_forced_missed: bool,
    high_risk_terminal: bool,
    endgame_issue: bool,
) -> str:
    if outcome == "incomplete":
        return "incomplete_stall_or_limit"
    if click_issue:
        return "execution_click"
    if read_issue:
        return "execution_read"
    if wrong_flag_issue:
        return "model_wrong_flag"
    if solver_forced_missed:
        return "model_missed_forced_move"
    if high_risk_terminal:
        return "model_high_risk_terminal_guess"
    if endgame_issue:
        return "model_endgame_guess"
    if execution_issue:
        return "execution_other"
    return "model_policy_or_probability"


def action_to_row(run_id: str, game_index: int, event: dict[str, Any]) -> tuple[Any, ...]:
    action = event.get("action") or {}
    click = event.get("click") or {}
    before = event.get("before_target") or {}
    after = event.get("after_target") or {}
    screen = event.get("screen_target") or click.get("target_screen") or {}
    quick = event.get("quick_number_read") or {}
    cursor_cell = click.get("cursor_cell_at_target") or {}
    return (
        run_id,
        game_index,
        int_or_none(event.get("step")),
        int_or_none(event.get("action_index")),
        action.get("kind", "unknown"),
        int_or_none(action.get("row")),
        int_or_none(action.get("col")),
        float_or_none(event.get("selection_elapsed")),
        as_int_bool(event.get("virtual_only")),
        event.get("virtual_change"),
        as_int_bool(click.get("issued")),
        click.get("method"),
        int_or_none(screen.get("x")),
        int_or_none(screen.get("y")),
        as_int_bool(before.get("revealed")),
        as_int_bool(before.get("flagged")),
        int_or_none(before.get("adjacent")),
        as_int_bool(after.get("revealed")),
        as_int_bool(after.get("flagged")),
        int_or_none(after.get("adjacent")),
        as_int_bool(event.get("changed")),
        as_int_bool(event.get("progress")),
        int_or_none(event.get("revealed_delta")),
        as_int_bool(event.get("target_revealed_after_open")),
        int_or_none(event.get("confirmed_open_cells")),
        int_or_none(event.get("persistent_revealed_cells")),
        as_int_bool(quick.get("attempted")),
        int_or_none(quick.get("cell_number")),
        float_or_none(quick.get("capture_seconds")),
        float_or_none(quick.get("classification_seconds")),
        float_or_none(event.get("after_read_elapsed")),
        as_int_bool(click.get("edge_click")),
        as_int_bool(cursor_cell.get("same_cell")),
    )


def should_ingest_actions(game_index: int, include_actions: str, streak_start: int, streak_end: int) -> bool:
    if include_actions == "all":
        return True
    if include_actions == "streak":
        return streak_start <= game_index <= streak_end
    return False


def count_virtual_flags(actions: Iterable[dict[str, Any]]) -> int:
    total = 0
    for event in actions:
        action = event.get("action") or {}
        if action.get("kind") == "flag" and event.get("virtual_only"):
            total += 1
    return total


def classify_outcome(summary: dict[str, Any]) -> str:
    if summary.get("won"):
        return "win"
    if summary.get("lost"):
        return "loss"
    if summary.get("done"):
        return "done_other"
    return "incomplete"


def game_sort_key(path: Path) -> int:
    match = GAME_RE.search(path.name)
    return int(match.group(1)) if match else 0


def game_index_from_path(path: Path) -> int | None:
    match = GAME_RE.search(path.name)
    return int(match.group(1)) if match else None


def safe_left_threshold(bucket: str) -> int | None:
    match = re.search(r"safe_left_le_(\d+)", bucket)
    return int(match.group(1)) if match else None


def build_batch_id() -> str:
    return f"batch_{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"


def training_profile_id(path: Path) -> str:
    return "training_profile_" + hashlib.sha256(normalize_path_text(str(path)).encode("utf-8")).hexdigest()[:16]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def add_source_file(
    rows: list[tuple[Any, ...]],
    batch_id: str,
    layer: str,
    source_type: str,
    path: Path,
    row_count: int,
    loaded_at: str,
) -> None:
    if not path.exists():
        return
    rows.append(
        (
            batch_id,
            layer,
            source_type,
            str(path),
            path.name,
            path.stat().st_size,
            sha256_path(path),
            row_count,
            loaded_at,
        )
    )


def json_row_count(path: Path, key: str) -> int:
    if not path.exists():
        return 0
    if key == "one":
        return 1
    data = load_json(path)
    value = data.get(key)
    if isinstance(value, (list, dict)):
        return len(value)
    return 1


def sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_existing_path(path: Path) -> Path:
    candidates = [path]
    if not path.is_absolute():
        candidates.extend([Path.cwd() / path, Path(__file__).resolve().parents[1] / path])
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return path


def normalize_path_text(path: str) -> str:
    return str(Path(path)).replace("/", "\\").lower()


def table_count(database: Path, table: str) -> int:
    with sqlite3.connect(database) as conn:
        return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def count_rows(conn: sqlite3.Connection, table: str) -> int:
    return int(conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0])


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def dump_json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True)


def as_int_bool(value: Any) -> int:
    return 1 if bool(value) else 0


def as_int_bool_or_none(value: Any) -> int | None:
    if value is None:
        return None
    return as_int_bool(value)


def int_or_none(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def first_or_none(values: Any) -> Any:
    if isinstance(values, list) and values:
        return values[0]
    return None


if __name__ == "__main__":
    main()
