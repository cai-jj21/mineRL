from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path("artifacts/report_assets/minesweeper_experiments.sqlite")
DEFAULT_OUTPUT_JSON = Path("artifacts/report_assets/database_analysis.json")
DEFAULT_OUTPUT_MD = Path("artifacts/report_assets/database_analysis.md")


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Analyze the Minesweeper RL SQLite experiment database.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    report = analyze_database(database=args.database)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


def analyze_database(database: Path = DEFAULT_DATABASE) -> dict[str, Any]:
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        report = {
            "database": str(database),
            "metadata": rows(conn, "SELECT key, value FROM database_metadata ORDER BY key"),
            "etl_batches": rows(
                conn,
                """
                SELECT
                    batch_id, status, started_at, finished_at, include_actions,
                    source_file_count, total_row_count
                FROM etl_batch
                ORDER BY started_at DESC
                """,
            ),
            "source_inventory": rows(conn, "SELECT * FROM v_ods_source_inventory"),
            "runs": rows(
                conn,
                """
                SELECT
                    run_id, run_type, games_total, completed_games, terminal_games, wins, losses,
                    win_rate_completed, longest_streak, avg_elapsed_seconds, avg_actions_per_second
                FROM runs
                ORDER BY run_type, run_id
                """,
            ),
            "warehouse_run_kpi": rows(conn, "SELECT * FROM v_dws_run_kpi ORDER BY run_id"),
            "experiment_leaderboard": rows(conn, "SELECT * FROM v_ads_experiment_dashboard"),
            "model_registry": rows(conn, "SELECT * FROM v_ads_model_registry ORDER BY model_id"),
            "experiment_registry": rows(conn, "SELECT * FROM v_ads_experiment_registry ORDER BY experiment_run_id"),
            "diagnostic_runs": rows(
                conn,
                """
                SELECT
                    diagnostic_run_id, games, wins, losses, win_rate, avg_elapsed_seconds,
                    primary_issue, next_focus, target_passed
                FROM diagnostic_runs
                ORDER BY diagnostic_run_id
                """,
            ),
            "diagnostic_signal_profile": rows(conn, "SELECT * FROM v_ads_diagnostic_signal_profile"),
            "diagnostic_game_summary": rows(conn, "SELECT * FROM v_ads_diagnostic_game_summary"),
            "diagnostic_recommendations": rows(conn, "SELECT * FROM v_ads_diagnostic_recommendation"),
            "failure_training_signal": rows(conn, "SELECT * FROM v_ads_failure_training_signal"),
            "training_dataset_profiles": rows(
                conn,
                "SELECT * FROM v_ads_training_dataset_profile",
            ),
            "training_feedback_slices": rows(
                conn,
                """
                SELECT
                    profile_id, region, safe_left_bucket, open_candidate_bucket,
                    behavior_regret_bucket, records, guess_records,
                    behavior_mine_records, avg_safe_left, avg_open_candidate_count,
                    avg_negative_counterfactual_rate, avg_behavior_regret, max_behavior_regret
                FROM v_ads_training_feedback_slice
                ORDER BY avg_behavior_regret DESC, records DESC
                """,
            ),
            "training_feedback_priority": rows(
                conn,
                """
                SELECT
                    profile_id, record_index, dataset, family, region, safe_left,
                    open_candidate_count, behavior_regret, behavior_mine, priority
                FROM v_ads_training_feedback_priority
                LIMIT 100
                """,
            ),
            "failure_attribution_summary": rows(
                conn,
                """
                SELECT
                    run_id, primary_failure_type, games, model_issue_games,
                    execution_issue_games, read_issue_games, click_issue_games,
                    endgame_issue_games, wrong_flag_issue_games, solver_forced_missed_games,
                    terminal_edge_games, terminal_corner_games,
                    avg_revealed_safe_cells, avg_safe_cells_left, avg_terminal_target_risk
                FROM v_ads_failure_attribution_summary
                ORDER BY games DESC, primary_failure_type
                """,
            ),
            "training_sample_candidates": rows(
                conn,
                """
                SELECT
                    run_id, game_index, primary_failure_type, safe_cells_left,
                    terminal_action_kind, terminal_row, terminal_col, terminal_target_risk,
                    terminal_best_guess_risk, terminal_edge, terminal_corner, sample_family, priority
                FROM v_ads_training_sample_candidates
                ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                         safe_cells_left ASC,
                         game_index ASC
                LIMIT 100
                """,
            ),
            "extreme_decision_candidates": rows_if_exists(
                conn,
                "v_ads_extreme_decision_candidates",
                """
                SELECT
                    run_id, game_index, step, decision_source, action_kind, row, col,
                    safe_cells_left, target_risk, best_guess_risk, risk_gap_to_best_guess,
                    edge_candidate, corner_candidate, sample_family, priority
                FROM v_ads_extreme_decision_candidates
                ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                         safe_cells_left ASC,
                         risk_gap_to_best_guess DESC,
                         game_index ASC,
                         step ASC
                LIMIT 100
                """,
            ),
            "decision_event_preview": rows(
                conn,
                """
                SELECT
                    run_id, game_index, step, decision_source, action_kind, row, col,
                    selected_by_solver_assist, solver_audit_available, solver_has_forced_moves,
                    solver_forced_safe_count, solver_forced_mine_count, target_known_safe,
                    target_known_mine, target_risk, best_guess_risk, revealed_delta,
                    target_revealed_after_open, click_issued, click_method
                FROM v_dwd_decision_event
                ORDER BY run_id, game_index, step
                LIMIT 100
                """,
            ),
            "failure_endgame_profile": rows(
                conn,
                """
                SELECT
                    failure_analysis_id, bucket, safe_left_threshold, loss_count, loss_rate,
                    has_wrong_flags, forced_available, target_known_mine,
                    avg_target_risk, avg_best_guess_risk
                FROM failure_endgame_bucket
                ORDER BY failure_analysis_id, safe_left_threshold
                """,
            ),
            "failure_exact_limit_compare": rows(
                conn,
                """
                SELECT
                    failure_analysis_id, exact_limit, terminal_states, forced_available,
                    target_known_mine, target_known_safe, avg_target_risk, avg_best_guess_risk
                FROM failure_exact_limit
                ORDER BY failure_analysis_id, exact_limit
                """,
            ),
            "windows_outcome_summary": rows(
                conn,
                """
                SELECT
                    run_id,
                    outcome,
                    COUNT(*) AS games,
                    AVG(elapsed_seconds) AS avg_elapsed_seconds,
                    AVG(agent_steps) AS avg_agent_steps,
                    AVG(revealed_safe_cells) AS avg_revealed_safe_cells
                FROM v_windows_game_outcomes
                GROUP BY run_id, outcome
                ORDER BY run_id, outcome
                """,
            ),
            "execution_anomalies": rows(conn, "SELECT * FROM v_execution_anomalies"),
            "action_mix": rows(conn, "SELECT * FROM v_action_mix ORDER BY run_id, kind"),
            "ten_streak_games": rows(
                conn,
                """
                SELECT
                    game_index, outcome, elapsed_seconds, agent_steps, physical_open_actions,
                    flags, revealed_safe_cells, actions_per_second
                FROM v_ten_streak_games
                ORDER BY game_index
                """,
            ),
            "quality_summary": rows(conn, "SELECT * FROM v_quality_summary ORDER BY check_set"),
            "late_loss_examples": rows(
                conn,
                """
                SELECT
                    game_index, elapsed_seconds, agent_steps, physical_open_actions,
                    flags, revealed_safe_cells, actions_per_second
                FROM game_summary
                WHERE outcome = 'loss'
                ORDER BY revealed_safe_cells DESC, agent_steps DESC
                LIMIT 10
                """,
            ),
            "slowest_games": rows(
                conn,
                """
                SELECT
                    game_index, outcome, elapsed_seconds, agent_steps, physical_open_actions,
                    revealed_safe_cells, actions_per_second
                FROM game_summary
                ORDER BY elapsed_seconds DESC
                LIMIT 10
                """,
            ),
        }
    finally:
        conn.close()
    return report


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# Minesweeper RL Experiment Warehouse",
        "",
        f"Database: `{report['database']}`",
        "",
        "## ETL 鎵规",
        "",
        table(
            report["etl_batches"],
            ["batch_id", "status", "started_at", "finished_at", "include_actions", "source_file_count", "total_row_count"],
        ),
        "",
        "## ODS 鏉ユ簮娓呭崟",
        "",
        table(
            report["source_inventory"],
            ["layer", "source_type", "file_count", "total_size_bytes", "total_row_count", "first_loaded_at", "last_loaded_at"],
        ),
        "",
        "## 杩愯姒傝",
        "",
        table(
            report["runs"],
            ["run_id", "run_type", "completed_games", "wins", "losses", "win_rate_completed", "longest_streak", "avg_elapsed_seconds"],
        ),
        "",
        "## DWS Run KPI",
        "",
        table(
            report["warehouse_run_kpi"],
            ["run_id", "games", "wins", "losses", "incomplete_games", "win_rate_completed", "avg_elapsed_seconds", "avg_actions_per_second", "total_reclicks", "total_unconfirmed_open_actions", "total_read_recoveries"],
        ),
        "",
        "## 瀹為獙鑳滅巼姒?",
        "",
        table(
            report["experiment_leaderboard"],
            ["name", "games", "wins", "win_rate", "target_win_rate", "target_gap", "wilson_low", "wilson_high", "point_target_pass"],
        ),
        "",
        "## Model Registry",
        "",
        table(
            report["model_registry"],
            ["checkpoint_name", "training_role", "checkpoint_exists", "checkpoint_size_bytes", "experiment_count", "best_observed_win_rate"],
        ),
        "",
        "## Experiment Registry",
        "",
        table(
            report["experiment_registry"],
            ["experiment_run_id", "run_type", "final_decision_mode", "solver_assist", "solver_exact_limit", "win_rate", "avg_elapsed_seconds"],
        ),
        "",
        "## 璇婃柇杩愯",
        "",
        table(
            report["diagnostic_runs"],
            ["diagnostic_run_id", "games", "wins", "losses", "win_rate", "avg_elapsed_seconds", "primary_issue", "next_focus", "target_passed"],
        ),
        "",
        "## 璇婃柇淇″彿",
        "",
        table(report["diagnostic_signal_profile"], ["diagnostic_run_id", "signal_name", "signal_count"]),
        "",
        "## 璇婃柇寤鸿",
        "",
        table(report["diagnostic_recommendations"], ["diagnostic_run_id", "recommendation_index", "recommendation"]),
        "",
        "## 璁粌鍙嶅摵淇″彿",
        "",
        table(
            report["failure_training_signal"],
            ["failure_analysis_id", "games", "losses", "wrong_flag_loss_rate", "terminal_forced_rate", "avg_risk_gap_to_best_guess", "selected_endgame_safe_left", "selected_endgame_loss_rate", "recommended_exact_limit"],
        ),
        "",
        "## Training Dataset Profiles",
        "",
        table(
            report["training_dataset_profiles"],
            [
                "profile_name",
                "record_count",
                "counterfactual_candidate_labels",
                "negative_counterfactual_rate",
                "avg_behavior_regret",
                "high_regret_rate",
                "behavior_mine_rate",
            ],
        ),
        "",
        "## Training Feedback Slices",
        "",
        table(
            report["training_feedback_slices"],
            [
                "region",
                "safe_left_bucket",
                "open_candidate_bucket",
                "behavior_regret_bucket",
                "records",
                "avg_behavior_regret",
                "max_behavior_regret",
                "behavior_mine_records",
            ],
        ),
        "",
        "## Training Feedback Priority",
        "",
        table(
            report["training_feedback_priority"],
            [
                "record_index",
                "family",
                "region",
                "safe_left",
                "open_candidate_count",
                "behavior_regret",
                "behavior_mine",
                "priority",
            ],
        ),
        "",
        "## Failure Attribution",
        "",
        table(
            report["failure_attribution_summary"],
            [
                "run_id",
                "primary_failure_type",
                "games",
                "model_issue_games",
                "execution_issue_games",
                "wrong_flag_issue_games",
                "solver_forced_missed_games",
                "terminal_edge_games",
                "terminal_corner_games",
                "avg_safe_cells_left",
            ],
        ),
        "",
        "## 娈嬪眬澶辫触鍒嗗竷",
        "",
        table(
            report["failure_endgame_profile"],
            ["failure_analysis_id", "bucket", "safe_left_threshold", "loss_count", "loss_rate", "has_wrong_flags", "forced_available", "avg_target_risk", "avg_best_guess_risk"],
        ),
        "",
        "## Exact Limit 瀵圭収",
        "",
        table(
            report["failure_exact_limit_compare"],
            ["failure_analysis_id", "exact_limit", "terminal_states", "forced_available", "target_known_mine", "target_known_safe", "avg_target_risk", "avg_best_guess_risk"],
        ),
        "",
        "## Training Sample Candidates",
        "",
        table(
            report["training_sample_candidates"],
            [
                "run_id",
                "game_index",
                "sample_family",
                "priority",
                "primary_failure_type",
                "safe_cells_left",
                "terminal_action_kind",
                "terminal_edge",
                "terminal_corner",
                "terminal_target_risk",
            ],
        ),
        "",
        "## Extreme Decision Candidates",
        "",
        table(
            report["extreme_decision_candidates"],
            [
                "run_id",
                "game_index",
                "step",
                "sample_family",
                "priority",
                "safe_cells_left",
                "row",
                "col",
                "target_risk",
                "best_guess_risk",
                "risk_gap_to_best_guess",
                "edge_candidate",
                "corner_candidate",
            ],
        ),
        "",
        "## Decision Event Preview",
        "",
        table(
            report["decision_event_preview"],
            ["run_id", "game_index", "step", "decision_source", "action_kind", "row", "col", "solver_audit_available", "solver_has_forced_moves", "target_risk", "best_guess_risk"],
        ),
        "",
        "## Windows 缁堝眬鍒嗗竷",
        "",
        table(
            report["windows_outcome_summary"],
            ["run_id", "outcome", "games", "avg_elapsed_seconds", "avg_agent_steps", "avg_revealed_safe_cells"],
        ),
        "",
        "## 鎵ц灞傚紓甯?",
        "",
        table(
            report["execution_anomalies"],
            ["run_id", "games", "total_reclicks", "total_click_unready_actions", "total_unconfirmed_open_actions", "total_read_recoveries", "total_execution_anomalies"],
        ),
        "",
        "## Action Mix",
        "",
        table(report["action_mix"], ["run_id", "kind", "actions", "virtual_actions", "avg_revealed_delta"]),
        "",
        "## 鍗佽繛鑳滃眬鏄庣粏",
        "",
        table(
            report["ten_streak_games"],
            ["game_index", "outcome", "elapsed_seconds", "agent_steps", "physical_open_actions", "flags", "revealed_safe_cells", "actions_per_second"],
        ),
        "",
        "## 璐ㄩ噺鏍￠獙姹囨€?",
        "",
        table(report["quality_summary"], ["check_set", "checks", "passed", "failed"]),
        "",
        "## 楂樻彮绀鸿緭灞€鏍锋湰",
        "",
        table(
            report["late_loss_examples"],
            ["game_index", "elapsed_seconds", "agent_steps", "physical_open_actions", "flags", "revealed_safe_cells", "actions_per_second"],
        ),
        "",
        "## 鏈€鎱㈠眬鏍锋湰",
        "",
        table(
            report["slowest_games"],
            ["game_index", "outcome", "elapsed_seconds", "agent_steps", "physical_open_actions", "revealed_safe_cells", "actions_per_second"],
        ),
        "",
    ]
    return "\n".join(lines)

def table(rows_: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows_:
        return "_鏃犳暟鎹甠"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows_:
        lines.append("| " + " | ".join(format_cell(row.get(column), column) for column in columns) + " |")
    return "\n".join(lines)


def format_cell(value: Any, column: str) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if column in {
            "win_rate_completed",
            "win_rate",
            "wilson_low",
            "wilson_high",
            "target_win_rate",
            "target_gap",
            "wrong_flag_loss_rate",
            "terminal_forced_rate",
            "terminal_target_known_mine_rate",
            "terminal_target_known_safe_rate",
            "selected_endgame_loss_rate",
            "loss_rate",
        }:
            return f"{value * 100:.2f}%"
        return f"{value:.2f}"
    return str(value)


def rows(conn: sqlite3.Connection, sql: str) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql)]


def rows_if_exists(conn: sqlite3.Connection, name: str, sql: str) -> list[dict[str, Any]]:
    if not table_exists(conn, name):
        return []
    return rows(conn, sql)


def table_exists(conn: sqlite3.Connection, name: str) -> bool:
    row = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type IN ('table', 'view') AND name = ?",
        (name,),
    ).fetchone()
    return row is not None


if __name__ == "__main__":
    main()
