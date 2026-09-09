from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path("artifacts/report_assets/minesweeper_experiments.sqlite")
DEFAULT_OUTPUT_JSON = Path("artifacts/report_assets/training_feedback_plan.json")
DEFAULT_OUTPUT_MD = Path("artifacts/report_assets/training_feedback_plan.md")
DEFAULT_CHECKPOINT = Path("artifacts/full_rlmix_20.pt")
DEFAULT_SAVE_PATH = Path("artifacts/full_rlmix_feedback_refine.pt")
DEFAULT_OUTPUT_DIR = Path("artifacts/training_feedback_refine")
DEFAULT_REPLAY_LOG_DIR = Path("artifacts/windows_agent")
DEFAULT_REPLAY_LOG_LIMIT = 8192
DEFAULT_EXTREME_DATASET = Path(
    "artifacts/report_assets/extreme_training_dataset/guess_counterfactual_policy_v5_stratified_mineonly_merged.npz"
)
DEFAULT_WINDOWS_EXTREME_DATASET = Path("artifacts/report_assets/extreme_training_dataset/windows_extreme_transitions.npz")
DEFAULT_SIM_GUESS_DATASET = Path("artifacts/report_assets/extreme_training_dataset/sim_guess_counterfactual.npz")
DEFAULT_SIM_SHORTBOARD_DATASET = Path("artifacts/report_assets/extreme_training_dataset/sim_guess_shortboard.npz")
DEFAULT_MODEL_FAILURE_GUESS_DATASET = Path("artifacts/report_assets/extreme_training_dataset/model_failure_guess_counterfactual.npz")
BOARD_ROWS = 16
BOARD_COLS = 30


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Generate a training feedback plan from the experiment warehouse.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--save-path", type=Path, default=DEFAULT_SAVE_PATH)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--replay-log-dir", type=Path, default=DEFAULT_REPLAY_LOG_DIR)
    parser.add_argument("--replay-log-limit", type=int, default=DEFAULT_REPLAY_LOG_LIMIT)
    parser.add_argument("--extreme-dataset", type=Path, default=DEFAULT_EXTREME_DATASET)
    parser.add_argument("--target-win-rate", type=float, default=0.47)
    parser.add_argument("--target-avg-seconds", type=float, default=60.0)
    args = parser.parse_args()

    report = build_training_feedback_plan(
        database=args.database,
        checkpoint=args.checkpoint,
        save_path=args.save_path,
        output_dir=args.output_dir,
        output_json=args.output_json,
        output_md=args.output_md,
        replay_log_dir=args.replay_log_dir,
        replay_log_limit=args.replay_log_limit,
        extreme_dataset=args.extreme_dataset,
        target_win_rate=args.target_win_rate,
        target_avg_seconds=args.target_avg_seconds,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_md.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    args.output_md.write_text(render_markdown(report), encoding="utf-8")
    print(json.dumps({"ok": True, "output_json": str(args.output_json), "output_md": str(args.output_md)}, ensure_ascii=False, indent=2))


def build_training_feedback_plan(
    *,
    database: Path = DEFAULT_DATABASE,
    checkpoint: Path | None = None,
    save_path: Path = DEFAULT_SAVE_PATH,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    output_json: Path = DEFAULT_OUTPUT_JSON,
    output_md: Path = DEFAULT_OUTPUT_MD,
    replay_log_dir: Path = DEFAULT_REPLAY_LOG_DIR,
    replay_log_limit: int = DEFAULT_REPLAY_LOG_LIMIT,
    extreme_dataset: Path = DEFAULT_EXTREME_DATASET,
    target_win_rate: float = 0.47,
    target_avg_seconds: float = 60.0,
) -> dict[str, Any]:
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        baseline = first_row(conn, "SELECT * FROM v_dws_run_kpi ORDER BY run_id LIMIT 1")
        leader = first_row(conn, "SELECT * FROM v_ads_experiment_dashboard ORDER BY win_rate DESC, games DESC LIMIT 1")
        model_registry = rows_if_exists(conn, "v_ads_model_registry", "SELECT * FROM v_ads_model_registry ORDER BY model_id")
        experiment_registry = rows_if_exists(
            conn,
            "v_ads_experiment_registry",
            "SELECT * FROM v_ads_experiment_registry ORDER BY experiment_run_id",
        )
        diagnostic = first_row(conn, "SELECT * FROM v_ads_failure_training_signal LIMIT 1")
        failure_attribution = rows_if_exists(
            conn,
            "v_ads_failure_attribution_summary",
            """
            SELECT *
            FROM v_ads_failure_attribution_summary
            ORDER BY games DESC, primary_failure_type
            """,
        )
        sample_candidates = rows_if_exists(
            conn,
            "v_ads_training_sample_candidates",
            """
            SELECT *
            FROM v_ads_training_sample_candidates
            ORDER BY CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                     safe_cells_left ASC,
                     game_index ASC
            LIMIT 50
            """,
        )
        decision_samples = rows_if_exists(
            conn,
            "decision_event",
            f"""
            SELECT
                step,
                action_kind,
                row,
                col,
                selected_by_solver_assist,
                solver_audit_available,
                solver_has_forced_moves,
                target_known_safe,
                target_known_mine,
                target_risk,
                best_guess_risk
            FROM decision_event
            WHERE action_kind = 'open'
            ORDER BY step
            LIMIT 2048
            """,
        )
        sample_family_profile = summarize_sample_family_profile(sample_candidates)
        decision_family_profile = summarize_decision_family_profile(decision_samples)
        execution = first_row(conn, "SELECT * FROM v_execution_anomalies ORDER BY total_execution_anomalies DESC LIMIT 1")
        analysis = first_row(
            conn,
            """
            SELECT primary_issue, next_focus, target_win_rate, target_avg_seconds, target_passed
            FROM diagnostic_runs
            ORDER BY diagnostic_run_id DESC
            LIMIT 1
            """,
        )
        execution_quality = (
            first_row(
                conn,
                """
                SELECT ok
                FROM quality_checks
                WHERE check_id = 'desktop.execution_anomalies'
                ORDER BY check_set
                LIMIT 1
                """,
            )
            if table_exists(conn, "quality_checks")
            else None
        )
        endgame_rows = rows(
            conn,
            """
            SELECT bucket, safe_left_threshold, loss_count, loss_rate, has_wrong_flags, forced_available,
                   target_known_mine, avg_target_risk, avg_best_guess_risk
            FROM failure_endgame_bucket
            ORDER BY safe_left_threshold
            """,
        )
        exact_rows = rows(
            conn,
            """
            SELECT exact_limit, terminal_states, forced_available, target_known_mine, target_known_safe,
                   avg_target_risk, avg_best_guess_risk
            FROM failure_exact_limit
            ORDER BY exact_limit DESC
            """,
        )
        risk_rows = rows(conn, "SELECT bucket, bucket_count FROM failure_risk_bucket ORDER BY bucket_count DESC, bucket")
        training_dataset_profiles = rows_if_exists(
            conn,
            "training_dataset_profile",
            "SELECT * FROM training_dataset_profile ORDER BY profile_name",
        )
        training_dataset_records = rows_if_exists(
            conn,
            "training_dataset_record",
            """
            SELECT family, region, safe_left_bucket, behavior_regret_bucket,
                   counterfactual_label_count, negative_counterfactual_candidates,
                   behavior_regret, behavior_mine
            FROM training_dataset_record
            """,
        )
    finally:
        conn.close()

    if not diagnostic:
        raise RuntimeError("failure training signals are missing from the warehouse")

    execution_data = dict(execution) if execution else None
    execution_quality_passed = bool(execution_quality["ok"]) if execution_quality and execution_quality["ok"] is not None else None
    current_win_rate = float(baseline["win_rate_completed"]) if baseline and baseline["win_rate_completed"] is not None else None
    current_actions_per_second = float(baseline["avg_actions_per_second"]) if baseline and baseline["avg_actions_per_second"] is not None else None
    current_reclicks = int(baseline["total_reclicks"]) if baseline and baseline["total_reclicks"] is not None else 0
    current_unconfirmed = int(baseline["total_unconfirmed_open_actions"]) if baseline and baseline["total_unconfirmed_open_actions"] is not None else 0
    current_read_recoveries = int(baseline["total_read_recoveries"]) if baseline and baseline["total_read_recoveries"] is not None else 0
    target_gap = max(0.0, target_win_rate - (current_win_rate or 0.0))

    wrong_flag_rate = float(diagnostic["wrong_flag_loss_rate"] or 0.0)
    terminal_forced_rate = float(diagnostic["terminal_forced_rate"] or 0.0)
    risk_gap = float(diagnostic["avg_risk_gap_to_best_guess"] or 0.0)
    selected_endgame_safe_left = int(diagnostic["selected_endgame_safe_left"] or 60)
    selected_endgame_loss_rate = float(diagnostic["selected_endgame_loss_rate"] or 0.0)
    recommended_exact_limit = int(diagnostic["recommended_exact_limit"] or 32)

    selected_checkpoint = _select_training_checkpoint(checkpoint, model_registry)
    training_asset_summary = summarize_training_dataset_assets(
        training_dataset_profiles,
        training_dataset_records,
    )

    hard_loss_args = derive_hard_loss_args(
        target_gap=target_gap,
        wrong_flag_rate=wrong_flag_rate,
        terminal_forced_rate=terminal_forced_rate,
        risk_gap=risk_gap,
        selected_endgame_safe_left=selected_endgame_safe_left,
        selected_endgame_loss_rate=selected_endgame_loss_rate,
        recommended_exact_limit=recommended_exact_limit,
        sample_family_profile=sample_family_profile,
        decision_family_profile=decision_family_profile,
        replay_log_dir=replay_log_dir,
        replay_log_limit=replay_log_limit,
        extreme_dataset=extreme_dataset,
        training_asset_summary=training_asset_summary,
    )
    focus = derive_training_focus(
        wrong_flag_rate=wrong_flag_rate,
        terminal_forced_rate=terminal_forced_rate,
        risk_gap=risk_gap,
        selected_endgame_safe_left=selected_endgame_safe_left,
        selected_endgame_loss_rate=selected_endgame_loss_rate,
        sample_family_profile=sample_family_profile,
        decision_family_profile=decision_family_profile,
        training_asset_summary=training_asset_summary,
        execution=execution_data,
        execution_quality_passed=execution_quality_passed,
    )
    commands = {
        "feedback_plan": f"python scripts/generate_training_feedback_plan.py --database {database} --output-json {output_json} --output-md {output_md}",
        "export_training_feedback_dataset": f"python scripts/export_training_feedback_dataset.py --database {database}",
        "build_extreme_training_dataset": " ".join(
            [
                "python",
                "scripts/build_extreme_training_dataset.py",
                "--checkpoint",
                str(selected_checkpoint),
                "--replay-log-dir",
                str(replay_log_dir),
                "--output",
                str(DEFAULT_WINDOWS_EXTREME_DATASET),
                "--device",
                "cpu",
                "--safe-left-threshold",
                str(selected_endgame_safe_left),
            ]
        ),
        "build_sim_guess_counterfactual_dataset": " ".join(
            [
                "python",
                "scripts/build_sim_guess_training_dataset.py",
                "--checkpoint",
                str(selected_checkpoint),
                "--output",
                str(DEFAULT_SIM_GUESS_DATASET),
                "--device",
                "cuda",
                "--games",
                str(max(512, int(hard_loss_args["mine_games"]) * 4)),
                "--safe-left-threshold",
                str(max(120, selected_endgame_safe_left)),
                "--guess-topk",
                str(hard_loss_args["guess_supervision_topk"]),
                "--counterfactual-labels",
                "--counterfactual-topk",
                str(hard_loss_args["counterfactual_topk"]),
            ]
        ),
        "build_sim_guess_shortboard_dataset": " ".join(
            [
                "python",
                "scripts/build_sim_guess_training_dataset.py",
                "--checkpoint",
                str(selected_checkpoint),
                "--output",
                str(DEFAULT_SIM_SHORTBOARD_DATASET),
                "--device",
                "cuda",
                "--games",
                str(max(1024, int(hard_loss_args["mine_games"]) * 8)),
                "--trajectory-mode",
                "policy",
                "--safe-left-threshold",
                str(selected_endgame_safe_left),
                "--collect-safe-left-min",
                "1",
                "--collect-safe-left-max",
                str(selected_endgame_safe_left),
                "--family-filter",
                "guess",
                "--family-filter",
                "guess_tail",
                "--family-filter",
                "edge_guess",
                "--family-filter",
                "edge_guess_tail",
                "--family-filter",
                "corner_guess_tail",
                "--behavior-selection",
                "mine-or-regret",
                "--min-behavior-regret",
                "4.0",
                "--counterfactual-labels",
                "--counterfactual-topk",
                str(hard_loss_args["counterfactual_topk"]),
                "--counterfactual-all-open",
                "--counterfactual-model-topk",
                str(hard_loss_args["counterfactual_topk"]),
                "--policy-candidate-checkpoint",
                str(selected_checkpoint),
            ]
        ),
        "build_model_failure_guess_counterfactual_dataset": " ".join(
            [
                "python",
                "scripts/build_model_failure_extreme_dataset.py",
                "--checkpoint",
                str(selected_checkpoint),
                "--output",
                str(DEFAULT_MODEL_FAILURE_GUESS_DATASET),
                "--device",
                "cuda",
                "--games",
                str(hard_loss_args["mine_games"]),
                "--tail-states",
                str(hard_loss_args["tail_states"]),
                "--endgame-safe-left",
                str(selected_endgame_safe_left),
                "--exact-limit",
                str(recommended_exact_limit),
                "--guess-topk",
                str(hard_loss_args["guess_supervision_topk"]),
                "--counterfactual-labels",
                "--counterfactual-topk",
                str(hard_loss_args["counterfactual_topk"]),
            ]
        ),
        "merge_extreme_datasets": " ".join(
            [
                "python",
                "scripts/merge_extreme_datasets.py",
                "--input",
                str(DEFAULT_WINDOWS_EXTREME_DATASET),
                "--input",
                str(DEFAULT_SIM_GUESS_DATASET),
                "--input",
                str(DEFAULT_MODEL_FAILURE_GUESS_DATASET),
                "--output",
                str(extreme_dataset),
            ]
        ),
        "hard_loss_refine": " ".join(
            [
                "python",
                "scripts/hard_loss_refine.py",
                "--checkpoint",
                str(selected_checkpoint),
                "--save-path",
                str(save_path),
                "--output-dir",
                str(output_dir),
                "--device",
                "cuda",
                "--feedback-plan",
                str(output_json),
                "--replay-log-dir",
                str(replay_log_dir),
                "--replay-log-limit",
                str(replay_log_limit),
                "--extreme-dataset",
                str(extreme_dataset),
            ]
        )
        + _build_hard_loss_refine_suffix(hard_loss_args),
        "analyze_extreme_policy_alignment": " ".join(
            [
                "python",
                "scripts/analyze_extreme_policy_alignment.py",
                "--dataset",
                str(extreme_dataset),
                "--checkpoint",
                str(selected_checkpoint),
                "--device",
                "cuda",
                "--output",
                str(extreme_dataset.with_suffix(".alignment.json")),
            ]
        ),
    }

    return {
        "ok": True,
        "database": str(database),
        "generated_at": utc_now(),
        "checkpoint": str(selected_checkpoint),
        "target": {
            "win_rate": target_win_rate,
            "avg_seconds": target_avg_seconds,
        },
        "baseline": {
            "run_id": baseline["run_id"] if baseline else None,
            "win_rate_completed": current_win_rate,
            "avg_elapsed_seconds": float(baseline["avg_elapsed_seconds"]) if baseline and baseline["avg_elapsed_seconds"] is not None else None,
            "avg_actions_per_second": current_actions_per_second,
            "total_reclicks": current_reclicks,
            "total_unconfirmed_open_actions": current_unconfirmed,
            "total_read_recoveries": current_read_recoveries,
            "leaderboard_name": leader["name"] if leader else None,
            "leaderboard_win_rate": float(leader["win_rate"]) if leader and leader["win_rate"] is not None else None,
            "target_gap": target_gap,
        },
        "signals": {
            "diagnostic_run_id": diagnostic["failure_analysis_id"],
            "wrong_flag_loss_rate": wrong_flag_rate,
            "terminal_forced_rate": terminal_forced_rate,
            "terminal_target_known_mine_rate": float(diagnostic["terminal_target_known_mine_rate"] or 0.0),
            "terminal_target_known_safe_rate": float(diagnostic["terminal_target_known_safe_rate"] or 0.0),
            "avg_risk_gap_to_best_guess": risk_gap,
            "selected_endgame_safe_left": selected_endgame_safe_left,
            "selected_endgame_loss_rate": selected_endgame_loss_rate,
            "recommended_exact_limit": recommended_exact_limit,
            "sample_family_profile": sample_family_profile,
            "decision_family_profile": decision_family_profile,
            "execution_anomalies": execution_data or {},
            "execution_quality_passed": execution_quality_passed,
            "training_asset_summary": training_asset_summary,
        },
        "model_registry": model_registry,
        "experiment_registry": experiment_registry,
        "failure_attribution_summary": failure_attribution,
        "training_sample_candidates": sample_candidates,
        "training_focus": focus,
        "data_asset_pipeline": [
            "build_extreme_training_dataset",
            "build_sim_guess_counterfactual_dataset",
            "build_sim_guess_shortboard_dataset",
            "build_model_failure_guess_counterfactual_dataset",
            "merge_extreme_datasets",
            "analyze_extreme_policy_alignment",
            "hard_loss_refine",
        ],
        "exact_limit_compare": exact_rows,
        "endgame_profile": endgame_rows,
        "risk_buckets": risk_rows,
        "training_dataset_profiles": training_dataset_profiles,
        "training_dataset_asset_summary": training_asset_summary,
        "hard_loss_refine_args": hard_loss_args,
        "commands": commands,
    }


def derive_hard_loss_args(
    *,
    target_gap: float,
    wrong_flag_rate: float,
    terminal_forced_rate: float,
    risk_gap: float,
    selected_endgame_safe_left: int,
    selected_endgame_loss_rate: float,
    recommended_exact_limit: int,
    sample_family_profile: dict[str, int],
    decision_family_profile: dict[str, int],
    replay_log_dir: Path,
    replay_log_limit: int,
    extreme_dataset: Path,
    training_asset_summary: dict[str, Any] | None = None,
) -> dict[str, Any]:
    tail_samples = int(sample_family_profile.get("tail_samples", 0))
    edge_samples = int(sample_family_profile.get("edge_samples", 0))
    corner_samples = int(sample_family_profile.get("corner_samples", 0))
    guess_samples = int(sample_family_profile.get("guess_samples", 0))
    guess_decisions = int(decision_family_profile.get("guess_decisions", 0))
    edge_guess_decisions = int(decision_family_profile.get("edge_guess_decisions", 0))
    corner_guess_decisions = int(decision_family_profile.get("corner_guess_decisions", 0))
    high_risk_guess_decisions = int(decision_family_profile.get("high_risk_guess_decisions", 0))
    endgame_weight = 4 if selected_endgame_loss_rate >= 0.6 else 3 if selected_endgame_loss_rate >= 0.35 else 2
    wrong_flag_weight = 5 if wrong_flag_rate >= 0.2 else 4 if wrong_flag_rate >= 0.1 else 3
    terminal_weight = 4 if terminal_forced_rate >= 0.25 else 3
    guess_weight = 4 if guess_samples > 0 or guess_decisions > 0 or risk_gap >= 0.04 else 3
    edge_weight = 3 if edge_samples > 0 or edge_guess_decisions > 0 or selected_endgame_loss_rate >= 0.35 else 2
    corner_weight = 4 if corner_samples > 0 or corner_guess_decisions > 0 or selected_endgame_loss_rate >= 0.35 else 3
    guess_imitation_weight = round(max(1.2, min(2.2, 1.4 + risk_gap * 4.5)), 2)
    risk_supervision_coef = round(max(0.06, min(0.14, 0.08 + risk_gap * 0.35)), 3)
    mine_aux_coef = 0.06 if wrong_flag_rate >= 0.2 else 0.04
    solver_imitation_coef = 0.12 if terminal_forced_rate >= 0.25 else 0.08
    risk_head_coef = 0.03 if risk_gap >= 0.04 else 0.02
    counterfactual_coef = round(max(0.08, min(0.22, 0.10 + risk_gap * 0.9)), 3)
    counterfactual_policy_coef = round(max(0.05, min(0.16, 0.06 + risk_gap * 0.7)), 3)
    counterfactual_value_head_coef = round(max(0.04, min(0.14, 0.05 + risk_gap * 0.5)), 3)
    counterfactual_value_rank_coef = round(max(0.2, min(0.65, 0.25 + risk_gap * 2.0)), 3)
    counterfactual_risk_head_coef = round(max(0.02, min(0.08, 0.025 + risk_gap * 0.35)), 3)
    asset_summary = training_asset_summary or {}
    behavior_mine_rate = float(asset_summary.get("behavior_mine_rate") or 0.0)
    behavior_mine_records = int(asset_summary.get("known_behavior_mine_records") or 0)
    behavior_mine_labeled_records = int(asset_summary.get("behavior_mine_labeled_records") or 0)
    extreme_filters = derive_extreme_bootstrap_filters(
        sample_family_profile=sample_family_profile,
        decision_family_profile=decision_family_profile,
        training_asset_summary=training_asset_summary,
        selected_endgame_safe_left=selected_endgame_safe_left,
        selected_endgame_loss_rate=selected_endgame_loss_rate,
        risk_gap=risk_gap,
    )
    train_policy_only = behavior_mine_labeled_records > 0 and selected_endgame_loss_rate >= 0.5
    behavior_mine_demotion_coef = 0.0
    if behavior_mine_labeled_records > 0 or behavior_mine_records > 0:
        behavior_mine_demotion_coef = round(
            max(0.04, min(0.10, 0.04 + behavior_mine_rate * 0.14 + risk_gap * 0.12)),
            3,
        )
        if selected_endgame_loss_rate >= 0.5:
            behavior_mine_demotion_coef = round(min(0.12, behavior_mine_demotion_coef + 0.02), 3)
    guess_survival_coef = 0.0
    guess_survival_mine_weight = 4.0
    guess_survival_topk = 16
    guess_survival_margin = 0.08
    tail_states = 16 if selected_endgame_safe_left >= 80 or tail_samples > 0 or edge_samples > 0 or corner_samples > 0 or guess_decisions > 0 else 12
    replay_size = 8192 if train_policy_only else (12288 if target_gap >= 0.03 else 8192)
    mine_games = 256 if train_policy_only else (384 if target_gap >= 0.03 else 256)
    rounds = 3 if train_policy_only else (5 if target_gap >= 0.03 else 4)
    rl_updates = 16 if train_policy_only else (24 if target_gap >= 0.03 else 16)
    eval_games = 200 if train_policy_only else (500 if target_gap >= 0.03 else 300)
    guess_supervision_topk = 10 if selected_endgame_loss_rate >= 0.5 else 8
    return {
        "exact_limit": recommended_exact_limit,
        "rounds": rounds,
        "mine_games": mine_games,
        "mine_batch_size": 64,
        "tail_states": tail_states,
        "replay_size": replay_size,
        "batch_size": 256,
        "imitation_updates": 64 if train_policy_only else (96 if target_gap >= 0.03 else 80),
        "rl_updates": rl_updates,
        "eval_games": eval_games,
        "lr": 5e-5 if train_policy_only else (8e-5 if target_gap < 0.05 else 9e-5),
        "pretrain_imitation_coef": 1.0,
        "solver_imitation_coef": solver_imitation_coef,
        "counterfactual_coef": counterfactual_coef,
        "counterfactual_margin": 0.08 if risk_gap >= 0.04 else 0.1,
        "counterfactual_temperature": 0.07 if risk_gap >= 0.04 else 0.08,
        "counterfactual_policy_coef": counterfactual_policy_coef,
        "counterfactual_policy_temperature": 0.22 if selected_endgame_loss_rate >= 0.5 else 0.25,
        "counterfactual_policy_topk": 16,
        "counterfactual_policy_min_gap": 0.03 if risk_gap >= 0.04 else 0.05,
        "counterfactual_policy_full_action": True,
        "counterfactual_only_guess": True,
        "counterfactual_value_head_coef": counterfactual_value_head_coef,
        "counterfactual_value_head_clip": 4.0,
        "counterfactual_value_rank_coef": counterfactual_value_rank_coef,
        "counterfactual_labels": True,
        "counterfactual_topk": 64,
        "mine_aux_coef": mine_aux_coef,
        "risk_supervision_coef": risk_supervision_coef,
        "risk_head_coef": risk_head_coef,
        "counterfactual_risk_head_coef": counterfactual_risk_head_coef,
        "counterfactual_risk_temperature": 1.0,
        "counterfactual_risk_loss_mode": "listwise",
        "guess_survival_coef": guess_survival_coef,
        "guess_survival_mine_weight": guess_survival_mine_weight,
        "guess_survival_topk": guess_survival_topk,
        "guess_survival_margin": guess_survival_margin,
        "behavior_mine_demotion_coef": behavior_mine_demotion_coef,
        "behavior_mine_demotion_topk": 4,
        "behavior_mine_demotion_margin": 0.08 if train_policy_only else 0.1,
        "risk_temperature": 0.05 if risk_gap >= 0.05 else 0.06,
        "guess_supervision_topk": guess_supervision_topk,
        "guess_imitation_weight": guess_imitation_weight,
        "train_policy_only": train_policy_only,
        "reset_optimizer": True,
        "replay_log_dir": str(replay_log_dir),
        "replay_log_limit": int(replay_log_limit),
        "extreme_dataset": str(extreme_dataset),
        "endgame_safe_left": selected_endgame_safe_left,
        "endgame_weight": endgame_weight,
        "wrong_flag_weight": wrong_flag_weight,
        "terminal_weight": terminal_weight,
        "guess_weight": guess_weight,
        "edge_weight": edge_weight,
        "corner_weight": corner_weight,
        "guess_decisions": guess_decisions,
        "edge_guess_decisions": edge_guess_decisions,
        "corner_guess_decisions": corner_guess_decisions,
        "high_risk_guess_decisions": high_risk_guess_decisions,
        **extreme_filters,
        "include_plain_tail": False,
        "inference_flips": True,
        "inference_ensemble": "probs",
    }


def derive_extreme_bootstrap_filters(
    *,
    sample_family_profile: dict[str, int],
    decision_family_profile: dict[str, int],
    training_asset_summary: dict[str, Any] | None,
    selected_endgame_safe_left: int,
    selected_endgame_loss_rate: float,
    risk_gap: float,
) -> dict[str, Any]:
    asset_summary = training_asset_summary or {}
    family_counts = asset_summary.get("families")
    family_counts = family_counts if isinstance(family_counts, dict) else {}

    # Keep the bootstrap replay concentrated on the weakest late-game regions.
    # Plain guess states are only kept as a fallback when we do not already have
    # tail/edge/corner coverage from the sample profile.
    tail_like_families = [
        family
        for family in ("guess_tail", "edge_guess_tail", "corner_guess_tail")
        if int(family_counts.get(family, 0)) > 0
    ]
    edge_like_families = [
        family
        for family in ("edge_guess", "corner_guess")
        if int(family_counts.get(family, 0)) > 0
    ]
    guess_samples = int(sample_family_profile.get("guess_samples", 0))
    tail_samples = int(sample_family_profile.get("tail_samples", 0))
    edge_samples = int(sample_family_profile.get("edge_samples", 0))
    corner_samples = int(sample_family_profile.get("corner_samples", 0))
    guess_decisions = int(decision_family_profile.get("guess_decisions", 0))
    edge_guess_decisions = int(decision_family_profile.get("edge_guess_decisions", 0))
    corner_guess_decisions = int(decision_family_profile.get("corner_guess_decisions", 0))

    extreme_family_filter: list[str] = []
    if tail_samples > 0 or selected_endgame_loss_rate >= 0.35 or risk_gap >= 0.04:
        extreme_family_filter.extend(tail_like_families)
    if edge_samples > 0 or edge_guess_decisions > 0 or corner_guess_decisions > 0:
        extreme_family_filter.extend(edge_like_families)
    if not extreme_family_filter and guess_samples > 0 and guess_decisions > 0:
        extreme_family_filter = ["guess"]
    extreme_family_filter = list(dict.fromkeys(extreme_family_filter))

    if extreme_family_filter:
        extreme_safe_left_min: int | None = 1 if tail_samples > 0 or selected_endgame_loss_rate >= 0.35 else None
        extreme_safe_left_max: int | None = int(selected_endgame_safe_left)
    elif selected_endgame_loss_rate >= 0.35 or risk_gap >= 0.04:
        extreme_safe_left_min = 1
        extreme_safe_left_max = int(selected_endgame_safe_left)
    else:
        extreme_safe_left_min = None
        extreme_safe_left_max = None

    return {
        "extreme_family_filter": extreme_family_filter,
        "extreme_safe_left_min": extreme_safe_left_min,
        "extreme_safe_left_max": extreme_safe_left_max,
        "extreme_behavior_mine_only": False,
        "extreme_min_behavior_regret": None,
    }


def _build_hard_loss_refine_suffix(hard_loss_args: dict[str, Any]) -> str:
    command: list[str] = []
    for family in hard_loss_args.get("extreme_family_filter", []) or []:
        command.extend(["--extreme-family-filter", str(family)])
    for option_name in ("extreme_safe_left_min", "extreme_safe_left_max"):
        value = hard_loss_args.get(option_name)
        if value is not None:
            command.extend([f"--{option_name.replace('_', '-')}", str(value)])
    if hard_loss_args.get("extreme_behavior_mine_only"):
        command.append("--extreme-behavior-mine-only")
    min_behavior_regret = hard_loss_args.get("extreme_min_behavior_regret")
    if min_behavior_regret is not None:
        command.extend(["--extreme-min-behavior-regret", str(min_behavior_regret)])
    return (" " + " ".join(command)) if command else ""


def summarize_sample_family_profile(sample_candidates: list[dict[str, Any]]) -> dict[str, int]:
    counts = Counter(str(row.get("sample_family") or "") for row in sample_candidates)
    edge_samples = counts.get("endgame_edge_tail_replay", 0) + counts.get("edge_guess_ranking", 0)
    corner_samples = counts.get("endgame_corner_tail_replay", 0) + counts.get("corner_guess_ranking", 0)
    guess_samples = counts.get("guess_risk_ranking", 0) + counts.get("edge_guess_ranking", 0) + counts.get("corner_guess_ranking", 0)
    tail_samples = counts.get("endgame_tail_replay", 0) + counts.get("endgame_edge_tail_replay", 0) + counts.get("endgame_corner_tail_replay", 0)
    extreme_samples = edge_samples + corner_samples + guess_samples
    return {
        "sample_count": len(sample_candidates),
        "tail_samples": tail_samples,
        "edge_samples": edge_samples,
        "corner_samples": corner_samples,
        "guess_samples": guess_samples,
        "extreme_samples": extreme_samples,
        "policy_samples": counts.get("policy_loss_review", 0),
    }


def summarize_decision_family_profile(decision_samples: list[dict[str, Any]]) -> dict[str, int]:
    open_decisions = len(decision_samples)
    guess_decisions = 0
    edge_guess_decisions = 0
    corner_guess_decisions = 0
    high_risk_guess_decisions = 0
    forced_open_decisions = 0

    for row in decision_samples:
        forced = bool(row.get("solver_has_forced_moves"))
        row_index = row.get("row")
        col_index = row.get("col")
        target_risk = row.get("target_risk")
        if forced:
            forced_open_decisions += 1
        else:
            guess_decisions += 1
            if target_risk is not None and float(target_risk) >= 0.33:
                high_risk_guess_decisions += 1
            if is_edge_cell(row_index, col_index):
                edge_guess_decisions += 1
            if is_corner_cell(row_index, col_index):
                corner_guess_decisions += 1

    extreme_guess_decisions = edge_guess_decisions + corner_guess_decisions + high_risk_guess_decisions
    return {
        "open_decisions": open_decisions,
        "guess_decisions": guess_decisions,
        "forced_open_decisions": forced_open_decisions,
        "edge_guess_decisions": edge_guess_decisions,
        "corner_guess_decisions": corner_guess_decisions,
        "high_risk_guess_decisions": high_risk_guess_decisions,
        "extreme_guess_decisions": extreme_guess_decisions,
    }


def is_edge_cell(row: Any, col: Any) -> bool:
    try:
        row_index = int(row)
        col_index = int(col)
    except (TypeError, ValueError):
        return False
    return row_index in {0, BOARD_ROWS - 1} or col_index in {0, BOARD_COLS - 1}


def is_corner_cell(row: Any, col: Any) -> bool:
    try:
        row_index = int(row)
        col_index = int(col)
    except (TypeError, ValueError):
        return False
    return row_index in {0, BOARD_ROWS - 1} and col_index in {0, BOARD_COLS - 1}


def _select_training_checkpoint(
    checkpoint: Path | None,
    model_registry: list[dict[str, Any]],
) -> Path:
    if checkpoint is not None:
        return checkpoint
    if not model_registry:
        return DEFAULT_CHECKPOINT

    def priority(row: dict[str, Any]) -> tuple[float, int, float, str]:
        win_rate = float(row.get("best_observed_win_rate") or 0.0)
        checkpoint_name = str(row.get("checkpoint_name") or "")
        best_bias = 0 if "best" in checkpoint_name.lower() else 1
        avg_elapsed = float(row.get("best_observed_avg_elapsed_seconds") or 1e9)
        checkpoint_path = str(row.get("checkpoint_path") or checkpoint_name)
        return (-win_rate, best_bias, avg_elapsed, checkpoint_path.lower())

    selected = min(model_registry, key=priority)
    selected_path = selected.get("checkpoint_path") or selected.get("checkpoint_name")
    if not isinstance(selected_path, str) or not selected_path:
        return DEFAULT_CHECKPOINT
    return Path(selected_path)


def derive_training_focus(
    *,
    wrong_flag_rate: float,
    terminal_forced_rate: float,
    risk_gap: float,
    selected_endgame_safe_left: int,
    selected_endgame_loss_rate: float,
    sample_family_profile: dict[str, int],
    decision_family_profile: dict[str, int],
    training_asset_summary: dict[str, Any] | None = None,
    execution: dict[str, Any] | None,
    execution_quality_passed: bool | None,
) -> list[dict[str, Any]]:
    focus: list[dict[str, Any]] = []
    edge_samples = int(sample_family_profile.get("edge_samples", 0))
    corner_samples = int(sample_family_profile.get("corner_samples", 0))
    guess_samples = int(sample_family_profile.get("guess_samples", 0))
    guess_decisions = int(decision_family_profile.get("guess_decisions", 0))
    edge_guess_decisions = int(decision_family_profile.get("edge_guess_decisions", 0))
    corner_guess_decisions = int(decision_family_profile.get("corner_guess_decisions", 0))
    high_risk_guess_decisions = int(decision_family_profile.get("high_risk_guess_decisions", 0))
    asset_summary = training_asset_summary or {}
    behavior_mine_records = int(asset_summary.get("known_behavior_mine_records") or 0)
    behavior_mine_rate = float(asset_summary.get("behavior_mine_rate") or 0.0)
    if behavior_mine_records > 0:
        focus.append(
            {
                "name": "behavior_mine_demotion",
                "priority": "medium",
                "reason": "已有带真踩雷标签的 guess 资产，可直接补 policy head 的生存信号，专盯“该开哪格才活得久”",
                "reason": "Use labelled behavior-mine counterfactuals to demote the exact guess cell that caused a loss, without globally reshaping all guesses.",
                "evidence": f"known_behavior_mine_records={behavior_mine_records}, behavior_mine_rate={behavior_mine_rate:.2%}",
            }
        )
    if behavior_mine_records > 0:
        focus.append(
            {
                "name": "guess_row_solver_isolation",
                "priority": "medium",
                "reason": "guess rows should be supervised by counterfactual survival instead of solver imitation",
                "evidence": f"known_behavior_mine_records={behavior_mine_records}, guess_decisions={guess_decisions}",
            }
        )
    if selected_endgame_loss_rate >= 0.5:
        focus.append(
            {
                "name": "endgame_tail_replay",
                "priority": "high",
                "reason": "残局区间输局占比高，应提升尾部状态采样和 endgame loss 权重",
                "evidence": f"safe_left <= {selected_endgame_safe_left}, loss_rate={selected_endgame_loss_rate:.2%}",
            }
        )
    if edge_samples > 0:
        focus.append(
            {
                "name": "edge_tail_replay",
                "priority": "medium",
                "reason": "边缘终局样本已经出现，应把边界棋盘的猜测和点开顺序单独强化。",
                "evidence": f"edge_samples={edge_samples}",
            }
        )
    if corner_samples > 0:
        focus.append(
            {
                "name": "corner_tail_replay",
                "priority": "high",
                "reason": "角落终局更容易触发误判，应单独提权。",
                "evidence": f"corner_samples={corner_samples}",
            }
        )
    if wrong_flag_rate >= 0.1:
        focus.append(
            {
                "name": "wrong_flag_suppression",
                "priority": "high",
                "reason": "错旗会扭曲剩余雷估计，优先做反向监督和更高的错旗惩罚",
                "evidence": f"wrong_flag_loss_rate={wrong_flag_rate:.2%}",
            }
        )
    if terminal_forced_rate >= 0.2:
        focus.append(
            {
                "name": "terminal_forced_move_imitation",
                "priority": "high",
                "reason": "终局前仍有大量 solver 可见强制安全步，应提高 solver imitation 和 exact-limit 训练权重",
                "evidence": f"terminal_forced_rate={terminal_forced_rate:.2%}",
            }
        )
    if guess_samples > 0 or guess_decisions > 0 or risk_gap >= 0.04:
        focus.append(
            {
                "name": "guess_risk_ranking",
                "priority": "medium",
                "reason": "目标格风险高于最佳猜测，应强化 guess 排序和 risk head 监督",
                "evidence": f"avg_risk_gap_to_best_guess={risk_gap:.3f}, guess_samples={guess_samples}",
            }
        )
    if edge_guess_decisions > 0:
        focus.append(
            {
                "name": "edge_guess_ranking",
                "priority": "medium",
                "reason": "边缘纯猜的排序更脆弱，应单独强化。",
                "evidence": f"edge_guess_decisions={edge_guess_decisions}",
            }
        )
    if corner_guess_decisions > 0:
        focus.append(
            {
                "name": "corner_guess_ranking",
                "priority": "high",
                "reason": "角落纯猜最容易误判，应单独提权。",
                "evidence": f"corner_guess_decisions={corner_guess_decisions}",
            }
        )
    if high_risk_guess_decisions > 0:
        focus.append(
            {
                "name": "high_risk_guess_ranking",
                "priority": "medium",
                "reason": "高风险纯猜需要更强的风险排序监督。",
                "evidence": f"high_risk_guess_decisions={high_risk_guess_decisions}",
            }
        )
    if execution_quality_passed is not True and execution and int(execution.get("total_execution_anomalies") or 0) > 0:
        focus.append(
            {
                "name": "execution_quality_review",
                "priority": "medium",
                "reason": "执行层仍有异常，需要先清理读盘/点击噪音再看模型训练收益",
                "evidence": execution,
            }
        )
    if not focus:
        focus.append(
            {
                "name": "baseline_curriculum_refresh",
                "priority": "medium",
                "reason": "当前反馈信号较弱，可先用稳定的硬样本回放和常规 RL 继续刷新基线。",
                "evidence": {},
            }
        )
    return focus


def summarize_training_dataset_assets(
    profile_rows: list[dict[str, Any]],
    record_rows: list[dict[str, Any]],
) -> dict[str, Any]:
    """Summarize persisted training assets separately from failure candidates."""

    def total_int(key: str) -> int:
        return int(sum(int(row.get(key) or 0) for row in profile_rows))

    total_records = total_int("record_count")
    total_labels = total_int("counterfactual_candidate_labels")
    total_negative = total_int("negative_counterfactual_candidates")
    weighted_regret = sum(
        float(row.get("avg_behavior_regret") or 0.0) * int(row.get("record_count") or 0)
        for row in profile_rows
    )
    family_counts = Counter(str(row.get("family") or "unclassified") for row in record_rows)
    region_counts = Counter(str(row.get("region") or "unclassified") for row in record_rows)
    safe_left_counts = Counter(str(row.get("safe_left_bucket") or "unknown") for row in record_rows)
    regret_bucket_counts = Counter(str(row.get("behavior_regret_bucket") or "unlabelled") for row in record_rows)
    behavior_mine_labeled_records = sum(1 for row in record_rows if row.get("behavior_mine") is not None)
    behavior_mine_records = sum(
        1
        for row in record_rows
        if row.get("behavior_mine") in (1, True, "1", "true", "True")
    )

    return {
        "profile_count": len(profile_rows),
        "total_records": total_records,
        "counterfactual_records": total_int("counterfactual_records"),
        "counterfactual_candidate_labels": total_labels,
        "negative_counterfactual_candidates": total_negative,
        "negative_counterfactual_rate": (
            total_negative / total_labels if total_labels > 0 else None
        ),
        "weighted_avg_behavior_regret": (
            weighted_regret / total_records if total_records > 0 else None
        ),
        "high_regret_records": total_int("high_regret_records"),
        "known_behavior_mine_records": (
            behavior_mine_records
            if behavior_mine_labeled_records > 0
            else total_int("known_behavior_mine_records")
        ),
        "behavior_mine_labeled_records": behavior_mine_labeled_records,
        "behavior_mine_rate": (
            behavior_mine_records / behavior_mine_labeled_records
            if behavior_mine_labeled_records > 0
            else None
        ),
        "families": dict(sorted(family_counts.items())),
        "regions": dict(sorted(region_counts.items())),
        "safe_left_buckets": dict(sorted(safe_left_counts.items())),
        "behavior_regret_buckets": dict(sorted(regret_bucket_counts.items())),
        "profile_names": [str(row.get("profile_name") or "") for row in profile_rows],
    }


def render_markdown(report: dict[str, Any]) -> str:
    baseline = report.get("baseline", {})
    signals = report.get("signals", {})
    commands = report.get("commands", {})
    focus = report.get("training_focus", [])
    args = report.get("hard_loss_refine_args", {})
    asset_summary = report.get("training_dataset_asset_summary", {})
    sample_profile = signals.get("sample_family_profile", {})
    decision_profile = signals.get("decision_family_profile", {})
    sample_profile_rows: list[dict[str, Any]] = []
    decision_profile_rows: list[dict[str, Any]] = []
    if isinstance(sample_profile, dict):
        sample_profile_rows = [
            {"name": "sample_count", "value": sample_profile.get("sample_count")},
            {"name": "tail_samples", "value": sample_profile.get("tail_samples")},
            {"name": "edge_samples", "value": sample_profile.get("edge_samples")},
            {"name": "corner_samples", "value": sample_profile.get("corner_samples")},
            {"name": "guess_samples", "value": sample_profile.get("guess_samples")},
            {"name": "extreme_samples", "value": sample_profile.get("extreme_samples")},
            {"name": "policy_samples", "value": sample_profile.get("policy_samples")},
        ]
    if isinstance(decision_profile, dict):
        decision_profile_rows = [
            {"name": "open_decisions", "value": decision_profile.get("open_decisions")},
            {"name": "forced_open_decisions", "value": decision_profile.get("forced_open_decisions")},
            {"name": "guess_decisions", "value": decision_profile.get("guess_decisions")},
            {"name": "edge_guess_decisions", "value": decision_profile.get("edge_guess_decisions")},
            {"name": "corner_guess_decisions", "value": decision_profile.get("corner_guess_decisions")},
            {"name": "high_risk_guess_decisions", "value": decision_profile.get("high_risk_guess_decisions")},
            {"name": "extreme_guess_decisions", "value": decision_profile.get("extreme_guess_decisions")},
        ]
    lines = [
        "# 数据反哺训练策略计划",
        "",
        f"数据库：`{report['database']}`",
        f"生成时间：`{report['generated_at']}`",
        "",
        "## 训练目标",
        "",
        f"- 目标胜率：`{fmt_pct(report['target']['win_rate'])}`",
        f"- 目标单局耗时：`{fmt_float(report['target']['avg_seconds'])}` 秒",
        f"- 当前基线胜率：`{fmt_pct(baseline.get('win_rate_completed'))}`",
        f"- 当前基线耗时：`{fmt_float(baseline.get('avg_elapsed_seconds'))}` 秒",
        "",
        "## 数据信号",
        "",
        table(
            [
                {
                    "signal": "错旗输局率",
                    "value": fmt_pct(signals.get("wrong_flag_loss_rate")),
                    "meaning": "错旗会污染剩余雷估计，直接影响后续动作质量。",
                },
                {
                    "signal": "终局强制步可见率",
                    "value": fmt_pct(signals.get("terminal_forced_rate")),
                    "meaning": "终局前仍能看到 solver 强制安全步，说明末盘监督仍有空间。",
                },
                {
                    "signal": "风险 gap",
                    "value": fmt_float(signals.get("avg_risk_gap_to_best_guess")),
                    "meaning": "目标格风险高于最佳猜测，适合拉高 guess 排序监督。",
                },
                {
                    "signal": "推荐残局阈值",
                    "value": f"safe_left <= {signals.get('selected_endgame_safe_left')}",
                    "meaning": "用数据里损失最重的残局区间做加权回放。",
                },
                {
                    "signal": "推荐 exact-limit",
                    "value": str(signals.get("recommended_exact_limit")),
                    "meaning": "exact solver 需要放到更容易产生强监督的深度。",
                },
            ],
            ["signal", "value", "meaning"],
        ),
        "",
        "## Extreme Sample Mix",
        "",
        table(sample_profile_rows, ["name", "value"]),
        "",
        "## Training Asset Inventory",
        "",
        table(
            [
                {"name": "profile_count", "value": asset_summary.get("profile_count")},
                {"name": "total_records", "value": asset_summary.get("total_records")},
                {"name": "counterfactual_records", "value": asset_summary.get("counterfactual_records")},
                {
                    "name": "counterfactual_candidate_labels",
                    "value": asset_summary.get("counterfactual_candidate_labels"),
                },
                {
                    "name": "negative_counterfactual_rate",
                    "value": asset_summary.get("negative_counterfactual_rate"),
                },
                {
                    "name": "weighted_avg_behavior_regret",
                    "value": asset_summary.get("weighted_avg_behavior_regret"),
                },
                {"name": "high_regret_records", "value": asset_summary.get("high_regret_records")},
                {
                    "name": "known_behavior_mine_records",
                    "value": asset_summary.get("known_behavior_mine_records"),
                },
                {
                    "name": "behavior_mine_rate",
                    "value": asset_summary.get("behavior_mine_rate"),
                },
                {"name": "families", "value": asset_summary.get("families")},
                {"name": "regions", "value": asset_summary.get("regions")},
            ],
            ["name", "value"],
        ),
        "",
        "## Extreme Decision Mix",
        "",
        table(decision_profile_rows, ["name", "value"]),
        "",
        "## 训练重点",
        "",
        table(focus, ["name", "priority", "reason", "evidence"]),
        "",
        "## 建议参数",
        "",
        table(
            [{"param": key, "value": format_value(value)} for key, value in args.items()],
            ["param", "value"],
        ),
        "",
        "## 运行命令",
        "",
        "```powershell",
        commands["hard_loss_refine"],
        "```",
        "",
        "## 生成命令",
        "",
        "```powershell",
        commands["feedback_plan"],
        "```",
        "",
    ]
    return "\n".join(lines)


def table(rows_: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows_:
        return "_无数据_"
    lines = [
        "| " + " | ".join(columns) + " |",
        "| " + " | ".join("---" for _ in columns) + " |",
    ]
    for row in rows_:
        lines.append("| " + " | ".join(format_value(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def format_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        if 0.0 <= value <= 1.0:
            return f"{value * 100:.2f}%"
        return f"{value:.3f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value)


def fmt_pct(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value) * 100:.2f}%"


def fmt_float(value: Any) -> str:
    if value is None:
        return ""
    return f"{float(value):.2f}"


def first_row(conn: sqlite3.Connection, sql: str) -> sqlite3.Row | None:
    row = conn.execute(sql).fetchone()
    return row


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


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


if __name__ == "__main__":
    main()
