from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


DEFAULT_DATABASE = Path("artifacts/report_assets/minesweeper_experiments.sqlite")
DEFAULT_OUTPUT_DIR = Path("artifacts/report_assets/training_feedback_dataset")
DEFAULT_MANIFEST = DEFAULT_OUTPUT_DIR / "training_feedback_dataset.json"
DEFAULT_JSONL = DEFAULT_OUTPUT_DIR / "training_feedback_dataset.jsonl"
DEFAULT_MD = DEFAULT_OUTPUT_DIR / "training_feedback_dataset.md"


def configure_utf8_stdout() -> None:
    reconfigure = getattr(sys.stdout, "reconfigure", None)
    if reconfigure is not None:
        reconfigure(encoding="utf-8")


def main() -> None:
    configure_utf8_stdout()
    parser = argparse.ArgumentParser(description="Export a training feedback dataset from the experiment warehouse.")
    parser.add_argument("--database", type=Path, default=DEFAULT_DATABASE)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--jsonl", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--markdown", type=Path, default=DEFAULT_MD)
    parser.add_argument("--decision-limit", type=int, default=2000)
    parser.add_argument("--extreme-decision-limit", type=int, default=2000)
    args = parser.parse_args()

    report = export_training_feedback_dataset(
        database=args.database,
        output_dir=args.output_dir,
        manifest=args.manifest,
        jsonl_path=args.jsonl,
        markdown_path=args.markdown,
        decision_limit=args.decision_limit,
        extreme_decision_limit=args.extreme_decision_limit,
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))
    raise SystemExit(0 if report["ok"] else 1)


def export_training_feedback_dataset(
    *,
    database: Path = DEFAULT_DATABASE,
    output_dir: Path = DEFAULT_OUTPUT_DIR,
    manifest: Path = DEFAULT_MANIFEST,
    jsonl_path: Path = DEFAULT_JSONL,
    markdown_path: Path = DEFAULT_MD,
    decision_limit: int = 2000,
    extreme_decision_limit: int = 2000,
) -> dict[str, Any]:
    output_dir.mkdir(parents=True, exist_ok=True)
    if manifest == DEFAULT_MANIFEST and output_dir != DEFAULT_OUTPUT_DIR:
        manifest = output_dir / manifest.name
    if jsonl_path == DEFAULT_JSONL and output_dir != DEFAULT_OUTPUT_DIR:
        jsonl_path = output_dir / jsonl_path.name
    if markdown_path == DEFAULT_MD and output_dir != DEFAULT_OUTPUT_DIR:
        markdown_path = output_dir / markdown_path.name
    conn = sqlite3.connect(database)
    conn.row_factory = sqlite3.Row
    try:
        model_registry = rows(conn, "SELECT * FROM v_ads_model_registry ORDER BY checkpoint_exists DESC, model_id")
        experiment_registry = rows(conn, "SELECT * FROM v_ads_experiment_registry ORDER BY experiment_run_id")
        failure_summary = rows(conn, "SELECT * FROM v_ads_failure_attribution_summary ORDER BY games DESC, primary_failure_type")
        sample_candidates = rows(
            conn,
            """
            SELECT *
            FROM v_ads_training_sample_candidates
            ORDER BY
                CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                safe_cells_left ASC,
                game_index ASC
            """,
        )
        decision_events = rows(
            conn,
            """
            SELECT *
            FROM decision_event
            ORDER BY run_id, game_index, step
            LIMIT ?
            """,
                (decision_limit,),
        )
        extreme_decision_candidates = rows_if_exists(
            conn,
            "v_ads_extreme_decision_candidates",
            """
            SELECT *
            FROM v_ads_extreme_decision_candidates
            ORDER BY
                CASE priority WHEN 'high' THEN 0 WHEN 'medium' THEN 1 ELSE 2 END,
                safe_cells_left ASC,
                risk_gap_to_best_guess DESC,
                game_index ASC,
                step ASC
            LIMIT ?
            """,
            (extreme_decision_limit,),
        )
        training_asset_records = rows_if_exists(
            conn,
            "training_dataset_record",
            """
            SELECT *
            FROM training_dataset_record
            ORDER BY profile_id, record_index
            """,
        )
    finally:
        conn.close()

    records: list[dict[str, Any]] = []
    for row in sample_candidates:
        records.append(
            {
                "sample_id": f"failure:{row['run_id']}:{row['game_index']}",
                "sample_type": "failure_attribution",
                "run_id": row["run_id"],
                "game_index": row["game_index"],
                "priority": row["priority"],
                "family": row["sample_family"],
                "primary_failure_type": row["primary_failure_type"],
                "safe_cells_left": row["safe_cells_left"],
                "terminal_step": row["terminal_step"],
                "terminal_action_kind": row["terminal_action_kind"],
                "terminal_row": row["terminal_row"],
                "terminal_col": row["terminal_col"],
                "terminal_target_risk": row["terminal_target_risk"],
                "terminal_best_guess_risk": row["terminal_best_guess_risk"],
                "terminal_edge": row["terminal_edge"],
                "terminal_corner": row["terminal_corner"],
                "label": row["sample_family"],
                "evidence": {
                    "primary_failure_type": row["primary_failure_type"],
                    "safe_cells_left": row["safe_cells_left"],
                    "terminal_target_risk": row["terminal_target_risk"],
                    "terminal_best_guess_risk": row["terminal_best_guess_risk"],
                    "terminal_edge": row["terminal_edge"],
                    "terminal_corner": row["terminal_corner"],
                },
            }
        )

    for row in extreme_decision_candidates:
        records.append(
            {
                "sample_id": f"extreme_decision:{row['run_id']}:{row['game_index']}:{row['step']}",
                "sample_type": "extreme_decision_candidate",
                "run_id": row["run_id"],
                "game_index": row["game_index"],
                "step": row["step"],
                "decision_source": row["decision_source"],
                "action_kind": row["action_kind"],
                "row": row["row"],
                "col": row["col"],
                "safe_cells_left": row["safe_cells_left"],
                "target_risk": row["target_risk"],
                "best_guess_risk": row["best_guess_risk"],
                "risk_gap_to_best_guess": row["risk_gap_to_best_guess"],
                "edge_candidate": row["edge_candidate"],
                "corner_candidate": row["corner_candidate"],
                "priority": row["priority"],
                "family": row["sample_family"],
                "label": row["sample_family"],
                "evidence": {
                    "safe_cells_left": row["safe_cells_left"],
                    "target_risk": row["target_risk"],
                    "best_guess_risk": row["best_guess_risk"],
                    "risk_gap_to_best_guess": row["risk_gap_to_best_guess"],
                    "edge_candidate": row["edge_candidate"],
                    "corner_candidate": row["corner_candidate"],
                },
            }
        )

    for row in decision_events:
        records.append(
            {
                "sample_id": f"decision:{row['run_id']}:{row['game_index']}:{row['step']}",
                "sample_type": "decision_event",
                "run_id": row["run_id"],
                "game_index": row["game_index"],
                "step": row["step"],
                "decision_source": row["decision_source"],
                "action_kind": row["action_kind"],
                "row": row["row"],
                "col": row["col"],
                "selected_by_solver_assist": row["selected_by_solver_assist"],
                "queued_from_solver_batch": row["queued_from_solver_batch"],
                "queued_from_chord": row["queued_from_chord"],
                "solver_assist_decision": row["solver_assist_decision"],
                "solver_audit_available": row["solver_audit_available"],
                "solver_has_forced_moves": row["solver_has_forced_moves"],
                "solver_forced_safe_count": row["solver_forced_safe_count"],
                "solver_forced_mine_count": row["solver_forced_mine_count"],
                "target_known_safe": row["target_known_safe"],
                "target_known_mine": row["target_known_mine"],
                "target_risk": row["target_risk"],
                "best_guess_risk": row["best_guess_risk"],
                "revealed_delta": row["revealed_delta"],
                "target_revealed_after_open": row["target_revealed_after_open"],
                "click_issued": row["click_issued"],
                "click_method": row["click_method"],
                "label": row["decision_source"],
                "evidence": {
                    "decision_source": row["decision_source"],
                    "solver_assist_decision": row["solver_assist_decision"],
                    "target_risk": row["target_risk"],
                    "best_guess_risk": row["best_guess_risk"],
                },
            }
        )

    for row in training_asset_records:
        records.append(
            {
                "sample_id": f"training_asset:{row['profile_id']}:{row['record_index']}",
                "sample_type": "training_asset_record",
                "profile_id": row["profile_id"],
                "dataset": row["dataset"],
                "record_index": row["record_index"],
                "family": row["family"],
                "region": row["region"],
                "expert_is_guess": row["expert_is_guess"],
                "source_quality": row["source_quality"],
                "extreme_score": row["extreme_score"],
                "action_row": row["action_row"],
                "action_col": row["action_col"],
                "safe_left": row["safe_left"],
                "safe_left_bucket": row["safe_left_bucket"],
                "open_candidate_count": row["open_candidate_count"],
                "open_candidate_bucket": row["open_candidate_bucket"],
                "counterfactual_label_count": row["counterfactual_label_count"],
                "negative_counterfactual_candidates": row["negative_counterfactual_candidates"],
                "positive_counterfactual_candidates": row["positive_counterfactual_candidates"],
                "negative_counterfactual_rate": row["negative_counterfactual_rate"],
                "best_counterfactual_value": row["best_counterfactual_value"],
                "behavior_counterfactual_value": row["behavior_counterfactual_value"],
                "behavior_regret": row["behavior_regret"],
                "behavior_regret_bucket": row["behavior_regret_bucket"],
                "behavior_mine": row["behavior_mine"],
                "label": row["family"],
                "evidence": {
                    "dataset": row["dataset"],
                    "family": row["family"],
                    "region": row["region"],
                    "safe_left": row["safe_left"],
                    "behavior_regret": row["behavior_regret"],
                    "behavior_mine": row["behavior_mine"],
                },
            }
        )

    jsonl_path.write_text("\n".join(json.dumps(record, ensure_ascii=False) for record in records) + ("\n" if records else ""), encoding="utf-8")
    manifest_payload = {
        "ok": True,
        "database": str(database),
        "generated_at": utc_now(),
        "output_dir": str(output_dir),
        "jsonl": str(jsonl_path),
        "markdown": str(markdown_path),
        "counts": {
            "model_registry": len(model_registry),
            "experiment_registry": len(experiment_registry),
            "failure_summary_rows": len(failure_summary),
            "sample_candidates": len(sample_candidates),
            "extreme_decision_candidates": len(extreme_decision_candidates),
            "decision_events": len(decision_events),
            "training_asset_records": len(training_asset_records),
            "records": len(records),
        },
        "schema": {
            "sample_types": [
                "failure_attribution",
                "extreme_decision_candidate",
                "decision_event",
                "training_asset_record",
            ],
            "decision_limit": decision_limit,
            "extreme_decision_limit": extreme_decision_limit,
        },
        "top_failure_types": failure_summary[:10],
        "models": model_registry,
        "experiments": experiment_registry,
    }
    manifest.write_text(json.dumps(manifest_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(render_markdown(manifest_payload), encoding="utf-8")
    return manifest_payload


def render_markdown(report: dict[str, Any]) -> str:
    lines = [
        "# 训练反哺数据集",
        "",
        f"数据仓库：`{report['database']}`",
        f"输出：`{report['output_dir']}`",
        "",
        "## 记录概览",
        "",
        table(
            [
                {"name": "model_registry", "value": report["counts"]["model_registry"]},
                {"name": "experiment_registry", "value": report["counts"]["experiment_registry"]},
                {"name": "failure_summary_rows", "value": report["counts"]["failure_summary_rows"]},
                {"name": "sample_candidates", "value": report["counts"]["sample_candidates"]},
                {"name": "extreme_decision_candidates", "value": report["counts"]["extreme_decision_candidates"]},
                {"name": "decision_events", "value": report["counts"]["decision_events"]},
                {"name": "training_asset_records", "value": report["counts"]["training_asset_records"]},
                {"name": "records", "value": report["counts"]["records"]},
            ],
            ["name", "value"],
        ),
        "",
        "## 数据类型",
        "",
        table(
            [{"sample_type": sample_type} for sample_type in report["schema"]["sample_types"]],
            ["sample_type"],
        ),
        "",
        "## 主要失败类型",
        "",
        table(report["top_failure_types"], ["run_id", "primary_failure_type", "games", "model_issue_games", "execution_issue_games"]),
        "",
        "## 使用命令",
        "",
        "```powershell",
        f"python scripts/hard_loss_refine.py --checkpoint artifacts/full_rlmix_20.pt --feedback-plan artifacts/report_assets/training_feedback_plan.json",
        "```",
        "",
    ]
    return "\n".join(lines)


def table(rows_: list[dict[str, Any]], columns: list[str]) -> str:
    if not rows_:
        return "_无数据_"
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join("---" for _ in columns) + " |"]
    for row in rows_:
        lines.append("| " + " | ".join(format_cell(row.get(column)) for column in columns) + " |")
    return "\n".join(lines)


def format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}".rstrip("0").rstrip(".")
    if isinstance(value, (dict, list)):
        return f"`{json.dumps(value, ensure_ascii=False, sort_keys=True)}`"
    return str(value)


def rows(conn: sqlite3.Connection, sql: str, params: tuple[Any, ...] = ()) -> list[dict[str, Any]]:
    return [dict(row) for row in conn.execute(sql, params)]


def rows_if_exists(
    conn: sqlite3.Connection,
    name: str,
    sql: str,
    params: tuple[Any, ...] = (),
) -> list[dict[str, Any]]:
    if not table_exists(conn, name):
        return []
    return rows(conn, sql, params)


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
