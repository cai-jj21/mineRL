from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "summarize_failure_analysis.py"
SPEC = importlib.util.spec_from_file_location("summarize_failure_analysis_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
failure_analysis = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = failure_analysis
SPEC.loader.exec_module(failure_analysis)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_build_failure_summary_extracts_loss_solver_metrics(tmp_path: Path) -> None:
    loss_path = tmp_path / "loss.json"
    exact_path = tmp_path / "exact.json"
    refine_path = tmp_path / "refine.json"
    write_json(
        loss_path,
        {
            "summary": {
                "games": 10,
                "wins": 4,
                "losses": 6,
                "win_rate": 0.4,
                "loss_wrong_flag_counts": {"clean_flags": 4, "has_wrong_flags": 2},
                "terminal_solver_forced_available": 3,
                "terminal_target_known_mine": 1,
                "terminal_target_known_safe_visible_solver": 2,
                "avg_target_risk": 0.3,
                "avg_best_guess_risk": 0.2,
                "risk_buckets": {"<0.20": 2, "0.50-0.75": 4},
                "endgame": {
                    "safe_left_le_100": {"count": 5, "has_wrong_flags": 2, "forced_available": 3},
                    "safe_left_le_20": {"count": 1, "target_known_mine": 1, "avg_target_risk": 0.8},
                },
            },
            "examples": [
                {
                    "seed": 7,
                    "steps_before_loss": 123,
                    "action_kind": "open",
                    "row": 1,
                    "col": 2,
                    "safe_left": 3,
                    "wrong_flags": 1,
                    "forced_safe": 2,
                    "forced_mines": 0,
                    "target_risk": 0.25,
                    "best_guess_risk": 0.1,
                    "target_known_safe": True,
                }
            ],
        },
    )
    write_json(
        exact_path,
        {"aggregate": {"8": {"terminal_states": 6, "forced_available": 2}, "32": {"terminal_states": 6, "target_known_mine": 1}}},
    )
    write_json(
        refine_path,
        {"metrics": {"games": 10, "win_rate": 0.4, "longest_streak": 2, "hard_refine_round": 1, "hard_replay_size": 12}},
    )

    summary = failure_analysis.build_failure_summary(loss_path, exact_path, [refine_path])

    assert summary["loss_analysis"]["wrong_flag_loss_rate"] == 2 / 6
    assert summary["loss_analysis"]["terminal_forced_rate"] == 3 / 6
    assert summary["loss_analysis"]["avg_risk_gap_to_best_guess"] == 0.09999999999999998
    assert list(summary["loss_analysis"]["endgame"]) == ["safe_left_le_20", "safe_left_le_100"]
    assert summary["exact_limit_compare"]["limits"][0]["limit"] == 8
    assert summary["examples"][0]["target_known_safe"] is True
    assert summary["hard_refine"][0]["hard_replay_size"] == 12


def test_write_outputs_emits_json_and_markdown(tmp_path: Path) -> None:
    summary = {
        "sources": {"loss_analysis": "loss.json", "exact_analysis": "exact.json", "hard_refine": []},
        "loss_analysis": {
            "games": 10,
            "wins": 4,
            "losses": 6,
            "win_rate": 0.4,
            "wrong_flag_losses": 2,
            "wrong_flag_loss_rate": 2 / 6,
            "terminal_forced_available": 3,
            "terminal_forced_rate": 0.5,
            "terminal_target_known_mine": 1,
            "terminal_target_known_mine_rate": 1 / 6,
            "terminal_target_known_safe_visible_solver": 2,
            "terminal_target_known_safe_rate": 2 / 6,
            "avg_target_risk": 0.3,
            "avg_best_guess_risk": 0.2,
            "avg_risk_gap_to_best_guess": 0.1,
            "risk_buckets": [{"bucket": "<0.20", "count": 2}],
            "endgame": {},
        },
        "exact_limit_compare": {"limits": []},
        "examples": [],
        "hard_refine": [],
        "interpretation": [],
    }
    output_json = tmp_path / "failure.json"
    output_md = tmp_path / "FAILURE_ANALYSIS.md"

    failure_analysis.write_outputs(summary, output_json, output_md)

    assert json.loads(output_json.read_text(encoding="utf-8"))["loss_analysis"]["games"] == 10
    markdown = output_md.read_text(encoding="utf-8")
    assert "失败分析附录" in markdown
    assert "40.00%" in markdown
    assert "重新生成" in markdown
