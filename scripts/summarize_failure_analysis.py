from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


DEFAULT_LOSS_ANALYSIS = Path("artifacts/analysis_loss_solver_compare_500_batched.json")
DEFAULT_EXACT_ANALYSIS = Path("artifacts/analysis_loss_solver_compare_200.json")
DEFAULT_HARD_REFINE = [
    Path("artifacts/full_rlmix_20_hard_refine_a.json"),
    Path("artifacts/full_rlmix_20_hard_refine_b.json"),
]
DEFAULT_OUTPUT_JSON = Path("artifacts/report_assets/failure_analysis_summary.json")
DEFAULT_OUTPUT_MD = Path("FAILURE_ANALYSIS.md")

RISK_BUCKET_ORDER = ["<0.20", "0.20-0.33", "0.33-0.50", "0.50-0.75", "0.75-1", "1.00"]


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize loss/solver comparison artifacts for the project report.")
    parser.add_argument("--loss-analysis", type=Path, default=DEFAULT_LOSS_ANALYSIS)
    parser.add_argument("--exact-analysis", type=Path, default=DEFAULT_EXACT_ANALYSIS)
    parser.add_argument("--hard-refine", type=Path, nargs="*", default=DEFAULT_HARD_REFINE)
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT_JSON)
    parser.add_argument("--output-md", type=Path, default=DEFAULT_OUTPUT_MD)
    args = parser.parse_args()

    summary = build_failure_summary(
        loss_analysis=args.loss_analysis,
        exact_analysis=args.exact_analysis,
        hard_refine_paths=args.hard_refine,
    )
    write_outputs(summary=summary, output_json=args.output_json, output_md=args.output_md)
    print(
        json.dumps(
            {
                "output_json": str(args.output_json),
                "output_md": str(args.output_md),
                "losses": summary["loss_analysis"].get("losses"),
                "wrong_flag_loss_rate": summary["loss_analysis"].get("wrong_flag_loss_rate"),
                "terminal_forced_rate": summary["loss_analysis"].get("terminal_forced_rate"),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


def build_failure_summary(
    loss_analysis: Path,
    exact_analysis: Path,
    hard_refine_paths: list[Path],
) -> dict[str, Any]:
    loss_data = load_json(loss_analysis)
    exact_data = load_json(exact_analysis)
    loss_summary = loss_data.get("summary", {})
    losses = int(loss_summary.get("losses", 0))
    wrong_flag_counts = loss_summary.get("loss_wrong_flag_counts", {})
    has_wrong_flags = int(wrong_flag_counts.get("has_wrong_flags", 0))
    terminal_forced = int(loss_summary.get("terminal_solver_forced_available", 0))
    target_known_mine = int(loss_summary.get("terminal_target_known_mine", 0))
    target_known_safe = int(loss_summary.get("terminal_target_known_safe_visible_solver", 0))
    avg_target_risk = none_or_float(loss_summary.get("avg_target_risk"))
    avg_best_guess_risk = none_or_float(loss_summary.get("avg_best_guess_risk"))

    return {
        "sources": {
            "loss_analysis": str(loss_analysis),
            "exact_analysis": str(exact_analysis),
            "hard_refine": [str(path) for path in hard_refine_paths],
        },
        "loss_analysis": {
            "games": int(loss_summary.get("games", 0)),
            "wins": int(loss_summary.get("wins", 0)),
            "losses": losses,
            "win_rate": none_or_float(loss_summary.get("win_rate")),
            "wrong_flag_losses": has_wrong_flags,
            "wrong_flag_loss_rate": ratio(has_wrong_flags, losses),
            "terminal_forced_available": terminal_forced,
            "terminal_forced_rate": ratio(terminal_forced, losses),
            "terminal_target_known_mine": target_known_mine,
            "terminal_target_known_mine_rate": ratio(target_known_mine, losses),
            "terminal_target_known_safe_visible_solver": target_known_safe,
            "terminal_target_known_safe_rate": ratio(target_known_safe, losses),
            "avg_wrong_flags_on_loss": none_or_float(loss_summary.get("avg_wrong_flags_on_loss")),
            "avg_flags_on_loss": none_or_float(loss_summary.get("avg_flags_on_loss")),
            "avg_target_risk": avg_target_risk,
            "avg_best_guess_risk": avg_best_guess_risk,
            "avg_risk_gap_to_best_guess": None
            if avg_target_risk is None or avg_best_guess_risk is None
            else avg_target_risk - avg_best_guess_risk,
            "risk_buckets": ordered_counts(loss_summary.get("risk_buckets", {}), RISK_BUCKET_ORDER),
            "endgame": summarize_endgame(loss_summary.get("endgame", {}), losses),
        },
        "exact_limit_compare": summarize_exact_compare(exact_data.get("aggregate", {})),
        "examples": summarize_examples(loss_data.get("examples", [])),
        "hard_refine": summarize_hard_refine(hard_refine_paths),
        "interpretation": [
            "Wrong flags are a distinct failure mode because they distort the global remaining-mine estimate.",
            "A substantial part of terminal losses still has solver-visible forced moves, especially in endgame states.",
            "The model's selected terminal targets have higher average risk than the solver's best visible guess.",
            "Small hard-state refinement did not produce a stable improvement, so future refinement should preserve the original state distribution.",
        ],
    }


def write_outputs(summary: dict[str, Any], output_json: Path, output_md: Path) -> None:
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    output_md.write_text(render_failure_markdown(summary), encoding="utf-8")


def render_failure_markdown(summary: dict[str, Any]) -> str:
    loss = summary["loss_analysis"]
    exact = summary["exact_limit_compare"]
    hard_refine = summary["hard_refine"]
    lines = [
        "# 失败分析附录",
        "",
        "本附录汇总 `full_rlmix_20.pt` 在内部仿真输局上的 solver 对照分析，用于解释当前模型的主要瓶颈和后续优化方向。",
        "",
        "## 数据来源",
        "",
        f"- Batched 输局分析：`{summary['sources']['loss_analysis']}`",
        f"- exact-limit 对照：`{summary['sources']['exact_analysis']}`",
        f"- hard-loss refine 记录：{', '.join(f'`{path}`' for path in summary['sources']['hard_refine'])}",
        "",
        "## 核心结论",
        "",
        "| 指标 | 数值 |",
        "| --- | ---: |",
        f"| 评估局数 | {loss['games']} |",
        f"| 胜率 | {format_percent(loss['win_rate'])} |",
        f"| 输局数 | {loss['losses']} |",
        f"| 输局中存在错旗 | {loss['wrong_flag_losses']} ({format_percent(loss['wrong_flag_loss_rate'])}) |",
        f"| 终局仍有 solver forced move | {loss['terminal_forced_available']} ({format_percent(loss['terminal_forced_rate'])}) |",
        f"| 终局目标被 solver 判定为已知雷 | {loss['terminal_target_known_mine']} ({format_percent(loss['terminal_target_known_mine_rate'])}) |",
        f"| 终局目标被 solver 判定为已知安全 | {loss['terminal_target_known_safe_visible_solver']} ({format_percent(loss['terminal_target_known_safe_rate'])}) |",
        f"| 平均终局目标风险 | {format_float(loss['avg_target_risk'])} |",
        f"| solver 最优可见猜测平均风险 | {format_float(loss['avg_best_guess_risk'])} |",
        f"| 平均风险差距 | {format_float(loss['avg_risk_gap_to_best_guess'])} |",
        "",
        "解释：模型并非只输在纯随机猜测；不少输局发生在 solver 仍能从可见约束推出 forced move，或模型在残局/错旗后对剩余雷数和风险排序处理不稳。",
        "",
        "## 终局风险分布",
        "",
        "| 风险区间 | 输局数 |",
        "| --- | ---: |",
    ]
    for bucket in loss["risk_buckets"]:
        lines.append(f"| {bucket['bucket']} | {bucket['count']} |")

    lines.extend(
        [
            "",
            "## 残局切片",
            "",
            "| 条件 | 数量 | 错旗 | forced available | known mine | target risk | best guess risk |",
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for key, row in loss["endgame"].items():
        lines.append(
            "| "
            + " | ".join(
                [
                    key.replace("_", " "),
                    str(row.get("count", "-")),
                    f"{row.get('has_wrong_flags', '-')}",
                    f"{row.get('forced_available', '-')}",
                    f"{row.get('target_known_mine', '-')}",
                    format_float(row.get("avg_target_risk")),
                    format_float(row.get("avg_best_guess_risk")),
                ]
            )
            + " |"
        )

    if exact:
        lines.extend(
            [
                "",
                "## exact-limit 对照",
                "",
                "| exact limit | forced available | known mine | target risk | best guess risk |",
                "| ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in exact.get("limits", []):
            lines.append(
                f"| {row['limit']} | {row['forced_available']} | {row['target_known_mine']} | "
                f"{format_float(row['avg_target_risk'])} | {format_float(row['avg_best_guess_risk'])} |"
            )

    lines.extend(
        [
            "",
            "## 典型输局样本",
            "",
            "| seed | safe left | wrong flags | forced safe | target risk | best guess risk | solver known safe |",
            "| ---: | ---: | ---: | ---: | ---: | ---: | --- |",
        ]
    )
    for example in summary["examples"]:
        lines.append(
            f"| {example['seed']} | {example['safe_left']} | {example['wrong_flags']} | "
            f"{example['forced_safe']} | {format_float(example['target_risk'])} | "
            f"{format_float(example['best_guess_risk'])} | {example['target_known_safe']} |"
        )

    if hard_refine:
        lines.extend(
            [
                "",
                "## hard-loss refine 结果",
                "",
                "| 文件 | round | games | win rate | longest streak | replay size |",
                "| --- | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for row in hard_refine:
            lines.append(
                f"| `{row['path']}` | {row['hard_refine_round']} | {row['games']} | "
                f"{format_percent(row['win_rate'])} | {row['longest_streak']} | {row['hard_replay_size']} |"
            )
        lines.extend(
            [
                "",
                "这些记录显示，小规模 hard-state refine 没有形成稳定增益；更合理的下一步是把 hard-state replay 与原始成功/中盘分布混合，并加强错旗后的长期惩罚。",
            ]
        )

    lines.extend(
        [
            "",
            "## 后续优化方向",
            "",
            "1. 增强错旗惩罚与 flag 校准，减少剩余雷数估计被污染。",
            "2. 针对 `safe_left <= 40` 的残局做课程学习，引入更多全局雷数约束状态。",
            "3. 训练时保留 solver 风险排序信号，但最终推理仍只使用 RL policy/ensemble。",
            "4. hard-state refine 需要混合原始分布，避免只拟合输局尾部导致策略退化。",
            "",
            "重新生成：",
            "",
            "```powershell",
            "python scripts/summarize_failure_analysis.py",
            "```",
            "",
        ]
    )
    return "\n".join(lines)


def summarize_endgame(endgame: dict[str, Any], losses: int) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in sorted(endgame.items(), key=lambda item: endgame_sort_key(item[0])):
        count = int(value.get("count", 0))
        result[key] = {
            "count": count,
            "loss_rate": ratio(count, losses),
            "has_wrong_flags": int(value.get("has_wrong_flags", 0)),
            "forced_available": int(value.get("forced_available", 0)),
            "target_known_mine": int(value.get("target_known_mine", 0)),
            "avg_target_risk": none_or_float(value.get("avg_target_risk")),
            "avg_best_guess_risk": none_or_float(value.get("avg_best_guess_risk")),
        }
    return result


def endgame_sort_key(key: str) -> int:
    try:
        return int(key.rsplit("_", 1)[-1])
    except ValueError:
        return 10**9


def summarize_exact_compare(aggregate: dict[str, Any]) -> dict[str, Any]:
    limits = []
    for key in ("8", "24", "32"):
        row = aggregate.get(key)
        if not row:
            continue
        limits.append(
            {
                "limit": int(key),
                "terminal_states": int(row.get("terminal_states", 0)),
                "forced_available": int(row.get("forced_available", 0)),
                "target_known_mine": int(row.get("target_known_mine", 0)),
                "target_known_safe": int(row.get("target_known_safe", 0)),
                "avg_target_risk": none_or_float(row.get("avg_target_risk")),
                "avg_best_guess_risk": none_or_float(row.get("avg_best_guess_risk")),
            }
        )
    return {"limits": limits}


def summarize_examples(examples: list[dict[str, Any]], limit: int = 8) -> list[dict[str, Any]]:
    rows = []
    for example in examples[:limit]:
        rows.append(
            {
                "seed": int(example.get("seed", 0)),
                "steps_before_loss": int(example.get("steps_before_loss", 0)),
                "action_kind": example.get("action_kind"),
                "row": int(example.get("row", 0)),
                "col": int(example.get("col", 0)),
                "safe_left": int(example.get("safe_left", 0)),
                "wrong_flags": int(example.get("wrong_flags", 0)),
                "forced_safe": int(example.get("forced_safe", 0)),
                "forced_mines": int(example.get("forced_mines", 0)),
                "target_risk": none_or_float(example.get("target_risk")),
                "best_guess_risk": none_or_float(example.get("best_guess_risk")),
                "target_known_mine": bool(example.get("target_known_mine")),
                "target_known_safe": bool(example.get("target_known_safe")),
            }
        )
    return rows


def summarize_hard_refine(paths: list[Path]) -> list[dict[str, Any]]:
    rows = []
    for path in paths:
        data = load_json(path)
        metrics = data.get("metrics", {})
        if not metrics:
            continue
        rows.append(
            {
                "path": str(path),
                "games": int(float(metrics.get("games", 0))),
                "win_rate": none_or_float(metrics.get("win_rate")),
                "longest_streak": int(float(metrics.get("longest_streak", 0))),
                "hard_refine_round": int(metrics.get("hard_refine_round", 0)),
                "source_checkpoint": metrics.get("source_checkpoint"),
                "hard_replay_size": int(metrics.get("hard_replay_size", 0)),
            }
        )
    return rows


def ordered_counts(counts: dict[str, Any], order: list[str]) -> list[dict[str, Any]]:
    return [{"bucket": bucket, "count": int(counts.get(bucket, 0))} for bucket in order if bucket in counts]


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def ratio(numerator: int, denominator: int) -> float | None:
    if denominator <= 0:
        return None
    return numerator / denominator


def none_or_float(value: Any) -> float | None:
    if value is None:
        return None
    return float(value)


def format_percent(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value) * 100.0:.2f}%"


def format_float(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.3f}"


if __name__ == "__main__":
    main()
