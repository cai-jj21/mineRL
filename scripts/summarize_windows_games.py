from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path
from typing import Any


GAME_RE = re.compile(r"game_(\d+)\.json$")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize per-game Windows Minesweeper JSON logs.")
    parser.add_argument("input", type=Path, help="directory containing game_*.json files")
    parser.add_argument("--streak-start", type=int, default=None)
    parser.add_argument("--streak-end", type=int, default=None)
    parser.add_argument("--output", type=Path, default=None)
    parser.add_argument(
        "--streak-report-output",
        type=Path,
        default=None,
        help="optional Markdown review for the selected streak; defaults to the longest streak when no range is given",
    )
    args = parser.parse_args()

    records = load_game_records(args.input)
    summary = summarize_records(records)
    result: dict[str, Any] = {
        "input": str(args.input),
        **summary,
    }

    streak_start = args.streak_start
    streak_end = args.streak_end
    if args.streak_report_output is not None and streak_start is None and streak_end is None:
        streak_start = summary["longest_streak_start"]
        streak_end = summary["longest_streak_end"]

    if streak_start is not None or streak_end is not None:
        if streak_start is None or streak_end is None:
            raise SystemExit("--streak-start and --streak-end must be provided together")
        result["selected_range"] = build_selected_range(records, streak_start, streak_end)

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    if args.streak_report_output is not None:
        if "selected_range" not in result:
            raise SystemExit("no streak range is available for --streak-report-output")
        args.streak_report_output.parent.mkdir(parents=True, exist_ok=True)
        args.streak_report_output.write_text(render_streak_markdown(result), encoding="utf-8")
    print(json.dumps(result, ensure_ascii=False, indent=2))


def load_game_records(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_dir():
        raise NotADirectoryError(path)

    records: list[dict[str, Any]] = []
    for file_path in sorted(path.glob("game_*.json"), key=game_sort_key):
        match = GAME_RE.match(file_path.name)
        if match is None:
            continue
        data = json.loads(file_path.read_text(encoding="utf-8"))
        summary = data.get("summary", {})
        actions = list(data.get("actions", []))
        first_open_action = first_open(summary, actions)
        terminal = terminal_action(actions)
        records.append(
            {
                "game_index": int(match.group(1)),
                "file": file_path.name,
                "won": summary.get("won") is True,
                "lost": summary.get("lost") is True,
                "done": summary.get("done") is True,
                "agent_steps": int(summary.get("agent_steps") or 0),
                "physical_open_actions": int(summary.get("physical_open_actions") or 0),
                "click_issued_actions": int(summary.get("click_issued_actions") or 0),
                "flags": int(summary.get("flags") or 0),
                "revealed_safe_cells": int(summary.get("revealed_safe_cells") or 0),
                "elapsed_seconds": float(summary.get("elapsed_seconds") or 0.0),
                "actions_per_second": float(summary.get("actions_per_second") or 0.0),
                "terminal_dialog": summary.get("terminal_dialog"),
                "reclicks": int(summary.get("reclicks") or 0),
                "click_unready_actions": int(summary.get("click_unready_actions") or 0),
                "unconfirmed_open_actions": int(summary.get("unconfirmed_open_actions") or 0),
                "read_recoveries": int(summary.get("read_recoveries") or 0),
                "read_repairs": int(summary.get("read_repairs") or 0),
                "read_restores": int(summary.get("read_restores") or 0),
                "quick_number_read_actions": int(summary.get("quick_number_read_actions") or 0),
                "quick_number_read_fallbacks": int(summary.get("quick_number_read_fallbacks") or 0),
                "avg_click_seconds": float(summary.get("avg_click_seconds") or 0.0),
                "avg_after_read_seconds": float(summary.get("avg_after_read_seconds") or 0.0),
                "open_zero_progress_actions": int(summary.get("open_zero_progress_actions") or 0),
                "open_target_miss_with_progress": int(summary.get("open_target_miss_with_progress") or 0),
                "virtual_flag_actions": count_virtual_flags(actions),
                "first_open": first_open_action,
                "first_open_revealed_delta": first_open_delta(summary),
                "terminal_action": action_cell(terminal),
                "terminal_detected_at": terminal.get("terminal_detected_at") if terminal else None,
            }
        )
    return records


def summarize_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    completed = [record for record in records if record["won"] or record["lost"] or record["done"]]
    terminal = [record for record in records if record["won"] or record["lost"]]
    wins = [record for record in terminal if record["won"]]
    losses = [record for record in terminal if record["lost"]]
    incomplete = [record for record in records if not record["won"] and not record["lost"]]
    longest = longest_win_streak(records)

    return {
        "file_count": len(records),
        "completed_games": len(completed),
        "terminal_games": len(terminal),
        "wins": len(wins),
        "losses": len(losses),
        "incomplete_games": len(incomplete),
        "win_rate_completed": len(wins) / len(terminal) if terminal else 0.0,
        "longest_streak": longest["length"],
        "longest_streak_start": longest["start"],
        "longest_streak_end": longest["end"],
        "avg_elapsed_seconds": average(record["elapsed_seconds"] for record in terminal),
        "avg_actions_per_second": average(record["actions_per_second"] for record in terminal),
        "avg_agent_steps": average(record["agent_steps"] for record in terminal),
        "avg_physical_open_actions": average(record["physical_open_actions"] for record in terminal),
        "total_reclicks": sum(record["reclicks"] for record in terminal),
        "total_click_unready_actions": sum(record["click_unready_actions"] for record in terminal),
        "total_unconfirmed_open_actions": sum(record["unconfirmed_open_actions"] for record in terminal),
        "total_read_recoveries": sum(record["read_recoveries"] for record in terminal),
        "total_read_repairs": sum(record["read_repairs"] for record in terminal),
        "total_read_restores": sum(record["read_restores"] for record in terminal),
        "total_open_zero_progress_actions": sum(record["open_zero_progress_actions"] for record in terminal),
        "total_open_target_miss_with_progress": sum(record["open_target_miss_with_progress"] for record in terminal),
    }


def build_selected_range(records: list[dict[str, Any]], start: int, end: int) -> dict[str, Any]:
    selected = [record for record in records if start <= record["game_index"] <= end]
    return {
        "start": start,
        "end": end,
        **summarize_records(selected),
        "games": [selected_game_record(record) for record in selected],
    }


def selected_game_record(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "game_index": record["game_index"],
        "file": record["file"],
        "won": record["won"],
        "lost": record["lost"],
        "done": record["done"],
        "agent_steps": record["agent_steps"],
        "physical_open_actions": record["physical_open_actions"],
        "virtual_flag_actions": record["virtual_flag_actions"],
        "flags": record["flags"],
        "revealed_safe_cells": record["revealed_safe_cells"],
        "elapsed_seconds": record["elapsed_seconds"],
        "actions_per_second": record["actions_per_second"],
        "terminal_dialog": record["terminal_dialog"],
        "terminal_action": record["terminal_action"],
        "terminal_detected_at": record["terminal_detected_at"],
        "first_open": record["first_open"],
        "first_open_revealed_delta": record["first_open_revealed_delta"],
        "quick_number_read_actions": record["quick_number_read_actions"],
        "quick_number_read_fallbacks": record["quick_number_read_fallbacks"],
        "avg_click_seconds": record["avg_click_seconds"],
        "avg_after_read_seconds": record["avg_after_read_seconds"],
        "reclicks": record["reclicks"],
        "click_unready_actions": record["click_unready_actions"],
        "unconfirmed_open_actions": record["unconfirmed_open_actions"],
        "read_recoveries": record["read_recoveries"],
        "read_repairs": record["read_repairs"],
        "read_restores": record["read_restores"],
        "open_zero_progress_actions": record["open_zero_progress_actions"],
        "open_target_miss_with_progress": record["open_target_miss_with_progress"],
    }


def render_streak_markdown(result: dict[str, Any]) -> str:
    selected = result.get("selected_range", {})
    games = list(selected.get("games", []))
    lines = [
        "# Windows Ten-Win Streak Review",
        "",
        f"- Source: `{result.get('input', '-')}`",
        f"- Range: `game_{int_or_dash(selected.get('start'))}.json` to `game_{int_or_dash(selected.get('end'))}.json`",
        f"- Result: {selected.get('wins', 0)} wins / {selected.get('terminal_games', 0)} terminal games",
        f"- Average time: {format_seconds(selected.get('avg_elapsed_seconds'))}",
        f"- Average agent steps: {format_number(selected.get('avg_agent_steps'))}",
        f"- Average physical opens: {format_number(selected.get('avg_physical_open_actions'))}",
        f"- Execution anomalies: reclicks={selected.get('total_reclicks', 0)}, "
        f"unconfirmed_opens={selected.get('total_unconfirmed_open_actions', 0)}, "
        f"read_recoveries={selected.get('total_read_recoveries', 0)}, "
        f"target_misses={selected.get('total_open_target_miss_with_progress', 0)}",
        "",
        "## Recorded Fields",
        "",
        "- Per-game outcome, elapsed time, agent steps, physical open count, virtual flag count, final flag count, and terminal dialog.",
        "- Per-action click evidence: target row/col, screen coordinate, click method, foreground window, cursor cell match, and down/up delivery.",
        "- Read stability evidence: quick number reads, fallback reads, repairs, restores, and read recoveries.",
        "- Execution anomaly counters: reclicks, unconfirmed opens, zero-progress opens, and target-miss-with-progress events.",
        "",
        "## Game Table",
        "",
        "| Game | Result | First open | Delta | Steps | Opens | Virtual flags | Flags | Safe cells | Seconds | APS | Terminal open | Dialog | Anomalies |",
        "| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- | --- |",
    ]
    for game in games:
        anomalies = (
            int(game.get("reclicks", 0) or 0)
            + int(game.get("unconfirmed_open_actions", 0) or 0)
            + int(game.get("read_recoveries", 0) or 0)
            + int(game.get("open_target_miss_with_progress", 0) or 0)
            + int(game.get("open_zero_progress_actions", 0) or 0)
        )
        lines.append(
            "| "
            f"{game.get('game_index')} | "
            f"{result_label(game)} | "
            f"{format_cell(game.get('first_open'))} | "
            f"{format_int(game.get('first_open_revealed_delta'))} | "
            f"{game.get('agent_steps', 0)} | "
            f"{game.get('physical_open_actions', 0)} | "
            f"{game.get('virtual_flag_actions', 0)} | "
            f"{game.get('flags', 0)} | "
            f"{game.get('revealed_safe_cells', 0)} | "
            f"{float(game.get('elapsed_seconds', 0.0)):.2f} | "
            f"{float(game.get('actions_per_second', 0.0)):.2f} | "
            f"{format_cell(game.get('terminal_action'))} | "
            f"{dialog_label(game.get('terminal_dialog'))} | "
            f"{anomalies} |"
        )
    lines.append("")
    return "\n".join(lines)


def longest_win_streak(records: list[dict[str, Any]]) -> dict[str, int | None]:
    best_length = 0
    best_start: int | None = None
    best_end: int | None = None
    current_length = 0
    current_start: int | None = None

    for record in records:
        if record["won"]:
            if current_length == 0:
                current_start = record["game_index"]
            current_length += 1
            if current_length > best_length:
                best_length = current_length
                best_start = current_start
                best_end = record["game_index"]
        else:
            current_length = 0
            current_start = None

    return {"length": best_length, "start": best_start, "end": best_end}


def first_open(summary: dict[str, Any], actions: list[dict[str, Any]]) -> dict[str, int] | None:
    summary_first = summary.get("first_open_action", {})
    summary_action = summary_first.get("action", {}) if isinstance(summary_first, dict) else {}
    cell = action_cell({"action": summary_action})
    if cell is not None:
        return cell
    for action in actions:
        if action.get("virtual_only"):
            continue
        payload = action.get("action", {})
        if isinstance(payload, dict) and payload.get("kind") == "open":
            return action_cell(action)
    return None


def first_open_delta(summary: dict[str, Any]) -> int | None:
    summary_first = summary.get("first_open_action", {})
    if not isinstance(summary_first, dict):
        return None
    delta = summary_first.get("revealed_delta")
    return int(delta) if delta is not None else None


def terminal_action(actions: list[dict[str, Any]]) -> dict[str, Any] | None:
    for action in reversed(actions):
        if action.get("terminal_dialog"):
            return action
    return None


def count_virtual_flags(actions: list[dict[str, Any]]) -> int:
    total = 0
    for action in actions:
        payload = action.get("action", {})
        if isinstance(payload, dict) and payload.get("kind") == "flag" and action.get("virtual_only"):
            total += 1
    return total


def action_cell(action_record: dict[str, Any] | None) -> dict[str, int] | None:
    if not action_record:
        return None
    action = action_record.get("action", {})
    if not isinstance(action, dict):
        return None
    if action.get("row") is None or action.get("col") is None:
        return None
    return {
        "kind": str(action.get("kind") or ""),
        "row": int(action["row"]),
        "col": int(action["col"]),
    }


def game_sort_key(path: Path) -> int:
    match = GAME_RE.match(path.name)
    return int(match.group(1)) if match is not None else sys.maxsize


def average(values: Any) -> float:
    items = [float(value) for value in values]
    return sum(items) / len(items) if items else 0.0


def result_label(game: dict[str, Any]) -> str:
    if game.get("won"):
        return "win"
    if game.get("lost"):
        return "loss"
    return "incomplete"


def format_cell(cell: Any) -> str:
    if not isinstance(cell, dict):
        return "-"
    if cell.get("row") is None or cell.get("col") is None:
        return "-"
    return f"r{int(cell['row'])}c{int(cell['col'])}"


def int_or_dash(value: Any) -> str:
    if value is None:
        return "-"
    return f"{int(value):03d}"


def format_seconds(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.2f}s"


def format_number(value: Any) -> str:
    if value is None:
        return "-"
    return f"{float(value):.1f}"


def format_int(value: Any) -> str:
    if value is None:
        return "-"
    return str(int(value))


def dialog_label(value: Any) -> str:
    if value is None:
        return "-"
    text = str(value)
    if text == "\u6e38\u620f\u80dc\u5229":
        return "game_win"
    if text == "\u6e38\u620f\u5931\u8d25":
        return "game_loss"
    return text


if __name__ == "__main__":
    main()
