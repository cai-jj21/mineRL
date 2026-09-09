from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import save_extreme_dataset
from minesweeper_rl.trainer import _flip_transition, load_checkpoint, transition_extreme_profile
from minesweeper_rl.windows_replay import load_windows_replay_transitions


DEFAULT_OUTPUT = Path("artifacts/report_assets/extreme_training_dataset/extreme_transitions.npz")
DEFAULT_RARE_FAMILY_AUGMENT_THRESHOLD = 128


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a stratified replay asset for endgame, edge, corner, and guess states."
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--replay-log-dir", type=Path, default=Path("artifacts/windows_agent"))
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--replay-limit", type=int, default=32768)
    parser.add_argument("--per-game-limit", type=int, default=256)
    parser.add_argument("--per-family-cap", type=int, default=2048)
    parser.add_argument("--rare-family-augment-threshold", type=int, default=DEFAULT_RARE_FAMILY_AUGMENT_THRESHOLD)
    parser.add_argument("--safe-left-threshold", type=int, default=100)
    parser.add_argument("--include-noisy-replay-paths", action="store_true")
    args = parser.parse_args()

    trainer = load_checkpoint(args.checkpoint, device=args.device)
    transitions, replay_report = load_windows_replay_transitions(
        trainer,
        args.replay_log_dir,
        limit=args.replay_limit,
        per_game_limit=args.per_game_limit,
        exclude_noisy_paths=not args.include_noisy_replay_paths,
    )

    ranked = []
    profile_counts: Counter[str] = Counter()
    for index, transition in enumerate(transitions):
        profile = transition_extreme_profile(
            transition,
            mines=trainer.config.mines,
            safe_left_threshold=args.safe_left_threshold,
        )
        if not bool(profile["extreme"]):
            continue
        transition.extreme_score = float(profile["score"])
        family = str(profile["family"])
        if family == "ordinary" and bool(profile.get("high_risk")):
            family = "high_risk"
        transition.extreme_family = family
        profile_counts[transition.extreme_family] += 1
        ranked.append((float(profile["score"]), index, transition))

    ranked.sort(key=lambda item: (-item[0], item[1]))
    selected: list[Any] = []
    selected_counts: Counter[str] = Counter()
    for _, _, transition in ranked:
        family = transition.extreme_family or "unclassified"
        if selected_counts[family] >= args.per_family_cap:
            continue
        selected.append(transition)
        selected_counts[family] += 1

    if not selected:
        raise RuntimeError("no extreme transitions were found in the replay corpus")

    augmented: list[Any] = []
    augmented_counts: Counter[str] = Counter()
    for transition in selected:
        family = transition.extreme_family or "unclassified"
        augmented.append(transition)
        augmented_counts[family] += 1
        if family != "tail" and selected_counts[family] <= args.rare_family_augment_threshold:
            for flip_vertical, flip_horizontal in ((True, False), (False, True), (True, True)):
                flipped = _flip_transition(
                    transition,
                    trainer.config.rows,
                    trainer.config.cols,
                    flip_vertical,
                    flip_horizontal,
                )
                flipped.extreme_score = transition.extreme_score
                flipped.extreme_family = family
                augmented.append(flipped)
                augmented_counts[family] += 1

    selected = augmented
    selected_counts = augmented_counts

    manifest = save_extreme_dataset(
        args.output,
        selected,
        metadata={
            "checkpoint": str(args.checkpoint),
            "replay_log_dir": str(args.replay_log_dir),
            "safe_left_threshold": args.safe_left_threshold,
            "per_family_cap": args.per_family_cap,
            "rare_family_augment_threshold": args.rare_family_augment_threshold,
            "replay_report": replay_report,
            "candidate_families": dict(sorted(profile_counts.items())),
            "selected_families": dict(sorted(selected_counts.items())),
        },
    )
    print(
        json.dumps(
            {
                "ok": True,
                "output": str(args.output),
                "manifest": manifest,
                "candidates": len(ranked),
                "selected": len(selected),
                "selected_families": dict(sorted(selected_counts.items())),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
