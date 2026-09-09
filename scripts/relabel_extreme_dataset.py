import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.extreme_replay import (
    load_extreme_dataset,
    relabel_counterfactual_values_risk_aware,
    save_extreme_dataset,
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Relabel extreme replay candidates with solver-risk-first offline targets."
    )
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--risk-weight", type=float, default=1.0)
    parser.add_argument("--outcome-weight", type=float, default=0.05)
    parser.add_argument("--mine-penalty", type=float, default=1.0)
    parser.add_argument(
        "--safe-left-max",
        type=int,
        default=None,
        help="Keep only states with at most this many safe cells left (classic board total is 391).",
    )
    args = parser.parse_args()

    transitions, input_manifest = load_extreme_dataset(args.input)
    if args.safe_left_max is not None:
        total_safe = 391
        transitions = [
            transition
            for transition in transitions
            if int(
                round(
                    (1.0 - float(np.clip(transition.global_features[1], 0.0, 1.0)))
                    * total_safe
                )
            )
            <= int(args.safe_left_max)
        ]
        if not transitions:
            raise RuntimeError("safe-left filter removed every transition")
    relabelled = relabel_counterfactual_values_risk_aware(
        transitions,
        risk_weight=args.risk_weight,
        outcome_weight=args.outcome_weight,
        mine_penalty=args.mine_penalty,
    )
    metadata = dict(input_manifest.get("metadata", input_manifest))
    metadata.update(
        {
            "source": "risk_aware_relabel",
            "input_dataset": str(args.input),
            "risk_weight": float(args.risk_weight),
            "outcome_weight": float(args.outcome_weight),
            "mine_penalty": float(args.mine_penalty),
            "safe_left_max": args.safe_left_max,
        }
    )
    manifest = save_extreme_dataset(args.output, relabelled, metadata=metadata)
    print(
        json.dumps(
            {
                "ok": True,
                "input": str(args.input),
                "output": str(args.output),
                "records": len(relabelled),
                "manifest": manifest,
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
