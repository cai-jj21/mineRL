from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.gate_calibration import GateCalibrator

GATED_PATH = ROOT / "scripts" / "evaluate_gated_specialist.py"
GATED_SPEC = importlib.util.spec_from_file_location("evaluate_gated_specialist_for_scan", GATED_PATH)
if GATED_SPEC is None or GATED_SPEC.loader is None:
    raise RuntimeError(f"could not load {GATED_PATH}")
gated = importlib.util.module_from_spec(GATED_SPEC)
sys.modules[GATED_SPEC.name] = gated
GATED_SPEC.loader.exec_module(gated)


def main() -> None:
    parser = argparse.ArgumentParser(description="Scan pure-model gated specialist settings.")
    parser.add_argument("--base-checkpoint", type=Path, required=True)
    parser.add_argument("--base-ensemble-checkpoint", dest="base_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--base-model-weight", dest="base_model_weights", action="append", type=float, default=[])
    parser.add_argument("--specialist-checkpoint", type=Path, required=True)
    parser.add_argument("--specialist-ensemble-checkpoint", dest="specialist_ensemble_checkpoints", action="append", type=Path, default=[])
    parser.add_argument("--specialist-model-weight", dest="specialist_model_weights", action="append", type=float, default=[])
    parser.add_argument("--model-ensemble-reduction", choices=["mean", "geomean"], default="mean")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--games", type=int, default=100)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--max-steps", type=int, default=600)
    parser.add_argument("--decision-actions", choices=["open", "full"], default="full")
    parser.add_argument("--safe-left-threshold", action="append", type=int, required=True)
    parser.add_argument("--specialist-weight", action="append", type=float, required=True)
    parser.add_argument("--blend-scope", choices=["all", "open", "open-disagree-margin"], default="open-disagree-margin")
    parser.add_argument("--specialist-margin", action="append", type=float, default=[])
    parser.add_argument(
        "--specialist-region-gate",
        choices=[
            "any",
            "current-edge-or-corner",
            "specialist-edge-or-corner",
            "either-edge-or-corner",
            "current-interior",
            "specialist-interior",
            "both-interior",
        ],
        default="any",
    )
    parser.add_argument("--gate-calibrator", type=Path)
    parser.add_argument("--gate-threshold", action="append", type=float, default=[])
    parser.add_argument("--inference-flips", action="store_true", default=True)
    parser.add_argument("--no-inference-flips", dest="inference_flips", action="store_false")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--risk-head-weight", type=float, default=0.0)
    parser.add_argument("--base-risk-head-weight", type=float)
    parser.add_argument("--specialist-risk-head-weight", type=float)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base_paths = [args.base_checkpoint, *args.base_ensemble_checkpoints]
    specialist_paths = [args.specialist_checkpoint, *args.specialist_ensemble_checkpoints]
    base_weights = gated.normalize_ensemble_weights(
        args.base_model_weights,
        expected=len(base_paths),
        label="base-model-weight",
    )
    specialist_weights = gated.normalize_ensemble_weights(
        args.specialist_model_weights,
        expected=len(specialist_paths),
        label="specialist-model-weight",
    )
    base_risk = args.risk_head_weight if args.base_risk_head_weight is None else args.base_risk_head_weight
    specialist_risk = (
        args.risk_head_weight
        if args.specialist_risk_head_weight is None
        else args.specialist_risk_head_weight
    )
    eval_args = SimpleNamespace(
        device=args.device,
        max_steps=args.max_steps,
        decision_actions=args.decision_actions,
        inference_flips=args.inference_flips,
        inference_ensemble=args.inference_ensemble,
        risk_head_weight=args.risk_head_weight,
    )
    bases = [gated.load_eval_trainer(path, eval_args, risk_head_weight=base_risk) for path in base_paths]
    specialists = [
        gated.load_eval_trainer(path, eval_args, risk_head_weight=specialist_risk)
        for path in specialist_paths
    ]
    calibrator = GateCalibrator.load(args.gate_calibrator) if args.gate_calibrator else None
    margins = args.specialist_margin or [0.05]
    thresholds = args.gate_threshold or ([float(calibrator.threshold)] if calibrator is not None else [None])

    results = []
    started_at = time.time()
    for safe_left in args.safe_left_threshold:
        for specialist_weight in args.specialist_weight:
            for margin in margins:
                for threshold in thresholds:
                    if calibrator is not None and threshold is not None:
                        calibrator.threshold = float(np.clip(threshold, 0.0, 1.0))
                    metrics = gated.evaluate_gated(
                        bases=bases,
                        base_model_weights=base_weights,
                        specialists=specialists,
                        specialist_model_weights=specialist_weights,
                        model_ensemble_reduction=args.model_ensemble_reduction,
                        games=args.games,
                        seed=args.seed,
                        batch_size=args.batch_size,
                        max_steps=args.max_steps,
                        safe_left_threshold=safe_left,
                        specialist_weight=specialist_weight,
                        blend_scope=args.blend_scope,
                        specialist_margin=margin,
                        specialist_region_gate=args.specialist_region_gate,
                        gate_calibrator=calibrator,
                    )
                    item = {
                        "safe_left_threshold": int(safe_left),
                        "specialist_weight": float(specialist_weight),
                        "blend_scope": args.blend_scope,
                        "specialist_margin": float(margin),
                        "specialist_region_gate": args.specialist_region_gate,
                        "gate_threshold": None if calibrator is None else float(calibrator.threshold),
                        **metrics,
                    }
                    results.append(item)
                    print(json.dumps(item, ensure_ascii=False, separators=(",", ":")))

    payload = {
        "ok": True,
        "elapsed_seconds": time.time() - started_at,
        "base_checkpoints": [str(path) for path in base_paths],
        "base_model_weights": base_weights,
        "specialist_checkpoints": [str(path) for path in specialist_paths],
        "specialist_model_weights": specialist_weights,
        "model_ensemble_reduction": args.model_ensemble_reduction,
        "gate_calibrator": str(args.gate_calibrator) if args.gate_calibrator else None,
        "games": int(args.games),
        "seed": int(args.seed),
        "results": sorted(results, key=lambda item: (item["win_rate"], item["wins"]), reverse=True),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
