from __future__ import annotations

import argparse
import json
import subprocess
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]


def _load_plan(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    params = data.get("hard_loss_refine_args")
    if not isinstance(params, dict):
        raise TypeError(f"feedback plan missing hard_loss_refine_args mapping: {path}")
    return data


def _stage_args(base_args: dict[str, Any], *, stage: str) -> dict[str, Any]:
    args = deepcopy(base_args)
    if stage == "stage1":
        args.update(
            {
                "train_policy_only": True,
                "freeze_backbone": True,
                "freeze_long_range": True,
                "counterfactual_only_guess": True,
                "counterfactual_policy_full_action": True,
                "reset_optimizer": True,
                "rounds": 2,
                "mine_games": 160,
                "imitation_updates": 96,
                "rl_updates": 12,
                "eval_games": 100,
                "guess_survival_coef": round(min(0.24, max(float(args.get("guess_survival_coef", 0.0)), 0.18)), 3),
                "guess_survival_topk": min(int(args.get("guess_survival_topk", 16)), 16),
                "guess_survival_margin": max(float(args.get("guess_survival_margin", 0.08)), 0.08),
                "lr": min(float(args.get("lr", 5e-5)), 5e-5),
            }
        )
    elif stage == "stage2":
        args.update(
            {
                "train_policy_only": True,
                "freeze_backbone": True,
                "freeze_long_range": True,
                "counterfactual_only_guess": True,
                "counterfactual_policy_full_action": True,
                "reset_optimizer": True,
                "rounds": 3,
                "mine_games": 256,
                "imitation_updates": 64,
                "rl_updates": 16,
                "eval_games": 200,
                "guess_survival_coef": 0.06,
                "guess_survival_topk": 16,
                "guess_survival_margin": 0.08,
                "lr": min(float(args.get("lr", 5e-5)), 5e-5),
            }
        )
    else:
        raise ValueError(f"unknown stage {stage!r}")
    return args


def _write_stage_plan(base_plan: dict[str, Any], stage_args: dict[str, Any], output_path: Path) -> Path:
    stage_plan = dict(base_plan)
    stage_plan["hard_loss_refine_args"] = stage_args
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(stage_plan, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def _build_command(*, checkpoint: Path, plan_path: Path, save_path: Path, save_last_path: Path, output_dir: Path, device: str) -> list[str]:
    return [
        sys.executable,
        str(ROOT / "scripts" / "hard_loss_refine.py"),
        "--checkpoint",
        str(checkpoint),
        "--save-path",
        str(save_path),
        "--save-last-path",
        str(save_last_path),
        "--output-dir",
        str(output_dir),
        "--device",
        device,
        "--feedback-plan",
        str(plan_path),
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Run a two-stage guess-survival curriculum on top of hard_loss_refine.")
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--plan", type=Path, default=Path("artifacts/report_assets/training_feedback_plan.json"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/guess_survival_curriculum"))
    parser.add_argument("--save-path", type=Path, default=Path("artifacts/full_rlmix_guess_survival_curriculum_best.pt"))
    parser.add_argument("--save-last-path", type=Path, default=Path("artifacts/full_rlmix_guess_survival_curriculum_last.pt"))
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    base_plan = _load_plan(args.plan)
    base_args = base_plan.get("hard_loss_refine_args")
    if not isinstance(base_args, dict):
        raise TypeError(f"feedback plan missing hard_loss_refine_args mapping: {args.plan}")

    stage1_dir = args.output_dir / "stage1"
    stage2_dir = args.output_dir / "stage2"
    stage1_plan = _write_stage_plan(base_plan, _stage_args(base_args, stage="stage1"), args.output_dir / "stage1_plan.json")
    stage2_plan = _write_stage_plan(base_plan, _stage_args(base_args, stage="stage2"), args.output_dir / "stage2_plan.json")

    stage1_save = args.output_dir / "stage1_best.pt"
    stage1_last = args.output_dir / "stage1_last.pt"
    stage2_save = args.save_path
    stage2_last = args.save_last_path

    stage1_cmd = _build_command(
        checkpoint=args.checkpoint,
        plan_path=stage1_plan,
        save_path=stage1_save,
        save_last_path=stage1_last,
        output_dir=stage1_dir,
        device=args.device,
    )
    stage2_cmd = _build_command(
        checkpoint=stage1_save,
        plan_path=stage2_plan,
        save_path=stage2_save,
        save_last_path=stage2_last,
        output_dir=stage2_dir,
        device=args.device,
    )

    preview = {
        "stage1_plan": str(stage1_plan),
        "stage2_plan": str(stage2_plan),
        "stage1_command": stage1_cmd,
        "stage2_command": stage2_cmd,
    }
    print(json.dumps(preview, ensure_ascii=False, indent=2))

    if args.dry_run:
        return

    subprocess.run(stage1_cmd, check=True)
    subprocess.run(stage2_cmd, check=True)


if __name__ == "__main__":
    main()
