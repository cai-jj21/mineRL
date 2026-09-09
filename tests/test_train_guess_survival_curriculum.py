from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "train_guess_survival_curriculum.py"
SPEC = importlib.util.spec_from_file_location("train_guess_survival_curriculum_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
curriculum = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = curriculum
SPEC.loader.exec_module(curriculum)


def test_stage_args_keep_curriculum_policy_only_through_both_phases() -> None:
    base_args = {
        "guess_survival_coef": 0.12,
        "guess_survival_topk": 32,
        "guess_survival_margin": 0.05,
        "lr": 1e-4,
    }

    stage1 = curriculum._stage_args(base_args, stage="stage1")
    stage2 = curriculum._stage_args(base_args, stage="stage2")

    assert stage1["train_policy_only"] is True
    assert stage1["freeze_backbone"] is True
    assert stage1["freeze_long_range"] is True
    assert stage1["counterfactual_only_guess"] is True
    assert stage1["rounds"] == 2
    assert stage1["mine_games"] == 160
    assert stage1["imitation_updates"] == 96
    assert stage1["rl_updates"] == 12
    assert stage1["eval_games"] == 100
    assert stage1["guess_survival_coef"] >= 0.18
    assert stage1["guess_survival_topk"] == 16
    assert stage1["guess_survival_margin"] == 0.08
    assert stage1["lr"] == 5e-5

    assert stage2["train_policy_only"] is True
    assert stage2["freeze_backbone"] is True
    assert stage2["freeze_long_range"] is True
    assert stage2["counterfactual_only_guess"] is True
    assert stage2["rounds"] == 3
    assert stage2["mine_games"] == 256
    assert stage2["imitation_updates"] == 64
    assert stage2["rl_updates"] == 16
    assert stage2["eval_games"] == 200
    assert stage2["guess_survival_coef"] == 0.06
    assert stage2["guess_survival_topk"] == 16
    assert stage2["guess_survival_margin"] == 0.08
    assert stage2["lr"] == 5e-5


def test_stage_plan_writer_and_command_builder(tmp_path: Path) -> None:
    base_plan = {"hard_loss_refine_args": {"lr": 1e-4}}
    stage_args = curriculum._stage_args(base_plan["hard_loss_refine_args"], stage="stage1")
    plan_path = curriculum._write_stage_plan(base_plan, stage_args, tmp_path / "stage1_plan.json")

    loaded = json.loads(plan_path.read_text(encoding="utf-8"))
    assert loaded["hard_loss_refine_args"]["train_policy_only"] is True
    assert loaded["hard_loss_refine_args"]["counterfactual_only_guess"] is True

    command = curriculum._build_command(
        checkpoint=Path("artifacts/full_rlmix_100_best.pt"),
        plan_path=plan_path,
        save_path=tmp_path / "best.pt",
        save_last_path=tmp_path / "last.pt",
        output_dir=tmp_path / "stage1",
        device="cuda",
    )

    assert command[0] == sys.executable
    assert "hard_loss_refine.py" in command[1]
    assert "--feedback-plan" in command
    assert str(plan_path) in command
