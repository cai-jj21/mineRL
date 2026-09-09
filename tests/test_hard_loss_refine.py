from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import torch

from minesweeper_rl.features import action_to_index
from minesweeper_rl.types import Action, ActionType, EpisodeTransition
from minesweeper_rl.trainer import MinesweeperTrainer, TrainingConfig


ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = ROOT / "scripts" / "hard_loss_refine.py"
SPEC = importlib.util.spec_from_file_location("hard_loss_refine_test", SCRIPT_PATH)
assert SPEC is not None and SPEC.loader is not None
hard_loss_refine = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = hard_loss_refine
SPEC.loader.exec_module(hard_loss_refine)


def make_transition(*, extreme_family: str = "") -> EpisodeTransition:
    rows, cols = 2, 2
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    return EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.5], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        extreme_family=extreme_family,
    )


def test_sample_transitions_keeps_extreme_states_in_batch() -> None:
    rng = np.random.default_rng(123)
    extreme = [make_transition(extreme_family="corner_tail") for _ in range(40)]
    ordinary = [make_transition() for _ in range(60)]
    batch = hard_loss_refine.sample_transitions(extreme + ordinary, 20, rng)

    assert len(batch) == 20
    assert sum(bool(transition.extreme_family) for transition in batch) == 15


def test_sample_transitions_family_balances_all_extreme_batches() -> None:
    rng = np.random.default_rng(123)
    corner = [make_transition(extreme_family="corner_guess_tail") for _ in range(5)]
    guess = [make_transition(extreme_family="guess_tail") for _ in range(200)]
    batch = hard_loss_refine.sample_transitions(corner + guess, 20, rng)

    assert len(batch) == 20
    assert sum(transition.extreme_family == "corner_guess_tail" for transition in batch) == 12
    assert sum(transition.extreme_family == "guess_tail" for transition in batch) == 8


def test_configure_trainer_can_freeze_backbone_and_reset_optimizer() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    args = type(
        "Args",
        (),
        {
            "rows": 16,
            "cols": 30,
            "mines": 99,
            "safe_radius": 1,
            "max_steps": 600,
            "exact_limit": 32,
            "lr": 1e-5,
            "pretrain_imitation_coef": 1.0,
            "solver_imitation_coef": 0.1,
            "mine_aux_coef": 0.0,
            "risk_supervision_coef": 0.1,
            "risk_head_coef": 0.0,
            "risk_temperature": 0.06,
            "guess_supervision_topk": 8,
            "guess_imitation_weight": 2.0,
            "inference_flips": True,
            "inference_ensemble": "probs",
            "freeze_backbone": True,
            "freeze_long_range": False,
            "reset_optimizer": True,
        },
    )()

    hard_loss_refine.configure_trainer(trainer, args)

    assert not any(param.requires_grad for param in trainer.model.backbone.parameters())
    assert any(param.requires_grad for param in trainer.model.policy_head.parameters())
    assert all(param.requires_grad is False for param in trainer.model.backbone.parameters())
    assert trainer.optimizer.param_groups
    assert all(param.requires_grad for group in trainer.optimizer.param_groups for param in group["params"])


def test_counterfactual_open_values_mark_mines_as_worst() -> None:
    game = hard_loss_refine.MinesweeperGame(rows=3, cols=3, mines=1, safe_radius=0, seed=7)
    game.open_cell(1, 1)
    _, _, mask = hard_loss_refine.encode_state(game)
    snapshot = hard_loss_refine.MinesweeperSolver(exact_limit=8).analyze(game)
    values = hard_loss_refine.build_counterfactual_open_values(
        game,
        mask,
        snapshot,
        hard_loss_refine.MinesweeperSolver(exact_limit=8),
        topk=64,
    )

    finite = np.isfinite(values)
    assert finite.any()
    assert np.nanmin(values) <= 0.25


def test_counterfactual_open_values_keep_behavior_candidate() -> None:
    game = hard_loss_refine.MinesweeperGame(rows=4, cols=4, mines=1, safe_radius=0, seed=8)
    game.open_cell(1, 1)
    _, _, mask = hard_loss_refine.encode_state(game)
    snapshot = hard_loss_refine.MinesweeperSolver(exact_limit=8).analyze(game)
    legal_indices = np.flatnonzero(mask[0].reshape(-1))
    assert legal_indices.size > 1
    requested = np.asarray([legal_indices[-1]], dtype=np.int64)
    values = hard_loss_refine.build_counterfactual_open_values(
        game,
        mask,
        snapshot,
        hard_loss_refine.MinesweeperSolver(exact_limit=8),
        topk=1,
        candidate_indices=requested,
    )

    assert np.isfinite(values.reshape(-1)[requested[0]])


def test_counterfactual_open_values_can_label_all_legal_open_candidates() -> None:
    game = hard_loss_refine.MinesweeperGame(rows=4, cols=4, mines=1, safe_radius=0, seed=8)
    game.open_cell(1, 1)
    _, _, mask = hard_loss_refine.encode_state(game)
    snapshot = hard_loss_refine.MinesweeperSolver(exact_limit=8).analyze(game)
    values = hard_loss_refine.build_counterfactual_open_values(
        game,
        mask,
        snapshot,
        hard_loss_refine.MinesweeperSolver(exact_limit=8),
        topk=1,
        include_all_open=True,
    )

    assert int(np.isfinite(values).sum()) == int(mask[0].sum())


def test_all_open_labels_keep_shortlist_candidates_expanded() -> None:
    game = hard_loss_refine.MinesweeperGame(rows=4, cols=4, mines=1, safe_radius=0, seed=8)
    game.open_cell(1, 1)
    _, _, mask = hard_loss_refine.encode_state(game)
    snapshot = hard_loss_refine.MinesweeperSolver(exact_limit=8).analyze(game)
    values = hard_loss_refine.build_counterfactual_open_values(
        game,
        mask,
        snapshot,
        hard_loss_refine.MinesweeperSolver(exact_limit=8),
        topk=1,
        include_all_open=True,
    )

    assert np.isfinite(values).sum() == mask[0].sum()
    assert np.isfinite(values).all() == bool(mask[0].all())


def test_top_open_candidate_indices_returns_legal_open_scores_in_order() -> None:
    rows, cols = 2, 3
    scores = np.full(4 * rows * cols, -10.0, dtype=np.float32)
    mask = np.zeros((4, rows, cols), dtype=bool)
    mask[0, 0, 0] = True
    mask[0, 0, 1] = True
    mask[0, 1, 2] = True
    scores[0] = 0.2
    scores[1] = 0.9
    scores[5] = 0.5
    scores[2] = 4.0
    scores[rows * cols + 1] = 10.0

    candidates = hard_loss_refine._top_open_candidate_indices(
        scores,
        mask,
        rows=rows,
        cols=cols,
        topk=2,
    )

    assert candidates.tolist() == [1, 5]


def test_counterfactual_model_topk_defaults_to_label_topk() -> None:
    args = type("Args", (), {"counterfactual_topk": 24, "counterfactual_model_topk": None})()
    override = type("Args", (), {"counterfactual_topk": 24, "counterfactual_model_topk": 7})()

    assert hard_loss_refine._counterfactual_model_topk(args) == 24
    assert hard_loss_refine._counterfactual_model_topk(override) == 7


def test_filter_extreme_replay_keeps_requested_tail_family() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0))

    def transition(progress: float) -> EpisodeTransition:
        action_mask = np.zeros((4, 3, 3), dtype=bool)
        action_mask[0, 1, 1] = True
        return EpisodeTransition(
            board=np.zeros((20, 3, 3), dtype=np.float32),
            global_features=np.array([0.0, progress, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            action_mask=action_mask,
            action_index=action_to_index(Action(ActionType.OPEN, 1, 1), 3, 3),
            expert_action_index=None,
            reward=0.0,
            done=False,
            expert_is_guess=True,
        )

    rows = [transition(0.9), transition(0.1)]

    filtered = hard_loss_refine.filter_extreme_replay(
        rows,
        trainer=trainer,
        family_filter=["guess_tail"],
        safe_left_min=1,
        safe_left_max=5,
        safe_left_threshold=2,
    )

    assert len(filtered) == 1
    assert filtered[0].global_features[1] == np.float32(0.9)


def test_filter_extreme_replay_can_keep_behavior_mine_high_regret_rows() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0))

    def transition(*, mine: bool, behavior_value: float) -> EpisodeTransition:
        action_mask = np.zeros((4, 3, 3), dtype=bool)
        action_mask[0, 1, 1] = True
        action_mask[0, 1, 2] = True
        mine_mask = np.zeros((3, 3), dtype=bool)
        mine_mask[1, 1] = mine
        counterfactual = np.full((3, 3), np.nan, dtype=np.float32)
        counterfactual[1, 1] = behavior_value
        counterfactual[1, 2] = 3.0
        return EpisodeTransition(
            board=np.zeros((20, 3, 3), dtype=np.float32),
            global_features=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
            action_mask=action_mask,
            action_index=action_to_index(Action(ActionType.OPEN, 1, 1), 3, 3),
            expert_action_index=None,
            reward=0.0,
            done=False,
            mine_mask=mine_mask,
            counterfactual_open_values=counterfactual,
            expert_is_guess=True,
        )

    filtered = hard_loss_refine.filter_extreme_replay(
        [
            transition(mine=True, behavior_value=-1.0),
            transition(mine=False, behavior_value=-1.0),
            transition(mine=True, behavior_value=2.5),
        ],
        trainer=trainer,
        behavior_mine_only=True,
        min_behavior_regret=2.0,
        safe_left_threshold=2,
    )

    assert len(filtered) == 1
    assert filtered[0].mine_mask is not None
    assert bool(filtered[0].mine_mask[1, 1])


def test_teacher_kl_ignores_illegal_student_actions() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    boards = torch.zeros((1, 20, 2, 2), dtype=torch.float32)
    global_features = torch.zeros((1, 6), dtype=torch.float32)
    with torch.no_grad():
        teacher_logits, _, teacher_risk = trainer.model.forward_with_risk(boards, global_features)
        teacher_logits = trainer._apply_model_risk_prior(teacher_logits, teacher_risk)

    flat_masks = torch.zeros((1, 16), dtype=torch.bool)
    flat_masks[0, 0] = True
    student_logits = torch.full((1, 16), -100.0, dtype=torch.float32)
    student_logits[0, 0] = 0.0
    student_logits[0, 15] = 100.0

    loss = hard_loss_refine.teacher_policy_kl_loss(
        student_logits=student_logits,
        teacher_trainer=trainer,
        boards=boards,
        global_features=global_features,
        flat_masks=flat_masks,
        temperature=1.0,
    )

    assert float(loss) < 1e-5


def test_teacher_kl_accepts_a_production_style_teacher_ensemble() -> None:
    trainer_a = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    trainer_b = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    trainer_b.model.load_state_dict(trainer_a.model.state_dict())
    boards = torch.zeros((1, 20, 2, 2), dtype=torch.float32)
    global_features = torch.zeros((1, 6), dtype=torch.float32)
    flat_masks = torch.ones((1, 16), dtype=torch.bool)
    student_logits = torch.zeros((1, 16), dtype=torch.float32)

    single = hard_loss_refine.teacher_policy_kl_loss(
        student_logits=student_logits,
        teacher_trainers=[trainer_a],
        teacher_weights=[1.0],
        boards=boards,
        global_features=global_features,
        flat_masks=flat_masks,
        temperature=1.0,
    )
    ensemble = hard_loss_refine.teacher_policy_kl_loss(
        student_logits=student_logits,
        teacher_trainers=[trainer_a, trainer_b],
        teacher_weights=[0.5, 0.5],
        boards=boards,
        global_features=global_features,
        flat_masks=flat_masks,
        temperature=1.0,
    )

    assert torch.allclose(single, ensemble, atol=1e-6)


def test_teacher_prior_ppo_loss_is_finite_and_has_policy_gradient() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    boards = torch.zeros((1, 20, 2, 2), dtype=torch.float32)
    global_features = torch.zeros((1, 6), dtype=torch.float32)
    flat_masks = torch.zeros((1, 16), dtype=torch.bool)
    flat_masks[0, :4] = True
    student_logits = torch.zeros((1, 16), dtype=torch.float32, requires_grad=True)
    transition = make_transition()
    transition.action_mask[0] = True
    transition.counterfactual_open_values = np.array(
        [[-1.0, 1.0], [0.5, 2.0]],
        dtype=np.float32,
    )

    loss = hard_loss_refine.teacher_prior_counterfactual_policy_loss(
        student_logits=student_logits,
        flat_masks=flat_masks,
        transitions=[transition],
        teacher_trainers=[trainer],
        teacher_weights=[1.0],
        boards=boards,
        global_features=global_features,
        strength=0.2,
        temperature=1.0,
        objective="ppo",
        clip_epsilon=0.2,
        teacher_topk=2,
    )
    loss.backward()

    assert torch.isfinite(loss)
    assert student_logits.grad is not None
    assert torch.isfinite(student_logits.grad).all()
    assert float(student_logits.grad.abs().sum()) > 0.0
