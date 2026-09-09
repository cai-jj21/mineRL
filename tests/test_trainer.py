from __future__ import annotations

from dataclasses import asdict

import numpy as np
import torch

from minesweeper_rl.features import action_to_index, decode_action_index, encode_state
from minesweeper_rl.features import COORDINATE_CHANNEL_START
from minesweeper_rl.types import Action, ActionType, EpisodeTransition
from minesweeper_rl.trainer import (
    MinesweeperTrainer,
    TrainingConfig,
    _expert_action_weights,
    _flip_transition,
    evaluate_policy,
    evaluate_policy_batched,
    load_checkpoint,
    make_game,
    transition_extreme_profile,
    transition_sample_weight,
    _normalize_counterfactual_values,
)


def test_solver_assisted_collection_records_legal_expert_actions() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20))

    transitions, _ = trainer.collect_episode(
        seed=100,
        deterministic=True,
        mode="rl",
        learn_from_solver=True,
    )

    assert transitions
    assert any(transition.expert_action_index is not None for transition in transitions)
    for transition in transitions:
        if transition.expert_action_index is not None:
            assert transition.action_mask.reshape(-1)[transition.expert_action_index]


def test_rl_collection_has_no_solver_labels_by_default() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20))

    transitions, _ = trainer.collect_episode(seed=100, deterministic=True, mode="rl")

    assert transitions
    assert all(transition.expert_action_index is None for transition in transitions)


def test_expert_episode_can_update_model() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20))

    transitions, summary = trainer.collect_expert_episode(seed=100)
    losses = trainer.update(transitions, imitation_coef=1.0)

    assert transitions
    assert summary.game_steps > 0
    assert all(transition.expert_action_index == transition.action_index for transition in transitions)
    assert losses["solver_imitation_loss"] > 0.0


def test_expert_action_weights_prefer_lower_risk_open_cells() -> None:
    rows, cols = 2, 3
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    expert_mask = np.zeros_like(action_mask)
    expert_mask[0, 0, 0] = True
    expert_mask[0, 0, 1] = True
    risk_map = np.full((rows, cols), 0.9, dtype=np.float32)
    risk_map[0, 0] = 0.1
    risk_map[0, 1] = 0.4
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        expert_action_mask=expert_mask,
        risk_map=risk_map,
    )

    weights = _expert_action_weights(
        transition,
        risk_temperature=0.1,
        allow_action_fallback=False,
    )

    assert weights[0, 0, 0] > weights[0, 0, 1] > 0.0
    assert float(weights.sum()) > 0.0


def test_expert_action_weights_ignore_counterfactual_open_values() -> None:
    rows, cols = 2, 3
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    expert_mask = np.zeros_like(action_mask)
    expert_mask[0, 0, 0] = True
    expert_mask[0, 0, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 0.1
    counterfactual[0, 1] = 1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        expert_action_mask=expert_mask,
        counterfactual_open_values=counterfactual,
    )

    weights = _expert_action_weights(
        transition,
        risk_temperature=0.1,
        allow_action_fallback=False,
    )

    assert weights[0, 0, 0] == weights[0, 0, 1] == 1.0
    assert float(weights.sum()) == 2.0


def test_counterfactual_policy_loss_prefers_higher_value_open_cells() -> None:
    rows, cols = 2, 3
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_policy_coef=1.0,
            counterfactual_policy_temperature=0.2,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 0.0
    counterfactual[0, 1] = 1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
    )
    masks = torch.tensor(np.stack([action_mask]), dtype=torch.bool)
    good_logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32)
    bad_logits = torch.zeros_like(good_logits)
    good_logits[0, 0, 0, 1] = 4.0
    bad_logits[0, 0, 0, 0] = 4.0

    good_loss = trainer._counterfactual_policy_loss(good_logits, masks, [transition])
    bad_loss = trainer._counterfactual_policy_loss(bad_logits, masks, [transition])

    assert good_loss < bad_loss


def test_counterfactual_policy_loss_keeps_low_value_behavior_cell() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_policy_coef=1.0,
            counterfactual_policy_temperature=0.2,
            counterfactual_policy_topk=1,
            counterfactual_policy_min_gap=0.1,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    action_mask[0, 0, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = -1.0
    counterfactual[0, 1] = 1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(1, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
        expert_is_guess=True,
    )
    masks = torch.tensor(np.stack([action_mask]), dtype=torch.bool)
    good_logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32)
    bad_logits = torch.zeros_like(good_logits)
    good_logits[0, 0, 0, 1] = 4.0
    bad_logits[0, 0, 0, 0] = 4.0

    good_loss = trainer._counterfactual_policy_loss(good_logits, masks, [transition])
    bad_loss = trainer._counterfactual_policy_loss(bad_logits, masks, [transition])

    assert good_loss < bad_loss


def test_counterfactual_policy_full_action_penalizes_non_open_mass() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_policy_coef=1.0,
            counterfactual_policy_temperature=0.2,
            counterfactual_policy_full_action=True,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    action_mask[0, 0, 1] = True
    action_mask[1, 1, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 0.0
    counterfactual[0, 1] = 1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
        expert_is_guess=True,
    )
    masks = torch.tensor(np.stack([action_mask]), dtype=torch.bool)
    good_logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32)
    bad_logits = torch.zeros_like(good_logits)
    good_logits[0, 0, 0, 1] = 4.0
    bad_logits[0, 1, 1, 1] = 4.0

    good_loss = trainer._counterfactual_policy_loss(good_logits, masks, [transition])
    bad_loss = trainer._counterfactual_policy_loss(bad_logits, masks, [transition])

    assert good_loss < bad_loss


def test_counterfactual_only_guess_excludes_guess_rows_from_solver_imitation() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_only_guess=True,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    expert_mask = np.zeros_like(action_mask)
    expert_mask[0, 0, 0] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 0.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        expert_action_mask=expert_mask,
        counterfactual_open_values=counterfactual,
        expert_is_guess=True,
    )
    logits = torch.zeros((1, 4 * rows * cols), dtype=torch.float32)

    loss = trainer._expert_policy_loss(logits, [transition], allow_action_fallback=True)

    assert float(loss) == 0.0


def test_transition_sample_weight_prioritizes_endgame_guess_states() -> None:
    rows, cols = 2, 2
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True

    early = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.1, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
    )
    late = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=-1.0,
        done=True,
        expert_is_guess=True,
    )

    assert transition_sample_weight(late) > transition_sample_weight(early)


def test_transition_sample_weight_rewards_edge_corner_tail_states() -> None:
    rows, cols = 3, 3
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True

    center = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 1, 1), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
    )
    corner = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.9, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
    )

    assert transition_sample_weight(corner) > transition_sample_weight(center)


def test_transition_sample_weight_rewards_counterfactual_regret() -> None:
    rows, cols = 2, 2
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    base = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=np.array([[1.0, 1.0], [np.nan, np.nan]], dtype=np.float32),
    )
    high_regret = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.5, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=np.array([[0.0, 3.0], [np.nan, np.nan]], dtype=np.float32),
    )

    assert transition_sample_weight(high_regret) > transition_sample_weight(base)


def test_transition_extreme_profile_identifies_corner_guess_tail() -> None:
    rows, cols = 16, 30
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.92, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        expert_is_guess=True,
        risk_map=np.full((rows, cols), 0.4, dtype=np.float32),
    )

    profile = transition_extreme_profile(transition, mines=99, safe_left_threshold=100)

    assert profile["extreme"] is True
    assert profile["family"] == "corner_guess_tail"
    assert profile["tail"] is True
    assert profile["guess"] is True
    assert profile["edge"] is True
    assert profile["corner"] is True
    assert profile["high_risk"] is True


def test_transition_extreme_profile_identifies_high_risk_family() -> None:
    rows, cols = 16, 30
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 5, 6] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.array([0.0, 0.2, 0.0, 0.0, 0.0, 0.0], dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 5, 6), rows, cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
        expert_is_guess=False,
        risk_map=np.full((rows, cols), 0.4, dtype=np.float32),
    )

    profile = transition_extreme_profile(transition, mines=99, safe_left_threshold=100)

    assert profile["extreme"] is True
    assert profile["family"] == "high_risk"
    assert profile["tail"] is False
    assert profile["guess"] is False
    assert profile["edge"] is False
    assert profile["corner"] is False
    assert profile["high_risk"] is True


def test_mine_auxiliary_ignores_hidden_mine_mask_when_risk_map_exists() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(rows=rows, cols=cols, mines=1, safe_radius=0, mine_aux_coef=1.0)
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        mine_mask=np.array([[True, False], [False, False]], dtype=bool),
        risk_map=np.full((rows, cols), 0.25, dtype=np.float32),
    )
    logits = torch.zeros((1, 4, rows, cols), requires_grad=True)
    masks = torch.tensor(action_mask[None], dtype=torch.bool)

    loss = trainer._mine_auxiliary_loss(logits, masks, [transition])

    assert float(loss) == 0.0


def test_counterfactual_risk_head_loss_uses_open_values() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_risk_head_coef=1.0,
            counterfactual_risk_temperature=1.0,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    action_mask[0, 0, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 2.0
    counterfactual[0, 1] = -1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
    )
    risk_logits = torch.zeros((1, 1, rows, cols), requires_grad=True)
    masks = torch.tensor(action_mask[None], dtype=torch.bool)

    loss = trainer._counterfactual_risk_head_loss(risk_logits, masks, [transition])

    assert float(loss) > 0.0
    loss.backward()
    assert risk_logits.grad is not None


def test_counterfactual_risk_head_listwise_loss_ranks_open_values() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_risk_head_coef=1.0,
            counterfactual_risk_temperature=1.0,
            counterfactual_risk_loss_mode="listwise",
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, 0, 0] = True
    action_mask[0, 0, 1] = True
    counterfactual = np.full((rows, cols), np.nan, dtype=np.float32)
    counterfactual[0, 0] = 2.0
    counterfactual[0, 1] = -1.0
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=0,
        expert_action_index=None,
        reward=0.0,
        done=False,
        counterfactual_open_values=counterfactual,
    )
    risk_logits = torch.zeros((1, 1, rows, cols), requires_grad=True)
    masks = torch.tensor(action_mask[None], dtype=torch.bool)

    loss = trainer._counterfactual_risk_head_loss(risk_logits, masks, [transition])

    assert float(loss) > 0.0
    loss.backward()
    assert risk_logits.grad is not None
    assert float(risk_logits.grad[0, 0, 0, 0]) > float(risk_logits.grad[0, 0, 0, 1])


def test_guess_survival_loss_uses_hidden_mine_labels_on_guess_rows() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            guess_survival_coef=1.0,
            guess_survival_mine_weight=4.0,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :, :] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=-1.0,
        done=True,
        expert_is_guess=True,
        mine_mask=np.asarray([[True, False], [False, False]], dtype=bool),
        risk_map=np.full((rows, cols), 0.25, dtype=np.float32),
    )
    logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32, requires_grad=True)
    masks = torch.tensor(action_mask[None], dtype=torch.bool)

    loss = trainer._guess_survival_loss(logits, masks, [transition])

    assert float(loss) > 0.0
    loss.backward()
    assert logits.grad is not None
    assert float(logits.grad[0, 0, 0, 0]) > 0.0
    assert float(logits.grad[0, 0, 0, 1]) < 0.0


def test_guess_survival_loss_penalizes_non_open_mass_on_guess_rows() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            guess_survival_coef=1.0,
            guess_survival_mine_weight=4.0,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :, :] = True
    action_mask[1, 1, 1] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=-1.0,
        done=True,
        expert_is_guess=True,
        mine_mask=np.asarray([[True, False], [False, False]], dtype=bool),
        risk_map=np.full((rows, cols), 0.25, dtype=np.float32),
    )
    masks = torch.tensor(action_mask[None], dtype=torch.bool)
    base_logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32)
    bad_logits = torch.zeros_like(base_logits)
    bad_logits[0, 1, 1, 1] = 4.0

    base_loss = trainer._guess_survival_loss(base_logits, masks, [transition])
    bad_loss = trainer._guess_survival_loss(bad_logits, masks, [transition])

    assert bad_loss > base_loss


def test_counterfactual_value_head_rank_loss_handles_sparse_rows() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_value_head_coef=1.0,
            counterfactual_value_rank_coef=0.5,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :, :] = True
    transitions = []
    for action_index, values in [
        (action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols), [[0.0, 2.0], [np.nan, np.nan]]),
        (action_to_index(Action(ActionType.OPEN, 0, 1), rows, cols), [[np.nan, np.nan], [1.0, 0.0]]),
    ]:
        transitions.append(
            EpisodeTransition(
                board=np.zeros((1, rows, cols), dtype=np.float32),
                global_features=np.zeros(6, dtype=np.float32),
                action_mask=action_mask,
                action_index=action_index,
                expert_action_index=None,
                reward=0.0,
                done=False,
                counterfactual_open_values=np.asarray(values, dtype=np.float32),
            )
        )
    prediction = torch.zeros((2, rows, cols), requires_grad=True)
    masks = torch.tensor(action_mask[None].repeat(2, axis=0), dtype=torch.bool)

    loss = trainer._counterfactual_value_head_loss(prediction, masks, transitions)

    assert float(loss) > 0.0
    loss.backward()
    assert prediction.grad is not None


def test_counterfactual_value_target_normalization_preserves_order_and_nan() -> None:
    values = np.asarray([[np.nan, -1.0, 2.0, 5.0]], dtype=np.float32)
    for mode in ("minmax", "zscore", "rank"):
        normalized = _normalize_counterfactual_values(values, mode)
        assert np.isnan(normalized[0, 0])
        assert np.argsort(normalized[0, 1:]).tolist() == [0, 1, 2]


def test_training_config_rejects_unknown_counterfactual_target_mode() -> None:
    try:
        TrainingConfig(counterfactual_value_target_mode="invalid")
    except ValueError as exc:
        assert "counterfactual_value_target_mode" in str(exc)
    else:
        raise AssertionError("unknown counterfactual target mode should fail")


def test_load_checkpoint_ignores_optimizer_state_mismatch(tmp_path) -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=2, cols=2, mines=1, safe_radius=0))
    checkpoint = {
        "config": asdict(trainer.config),
        "model": trainer.model.state_dict(),
        "optimizer": {
            "state": {},
            "param_groups": [
                {"lr": 1e-3, "params": [0]},
                {"lr": 1e-3, "params": [1]},
            ],
        },
    }
    path = tmp_path / "checkpoint.pt"
    torch.save(checkpoint, path)

    loaded = load_checkpoint(path)

    assert loaded.config.rows == 2
    assert loaded.config.cols == 2


def test_counterfactual_preference_loss_penalizes_bad_behavior_action() -> None:
    rows, cols = 2, 2
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            counterfactual_coef=1.0,
            counterfactual_margin=0.1,
        )
    )
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :, :] = True
    expert_mask = np.zeros_like(action_mask)
    expert_mask[0, 0, 0] = True
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 1, 1), rows, cols),
        expert_action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        reward=-1.0,
        done=True,
        expert_action_mask=expert_mask,
    )
    logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32)
    loss = trainer._counterfactual_preference_loss(logits.view(1, -1), [transition])

    assert float(loss) > 0.0


def test_behavior_mine_demotion_only_pushes_bad_behavior_below_better_candidates() -> None:
    rows, cols = 2, 2
    action_mask = np.zeros((4, rows, cols), dtype=bool)
    action_mask[0, :, :] = True
    mine_mask = np.zeros((rows, cols), dtype=bool)
    mine_mask[0, 0] = True
    counterfactual = np.asarray(
        [
            [-1.0, 2.0],
            [1.0, 0.2],
        ],
        dtype=np.float32,
    )
    transition = EpisodeTransition(
        board=np.zeros((1, rows, cols), dtype=np.float32),
        global_features=np.zeros(6, dtype=np.float32),
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 0, 0), rows, cols),
        expert_action_index=None,
        reward=-1.0,
        done=True,
        mine_mask=mine_mask,
        counterfactual_open_values=counterfactual,
        expert_is_guess=True,
    )
    trainer = MinesweeperTrainer(
        TrainingConfig(
            rows=rows,
            cols=cols,
            mines=1,
            safe_radius=0,
            behavior_mine_demotion_coef=1.0,
            behavior_mine_demotion_topk=0,
            behavior_mine_demotion_margin=0.1,
        )
    )
    logits = torch.zeros((1, 4, rows, cols), dtype=torch.float32, requires_grad=True)

    loss = trainer._behavior_mine_demotion_loss(
        logits,
        torch.tensor(action_mask[None], dtype=torch.bool),
        [transition],
    )
    loss.backward()

    gradients = logits.grad[0, 0].detach().cpu().numpy()
    assert float(loss) > 0.0
    assert gradients[0, 0] > 0.0
    assert gradients[0, 1] < 0.0
    assert gradients[1, 0] < 0.0


def test_training_expert_prioritizes_unflagging_wrong_flags() -> None:
    trainer = MinesweeperTrainer(
        TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20, decision_actions="full")
    )
    game = make_game(trainer.config, seed=100)
    game.open_cell(1, 1)

    wrong_flag: tuple[int, int] | None = None
    for row in range(game.rows):
        for col in range(game.cols):
            if game.hidden_mask()[row, col] and not game.mines[row, col]:
                wrong_flag = (row, col)
                break
        if wrong_flag is not None:
            break
    assert wrong_flag is not None
    game.flag_cell(*wrong_flag)

    _, _, action_mask = encode_state(game)
    assert action_mask.reshape(-1)[action_to_index(Action(ActionType.UNFLAG, wrong_flag[0], wrong_flag[1]), game.rows, game.cols)]
    action_index = trainer._solver_expert_action(game, action_mask)

    assert action_index == action_to_index(
        Action(ActionType.UNFLAG, wrong_flag[0], wrong_flag[1]),
        game.rows,
        game.cols,
    )


def test_open_only_decision_mode_masks_non_open_actions() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20))

    transitions, _ = trainer.collect_episode(seed=100, deterministic=True, mode="rl", learn_from_solver=True)

    assert transitions
    assert all(decode_action_index(transition.action_index, trainer.config.rows, trainer.config.cols).kind == ActionType.OPEN for transition in transitions)
    assert all(
        transition.expert_action_index is None
        or decode_action_index(transition.expert_action_index, trainer.config.rows, trainer.config.cols).kind == ActionType.OPEN
        for transition in transitions
    )


def test_flip_augmentation_rebuilds_coordinate_channels() -> None:
    trainer = MinesweeperTrainer(
        TrainingConfig(rows=4, cols=5, mines=3, safe_radius=0, seed=100, max_steps=20, decision_actions="full")
    )
    game = make_game(trainer.config, seed=100)
    game.open_cell(1, 1)
    board, global_features, action_mask = encode_state(game)
    transition = EpisodeTransition(
        board=board,
        global_features=global_features,
        action_mask=action_mask,
        action_index=action_to_index(Action(ActionType.OPEN, 2, 3), game.rows, game.cols),
        expert_action_index=None,
        reward=0.0,
        done=False,
    )

    flipped = _flip_transition(transition, trainer.config.rows, trainer.config.cols, True, True)

    assert np.allclose(flipped.board[COORDINATE_CHANNEL_START:], board[COORDINATE_CHANNEL_START:])
    assert not np.allclose(flipped.board[:COORDINATE_CHANNEL_START], board[:COORDINATE_CHANNEL_START])


def test_exploration_logits_keep_topk_and_scale_temperature() -> None:
    trainer = MinesweeperTrainer(
        TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, exploration_temperature=0.5, exploration_topk=2)
    )

    logits = torch.tensor([[0.0, 2.0, 1.0, -1e9]], dtype=torch.float32)
    filtered = trainer._exploration_logits(logits)

    assert torch.isclose(filtered[0, 1], torch.tensor(4.0))
    assert torch.isclose(filtered[0, 2], torch.tensor(2.0))
    assert filtered[0, 0] < -1e8


def test_checkpoint_loader_can_expand_model_width(tmp_path) -> None:
    small = MinesweeperTrainer(
        TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, hidden_channels=8, residual_blocks=1)
    )
    with torch.no_grad():
        for param in small.model.parameters():
            param.fill_(0.125)

    checkpoint_path = tmp_path / "small.pt"
    from minesweeper_rl.trainer import save_checkpoint

    save_checkpoint(checkpoint_path, small, episode=1, metrics={"win_rate": 0.0})

    loaded = load_checkpoint(checkpoint_path, config_overrides={"hidden_channels": 16, "residual_blocks": 2})
    loaded_weight = loaded.model.state_dict()["backbone.0.weight"]

    assert loaded.config.hidden_channels == 16
    assert loaded.config.residual_blocks == 2
    assert torch.allclose(loaded_weight[:8], torch.full_like(loaded_weight[:8], 0.125))


def test_batched_evaluation_matches_sequential_evaluation() -> None:
    trainer = MinesweeperTrainer(
        TrainingConfig(rows=3, cols=3, mines=1, safe_radius=0, seed=100, max_steps=20, decision_actions="full")
    )

    sequential = evaluate_policy(trainer, games=5, seed=200, mode="rl", risk_weight=0.0)
    batched = evaluate_policy_batched(trainer, games=5, seed=200, mode="rl", risk_weight=0.0, batch_size=3)

    assert batched == sequential
