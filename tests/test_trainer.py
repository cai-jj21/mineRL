from __future__ import annotations

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

    weights = _expert_action_weights(transition, risk_temperature=0.1, allow_action_fallback=False)

    assert weights[0, 0, 0] > weights[0, 0, 1] > 0.0
    assert float(weights.sum()) > 0.0


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
