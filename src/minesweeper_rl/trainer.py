from __future__ import annotations

import json
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from statistics import mean

import numpy as np
import torch
from torch import nn
from torch.distributions import Categorical
from torch.nn import functional as F
from tqdm import tqdm

from minesweeper_rl.features import (
    COORDINATE_CHANNEL_START,
    action_channel,
    action_to_index,
    coordinate_channels,
    decode_action_index,
    encode_state,
)
from minesweeper_rl.game import MinesweeperGame
from minesweeper_rl.model import MinesweeperNet
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.types import Action, ActionType, EpisodeSummary, EpisodeTransition, GameConfig, SolverSnapshot


MODEL_MODES = {"rl", "policy", "hybrid"}
VALID_MODES = ("rl", "policy", "solver", "hybrid")
DECISION_ACTION_MODES = ("open", "full")
INFERENCE_ENSEMBLES = ("logits", "probs")


@dataclass
class TrainingConfig:
    rows: int = 16
    cols: int = 30
    mines: int = 99
    safe_radius: int = 1
    exact_limit: int = 24
    seed: int = 0
    episodes: int = 1000
    gamma: float = 0.995
    lr: float = 3e-4
    entropy_coef: float = 0.01
    value_coef: float = 0.5
    solver_imitation_coef: float = 0.2
    mine_aux_coef: float = 0.0
    risk_supervision_coef: float = 0.0
    risk_temperature: float = 0.08
    grad_clip: float = 1.0
    hidden_channels: int = 64
    residual_blocks: int = 3
    eval_every: int = 100
    eval_games: int = 50
    pretrain_episodes: int = 0
    pretrain_imitation_coef: float = 1.0
    pretrain_epochs_per_episode: int = 1
    pretrain_batch_size: int = 256
    expert_replay_size: int = 8192
    dagger_replay_size: int = 8192
    dagger_batch_size: int = 256
    dagger_updates_per_episode: int = 1
    guess_supervision_topk: int = 5
    exploration_temperature: float = 1.0
    exploration_topk: int = 0
    inference_augment_flips: bool = False
    inference_ensemble: str = "logits"
    augment_flips: bool = True
    decision_actions: str = "open"
    risk_weight: float = 0.0
    max_steps: int = 2000
    device: str = "cpu"

    def __post_init__(self) -> None:
        if self.decision_actions not in DECISION_ACTION_MODES:
            raise ValueError(f"unknown decision_actions {self.decision_actions!r}; expected one of {DECISION_ACTION_MODES}")
        if self.exploration_temperature <= 0.0:
            raise ValueError("exploration_temperature must be positive")
        if self.exploration_topk < 0:
            raise ValueError("exploration_topk must be non-negative")
        if self.inference_ensemble not in INFERENCE_ENSEMBLES:
            raise ValueError(f"unknown inference_ensemble {self.inference_ensemble!r}; expected one of {INFERENCE_ENSEMBLES}")

    @property
    def game_config(self) -> GameConfig:
        return GameConfig(rows=self.rows, cols=self.cols, mines=self.mines, safe_radius=self.safe_radius)


def make_game(config: TrainingConfig | GameConfig, seed: int | None = None) -> MinesweeperGame:
    return MinesweeperGame(
        rows=config.rows,
        cols=config.cols,
        mines=config.mines,
        safe_radius=config.safe_radius,
        seed=seed,
    )


def center_first_open(game: MinesweeperGame) -> tuple[float, bool]:
    if game.mines_placed or game.done:
        return 0.0, game.done
    row, col = game.rows // 2, game.cols // 2
    _, reward, done, _ = game.open_cell(row, col)
    return reward, done


def resolve_forced_moves(
    game: MinesweeperGame,
    solver: MinesweeperSolver,
    max_rounds: int = 256,
) -> tuple[SolverSnapshot, float, int]:
    total_reward = 0.0
    forced_steps = 0
    snapshot = solver.analyze(game)

    for _ in range(max_rounds):
        if game.done or not snapshot.has_forced_moves:
            return snapshot, total_reward, forced_steps

        progressed = False

        for row, col in zip(*np.where(snapshot.mine_mask)):
            if game.done:
                break
            if not game.revealed[row, col] and not game.flagged[row, col]:
                _, reward, _, _ = game.flag_cell(int(row), int(col))
                total_reward += reward
                forced_steps += 1
                progressed = True

        for row, col in zip(*np.where(snapshot.safe_mask)):
            if game.done:
                break
            if not game.revealed[row, col] and not game.flagged[row, col]:
                _, reward, _, _ = game.open_cell(int(row), int(col))
                total_reward += reward
                forced_steps += 1
                progressed = True

        if not progressed:
            return snapshot, total_reward, forced_steps
        snapshot = solver.analyze(game)

    return snapshot, total_reward, forced_steps


class MinesweeperTrainer:
    def __init__(self, config: TrainingConfig) -> None:
        self.config = config
        self.device = torch.device(config.device if torch.cuda.is_available() or config.device == "cpu" else "cpu")
        if self.device.type == "cuda":
            try:
                torch.backends.cuda.matmul.allow_tf32 = True
                torch.backends.cudnn.allow_tf32 = True
                torch.backends.cudnn.benchmark = True
                torch.set_float32_matmul_precision("high")
            except Exception:
                pass
        self.solver = MinesweeperSolver(exact_limit=config.exact_limit)
        self.model = MinesweeperNet(
            hidden_channels=config.hidden_channels,
            residual_blocks=config.residual_blocks,
        ).to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=config.lr)
        self.rng = np.random.default_rng(config.seed)

    def collect_episode(
        self,
        seed: int | None,
        deterministic: bool = False,
        mode: str = "rl",
        risk_weight: float | None = None,
        learn_from_solver: bool = False,
    ) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        if mode == "solver":
            return self._collect_solver_episode(seed=seed)
        if mode not in MODEL_MODES:
            raise ValueError(f"unknown mode {mode!r}; expected one of {VALID_MODES}")
        return self._collect_model_episode(
            seed=seed,
            deterministic=deterministic,
            mode=mode,
            risk_weight=self.config.risk_weight if risk_weight is None else risk_weight,
            learn_from_solver=learn_from_solver,
        )

    def _collect_model_episode(
        self,
        seed: int | None,
        deterministic: bool,
        mode: str,
        risk_weight: float,
        learn_from_solver: bool,
    ) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward = 0.0
        agent_steps = 0
        transitions: list[EpisodeTransition] = []

        while not game.done and agent_steps < self.config.max_steps:
            board, global_features, action_mask = encode_state(game)
            action_mask = self._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            mine_mask = game.mines.copy() if learn_from_solver and game.mines_placed else None
            risk_map = None
            snapshot = None
            if learn_from_solver and game.mines_placed:
                snapshot = self.solver.analyze(game)
            if learn_from_solver and self.config.risk_supervision_coef > 0.0 and game.mines_placed:
                risk_map = snapshot.risk_map if snapshot is not None else None
            expert_action_mask = self._solver_expert_action_mask(game, action_mask, snapshot) if learn_from_solver else None
            expert_action_index = None
            if expert_action_mask is not None and expert_action_mask.any():
                expert_action_index = int(np.flatnonzero(expert_action_mask.reshape(-1))[0])
            action_index = self._select_action(
                board=board,
                global_features=global_features,
                action_mask=action_mask,
                game=game,
                deterministic=deterministic,
                mode=mode,
                risk_weight=risk_weight,
            )
            action = decode_action_index(action_index, game.rows, game.cols)
            _, reward, _, _ = game.step(action)
            total_reward += reward
            agent_steps += 1

            transitions.append(
                EpisodeTransition(
                    board=board,
                    global_features=global_features,
                    action_mask=action_mask,
                    action_index=action_index,
                    expert_action_index=expert_action_index,
                    reward=float(reward),
                    done=bool(game.done),
                    expert_action_mask=expert_action_mask,
                    mine_mask=mine_mask,
                    risk_map=risk_map,
                )
            )

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=agent_steps,
            forced_steps=0,
        )
        return transitions, summary

    def _collect_solver_episode(self, seed: int | None) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward, done = center_first_open(game)
        forced_steps = 0
        guess_steps = 0

        if not done:
            snapshot, reward, steps = resolve_forced_moves(game, self.solver)
            total_reward += reward
            forced_steps += steps
        else:
            snapshot = self.solver.analyze(game)

        while not game.done and guess_steps + forced_steps < self.config.max_steps:
            action_mask = game.legal_open_mask().astype(bool)
            if not action_mask.any():
                break

            action_index = self._select_solver_guess(snapshot, action_mask)
            row, col = divmod(action_index, game.cols)
            _, reward, done, _ = game.open_cell(row, col)
            total_reward += reward
            guess_steps += 1

            if not done:
                snapshot, forced_reward, steps = resolve_forced_moves(game, self.solver)
                total_reward += forced_reward
                forced_steps += steps
            else:
                snapshot = self.solver.analyze(game)

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=guess_steps,
            forced_steps=forced_steps,
        )
        return [], summary

    def collect_expert_episode(self, seed: int | None) -> tuple[list[EpisodeTransition], EpisodeSummary]:
        game = make_game(self.config, seed=seed)
        total_reward = 0.0
        expert_steps = 0
        transitions: list[EpisodeTransition] = []

        while not game.done and expert_steps < self.config.max_steps:
            board, global_features, action_mask = encode_state(game)
            action_mask = self._decision_action_mask(action_mask)
            if not action_mask.any():
                break

            if not game.mines_placed:
                expert_action_mask = self._solver_expert_action_mask(game, action_mask)
                if expert_action_mask is None or not expert_action_mask.any():
                    break
                action_index = int(np.flatnonzero(expert_action_mask.reshape(-1))[0])
                reward = self._apply_expert_action(game, transitions, board, global_features, action_mask, action_index)
                total_reward += reward
                expert_steps += 1
                continue

            snapshot = self.solver.analyze(game)
            forced_actions: list[int] = []
            mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
            for row, col in zip(*np.where(mine_cells)):
                forced_actions.append(action_to_index(Action(ActionType.FLAG, int(row), int(col)), game.rows, game.cols))

            safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
            for row, col in zip(*np.where(safe_cells)):
                forced_actions.append(action_to_index(Action(ActionType.OPEN, int(row), int(col)), game.rows, game.cols))

            for action_index in forced_actions:
                if expert_steps >= self.config.max_steps or game.done:
                    break
                current_board, current_global_features, current_action_mask = encode_state(game)
                current_action_mask = self._decision_action_mask(current_action_mask)
                current_expert_mask = _expert_mask_from_indices(current_action_mask, forced_actions)
                if not current_expert_mask.any():
                    continue
                if not current_action_mask.reshape(-1)[action_index]:
                    action_index = int(np.flatnonzero(current_expert_mask.reshape(-1))[0])
                reward = self._apply_expert_action(
                    game,
                    transitions,
                    current_board,
                    current_global_features,
                    current_action_mask,
                    action_index,
                    current_expert_mask,
                    snapshot.risk_map,
                )
                total_reward += reward
                expert_steps += 1

            if forced_actions:
                continue

            action_index = self._solver_expert_action_from_snapshot(game, snapshot, action_mask)
            if action_index is None:
                break
            reward = self._apply_expert_action(
                game,
                transitions,
                board,
                global_features,
                action_mask,
                action_index,
                _single_action_mask(action_mask, action_index),
                snapshot.risk_map,
            )
            total_reward += reward
            expert_steps += 1

        summary = self._episode_summary(
            game=game,
            seed=seed,
            total_reward=total_reward,
            guess_steps=expert_steps,
            forced_steps=0,
        )
        return transitions, summary

    def _apply_expert_action(
        self,
        game: MinesweeperGame,
        transitions: list[EpisodeTransition],
        board: np.ndarray,
        global_features: np.ndarray,
        action_mask: np.ndarray,
        action_index: int,
        expert_action_mask: np.ndarray | None = None,
        risk_map: np.ndarray | None = None,
    ) -> float:
        mine_mask = game.mines.copy() if game.mines_placed else None
        action = decode_action_index(action_index, game.rows, game.cols)
        _, reward, _, _ = game.step(action)
        transitions.append(
            EpisodeTransition(
                board=board,
                global_features=global_features,
                action_mask=action_mask,
                action_index=action_index,
                expert_action_index=action_index,
                reward=float(reward),
                done=bool(game.done),
                expert_action_mask=expert_action_mask,
                mine_mask=mine_mask,
                risk_map=risk_map,
            )
        )
        return float(reward)

    def _episode_summary(
        self,
        game: MinesweeperGame,
        seed: int | None,
        total_reward: float,
        guess_steps: int,
        forced_steps: int,
    ) -> EpisodeSummary:
        return EpisodeSummary(
            won=bool(game.won),
            lost=bool(game.lost),
            reward=float(total_reward),
            game_steps=int(game.step_count),
            guess_steps=guess_steps,
            forced_steps=forced_steps,
            revealed_safe_cells=int((game.revealed & ~game.mines).sum()) if game.mines_placed else int(game.revealed.sum()),
            flags=int(game.flagged.sum()),
            seed=seed,
        )

    def train(self, save_path: str | Path | None = None) -> dict[str, float | int | bool | None]:
        save_path = Path(save_path) if save_path else None
        best_win_rate = -1.0
        best_saved = False
        recent: list[EpisodeSummary] = []
        expert_replay: list[EpisodeTransition] = []
        dagger_replay: list[EpisodeTransition] = []

        if self.config.pretrain_episodes > 0:
            pretrain = tqdm(
                range(1, self.config.pretrain_episodes + 1),
                desc="solver pretrain",
                unit="episode",
            )
            for episode in pretrain:
                seed = int(self.rng.integers(0, 2**31 - 1))
                transitions, summary = self.collect_expert_episode(seed=seed)
                if transitions:
                    if self.config.expert_replay_size > 0:
                        expert_replay.extend(transitions)
                        overflow = len(expert_replay) - self.config.expert_replay_size
                        if overflow > 0:
                            del expert_replay[:overflow]
                        train_source = expert_replay
                    else:
                        train_source = transitions
                    losses = self._zero_losses()
                    for _ in range(max(1, self.config.pretrain_epochs_per_episode)):
                        batch = self._sample_transitions(train_source, self.config.pretrain_batch_size)
                        losses = self.update_imitation(batch)
                else:
                    losses = self._zero_losses()
                pretrain.set_postfix(
                    win_rate=f"{float(summary.won):.2f}",
                    expert_steps=summary.guess_steps,
                    replay=len(expert_replay),
                    loss=f"{losses['loss']:.3f}",
                    solver=f"{losses['solver_imitation_loss']:.3f}",
                    mine=f"{losses['mine_aux_loss']:.3f}",
                    risk=f"{losses['risk_supervision_loss']:.3f}",
                )

        progress = tqdm(range(1, self.config.episodes + 1), desc="training", unit="episode")
        for episode in progress:
            seed = int(self.rng.integers(0, 2**31 - 1))
            collect_training_labels = (
                self.config.solver_imitation_coef > 0.0
                or self.config.dagger_updates_per_episode > 0
                or self.config.mine_aux_coef > 0.0
                or self.config.risk_supervision_coef > 0.0
            )
            transitions, summary = self.collect_episode(
                seed=seed,
                deterministic=False,
                mode="rl",
                learn_from_solver=collect_training_labels,
            )
            if transitions:
                losses = self.update(transitions)
                expert_labeled = [transition for transition in transitions if transition.expert_action_index is not None]
                if expert_labeled and self.config.dagger_updates_per_episode > 0:
                    if self.config.dagger_replay_size > 0:
                        dagger_replay.extend(expert_labeled)
                        overflow = len(dagger_replay) - self.config.dagger_replay_size
                        if overflow > 0:
                            del dagger_replay[:overflow]
                        train_source = dagger_replay
                    else:
                        train_source = expert_labeled

                    dagger_losses = self._zero_losses()
                    for _ in range(self.config.dagger_updates_per_episode):
                        batch = self._sample_transitions(train_source, self.config.dagger_batch_size)
                        dagger_losses = self.update_imitation(batch)
                    losses["dagger_loss"] = dagger_losses["loss"]
            else:
                losses = self._zero_losses()
            recent.append(summary)
            recent = recent[-100:]

            progress.set_postfix(
                win_rate=f"{mean(s.won for s in recent):.2f}",
                reward=f"{summary.reward:.2f}",
                agent_steps=summary.guess_steps,
                loss=f"{losses['loss']:.3f}",
                solver=f"{losses['solver_imitation_loss']:.3f}",
                dagger=f"{losses.get('dagger_loss', 0.0):.3f}",
                mine=f"{losses['mine_aux_loss']:.3f}",
                risk=f"{losses['risk_supervision_loss']:.3f}",
            )

            if self.config.eval_every > 0 and episode % self.config.eval_every == 0:
                metrics = evaluate_policy(
                    self,
                    games=self.config.eval_games,
                    seed=self.config.seed + episode * 997,
                    mode="rl",
                )
                if metrics["win_rate"] > best_win_rate:
                    best_win_rate = metrics["win_rate"]
                    if save_path is not None:
                        save_checkpoint(save_path, self, episode=episode, metrics=metrics)
                        best_saved = True

        final_metrics = evaluate_policy(
            self,
            games=max(1, self.config.eval_games),
            seed=self.config.seed + 100000,
            mode="rl",
        )
        if save_path is not None and (not best_saved or final_metrics["win_rate"] >= best_win_rate):
            save_checkpoint(save_path, self, episode=self.config.episodes, metrics=final_metrics)
        return final_metrics

    def update(self, transitions: list[EpisodeTransition], imitation_coef: float | None = None) -> dict[str, float]:
        transitions = self._augment_transitions(transitions)
        boards = torch.tensor(np.stack([t.board for t in transitions]), dtype=torch.float32, device=self.device)
        global_features = torch.tensor(
            np.stack([t.global_features for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=self.device)
        actions = torch.tensor([t.action_index for t in transitions], dtype=torch.long, device=self.device)
        returns = self._discounted_returns([t.reward for t in transitions]).to(self.device)

        logits, values = self.model(boards, global_features)
        flat_logits = logits.view(logits.shape[0], -1)
        flat_masks = masks.view(masks.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~flat_masks, -1e9)
        dist = Categorical(logits=self._exploration_logits(masked_logits, actions))

        log_probs = dist.log_prob(actions)
        entropy = dist.entropy().mean()
        advantages = returns - values.detach()
        if len(advantages) > 1:
            advantages = (advantages - advantages.mean()) / (advantages.std(unbiased=False) + 1e-8)

        policy_loss = -(log_probs * advantages).mean()
        value_loss = F.mse_loss(values, returns)
        solver_imitation_loss = self._expert_policy_loss(masked_logits, transitions, allow_action_fallback=False)
        loss = policy_loss + self.config.value_coef * value_loss
        loss = loss + (self.config.solver_imitation_coef if imitation_coef is None else imitation_coef) * solver_imitation_loss
        mine_aux_loss = self._mine_auxiliary_loss(logits, masks, transitions)
        risk_supervision_loss = self._risk_supervision_loss(logits, masks, transitions)
        loss = loss + self.config.mine_aux_coef * mine_aux_loss
        loss = loss + self.config.risk_supervision_coef * risk_supervision_loss
        loss = loss - self.config.entropy_coef * entropy

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": float(policy_loss.detach().cpu()),
            "value_loss": float(value_loss.detach().cpu()),
            "solver_imitation_loss": float(solver_imitation_loss.detach().cpu()),
            "mine_aux_loss": float(mine_aux_loss.detach().cpu()),
            "risk_supervision_loss": float(risk_supervision_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def update_imitation(self, transitions: list[EpisodeTransition]) -> dict[str, float]:
        transitions = self._augment_transitions(transitions)
        boards = torch.tensor(np.stack([t.board for t in transitions]), dtype=torch.float32, device=self.device)
        global_features = torch.tensor(
            np.stack([t.global_features for t in transitions]),
            dtype=torch.float32,
            device=self.device,
        )
        masks = torch.tensor(np.stack([t.action_mask for t in transitions]), dtype=torch.bool, device=self.device)
        logits, values = self.model(boards, global_features)
        flat_logits = logits.view(logits.shape[0], -1)
        flat_masks = masks.view(masks.shape[0], -1)
        masked_logits = flat_logits.masked_fill(~flat_masks, -1e9)
        imitation_loss = self._expert_policy_loss(masked_logits, transitions, allow_action_fallback=True)
        mine_aux_loss = self._mine_auxiliary_loss(logits, masks, transitions)
        risk_supervision_loss = self._risk_supervision_loss(logits, masks, transitions)
        loss = imitation_loss * self.config.pretrain_imitation_coef
        loss = loss + self.config.mine_aux_coef * mine_aux_loss
        loss = loss + self.config.risk_supervision_coef * risk_supervision_loss
        entropy = Categorical(logits=masked_logits).entropy().mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip)
        self.optimizer.step()

        return {
            "loss": float(loss.detach().cpu()),
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "solver_imitation_loss": float(imitation_loss.detach().cpu()),
            "mine_aux_loss": float(mine_aux_loss.detach().cpu()),
            "risk_supervision_loss": float(risk_supervision_loss.detach().cpu()),
            "entropy": float(entropy.detach().cpu()),
        }

    def _expert_policy_loss(
        self,
        masked_logits: torch.Tensor,
        transitions: list[EpisodeTransition],
        allow_action_fallback: bool,
    ) -> torch.Tensor:
        expert_masks = torch.tensor(
            np.stack([_transition_expert_mask(transition, allow_action_fallback) for transition in transitions]),
            dtype=torch.bool,
            device=self.device,
        ).view(masked_logits.shape[0], -1)
        valid_rows = expert_masks.any(dim=1)
        if not bool(valid_rows.any()):
            return masked_logits.new_tensor(0.0)

        targets = expert_masks[valid_rows].float()
        targets = targets / targets.sum(dim=1, keepdim=True).clamp_min(1.0)
        log_probs = F.log_softmax(masked_logits[valid_rows], dim=1)
        return -(targets * log_probs).sum(dim=1).mean()

    def _zero_losses(self) -> dict[str, float]:
        return {
            "loss": 0.0,
            "policy_loss": 0.0,
            "value_loss": 0.0,
            "solver_imitation_loss": 0.0,
            "dagger_loss": 0.0,
            "mine_aux_loss": 0.0,
            "risk_supervision_loss": 0.0,
            "entropy": 0.0,
        }

    def _mine_auxiliary_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.mine_aux_coef <= 0.0:
            return logits.new_tensor(0.0)
        mine_targets = torch.tensor(
            np.stack(
                [
                    np.zeros((self.config.rows, self.config.cols), dtype=bool)
                    if transition.mine_mask is None
                    else transition.mine_mask
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        known_targets = torch.tensor(
            [transition.mine_mask is not None for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        ).view(-1, 1, 1)
        open_channel = action_channel(ActionType.OPEN)
        valid_cells = known_targets & masks[:, open_channel, :, :]
        if not bool(valid_cells.any()):
            return logits.new_tensor(0.0)

        safe_targets = 1.0 - mine_targets
        open_logits = logits[:, open_channel, :, :]
        return F.binary_cross_entropy_with_logits(open_logits[valid_cells], safe_targets[valid_cells])

    def _risk_supervision_loss(
        self,
        logits: torch.Tensor,
        masks: torch.Tensor,
        transitions: list[EpisodeTransition],
    ) -> torch.Tensor:
        if self.config.risk_supervision_coef <= 0.0:
            return logits.new_tensor(0.0)

        open_channel = action_channel(ActionType.OPEN)
        cells = self.config.rows * self.config.cols
        expert_actions = torch.tensor(
            [transition.expert_action_index if transition.expert_action_index is not None else transition.action_index for transition in transitions],
            dtype=torch.long,
            device=self.device,
        )
        expert_kinds = expert_actions // cells
        risk_maps = torch.tensor(
            np.stack(
                [
                    np.zeros((self.config.rows, self.config.cols), dtype=np.float32)
                    if transition.risk_map is None
                    else transition.risk_map.astype(np.float32)
                    for transition in transitions
                ]
            ),
            dtype=torch.float32,
            device=self.device,
        )
        risk_valid = torch.tensor(
            [transition.risk_map is not None for transition in transitions],
            dtype=torch.bool,
            device=self.device,
        )
        guess_mask = risk_valid & (expert_kinds == open_channel)
        if not bool(guess_mask.any()):
            return logits.new_tensor(0.0)

        open_logits = logits[:, open_channel, :, :].view(logits.shape[0], -1)
        open_masks = masks[:, open_channel, :, :].view(masks.shape[0], -1)
        selected_logits = open_logits[guess_mask].masked_fill(~open_masks[guess_mask], -1e9)
        selected_risk = risk_maps[guess_mask].view(risk_maps[guess_mask].shape[0], -1).masked_fill(
            ~open_masks[guess_mask],
            float("inf"),
        )
        target = torch.softmax(-selected_risk / max(self.config.risk_temperature, 1e-6), dim=1)
        log_probs = F.log_softmax(selected_logits, dim=1)
        return -(target * log_probs).sum(dim=1).mean()

    def _sample_transitions(
        self,
        transitions: list[EpisodeTransition],
        batch_size: int,
    ) -> list[EpisodeTransition]:
        if batch_size <= 0 or len(transitions) <= batch_size:
            return list(transitions)
        indices = self.rng.choice(len(transitions), size=batch_size, replace=False)
        return [transitions[int(index)] for index in indices]

    def _augment_transitions(self, transitions: list[EpisodeTransition]) -> list[EpisodeTransition]:
        if not self.config.augment_flips or not transitions:
            return transitions
        augmented: list[EpisodeTransition] = []
        for transition in transitions:
            transform = int(self.rng.integers(0, 4))
            flip_vertical = bool(transform & 1)
            flip_horizontal = bool(transform & 2)
            if not flip_vertical and not flip_horizontal:
                augmented.append(transition)
                continue
            augmented.append(_flip_transition(transition, self.config.rows, self.config.cols, flip_vertical, flip_horizontal))
        return augmented

    def _decision_action_mask(self, action_mask: np.ndarray) -> np.ndarray:
        if self.config.decision_actions == "full":
            return action_mask
        mask = np.zeros_like(action_mask, dtype=bool)
        mask[action_channel(ActionType.OPEN)] = action_mask[action_channel(ActionType.OPEN)]
        return mask

    def _select_action(
        self,
        board: np.ndarray,
        global_features: np.ndarray,
        action_mask: np.ndarray,
        game: MinesweeperGame,
        deterministic: bool,
        mode: str,
        risk_weight: float,
    ) -> int:
        with torch.no_grad():
            mask_t = torch.tensor(action_mask.reshape(1, -1), dtype=torch.bool, device=self.device)
            use_flip_ensemble = deterministic and self.config.inference_augment_flips
            use_probability_ensemble = use_flip_ensemble and self.config.inference_ensemble == "probs" and mode != "hybrid"
            if use_probability_ensemble:
                masked_scores = self._predict_policy_scores_batch(
                    boards=np.stack([board]),
                    global_features_batch=np.stack([global_features]),
                    action_masks=np.stack([action_mask]),
                    use_flip_ensemble=True,
                )
            else:
                logits = self._predict_policy_logits(board, global_features, use_flip_ensemble=use_flip_ensemble)
                if mode == "hybrid":
                    logits = self._apply_solver_risk_prior(logits, game, risk_weight)
                flat_logits = logits.view(1, -1)
                masked_scores = flat_logits.masked_fill(~mask_t, -1e9)
            if deterministic:
                return int(masked_scores.argmax(dim=1).item())
            dist = Categorical(logits=self._exploration_logits(masked_scores))
            return int(dist.sample().item())

    def _predict_policy_scores_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        action_masks: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        if not use_flip_ensemble or self.config.inference_ensemble == "logits":
            logits = self._predict_policy_logits_batch(
                boards=boards,
                global_features_batch=global_features_batch,
                use_flip_ensemble=use_flip_ensemble,
            )
            flat_masks = torch.tensor(action_masks.reshape(boards.shape[0], -1), dtype=torch.bool, device=self.device)
            return logits.view(boards.shape[0], -1).masked_fill(~flat_masks, -1e9)

        parts = self._predict_flip_policy_logits_parts(boards=boards, global_features_batch=global_features_batch)
        flat_masks = torch.tensor(action_masks.reshape(boards.shape[0], -1), dtype=torch.bool, device=self.device)
        probabilities = [
            F.softmax(part.view(boards.shape[0], -1).masked_fill(~flat_masks, -1e9), dim=1)
            for part in parts
        ]
        return torch.stack(probabilities, dim=0).mean(dim=0).masked_fill(~flat_masks, -1.0)

    def _predict_flip_policy_logits_parts(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
    ) -> list[torch.Tensor]:
        transforms = [(False, False), (True, False), (False, True), (True, True)]
        transformed_boards = np.concatenate(
            [
                np.stack([_flip_board(board, self.config.rows, self.config.cols, flip_vertical, flip_horizontal) for board in boards])
                for flip_vertical, flip_horizontal in transforms
            ],
            axis=0,
        )
        transformed_globals = np.concatenate([global_features_batch for _ in transforms], axis=0)
        board_t = torch.tensor(transformed_boards, dtype=torch.float32, device=self.device)
        global_t = torch.tensor(transformed_globals, dtype=torch.float32, device=self.device)
        logits, _ = self.model(board_t, global_t)
        logits_by_transform = logits.view(len(transforms), boards.shape[0], logits.shape[1], logits.shape[2], logits.shape[3])
        return [
            _unflip_policy_logits(logits_by_transform[index], flip_vertical, flip_horizontal)
            for index, (flip_vertical, flip_horizontal) in enumerate(transforms)
        ]

    def _predict_policy_logits(
        self,
        board: np.ndarray,
        global_features: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        return self._predict_policy_logits_batch(
            boards=np.stack([board]),
            global_features_batch=np.stack([global_features]),
            use_flip_ensemble=use_flip_ensemble,
        )

    def _predict_policy_logits_batch(
        self,
        boards: np.ndarray,
        global_features_batch: np.ndarray,
        use_flip_ensemble: bool,
    ) -> torch.Tensor:
        if not use_flip_ensemble:
            board_t = torch.tensor(boards, dtype=torch.float32, device=self.device)
            global_t = torch.tensor(global_features_batch, dtype=torch.float32, device=self.device)
            logits, _ = self.model(board_t, global_t)
            return logits

        transforms = [(False, False), (True, False), (False, True), (True, True)]
        logits_by_transform = self._predict_flip_policy_logits_parts(boards=boards, global_features_batch=global_features_batch)
        unflipped = [
            logits_by_transform[index]
            for index, _ in enumerate(transforms)
        ]
        return torch.stack(unflipped, dim=0).mean(dim=0)

    def _exploration_logits(
        self,
        masked_logits: torch.Tensor,
        actions: torch.Tensor | None = None,
    ) -> torch.Tensor:
        logits = masked_logits
        topk = int(self.config.exploration_topk)
        if topk > 0 and topk < logits.shape[1]:
            _, top_indices = torch.topk(logits, k=topk, dim=1)
            top_mask = torch.zeros_like(logits, dtype=torch.bool)
            top_mask.scatter_(1, top_indices, True)
            if actions is not None:
                top_mask.scatter_(1, actions.view(-1, 1), True)
            logits = logits.masked_fill(~top_mask, -1e9)

        temperature = max(float(self.config.exploration_temperature), 1e-6)
        return logits / temperature

    def _apply_solver_risk_prior(
        self,
        logits: torch.Tensor,
        game: MinesweeperGame,
        risk_weight: float,
    ) -> torch.Tensor:
        if risk_weight == 0.0 or not game.mines_placed:
            return logits
        adjusted = logits.clone()
        snapshot = self.solver.analyze(game)
        risk = torch.tensor(snapshot.risk_map, dtype=torch.float32, device=self.device)
        adjusted[:, action_channel(ActionType.OPEN), :, :] -= risk_weight * risk
        adjusted[:, action_channel(ActionType.FLAG), :, :] += risk_weight * risk
        return adjusted

    def _select_solver_guess(self, snapshot: SolverSnapshot, action_mask: np.ndarray) -> int:
        if snapshot.best_guess_index is not None and action_mask.ravel()[snapshot.best_guess_index]:
            return int(snapshot.best_guess_index)
        return self._solver_guess_from_risk(snapshot, action_mask)

    def _solver_expert_action(self, game: MinesweeperGame, action_mask: np.ndarray) -> int | None:
        if not game.mines_placed:
            row, col = game.rows // 2, game.cols // 2
            action = Action(ActionType.OPEN, row, col)
            index = action_to_index(action, game.rows, game.cols)
            return index if action_mask.reshape(-1)[index] else None

        snapshot = self.solver.analyze(game)
        expert_mask = self._solver_expert_action_mask(game, action_mask, snapshot)
        if expert_mask is None or not expert_mask.any():
            return None
        for kind in (ActionType.UNFLAG, ActionType.FLAG, ActionType.OPEN, ActionType.CHORD):
            kind_mask = expert_mask[action_channel(kind)]
            if kind_mask.any():
                if kind == ActionType.OPEN:
                    flat_index = self._select_open_index_from_snapshot(snapshot, kind_mask)
                else:
                    flat_index = int(np.flatnonzero(kind_mask.reshape(-1))[0])
                return action_channel(kind) * game.rows * game.cols + flat_index
        return None

    def _solver_expert_action_mask(
        self,
        game: MinesweeperGame,
        action_mask: np.ndarray,
        snapshot: SolverSnapshot | None = None,
    ) -> np.ndarray | None:
        if not game.mines_placed:
            row, col = game.rows // 2, game.cols // 2
            action = Action(ActionType.OPEN, row, col)
            index = action_to_index(action, game.rows, game.cols)
            return _single_action_mask(action_mask, index) if action_mask.reshape(-1)[index] else None

        if snapshot is None:
            snapshot = self.solver.analyze(game)
        expert_mask = np.zeros_like(action_mask, dtype=bool)

        wrong_flags = game.flagged & ~game.mines & action_mask[action_channel(ActionType.UNFLAG)]
        if self.config.decision_actions == "full":
            expert_mask[action_channel(ActionType.UNFLAG)] = wrong_flags

            mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
            expert_mask[action_channel(ActionType.FLAG)] = mine_cells

        safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
        expert_mask[action_channel(ActionType.OPEN)] = safe_cells

        if expert_mask.any():
            return expert_mask

        open_mask = action_mask[action_channel(ActionType.OPEN)]
        if not open_mask.any():
            return None
        guess_mask = self._topk_guess_open_mask(snapshot, open_mask)
        if guess_mask is None:
            return None
        expert_mask[action_channel(ActionType.OPEN)] = guess_mask
        return expert_mask

    def _solver_expert_action_from_snapshot(
        self,
        game: MinesweeperGame,
        snapshot: SolverSnapshot,
        action_mask: np.ndarray,
    ) -> int | None:
        mine_cells = snapshot.mine_mask & action_mask[action_channel(ActionType.FLAG)]
        for row, col in zip(*np.where(mine_cells)):
            return action_to_index(Action(ActionType.FLAG, int(row), int(col)), game.rows, game.cols)

        safe_cells = snapshot.safe_mask & action_mask[action_channel(ActionType.OPEN)]
        if safe_cells.any():
            open_index = self._select_open_index_from_snapshot(snapshot, safe_cells)
            row, col = divmod(open_index, game.cols)
            return action_to_index(Action(ActionType.OPEN, row, col), game.rows, game.cols)

        open_mask = action_mask[action_channel(ActionType.OPEN)]
        if not open_mask.any():
            return None
        open_index = self._select_open_index_from_snapshot(snapshot, open_mask)
        row, col = divmod(open_index, game.cols)
        return action_to_index(Action(ActionType.OPEN, row, col), game.rows, game.cols)

    def _select_open_index_from_snapshot(self, snapshot: SolverSnapshot, open_mask: np.ndarray) -> int:
        safe_mask = snapshot.safe_mask & open_mask
        if safe_mask.any():
            frontier = snapshot.frontier_degree_map.copy()
            frontier[~safe_mask] = -np.inf
            return int(np.argmax(frontier))

        open_flat = open_mask.reshape(-1)
        if snapshot.best_guess_index is not None and open_flat[snapshot.best_guess_index]:
            return int(snapshot.best_guess_index)
        return self._solver_guess_from_risk(snapshot, open_mask)

    def _topk_guess_open_mask(self, snapshot: SolverSnapshot, open_mask: np.ndarray) -> np.ndarray | None:
        open_flat = open_mask.reshape(-1)
        candidate_indices = np.flatnonzero(open_flat)
        if candidate_indices.size == 0:
            return None

        risk_flat = snapshot.risk_map.reshape(-1)
        topk = max(1, int(self.config.guess_supervision_topk))
        k = min(topk, candidate_indices.size)
        ranked_candidates = candidate_indices[np.argpartition(risk_flat[candidate_indices], k - 1)[:k]]
        ranked_candidates = ranked_candidates[np.argsort(risk_flat[ranked_candidates], kind="stable")]
        if snapshot.best_guess_index is not None and open_flat[snapshot.best_guess_index]:
            ranked_candidates = np.unique(np.concatenate([ranked_candidates, np.array([snapshot.best_guess_index])]))

        guess_mask = np.zeros_like(open_mask, dtype=bool)
        guess_mask.reshape(-1)[ranked_candidates] = True
        return guess_mask

    def _solver_guess_from_risk(self, snapshot: SolverSnapshot, action_mask: np.ndarray) -> int:
        risk = snapshot.risk_map.copy()
        risk[~action_mask] = np.inf
        return int(np.argmin(risk))

    def _discounted_returns(self, rewards: list[float]) -> torch.Tensor:
        returns: list[float] = []
        running = 0.0
        for reward in reversed(rewards):
            running = reward + self.config.gamma * running
            returns.append(running)
        returns.reverse()
        return torch.tensor(returns, dtype=torch.float32)


def evaluate_policy(
    trainer: MinesweeperTrainer,
    games: int = 100,
    seed: int = 0,
    mode: str = "rl",
    risk_weight: float = 0.0,
) -> dict[str, float | int | bool | None]:
    summaries: list[EpisodeSummary] = []
    for i in range(games):
        _, summary = trainer.collect_episode(
            seed=seed + i,
            deterministic=True,
            mode=mode,
            risk_weight=risk_weight,
        )
        summaries.append(summary)

    return _metrics_from_summaries(
        summaries=summaries,
        games=games,
        seed=seed,
        mode=mode,
        decision_actions=trainer.config.decision_actions,
        inference_augment_flips=trainer.config.inference_augment_flips,
        inference_ensemble=trainer.config.inference_ensemble,
    )


def evaluate_policy_batched(
    trainer: MinesweeperTrainer,
    games: int = 100,
    seed: int = 0,
    mode: str = "rl",
    risk_weight: float = 0.0,
    batch_size: int = 64,
) -> dict[str, float | int | bool | None]:
    if mode not in {"rl", "policy"} or risk_weight != 0.0:
        return evaluate_policy(trainer, games=games, seed=seed, mode=mode, risk_weight=risk_weight)

    summaries: list[EpisodeSummary] = []
    batch_size = max(1, int(batch_size))

    for start in range(0, games, batch_size):
        current_batch = min(batch_size, games - start)
        game_batch = [make_game(trainer.config, seed=seed + start + offset) for offset in range(current_batch)]
        rewards = [0.0 for _ in range(current_batch)]
        agent_steps = [0 for _ in range(current_batch)]
        active = set(range(current_batch))

        while active:
            ready: list[int] = []
            boards: list[np.ndarray] = []
            global_features: list[np.ndarray] = []
            action_masks: list[np.ndarray] = []

            for index in list(active):
                game = game_batch[index]
                if game.done or agent_steps[index] >= trainer.config.max_steps:
                    active.remove(index)
                    continue

                board, global_feature, action_mask = encode_state(game)
                action_mask = trainer._decision_action_mask(action_mask)
                if not action_mask.any():
                    active.remove(index)
                    continue

                ready.append(index)
                boards.append(board)
                global_features.append(global_feature)
                action_masks.append(action_mask)

            if not ready:
                continue

            scores = trainer._predict_policy_scores_batch(
                boards=np.stack(boards),
                global_features_batch=np.stack(global_features),
                action_masks=np.stack(action_masks),
                use_flip_ensemble=trainer.config.inference_augment_flips,
            )
            action_indices = scores.argmax(dim=1).detach().cpu().numpy()

            for local_index, game_index in enumerate(ready):
                game = game_batch[game_index]
                action = decode_action_index(int(action_indices[local_index]), trainer.config.rows, trainer.config.cols)
                _, reward, _, _ = game.step(action)
                rewards[game_index] += float(reward)
                agent_steps[game_index] += 1
                if game.done or agent_steps[game_index] >= trainer.config.max_steps:
                    active.discard(game_index)

        for offset, game in enumerate(game_batch):
            summaries.append(
                trainer._episode_summary(
                    game=game,
                    seed=seed + start + offset,
                    total_reward=rewards[offset],
                    guess_steps=agent_steps[offset],
                    forced_steps=0,
                )
            )

    return _metrics_from_summaries(
        summaries=summaries,
        games=games,
        seed=seed,
        mode=mode,
        decision_actions=trainer.config.decision_actions,
        inference_augment_flips=trainer.config.inference_augment_flips,
        inference_ensemble=trainer.config.inference_ensemble,
    )


def _metrics_from_summaries(
    summaries: list[EpisodeSummary],
    games: int,
    seed: int,
    mode: str,
    decision_actions: str,
    inference_augment_flips: bool,
    inference_ensemble: str,
) -> dict[str, float | int | bool | None]:
    wins = [summary.won for summary in summaries]
    longest_streak = 0
    current = 0
    current_start = 0
    longest_start: int | None = None
    longest_end: int | None = None
    for index, won in enumerate(wins):
        if won:
            if current == 0:
                current_start = index
            current += 1
            if current > longest_streak:
                longest_streak = current
                longest_start = current_start
                longest_end = index
        else:
            current = 0

    avg_agent_steps = float(mean(summary.guess_steps for summary in summaries) if summaries else 0.0)
    return {
        "mode": mode,
        "decision_actions": decision_actions,
        "inference_augment_flips": inference_augment_flips,
        "inference_ensemble": inference_ensemble,
        "solver_decision": mode in {"solver", "hybrid"},
        "games": float(games),
        "win_rate": float(mean(wins) if wins else 0.0),
        "longest_streak": float(longest_streak),
        "longest_streak_start_index": longest_start,
        "longest_streak_end_index": longest_end,
        "longest_streak_start_seed": None if longest_start is None else seed + longest_start,
        "longest_streak_end_seed": None if longest_end is None else seed + longest_end,
        "avg_reward": float(mean(summary.reward for summary in summaries) if summaries else 0.0),
        "avg_agent_steps": avg_agent_steps,
        "avg_guess_steps": avg_agent_steps,
        "avg_forced_steps": float(mean(summary.forced_steps for summary in summaries) if summaries else 0.0),
        "avg_revealed_safe_cells": float(mean(summary.revealed_safe_cells for summary in summaries) if summaries else 0.0),
    }


def save_checkpoint(
    path: str | Path,
    trainer: MinesweeperTrainer,
    episode: int,
    metrics: dict[str, float | int | bool | None],
) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "episode": episode,
            "config": asdict(trainer.config),
            "model": trainer.model.state_dict(),
            "optimizer": trainer.optimizer.state_dict(),
            "metrics": metrics,
        },
        path,
    )
    metrics_path = path.with_suffix(".json")
    metrics_path.write_text(json.dumps({"episode": episode, "metrics": metrics}, indent=2), encoding="utf-8")


def load_checkpoint(
    path: str | Path,
    device: str = "cpu",
    config_overrides: dict[str, object] | None = None,
) -> MinesweeperTrainer:
    checkpoint = torch.load(path, map_location=device)
    known_fields = {field.name for field in fields(TrainingConfig)}
    config_values = {key: value for key, value in checkpoint["config"].items() if key in known_fields}
    if config_overrides:
        config_values.update({key: value for key, value in config_overrides.items() if key in known_fields})
    config = TrainingConfig(**config_values)
    config.device = device
    trainer = MinesweeperTrainer(config)
    partial_model_load = False
    try:
        trainer.model.load_state_dict(checkpoint["model"])
    except RuntimeError as exc:
        try:
            partial_model_load = _load_compatible_model_state(trainer.model, checkpoint["model"])
        except RuntimeError as fallback_exc:
            raise RuntimeError(
                "Checkpoint is incompatible with the current action-space model. "
                "Train a new checkpoint with `python -m minesweeper_rl.cli train`."
            ) from fallback_exc
    if "optimizer" in checkpoint and not partial_model_load:
        trainer.optimizer.load_state_dict(checkpoint["optimizer"])
    return trainer


def _load_compatible_model_state(model: nn.Module, state_dict: dict[str, torch.Tensor]) -> bool:
    current_state = model.state_dict()
    loaded_state: dict[str, torch.Tensor] = {}
    partial = False

    for key, current_tensor in current_state.items():
        checkpoint_tensor = state_dict.get(key)
        if checkpoint_tensor is None:
            partial = True
            continue
        if checkpoint_tensor.shape == current_tensor.shape:
            loaded_state[key] = checkpoint_tensor
            continue
        if checkpoint_tensor.ndim != current_tensor.ndim:
            raise RuntimeError(
                f"incompatible tensor shape for {key}: checkpoint {tuple(checkpoint_tensor.shape)} vs current {tuple(current_tensor.shape)}"
            )

        merged = current_tensor.clone()
        slices = tuple(slice(0, min(checkpoint_dim, current_dim)) for checkpoint_dim, current_dim in zip(checkpoint_tensor.shape, current_tensor.shape))
        merged[slices] = checkpoint_tensor[slices]
        loaded_state[key] = merged
        partial = True

    model.load_state_dict(loaded_state, strict=False)
    return partial


def _single_action_mask(action_mask: np.ndarray, action_index: int) -> np.ndarray:
    expert_mask = np.zeros_like(action_mask, dtype=bool)
    flat = expert_mask.reshape(-1)
    if action_mask.reshape(-1)[action_index]:
        flat[action_index] = True
    return expert_mask


def _expert_mask_from_indices(action_mask: np.ndarray, action_indices: list[int]) -> np.ndarray:
    expert_mask = np.zeros_like(action_mask, dtype=bool)
    flat_mask = action_mask.reshape(-1)
    flat_expert = expert_mask.reshape(-1)
    for action_index in action_indices:
        if flat_mask[action_index]:
            flat_expert[action_index] = True
    return expert_mask


def _transition_expert_mask(transition: EpisodeTransition, allow_action_fallback: bool) -> np.ndarray:
    if transition.expert_action_mask is not None and transition.expert_action_mask.any():
        return transition.expert_action_mask & transition.action_mask
    if allow_action_fallback:
        action_index = transition.action_index if transition.expert_action_index is None else transition.expert_action_index
        return _single_action_mask(transition.action_mask, action_index)
    if transition.expert_action_index is None:
        return np.zeros_like(transition.action_mask, dtype=bool)
    return _single_action_mask(transition.action_mask, transition.expert_action_index)


def _flip_transition(
    transition: EpisodeTransition,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> EpisodeTransition:
    return EpisodeTransition(
        board=_flip_board(transition.board, rows, cols, flip_vertical, flip_horizontal),
        global_features=transition.global_features.copy(),
        action_mask=_flip_spatial(transition.action_mask, flip_vertical, flip_horizontal),
        action_index=_flip_action_index(transition.action_index, rows, cols, flip_vertical, flip_horizontal),
        expert_action_index=None
        if transition.expert_action_index is None
        else _flip_action_index(transition.expert_action_index, rows, cols, flip_vertical, flip_horizontal),
        reward=transition.reward,
        done=transition.done,
        expert_action_mask=None
        if transition.expert_action_mask is None
        else _flip_spatial(transition.expert_action_mask, flip_vertical, flip_horizontal),
        mine_mask=None
        if transition.mine_mask is None
        else _flip_spatial(transition.mine_mask, flip_vertical, flip_horizontal),
        risk_map=None
        if transition.risk_map is None
        else _flip_spatial(transition.risk_map, flip_vertical, flip_horizontal),
    )


def _flip_board(
    board: np.ndarray,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> np.ndarray:
    flipped = _flip_spatial(board, flip_vertical, flip_horizontal)
    coord_layers = coordinate_channels(rows, cols)
    coord_end = COORDINATE_CHANNEL_START + len(coord_layers)
    if flipped.shape[0] < coord_end:
        return flipped

    flipped = flipped.copy()
    for offset, channel in enumerate(coord_layers):
        flipped[COORDINATE_CHANNEL_START + offset] = channel
    return flipped


def _flip_spatial(array: np.ndarray, flip_vertical: bool, flip_horizontal: bool) -> np.ndarray:
    flipped = array
    if flip_vertical:
        flipped = np.flip(flipped, axis=-2)
    if flip_horizontal:
        flipped = np.flip(flipped, axis=-1)
    return np.ascontiguousarray(flipped)


def _unflip_policy_logits(
    logits: torch.Tensor,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> torch.Tensor:
    unflipped = logits
    if flip_vertical:
        unflipped = torch.flip(unflipped, dims=(-2,))
    if flip_horizontal:
        unflipped = torch.flip(unflipped, dims=(-1,))
    return unflipped


def _flip_action_index(
    action_index: int,
    rows: int,
    cols: int,
    flip_vertical: bool,
    flip_horizontal: bool,
) -> int:
    action = decode_action_index(action_index, rows, cols)
    row = rows - 1 - action.row if flip_vertical else action.row
    col = cols - 1 - action.col if flip_horizontal else action.col
    return action_to_index(Action(action.kind, row, col), rows, cols)
