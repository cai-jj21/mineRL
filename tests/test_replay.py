from __future__ import annotations

from minesweeper_rl.replay import load_trace, run_episode_trace, save_trace
from minesweeper_rl.trainer import MinesweeperTrainer, TrainingConfig


def test_run_episode_trace_records_a_winning_history() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(seed=100))
    trace = run_episode_trace(trainer, seed=100, mode="solver")

    assert trace["summary"]["won"] is True
    assert trace["frames"]
    assert trace["frames"][0]["event"] == "first_open"
    assert any(frame["event"] == "agent_guess" for frame in trace["frames"])
    assert trace["frames"][-1]["summary"]["done"] is True


def test_run_episode_trace_rl_uses_only_agent_actions() -> None:
    trainer = MinesweeperTrainer(TrainingConfig(seed=100))
    trace = run_episode_trace(trainer, seed=100, mode="rl", deterministic=True)

    assert trace["frames"]
    assert trace["frames"][0]["event"].startswith("agent_")
    assert not any(frame["event"].startswith("forced_") for frame in trace["frames"])
    assert trace["summary"]["forced_steps"] == 0


def test_trace_roundtrip(tmp_path) -> None:
    trainer = MinesweeperTrainer(TrainingConfig(seed=100))
    trace = run_episode_trace(trainer, seed=100, mode="solver")

    path = tmp_path / "replay.json"
    save_trace(path, trace)
    loaded = load_trace(path)

    assert loaded["seed"] == trace["seed"]
    assert loaded["summary"]["won"] == trace["summary"]["won"]
    assert len(loaded["frames"]) == len(trace["frames"])
