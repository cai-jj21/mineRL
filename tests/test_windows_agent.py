from __future__ import annotations

import importlib.util
import json
import sys
from collections import Counter
from argparse import Namespace
from types import SimpleNamespace
from pathlib import Path

import numpy as np
from PIL import Image
import torch


ROOT = Path(__file__).resolve().parents[1]
MODULE_PATH = ROOT / "scripts" / "windows_minesweeper_agent.py"
SPEC = importlib.util.spec_from_file_location("windows_minesweeper_agent_test", MODULE_PATH)
assert SPEC is not None and SPEC.loader is not None
windows_agent = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = windows_agent
SPEC.loader.exec_module(windows_agent)


def test_default_desktop_decision_path_keeps_basic_filter_off() -> None:
    assert windows_agent.BASE_DEFAULTS["basic_safety_filter"] == "none"
    assert windows_agent.BASE_DEFAULTS["solver_safety_filter"] == "none"
    assert not windows_agent.BASE_DEFAULTS["audit_solver"]
    assert not windows_agent.BASE_DEFAULTS["audit_basic"]
    assert windows_agent.BASE_DEFAULTS["center_first_open"] is True
    assert windows_agent.BASE_DEFAULTS["risk_head_weight"] == 0.0


def test_load_configured_trainer_applies_risk_head_weight(monkeypatch) -> None:
    captured = {}

    class FakeModel:
        def eval(self):
            captured["model_eval"] = True

    fake_trainer = SimpleNamespace(
        config=SimpleNamespace(),
        model=FakeModel(),
    )

    def fake_load_checkpoint(checkpoint, device):
        captured["checkpoint"] = checkpoint
        captured["device"] = device
        return fake_trainer

    monkeypatch.setattr(windows_agent, "load_checkpoint", fake_load_checkpoint)
    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        device="cuda",
        max_steps=600,
        flag_mode="memory",
        inference_flips=True,
        inference_ensemble="probs",
        risk_head_weight=0.35,
    )

    trainer = windows_agent.load_configured_trainer(args)

    assert trainer is fake_trainer
    assert captured["checkpoint"] == Path("artifacts/full_rlmix_20.pt")
    assert captured["device"] == "cuda"
    assert captured["model_eval"] is True
    assert trainer.config.rows == windows_agent.ROWS
    assert trainer.config.cols == windows_agent.COLS
    assert trainer.config.mines == windows_agent.MINES
    assert trainer.config.safe_radius == 1
    assert trainer.config.max_steps == 600
    assert trainer.config.decision_actions == "full"
    assert trainer.config.inference_augment_flips is True
    assert trainer.config.inference_ensemble == "probs"
    assert trainer.config.risk_head_weight == 0.35


def test_load_configured_trainer_builds_checkpoint_ensemble(monkeypatch) -> None:
    loaded: list[Path] = []

    class FakeModel:
        def eval(self):
            pass

    def fake_load_checkpoint(checkpoint, device):
        loaded.append(Path(checkpoint))
        return SimpleNamespace(config=SimpleNamespace(), model=FakeModel())

    monkeypatch.setattr(windows_agent, "load_checkpoint", fake_load_checkpoint)
    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        ensemble_checkpoints=[Path("artifacts/full_rlmix_20_refine.pt")],
        device="cuda",
        max_steps=600,
        flag_mode="memory",
        inference_flips=True,
        inference_ensemble="probs",
        risk_head_weight=0.0,
    )

    trainer = windows_agent.load_configured_trainer(args)

    assert isinstance(trainer, windows_agent.DesktopPolicyEnsemble)
    assert loaded == [
        Path("artifacts/full_rlmix_20.pt"),
        Path("artifacts/full_rlmix_20_refine.pt"),
    ]
    assert len(trainer.trainers) == 2
    assert trainer.checkpoints == loaded


def test_desktop_policy_ensemble_averages_scores() -> None:
    first_action = windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0)
    second_action = windows_agent.Action(windows_agent.ActionType.OPEN, 0, 1)
    first_index = windows_agent.action_to_index(first_action, windows_agent.ROWS, windows_agent.COLS)
    second_index = windows_agent.action_to_index(second_action, windows_agent.ROWS, windows_agent.COLS)

    class FakeTrainer:
        def __init__(self, first_score: float, second_score: float) -> None:
            self.config = SimpleNamespace(inference_augment_flips=False)
            self.model = SimpleNamespace()
            self.first_score = first_score
            self.second_score = second_score

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, **kwargs):
            return first_index

        def _predict_policy_scores_batch(self, **kwargs):
            scores = torch.full((1, 4 * windows_agent.ROWS * windows_agent.COLS), -1e9)
            scores[0, first_index] = self.first_score
            scores[0, second_index] = self.second_score
            return scores

    ensemble = windows_agent.DesktopPolicyEnsemble(
        trainers=[
            FakeTrainer(first_score=4.0, second_score=1.0),
            FakeTrainer(first_score=0.0, second_score=8.0),
        ],
        checkpoints=[Path("a.pt"), Path("b.pt")],
    )
    action_mask = np.zeros((4, windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    action_mask[windows_agent.action_channel(windows_agent.ActionType.OPEN), 0, 0] = True
    action_mask[windows_agent.action_channel(windows_agent.ActionType.OPEN), 0, 1] = True

    selected = ensemble._select_action(
        board=np.zeros((1, windows_agent.ROWS, windows_agent.COLS), dtype=np.float32),
        global_features=np.zeros((1,), dtype=np.float32),
        action_mask=action_mask,
        game=SimpleNamespace(),
        deterministic=True,
        mode="rl",
        risk_weight=0.0,
    )

    assert selected == second_index


def test_choose_initial_open_action_uses_center_only_on_empty_board() -> None:
    empty_board = SimpleNamespace(revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool))
    non_empty_board = SimpleNamespace(revealed=np.pad(np.array([[True]], dtype=bool), ((0, windows_agent.ROWS - 1), (0, windows_agent.COLS - 1))))

    action = windows_agent.choose_initial_open_action(empty_board, center_first_open=True)
    assert action is not None
    assert action.kind == windows_agent.ActionType.OPEN
    assert (action.row, action.col) == (windows_agent.ROWS // 2, windows_agent.COLS // 2)

    assert windows_agent.choose_initial_open_action(empty_board, center_first_open=False) is None
    assert windows_agent.choose_initial_open_action(non_empty_board, center_first_open=True) is None


def test_solver_assist_prefers_forced_safe_open_and_keeps_mines_virtual() -> None:
    shape = (windows_agent.ROWS, windows_agent.COLS)
    revealed = np.zeros(shape, dtype=bool)
    revealed[0, 0] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros(shape, dtype=bool),
        adjacent=np.zeros(shape, dtype=np.int8),
        mine_like=np.zeros(shape, dtype=bool),
        grid=windows_agent.Grid(x_lines=list(range(windows_agent.COLS + 1)), y_lines=list(range(windows_agent.ROWS + 1))),
        screenshot=None,
    )
    safe_mask = np.zeros(shape, dtype=bool)
    safe_mask[0, 1] = True
    safe_mask[1, 1] = True
    mine_mask = np.zeros(shape, dtype=bool)
    mine_mask[1, 0] = True
    frontier_degree = np.zeros(shape, dtype=np.float32)
    frontier_degree[0, 1] = 1.0
    frontier_degree[1, 1] = 3.0

    class FakeSolver:
        def analyze(self, game):
            return SimpleNamespace(
                safe_mask=safe_mask,
                mine_mask=mine_mask,
                frontier_degree_map=frontier_degree,
                best_guess=(0, 1),
                best_guess_risk=0.2,
            )

    action, virtual_mines, record = windows_agent.select_solver_assist_action(
        FakeSolver(),
        board,
        forbidden_open_cells=set(),
        use_memory_flags=True,
    )

    assert action == windows_agent.Action(windows_agent.ActionType.OPEN, 1, 1)
    assert virtual_mines == [(1, 0)]
    assert record["decision"] == "forced_safe_open"
    assert record["applied"]


def test_solver_assist_falls_back_to_rl_when_no_forced_move() -> None:
    shape = (windows_agent.ROWS, windows_agent.COLS)
    revealed = np.zeros(shape, dtype=bool)
    revealed[0, 0] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros(shape, dtype=bool),
        adjacent=np.zeros(shape, dtype=np.int8),
        mine_like=np.zeros(shape, dtype=bool),
        grid=windows_agent.Grid(x_lines=list(range(windows_agent.COLS + 1)), y_lines=list(range(windows_agent.ROWS + 1))),
        screenshot=None,
    )

    class FakeSolver:
        def analyze(self, game):
            return SimpleNamespace(
                safe_mask=np.zeros(shape, dtype=bool),
                mine_mask=np.zeros(shape, dtype=bool),
                frontier_degree_map=np.zeros(shape, dtype=np.float32),
                best_guess=(0, 1),
                best_guess_risk=0.5,
            )

    action, virtual_mines, record = windows_agent.select_solver_assist_action(
        FakeSolver(),
        board,
        forbidden_open_cells=set(),
        use_memory_flags=True,
    )

    assert action is None
    assert virtual_mines == []
    assert record["decision"] == "rl_fallback"
    assert not record["applied"]


def test_restore_revealed_cells_keeps_prior_open_cells_visible() -> None:
    prev_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    prev_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    prev_revealed[0, 2] = True
    prev_adjacent[0, 2] = 0
    previous = windows_agent.ScreenBoard(
        revealed=prev_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=prev_adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    current = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    restored = windows_agent.restore_revealed_cells(current, previous)

    assert restored.revealed[0, 2]
    assert restored.adjacent[0, 2] == 0
    assert restored.read_restores == 1


def test_repair_impossible_zero_reveals_hides_zero_next_to_hidden_cell() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    flagged = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    mine_like = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[5, 5] = True

    repairs = windows_agent.repair_impossible_zero_reveals(revealed, flagged, adjacent, mine_like)

    assert repairs == 1
    assert not revealed[5, 5]


def test_repair_impossible_zero_reveals_keeps_consistent_zero_region() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    flagged = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    mine_like = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[4:7, 4:7] = True
    adjacent[4:7, 4:7] = 1
    adjacent[5, 5] = 0

    repairs = windows_agent.repair_impossible_zero_reveals(revealed, flagged, adjacent, mine_like)

    assert repairs == 0
    assert revealed[5, 5]


def test_read_stable_board_uses_accurate_recovery_for_restored_fast_read(monkeypatch) -> None:
    previous_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    previous_revealed[0, 0] = True
    previous = windows_agent.ScreenBoard(
        revealed=previous_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    fast_hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    accurate_revealed = previous_revealed.copy()
    accurate_revealed[0, 1] = True
    accurate = windows_agent.ScreenBoard(
        revealed=accurate_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    class Desktop:
        read_mode = "fast"

        def read_board(self, **kwargs):
            return fast_hidden

        def _read_board_accurate(self, **kwargs):
            return accurate

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board = windows_agent.read_stable_board(
        Desktop(),
        step_count=1,
        keep_screenshot=False,
        reads=1,
        previous_board=previous,
    )

    assert board.read_restores == 0
    assert board.read_recoveries == 1
    assert board.revealed[0, 0]
    assert board.revealed[0, 1]


def test_read_stable_board_prefers_cleaner_matching_signature(monkeypatch) -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    recovered = windows_agent.ScreenBoard(
        revealed=revealed.copy(),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_recoveries=1,
    )
    clean = windows_agent.ScreenBoard(
        revealed=revealed.copy(),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    reads = [recovered, clean]

    class Desktop:
        read_mode = "fast"

        def read_board(self, **kwargs):
            return reads.pop(0)

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board = windows_agent.read_stable_board(
        Desktop(),
        step_count=1,
        keep_screenshot=False,
        reads=2,
    )

    assert board.read_recoveries == 0
    assert board.revealed[0, 0]


def test_read_stable_board_returns_best_read_when_later_read_is_worse(monkeypatch) -> None:
    clean_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    repaired_revealed = clean_revealed.copy()
    repaired_revealed[0, 1] = True
    repaired = windows_agent.ScreenBoard(
        revealed=repaired_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    reads = [clean, repaired]

    class Desktop:
        read_mode = "fast"

        def read_board(self, **kwargs):
            return reads.pop(0)

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board = windows_agent.read_stable_board(
        Desktop(),
        step_count=1,
        keep_screenshot=False,
        reads=2,
    )

    assert board.read_repairs == 0
    assert int(board.revealed.sum()) == 1
    assert board.revealed[0, 0]


def test_repaired_fast_read_retries_until_accurate_board_is_clean(monkeypatch) -> None:
    shape = (windows_agent.ROWS, windows_agent.COLS)
    repaired = windows_agent.ScreenBoard(
        revealed=np.zeros(shape, dtype=bool),
        flagged=np.zeros(shape, dtype=bool),
        adjacent=np.zeros(shape, dtype=np.int8),
        mine_like=np.zeros(shape, dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=2,
    )
    clean_revealed = np.zeros(shape, dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros(shape, dtype=bool),
        adjacent=np.zeros(shape, dtype=np.int8),
        mine_like=np.zeros(shape, dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    sleeps: list[float] = []

    class Desktop:
        def __init__(self) -> None:
            self.reads = [repaired, clean]

        def dialog_is_open(self) -> bool:
            return False

        def _read_board_accurate(self, **kwargs):
            return self.reads.pop(0)

    monkeypatch.setattr(windows_agent.time, "sleep", sleeps.append)

    result = windows_agent.WindowsMinesweeper._accurate_fallback_for_repaired_fast_board(
        Desktop(),
        repaired,
        step_count=1,
        keep_screenshot=False,
    )

    assert result.read_repairs == 0
    assert result.read_recoveries == 1
    assert sleeps == [0.025, 0.05]


def test_read_stable_board_does_not_use_worse_read_as_next_baseline(monkeypatch) -> None:
    clean_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    repaired_revealed = clean_revealed.copy()
    repaired_revealed[0, 1] = True
    repaired = windows_agent.ScreenBoard(
        revealed=repaired_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    final_clean_revealed = clean_revealed.copy()
    final_clean_revealed[0, 2] = True
    final_clean = windows_agent.ScreenBoard(
        revealed=final_clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    reads = [clean, repaired, final_clean]
    previous_repairs: list[int | None] = []

    class Desktop:
        read_mode = "fast"

        def read_board(self, **kwargs):
            previous = kwargs.get("previous_board")
            previous_repairs.append(None if previous is None else previous.read_repairs)
            return reads.pop(0)

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board = windows_agent.read_stable_board(
        Desktop(),
        step_count=1,
        keep_screenshot=False,
        reads=3,
    )

    assert previous_repairs == [None, 0, 0]
    assert board.read_repairs == 0
    assert board.revealed[0, 2]


def test_read_stable_board_returns_terminal_read_immediately(monkeypatch) -> None:
    clean_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    lost_mines = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    lost_mines[0, 1] = True
    lost_revealed = clean_revealed | lost_mines
    lost = windows_agent.ScreenBoard(
        revealed=lost_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=lost_mines,
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    reads = [clean, lost]

    class Desktop:
        read_mode = "fast"

        def read_board(self, **kwargs):
            return reads.pop(0)

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board = windows_agent.read_stable_board(
        Desktop(),
        step_count=1,
        keep_screenshot=False,
        reads=2,
    )

    assert board.done
    assert board.lost
    assert board.mine_like[0, 1]


def test_classify_cell_fast_prefers_center_over_blue_border() -> None:
    open_zero = np.full((24, 24, 3), [166, 212, 247], dtype=np.uint8)
    open_zero[6:18, 6:18] = [138, 143, 147]
    cell = windows_agent.classify_cell_fast(open_zero)

    assert cell["kind"] == "revealed"
    assert cell["number"] == 0


def test_classify_cell_fast_keeps_blue_hidden_cells_hidden() -> None:
    hidden = np.full((24, 24, 3), [166, 212, 247], dtype=np.uint8)
    cell = windows_agent.classify_cell_fast(hidden)

    assert cell["kind"] == "hidden"


def test_classify_cell_fast_keeps_gradient_blue_hidden_cells_hidden() -> None:
    rows = np.linspace(220, 145, 80, dtype=np.uint8)[:, None]
    cols = np.linspace(18, -18, 80, dtype=np.int16)[None, :]
    blue = np.clip(rows.astype(np.int16) + cols, 0, 255).astype(np.uint8)
    hidden = np.zeros((80, 80, 3), dtype=np.uint8)
    hidden[:, :, 0] = np.clip(blue.astype(np.int16) - 86, 0, 255).astype(np.uint8)
    hidden[:, :, 1] = np.clip(blue.astype(np.int16) - 34, 0, 255).astype(np.uint8)
    hidden[:, :, 2] = blue

    assert windows_agent.classify_cell_fast(hidden) == {"kind": "hidden", "number": 0}


def test_classify_cell_keeps_dark_gradient_blue_tile_hidden() -> None:
    rows = np.linspace(228, 188, 80, dtype=np.int16)[:, None]
    cols = np.linspace(10, -10, 80, dtype=np.int16)[None, :]
    blue = np.clip(rows + cols, 0, 255)
    hidden = np.zeros((80, 80, 3), dtype=np.uint8)
    hidden[:, :, 0] = np.clip(blue - 138, 0, 255).astype(np.uint8)
    hidden[:, :, 1] = np.clip(blue - 108, 0, 255).astype(np.uint8)
    hidden[:, :, 2] = blue.astype(np.uint8)

    assert windows_agent.classify_cell(Image.fromarray(hidden))["kind"] == "hidden"
    assert windows_agent.classify_cell_fast(hidden) == {"kind": "hidden", "number": 0}


def test_crop_grid_array_uses_numpy_axis_order() -> None:
    array = np.arange(6 * 8 * 3, dtype=np.uint8).reshape((6, 8, 3))
    grid = windows_agent.Grid(x_lines=[2, 5], y_lines=[1, 4])

    crop = windows_agent.crop_grid_array(array, grid)

    np.testing.assert_array_equal(crop, array[1:5, 2:6])


def test_click_point_defaults_to_cell_center() -> None:
    desktop = windows_agent.WindowsMinesweeper.__new__(windows_agent.WindowsMinesweeper)
    desktop.grid = windows_agent.Grid(x_lines=[100, 200], y_lines=[50, 150])
    desktop.click_fraction = 0.5

    assert desktop.click_point(0, 0) == desktop.grid.center(0, 0) == (150, 100)


def test_dialog_button_matches_known_chinese_labels() -> None:
    assert windows_agent.dialog_button_matches("开始新游戏", "new")
    assert windows_agent.dialog_button_matches("重新开始", "restart")
    assert windows_agent.dialog_button_matches("继续", "continue")


def test_dialog_button_matches_close_and_statistics_title() -> None:
    assert windows_agent.dialog_button_matches("\u5173\u95ed(&C)", "close")
    assert windows_agent.is_statistics_dialog_title("\u626b\u96f7\u7edf\u8ba1\u4fe1\u606f - user")


def test_dialog_fallback_point_uses_current_new_game_row_layout() -> None:
    rect = (100, 200, 800, 600)

    new_point = windows_agent.dialog_fallback_point("新游戏", rect, "new")
    restart_point = windows_agent.dialog_fallback_point("新游戏", rect, "restart")
    continue_point = windows_agent.dialog_fallback_point("新游戏", rect, "continue")

    assert new_point == (450, 392)
    assert restart_point == (450, 472)
    assert continue_point == (450, 552)
    assert new_point[1] < restart_point[1] < continue_point[1]


def test_dialog_fallback_point_uses_bottom_buttons_for_terminal_dialog() -> None:
    rect = (100, 200, 800, 600)

    restart_point = windows_agent.dialog_fallback_point("游戏失败", rect, "restart")
    new_point = windows_agent.dialog_fallback_point("游戏失败", rect, "new")

    assert restart_point == (450, 560)
    assert new_point == (681, 560)


def test_attribute_terminal_to_previous_action_does_not_overwrite_specific_source() -> None:
    actions = [{"terminal_dialog": "游戏失败", "terminal_detected_at": "after_click"}]

    windows_agent.attribute_terminal_to_previous_action(actions, "游戏失败", "final_check")

    assert actions[0]["terminal_detected_at"] == "after_click"


def test_summary_from_board_includes_first_open_action_details() -> None:
    board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    actions = [
        {
            "action": {"kind": "open", "row": 4, "col": 7},
            "click": {"kind": "open", "row": 4, "col": 7, "method": "mouse_event"},
            "revealed_delta": 12,
            "target_revealed_after_open": True,
            "after_read_repairs": 0,
            "after_read_restores": 0,
            "after_grid": {"x0": 1},
            "open_effect": {"target_revealed": True},
        }
    ]

    summary = windows_agent.summary_from_board(board, actions, elapsed=1.5)

    assert summary["first_open_action"]["action"] == {"kind": "open", "row": 4, "col": 7}
    assert summary["first_open_action"]["revealed_delta"] == 12
    assert summary["first_open_action"]["target_revealed_after_open"] is True


def test_fast_timing_respects_reliable_user_delays() -> None:
    timing = windows_agent.live_timing_settings(
        Namespace(
            speed_profile="fast",
            capture_delay=0.003,
            action_delay=0.02,
            stable_reads=2,
            stable_read_delay=0.02,
            no_progress_reclicks=1,
            reclick_delay=0.03,
            post_click_settle=0.02,
        )
    )

    assert timing["action_delay"] == 0.02
    assert timing["settle_read_delay"] == 0.02
    assert timing["reclick_delay"] == 0.08
    assert timing["no_progress_reclicks"] == 1
    assert timing["click_pause"] == 0.008
    assert timing["cursor_settle"] == 0.003
    assert timing["post_click_settle"] == 0.08


def test_custom_timing_respects_low_user_delays() -> None:
    timing = windows_agent.live_timing_settings(
        Namespace(
            speed_profile="custom",
            capture_delay=0.0005,
            action_delay=0.0,
            stable_reads=1,
            stable_read_delay=0.001,
            no_progress_reclicks=0,
            reclick_delay=0.015,
            click_confirm_retries=2,
            click_hold=0.03,
            cursor_settle=0.008,
            post_click_settle=0.025,
        )
    )

    assert timing["capture_delay"] == 0.0005
    assert timing["action_delay"] == 0.0
    assert timing["settle_read_delay"] == 0.001
    assert timing["click_hold"] == 0.03
    assert timing["click_pause"] == 0.03
    assert timing["cursor_settle"] == 0.008
    assert timing["post_click_settle"] == 0.025


def test_classify_cell_fast_ignores_border_noise_on_empty_revealed_cell() -> None:
    cell = np.full((80, 80, 3), [214, 220, 232], dtype=np.uint8)
    cell[:6, :] = [35, 40, 60]
    cell[-6:, :] = [35, 40, 60]
    cell[:, :6] = [35, 40, 60]
    cell[:, -6:] = [35, 40, 60]
    cell[8:-8, 8:-8] = [238, 240, 244]

    assert windows_agent.classify_cell_fast(cell[5:-5, 5:-5]) == {"kind": "revealed", "number": 7}
    assert windows_agent.classify_cell_fast(cell[6:-6, 6:-6]) == {"kind": "revealed", "number": 0}


def test_classify_cell_fast_reads_blue_four_on_revealed_background() -> None:
    cell = np.full((80, 80, 3), [190, 198, 218], dtype=np.uint8)
    blue_four = [45, 45, 125]
    cell[14:54, 24:34] = blue_four
    cell[36:46, 24:58] = blue_four
    cell[14:66, 50:60] = blue_four

    assert windows_agent.classify_cell_fast(cell) == {"kind": "revealed", "number": 4}


def test_classify_cell_fast_keeps_dark_blue_four_revealed() -> None:
    cell = np.full((80, 80, 3), [128, 136, 193], dtype=np.uint8)
    blue_four = [2, 2, 133]
    cell[14:54, 24:34] = blue_four
    cell[36:46, 24:58] = blue_four
    cell[14:66, 50:60] = blue_four

    assert windows_agent.classify_cell_fast(cell) == {"kind": "revealed", "number": 4}


def test_classify_cell_fast_reads_blue_one_on_revealed_background() -> None:
    cell = np.full((80, 80, 3), [205, 214, 232], dtype=np.uint8)
    blue_one = [55, 75, 175]
    cell[14:66, 36:46] = blue_one
    cell[14:24, 30:46] = blue_one
    cell[56:66, 26:58] = blue_one

    assert windows_agent.classify_cell_fast(cell) == {"kind": "revealed", "number": 1}


def test_cell_pixel_change_detects_open_transition() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    flagged = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    mine_like = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    grid = windows_agent.Grid(
        x_lines=[index * 20 for index in range(windows_agent.COLS + 1)],
        y_lines=[index * 20 for index in range(windows_agent.ROWS + 1)],
    )
    before_pixels = np.zeros((windows_agent.ROWS * 20, windows_agent.COLS * 20, 3), dtype=np.uint8)
    after_pixels = before_pixels.copy()
    after_pixels[100:120, 140:160] = (200, 200, 200)
    before = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=flagged,
        adjacent=adjacent,
        mine_like=mine_like,
        grid=grid,
        screenshot=None,
        pixels=before_pixels,
    )
    after = windows_agent.ScreenBoard(
        revealed=revealed.copy(),
        flagged=flagged.copy(),
        adjacent=adjacent.copy(),
        mine_like=mine_like.copy(),
        grid=grid,
        screenshot=None,
        pixels=after_pixels,
    )

    changed = windows_agent.cell_pixel_change(before, after, 5, 7)
    unchanged = windows_agent.cell_pixel_change(before, before, 5, 7)

    assert changed["available"]
    assert changed["changed"]
    assert changed["changed_ratio"] == 1.0
    assert unchanged["available"]
    assert not unchanged["changed"]

    after_with_one_pixel_drift = windows_agent.ScreenBoard(
        revealed=revealed.copy(),
        flagged=flagged.copy(),
        adjacent=adjacent.copy(),
        mine_like=mine_like.copy(),
        grid=windows_agent.Grid(
            x_lines=grid.x_lines,
            y_lines=[*grid.y_lines[:-1], grid.y_lines[-1] + 1],
        ),
        screenshot=None,
        pixels=np.pad(after_pixels, ((0, 1), (0, 0), (0, 0))),
    )
    drifted = windows_agent.cell_pixel_change(before, after_with_one_pixel_drift, 5, 7)

    assert drifted["available"]
    assert drifted["changed"]


def test_classify_cell_fast_keeps_broad_blue_digit_revealed() -> None:
    cell = np.full((80, 80, 3), [198, 210, 231], dtype=np.uint8)
    blue_one = [55, 75, 175]
    cell[8:72, 30:50] = blue_one
    cell[8:20, 24:56] = blue_one
    cell[60:72, 20:60] = blue_one

    assert windows_agent._looks_like_hidden_blue_cell(cell)
    assert windows_agent.classify_cell_fast(cell) == {"kind": "revealed", "number": 1}


def test_best_grid_sequence_recovers_clipped_outer_lines() -> None:
    candidates = [89, 179, 268, 357, 447, 538, 627, 716, 806, 896, 985, 1074, 1162, 1250]

    lines = windows_agent._best_grid_sequence(
        candidates,
        expected=windows_agent.ROWS + 1,
        min_spacing=45,
        max_spacing=130,
    )

    assert len(lines) == windows_agent.ROWS + 1
    assert lines[0] == 0
    assert lines[-1] in {1429, 1430, 1431}
    assert max(np.diff(lines)) - min(np.diff(lines)) <= 2


def test_classify_cell_fast_keeps_red_three_revealed() -> None:
    cell = np.full((80, 80, 3), [190, 198, 218], dtype=np.uint8)
    red_three = [175, 35, 35]
    cell[14:24, 24:58] = red_three
    cell[36:46, 30:58] = red_three
    cell[58:68, 24:58] = red_three
    cell[14:68, 50:60] = red_three

    assert windows_agent.classify_cell_fast(cell) == {"kind": "revealed", "number": 3}


def test_open_confirmation_skips_when_target_opened_cleanly() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    revealed[0, 1] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    action = windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3)
    revealed[action.row, action.col] = True

    assert not windows_agent.open_needs_confirmation(action, board, before_revealed=1)


def test_open_confirmation_runs_when_target_stayed_hidden() -> None:
    board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    action = windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3)

    assert windows_agent.open_needs_confirmation(action, board, before_revealed=0)


def test_open_confirmation_runs_when_other_cells_opened_but_target_did_not() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    revealed[0, 1] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    action = windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3)

    assert windows_agent.open_needs_confirmation(action, board, before_revealed=1)


def test_confirm_open_read_passively_rechecks_before_reclick(monkeypatch) -> None:
    hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    revealed_mask = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed_mask[3, 3] = True
    revealed = windows_agent.ScreenBoard(
        revealed=revealed_mask,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    class Desktop:
        clicks = 0

        def click_action(self, action):
            self.clicks += 1
            return {"issued": True}

    desktop = Desktop()

    def fake_read_stable_board(*args, **kwargs):
        return revealed

    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    result = windows_agent.confirm_open_read(
        desktop=desktop,
        action=windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3),
        previous_board=hidden,
        step_count=1,
        keep_screenshot=False,
        settle_reads=1,
        settle_read_delay=0.0,
        reclicks=1,
        reclick_delay=0.0,
        virtual_flags=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        use_memory_flags=False,
    )

    assert result is not None
    _, board, info = result
    assert board.revealed[3, 3]
    assert info["confirmed"]
    assert info["passive_reads"] == 1
    assert info["reclicks"] == 0
    assert info["final_read_repairs"] == 0
    assert info["final_read_restores"] == 0
    assert desktop.clicks == 0


def test_confirm_open_read_keeps_cleaner_best_board(monkeypatch) -> None:
    hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    clean_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    repaired_revealed = clean_revealed.copy()
    repaired_revealed[3, 4] = True
    repaired = windows_agent.ScreenBoard(
        revealed=repaired_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    reads = [clean, repaired, repaired]

    class Desktop:
        clicks = 0

        def click_action(self, action):
            self.clicks += 1
            return {"issued": True}

    desktop = Desktop()

    def fake_read_stable_board(*args, **kwargs):
        return reads.pop(0)

    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    result = windows_agent.confirm_open_read(
        desktop=desktop,
        action=windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3),
        previous_board=hidden,
        step_count=1,
        keep_screenshot=False,
        settle_reads=1,
        settle_read_delay=0.0,
        reclicks=1,
        reclick_delay=0.0,
        virtual_flags=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        use_memory_flags=False,
    )

    assert result is not None
    _, board, info = result
    assert board.read_repairs == 0
    assert int(board.revealed.sum()) == 1
    assert board.revealed[0, 0]
    assert info["best_revealed"] == 1
    assert info["final_read_repairs"] == 0


def test_confirm_open_read_prefers_target_revealed_even_if_dirty(monkeypatch) -> None:
    hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    clean_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    clean_revealed[0, 0] = True
    clean = windows_agent.ScreenBoard(
        revealed=clean_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    dirty_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    dirty_revealed[3, 3] = True
    dirty_revealed[3, 4] = True
    dirty = windows_agent.ScreenBoard(
        revealed=dirty_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    reads = [clean, dirty, dirty]

    class Desktop:
        clicks = 0

        def click_action(self, action):
            self.clicks += 1
            return {"issued": True}

    desktop = Desktop()

    def fake_read_stable_board(*args, **kwargs):
        return reads.pop(0)

    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    result = windows_agent.confirm_open_read(
        desktop=desktop,
        action=windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3),
        previous_board=hidden,
        step_count=1,
        keep_screenshot=False,
        settle_reads=1,
        settle_read_delay=0.0,
        reclicks=1,
        reclick_delay=0.0,
        virtual_flags=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        use_memory_flags=False,
    )

    assert result is not None
    _, board, info = result
    assert board.revealed[3, 3]
    assert not board.revealed[0, 0]
    assert info["best_revealed"] == int(board.revealed.sum())
    assert info["final_read_repairs"] == 1


def test_board_difference_counts_reports_changed_cells() -> None:
    previous_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    current_revealed = previous_revealed.copy()
    current_revealed[2, 3] = True
    previous = windows_agent.ScreenBoard(
        revealed=previous_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    current = windows_agent.ScreenBoard(
        revealed=current_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    diff = windows_agent.board_difference_counts(current, previous)

    assert diff["available"]
    assert diff["revealed_changed"] == 1
    assert diff["signature_changed"]


def test_board_compare_counts_reports_fast_hidden_accurate_revealed() -> None:
    fast_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    accurate_revealed = fast_revealed.copy()
    accurate_revealed[2, 3] = True
    fast = windows_agent.ScreenBoard(
        revealed=fast_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    accurate = windows_agent.ScreenBoard(
        revealed=accurate_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    diff = windows_agent.board_compare_counts(fast, accurate)

    assert diff["fast_hidden_accurate_revealed"] == 1
    assert diff["fast_revealed_accurate_hidden"] == 0
    assert diff["revealed_mismatch"] == 1
    assert not diff["same_signature"]


def test_summarize_accurate_compare_records_counts_mismatches() -> None:
    summary = windows_agent.summarize_accurate_compare_records(
        [
            {
                "available": True,
                "fast_hidden_accurate_revealed": 1,
                "fast_revealed_accurate_hidden": 0,
                "number_mismatch": 2,
                "same_signature": False,
                "accurate_elapsed_seconds": 0.2,
            },
            {
                "available": True,
                "fast_hidden_accurate_revealed": 0,
                "fast_revealed_accurate_hidden": 1,
                "number_mismatch": 0,
                "same_signature": True,
                "accurate_elapsed_seconds": 0.4,
            },
        ]
    )

    assert summary["records"] == 2
    assert summary["fast_hidden_accurate_revealed"] == 1
    assert summary["fast_revealed_accurate_hidden"] == 1
    assert summary["number_mismatch"] == 2
    assert not summary["all_same_signature"]
    assert summary["avg_accurate_seconds"] == 0.30000000000000004


def test_read_compare_preflight_warns_on_fast_accurate_mismatch(monkeypatch) -> None:
    fast_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    accurate_revealed = fast_revealed.copy()
    accurate_revealed[2, 3] = True
    grid = windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1])
    fast = windows_agent.ScreenBoard(
        revealed=fast_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    accurate = windows_agent.ScreenBoard(
        revealed=accurate_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )

    class Desktop:
        read_mode = "fast"

        def _read_board_accurate(self, *args, **kwargs):
            return accurate

    monkeypatch.setattr(windows_agent, "read_stable_board", lambda *args, **kwargs: fast)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    report = windows_agent.read_compare_preflight(
        Desktop(),
        timing={"settle_reads": 1, "settle_read_delay": 0.0},
        reads=1,
        interval=0.0,
    )

    assert report["status"] == "warning"
    assert report["accurate_compare_summary"]["fast_hidden_accurate_revealed"] == 1
    assert not report["accurate_compare_summary"]["all_same_signature"]


def test_open_effect_summary_reports_nearest_revealed_offset() -> None:
    before = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    after_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    after_revealed[5, 6] = True
    after_revealed[8, 9] = True
    after = windows_agent.ScreenBoard(
        revealed=after_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    effect = windows_agent.open_effect_summary(
        before,
        after,
        windows_agent.Action(windows_agent.ActionType.OPEN, 5, 5),
    )

    assert effect["new_revealed_cells"] == 2
    assert effect["target_missed_with_progress"]
    assert effect["nearest_new_revealed"] == {
        "row": 5,
        "col": 6,
        "dr": 0,
        "dc": 1,
        "chebyshev": 1,
        "manhattan": 1,
    }


def test_memory_board_overlays_persistent_reveals() -> None:
    raw_board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    persistent_mask = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    persistent_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    persistent_mask[2, 3] = True
    persistent_adjacent[2, 3] = 4

    board = windows_agent.memory_board(
        raw_board,
        np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        False,
        persistent_revealed_mask=persistent_mask,
        persistent_revealed_adjacent=persistent_adjacent,
    )

    assert board.revealed[2, 3]
    assert board.adjacent[2, 3] == 4
    assert not board.mine_like[2, 3]


def test_memory_board_persistent_reveal_clears_virtual_flag() -> None:
    raw_board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    virtual_flags = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    virtual_flags[2, 3] = True
    persistent_mask = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    persistent_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    persistent_mask[2, 3] = True
    persistent_adjacent[2, 3] = 4

    board = windows_agent.memory_board(
        raw_board,
        virtual_flags,
        True,
        persistent_revealed_mask=persistent_mask,
        persistent_revealed_adjacent=persistent_adjacent,
    )

    assert board.revealed[2, 3]
    assert not board.flagged[2, 3]
    assert board.adjacent[2, 3] == 4
    assert not virtual_flags[2, 3]


def test_remember_confirmed_open_cells_records_clean_reveals() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    revealed[4, 5] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    cells: set[tuple[int, int]] = set()

    added = windows_agent.remember_confirmed_open_cells(board, cells)

    assert added == 2
    assert cells == {(2, 3), (4, 5)}


def test_remember_confirmed_open_cells_records_dirty_target_only() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[1, 1] = True
    revealed[3, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    cells: set[tuple[int, int]] = set()

    added = windows_agent.remember_confirmed_open_cells(board, cells, target=(3, 3))

    assert added == 1
    assert cells == {(3, 3)}


def test_remember_confirmed_open_cells_forces_target_when_progress_seen() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[1, 1] = True
    revealed[3, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    cells: set[tuple[int, int]] = set()

    added = windows_agent.remember_confirmed_open_cells(board, cells, target=(3, 3), force_target=True)

    assert added == 1
    assert cells == {(3, 3)}


def test_confirmed_open_action_mask_blocks_reopened_and_reflagged_cells() -> None:
    action_mask = np.ones((4, windows_agent.ROWS, windows_agent.COLS), dtype=bool)

    masked = windows_agent.apply_confirmed_open_action_mask(
        action_mask,
        open_forbidden_cells={(1, 1)},
        confirmed_open_cells={(2, 3)},
    )

    assert action_mask[windows_agent.action_channel(windows_agent.ActionType.OPEN), 1, 1]
    assert not masked[windows_agent.action_channel(windows_agent.ActionType.OPEN), 1, 1]
    assert not masked[windows_agent.action_channel(windows_agent.ActionType.OPEN), 2, 3]
    assert not masked[windows_agent.action_channel(windows_agent.ActionType.FLAG), 2, 3]
    assert not masked[windows_agent.action_channel(windows_agent.ActionType.UNFLAG), 2, 3]
    assert masked[windows_agent.action_channel(windows_agent.ActionType.FLAG), 1, 1]
    assert masked[windows_agent.action_channel(windows_agent.ActionType.CHORD), 2, 3]


def test_clear_virtual_flags_for_confirmed_cells_only() -> None:
    virtual_flags = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    virtual_flags[2, 3] = True
    virtual_flags[4, 5] = True

    cleared = windows_agent.clear_virtual_flags_for_cells(virtual_flags, {(2, 3), (20, 20)})

    assert cleared == 1
    assert not virtual_flags[2, 3]
    assert virtual_flags[4, 5]


def test_clear_blocked_open_cells_for_confirmed_removes_attempt_state() -> None:
    blocked_open_cells = {(2, 3), (4, 5)}
    blocked_open_attempts = {(2, 3): 2, (4, 5): 1}

    cleared = windows_agent.clear_blocked_open_cells_for_confirmed(
        blocked_open_cells,
        blocked_open_attempts,
        {(2, 3)},
    )

    assert cleared == 1
    assert blocked_open_cells == {(4, 5)}
    assert blocked_open_attempts == {(4, 5): 1}


def test_open_needs_confirmation_runs_when_progress_misses_target() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    revealed[0, 1] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    action = windows_agent.Action(windows_agent.ActionType.OPEN, 3, 3)

    assert windows_agent.open_needs_confirmation(action, board, before_revealed=1)


def test_summary_reports_open_target_misses_and_zero_progress() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    actions = [
        {
            "action": {"kind": "open", "row": 0, "col": 1},
            "click": {
                "issued": True,
                "cursor_ready": True,
                "elapsed_seconds": 0.01,
                "sendinput_fallback": "mouse_event",
                "sendinput_error": "OSError: boom",
            },
            "revealed_delta": 3,
            "target_revealed_after_open": False,
            "open_effect": {
                "target_missed_with_progress": True,
                "nearest_new_revealed": {"row": 0, "col": 2, "dr": 0, "dc": 1, "chebyshev": 1, "manhattan": 1},
            },
            "after_read_recoveries": 1,
            "confirmed_open_added_after": 3,
            "confirmed_open_cells": 7,
            "basic_audit": {
                "available": True,
                "target": {"known_mine": True, "known_safe": False, "conflict": False},
            },
        },
        {
            "action": {"kind": "open", "row": 0, "col": 2},
            "click": {"issued": True, "cursor_ready": True, "elapsed_seconds": 0.01},
            "revealed_delta": 0,
            "target_revealed_after_open": False,
            "after_target": {"mine_like": False},
        },
    ]

    summary = windows_agent.summary_from_board(board, actions=actions, elapsed=1.0)

    assert summary["open_target_miss_with_progress"] == 1
    assert summary["open_target_miss_nearest_avg_manhattan"] == 1.0
    assert summary["open_target_miss_nearest_max_manhattan"] == 1
    assert summary["open_zero_progress_actions"] == 1
    assert summary["sendinput_fallback_actions"] == 1
    assert summary["sendinput_error_actions"] == 1
    assert summary["read_recovery_actions"] == 1
    assert summary["avg_revealed_delta_per_open"] == 1.5
    assert summary["max_confirmed_open_cells"] == 7
    assert summary["confirmed_open_remembered_cells"] == 3
    assert summary["cleared_confirmed_virtual_flags"] == 0
    assert summary["cleared_confirmed_blocked_opens"] == 0
    assert summary["basic_audited_opens"] == 1
    assert summary["basic_known_mine_opens"] == 1


def test_summary_reports_repeated_open_targets() -> None:
    board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    actions = [
        {
            "action": {"kind": "open", "row": 0, "col": 1},
            "click": {"issued": True, "cursor_ready": True, "elapsed_seconds": 0.01},
            "revealed_delta": 1,
            "target_revealed_after_open": True,
        },
        {
            "action": {"kind": "open", "row": 0, "col": 1},
            "click": {"issued": True, "cursor_ready": True, "elapsed_seconds": 0.01},
            "revealed_delta": 0,
            "target_revealed_after_open": True,
        },
    ]

    summary = windows_agent.summary_from_board(board, actions=actions, elapsed=1.0)

    assert summary["open_target_repeat_actions"] == 1


def test_remember_revealed_cells_skips_repaired_reads() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_repairs=1,
    )
    cells: set[tuple[int, int]] = set()

    added = windows_agent.remember_revealed_cells(board, cells)

    assert added == 0
    assert cells == set()


def test_remember_revealed_cells_skips_restored_reads() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
        read_restores=1,
    )
    cells: set[tuple[int, int]] = set()

    added = windows_agent.remember_revealed_cells(board, cells)

    assert added == 0
    assert cells == set()


def test_remember_revealed_cells_tracks_stable_reveals() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    revealed[4, 5] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    cells: set[tuple[int, int]] = {(2, 3)}

    added = windows_agent.remember_revealed_cells(board, cells)

    assert added == 1
    assert cells == {(2, 3), (4, 5)}


def test_remember_revealed_cells_mask_mode_stores_numbers() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    revealed[2, 3] = True
    adjacent[2, 3] = 4
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    persistent_mask = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    persistent_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)

    added = windows_agent.remember_revealed_cells(board, persistent_mask, persistent_adjacent)

    assert added == 1
    assert persistent_mask[2, 3]
    assert persistent_adjacent[2, 3] == 4


def test_remember_revealed_cells_requires_consecutive_observations() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    cells: set[tuple[int, int]] = set()
    observations = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.uint8)

    first_added = windows_agent.remember_revealed_cells(
        board,
        cells,
        observations=observations,
        min_observations=2,
    )
    second_added = windows_agent.remember_revealed_cells(
        board,
        cells,
        observations=observations,
        min_observations=2,
    )

    assert first_added == 0
    assert second_added == 1
    assert cells == {(2, 3)}


def test_remember_revealed_cells_resets_unconfirmed_observation() -> None:
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[2, 3] = True
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    hidden_board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    cells: set[tuple[int, int]] = set()
    observations = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.uint8)

    windows_agent.remember_revealed_cells(board, cells, observations=observations, min_observations=2)
    windows_agent.remember_revealed_cells(hidden_board, cells, observations=observations, min_observations=2)
    added = windows_agent.remember_revealed_cells(board, cells, observations=observations, min_observations=2)

    assert added == 0
    assert observations[2, 3] == 1
    assert cells == set()


def test_wait_for_fresh_board_requires_consecutive_fresh_reads(monkeypatch) -> None:
    dirty = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    dirty.revealed[0, 0] = True
    fresh = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    reads = [dirty, fresh, fresh]

    class FakeDesktop:
        def dialog_is_open(self) -> bool:
            return False

        def read_board(self, keep_screenshot: bool = False) -> windows_agent.ScreenBoard:
            return reads.pop(0)

    clock = {"value": 0.0}

    def fake_time() -> float:
        return clock["value"]

    def fake_sleep(seconds: float) -> None:
        clock["value"] += seconds

    monkeypatch.setattr(windows_agent.time, "time", fake_time)
    monkeypatch.setattr(windows_agent.time, "sleep", fake_sleep)

    assert windows_agent.wait_for_fresh_board(FakeDesktop(), timeout=0.5, reads=2)


def test_wait_for_fresh_board_closes_statistics_dialog(monkeypatch) -> None:
    fresh = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    class FakeDesktop:
        def __init__(self) -> None:
            self.closed = False

        def dialog_is_open(self) -> bool:
            return not self.closed

        def dialog_title(self) -> str:
            return "\u626b\u96f7\u7edf\u8ba1\u4fe1\u606f"

        def choose_dialog_option(self, option: str, timeout: float, attempts: int) -> bool:
            assert option == "close"
            self.closed = True
            return True

        def read_board(self, keep_screenshot: bool = False) -> windows_agent.ScreenBoard:
            return fresh

    clock = {"value": 0.0}

    def fake_time() -> float:
        return clock["value"]

    def fake_sleep(seconds: float) -> None:
        clock["value"] += seconds

    desktop = FakeDesktop()
    monkeypatch.setattr(windows_agent.time, "time", fake_time)
    monkeypatch.setattr(windows_agent.time, "sleep", fake_sleep)

    assert windows_agent.wait_for_fresh_board(desktop, timeout=0.5, reads=1)
    assert desktop.closed


def test_quick_number_board_after_open_updates_single_cell(monkeypatch) -> None:
    grid = windows_agent.Grid(
        x_lines=[index * 30 for index in range(windows_agent.COLS + 1)],
        y_lines=[index * 30 for index in range(windows_agent.ROWS + 1)],
    )
    previous_pixels = np.zeros((windows_agent.ROWS * 30, windows_agent.COLS * 30, 3), dtype=np.uint8)
    previous = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
        pixels=previous_pixels,
    )
    fake_grid = grid

    class FakeDesktop:
        grid = fake_grid

        def capture_region_array(self, left: int, top: int, right: int, bottom: int) -> np.ndarray:
            assert (left, top, right, bottom) == grid.crop_box(4, 5)
            return np.full((bottom - top, right - left, 3), 220, dtype=np.uint8)

    monkeypatch.setattr(
        windows_agent,
        "classify_cell_fast",
        lambda array: {"kind": "revealed", "number": 3},
    )

    board, info = windows_agent.quick_number_board_after_open(
        FakeDesktop(),
        previous,
        windows_agent.Action(windows_agent.ActionType.OPEN, 4, 5),
        step_count=7,
        keep_screenshot=False,
    )

    assert board is not None
    assert info["accepted"]
    assert board.revealed[4, 5]
    assert board.adjacent[4, 5] == 3
    assert not previous.revealed[4, 5]
    assert board.read_timing["mode"] == "quick_number"
    assert board.step_count == 7


def test_quick_number_board_after_open_falls_back_on_zero(monkeypatch) -> None:
    grid = windows_agent.Grid(
        x_lines=[index * 30 for index in range(windows_agent.COLS + 1)],
        y_lines=[index * 30 for index in range(windows_agent.ROWS + 1)],
    )
    previous = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
        pixels=np.zeros((windows_agent.ROWS * 30, windows_agent.COLS * 30, 3), dtype=np.uint8),
    )
    fake_grid = grid

    class FakeDesktop:
        grid = fake_grid

        def capture_region_array(self, left: int, top: int, right: int, bottom: int) -> np.ndarray:
            return np.zeros((bottom - top, right - left, 3), dtype=np.uint8)

    monkeypatch.setattr(
        windows_agent,
        "classify_cell_fast",
        lambda array: {"kind": "revealed", "number": 0},
    )

    board, info = windows_agent.quick_number_board_after_open(
        FakeDesktop(),
        previous,
        windows_agent.Action(windows_agent.ActionType.OPEN, 2, 3),
        step_count=1,
        keep_screenshot=False,
    )

    assert board is None
    assert info["reason"] == "zero_or_ambiguous_reveal"


def test_click_action_verifies_cursor_then_parks_cursor(monkeypatch) -> None:
    events: list[object] = []

    class FakeDesktop:
        grid = windows_agent.Grid(x_lines=[0, 10, 20], y_lines=[0, 10])
        click_method = "auto"
        click_pause = 0.0
        cursor_settle = 0.0
        post_click_settle = 0.0
        _resolved_click_method = None

        def dialog_is_open(self) -> bool:
            return False

        def ensure_foreground(self) -> bool:
            return True

        def park_cursor(self) -> None:
            events.append("park")

    def fake_set_cursor_pos(pos: tuple[int, int]) -> None:
        events.append(("move", pos))

    def fake_click_mouse(
        down: int,
        up: int,
        x: int,
        y: int,
        pause: float = 0.0,
        method: str = "auto",
        report: dict | None = None,
    ) -> str:
        events.append(("click", x, y))
        if report is not None:
            report["fake_report_seen"] = True
        return "fake"

    monkeypatch.setattr(windows_agent.win32api, "SetCursorPos", fake_set_cursor_pos)
    monkeypatch.setattr(windows_agent.win32api, "GetCursorPos", lambda: (5, 5))
    monkeypatch.setattr(windows_agent, "click_mouse", fake_click_mouse)

    windows_agent.WindowsMinesweeper.click_action(
        FakeDesktop(),
        windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
    )

    assert events == [("move", (5, 5)), ("click", 5, 5), "park"]


def test_click_action_locks_auto_click_method_after_fallback(monkeypatch) -> None:
    calls: list[str] = []
    cursor = {"pos": (5, 5)}

    class FakeDesktop:
        grid = windows_agent.Grid(x_lines=[0, 10, 20], y_lines=[0, 10])
        click_method = "auto"
        click_pause = 0.0
        cursor_settle = 0.0
        post_click_settle = 0.0
        _resolved_click_method = None

        def dialog_is_open(self) -> bool:
            return False

        def ensure_foreground(self) -> bool:
            return True

        def park_cursor(self) -> None:
            pass

    def fake_set_cursor_pos(pos: tuple[int, int]) -> None:
        cursor["pos"] = pos

    def fake_click_mouse(
        down: int,
        up: int,
        x: int,
        y: int,
        pause: float = 0.0,
        method: str = "auto",
        report: dict | None = None,
    ) -> str:
        calls.append(method)
        return "mouse_event"

    monkeypatch.setattr(windows_agent.win32api, "SetCursorPos", fake_set_cursor_pos)
    monkeypatch.setattr(windows_agent.win32api, "GetCursorPos", lambda: cursor["pos"])
    monkeypatch.setattr(windows_agent, "click_mouse", fake_click_mouse)

    desktop = FakeDesktop()
    windows_agent.WindowsMinesweeper.click_action(
        desktop,
        windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
    )
    windows_agent.WindowsMinesweeper.click_action(
        desktop,
        windows_agent.Action(windows_agent.ActionType.OPEN, 0, 1),
    )

    assert calls == ["auto", "mouse_event"]
    assert desktop._resolved_click_method == "mouse_event"


def test_click_action_replaces_stale_click_report_before_dialog_error() -> None:
    class FakeDesktop:
        click_method = "auto"
        last_click_report = {"row": 9, "col": 9, "issued": True}

        def dialog_is_open(self) -> bool:
            return True

    desktop = FakeDesktop()

    try:
        windows_agent.WindowsMinesweeper.click_action(
            desktop,
            windows_agent.Action(windows_agent.ActionType.OPEN, 2, 3),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected dialog-open click to fail")

    assert desktop.last_click_report["row"] == 2
    assert desktop.last_click_report["col"] == 3
    assert not desktop.last_click_report["issued"]


def test_click_action_keeps_current_report_when_click_backend_fails(monkeypatch) -> None:
    class FakeDesktop:
        grid = windows_agent.Grid(x_lines=[0, 10, 20, 30], y_lines=[0, 10, 20, 30])
        click_method = "sendinput"
        click_pause = 0.0
        cursor_settle = 0.0
        post_click_settle = 0.0
        _resolved_click_method = None
        last_click_report = {"row": 9, "col": 9, "issued": True}

        def dialog_is_open(self) -> bool:
            return False

        def ensure_foreground(self) -> bool:
            return True

    cursor = {"pos": (15, 15)}

    def fake_set_cursor_pos(pos: tuple[int, int]) -> None:
        cursor["pos"] = pos

    def fake_click_mouse(*args, **kwargs):
        raise RuntimeError("backend failed")

    desktop = FakeDesktop()
    monkeypatch.setattr(windows_agent.win32api, "SetCursorPos", fake_set_cursor_pos)
    monkeypatch.setattr(windows_agent.win32api, "GetCursorPos", lambda: cursor["pos"])
    monkeypatch.setattr(windows_agent, "click_mouse", fake_click_mouse)

    try:
        windows_agent.WindowsMinesweeper.click_action(
            desktop,
            windows_agent.Action(windows_agent.ActionType.OPEN, 1, 1),
        )
    except RuntimeError:
        pass
    else:
        raise AssertionError("expected backend failure")

    assert desktop.last_click_report["row"] == 1
    assert desktop.last_click_report["col"] == 1
    assert "backend failed" in desktop.last_click_report["click_error"]


def test_click_mouse_can_force_legacy_mouse_event(monkeypatch) -> None:
    events: list[object] = []

    def fake_send_mouse_button(flags: int) -> None:
        events.append(("sendinput", flags))

    def fake_mouse_event(flags: int, dx: int, dy: int, data: int, extra: int) -> None:
        events.append(("mouse_event", flags))

    monkeypatch.setattr(windows_agent, "send_mouse_button", fake_send_mouse_button)
    monkeypatch.setattr(windows_agent.win32api, "mouse_event", fake_mouse_event)
    monkeypatch.setattr(windows_agent.win32api, "SetCursorPos", lambda pos: None)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    method = windows_agent.click_mouse(1, 2, 5, 5, method="mouse_event")

    assert method == "mouse_event"
    assert events == [("mouse_event", 1), ("mouse_event", 2)]


def test_click_mouse_absolute_sendinput_moves_then_clicks(monkeypatch) -> None:
    events: list[object] = []

    def fake_absolute_move(x: int, y: int) -> None:
        events.append(("move", x, y))

    def fake_send_mouse_button(flags: int) -> None:
        events.append(("button", flags))

    monkeypatch.setattr(windows_agent, "send_mouse_absolute_move", fake_absolute_move)
    monkeypatch.setattr(windows_agent, "send_mouse_button", fake_send_mouse_button)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    method = windows_agent.click_mouse(1, 2, 2807, 432, method="sendinput_absolute")

    assert method == "sendinput_absolute"
    assert events == [("move", 2807, 432), ("button", 1), ("button", 2)]


def test_click_mouse_auto_prefers_sendinput(monkeypatch) -> None:
    events: list[object] = []

    def fake_send_mouse_button(flags: int) -> None:
        events.append(("sendinput", flags))

    def fake_mouse_event(flags: int, dx: int, dy: int, data: int, extra: int) -> None:
        events.append(("mouse_event", flags))

    report: dict[str, object] = {}
    monkeypatch.setattr(windows_agent, "send_mouse_button", fake_send_mouse_button)
    monkeypatch.setattr(windows_agent.win32api, "mouse_event", fake_mouse_event)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    method = windows_agent.click_mouse(1, 2, 5, 5, method="auto", report=report)

    assert method == "sendinput"
    assert report["sendinput_attempted"]
    assert "sendinput_error" not in report
    assert events == [("sendinput", 1), ("sendinput", 2)]


def test_click_mouse_auto_reports_sendinput_fallback(monkeypatch) -> None:
    events: list[object] = []

    def fake_send_mouse_button(flags: int) -> None:
        events.append(("sendinput", flags))
        raise OSError("boom")

    def fake_mouse_event(flags: int, dx: int, dy: int, data: int, extra: int) -> None:
        events.append(("mouse_event", flags))

    report: dict[str, object] = {}
    monkeypatch.setattr(windows_agent, "send_mouse_button", fake_send_mouse_button)
    monkeypatch.setattr(windows_agent.win32api, "mouse_event", fake_mouse_event)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    method = windows_agent.click_mouse(1, 2, 5, 5, method="auto", report=report)

    assert method == "mouse_event"
    assert report["sendinput_attempted"]
    assert report["sendinput_fallback"] == "mouse_event"
    assert "boom" in str(report["sendinput_error"])
    assert events == [("sendinput", 1), ("mouse_event", 1), ("mouse_event", 2)]


def test_basic_action_audit_marks_target_known_mine_from_single_clue() -> None:
    revealed = np.ones((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = False
    adjacent = np.full((windows_agent.ROWS, windows_agent.COLS), 8, dtype=np.int8)
    adjacent[1, 1] = 1
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )

    audit = windows_agent.basic_action_audit(
        board,
        windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
    )

    assert audit["available"]
    assert audit["target"]["known_mine"]
    assert not audit["target"]["known_safe"]
    assert audit["target"]["witnesses"][0]["conclusion"] == "mine"


def test_basic_safety_filter_blocks_only_known_mine_opens() -> None:
    revealed = np.ones((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = False
    adjacent = np.full((windows_agent.ROWS, windows_agent.COLS), 8, dtype=np.int8)
    adjacent[1, 1] = 1
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    action_mask = np.ones((4, windows_agent.ROWS, windows_agent.COLS), dtype=bool)

    record = windows_agent.apply_basic_safety_filter(board, action_mask, "avoid-known-mines")

    assert record["applied"]
    assert record["blocked_known_mine_opens"] == 1
    assert not action_mask[windows_agent.action_channel(windows_agent.ActionType.OPEN), 0, 0]
    assert action_mask[windows_agent.action_channel(windows_agent.ActionType.FLAG), 0, 0]


def test_basic_safety_filter_does_not_block_conflicting_inference() -> None:
    revealed = np.ones((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = False
    adjacent = np.full((windows_agent.ROWS, windows_agent.COLS), 8, dtype=np.int8)
    adjacent[1, 0] = 0
    adjacent[1, 1] = 1
    board = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(x_lines=[0, 1], y_lines=[0, 1]),
        screenshot=None,
    )
    action_mask = np.ones((4, windows_agent.ROWS, windows_agent.COLS), dtype=bool)

    record = windows_agent.apply_basic_safety_filter(board, action_mask, "avoid-known-mines")

    assert not record["applied"]
    assert record["blocked_known_mine_opens"] == 0
    assert record["conflict_count"] == 1
    assert action_mask[windows_agent.action_channel(windows_agent.ActionType.OPEN), 0, 0]


def test_play_game_default_path_does_not_apply_safety_filters(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=list(range(windows_agent.COLS + 1)),
        y_lines=list(range(windows_agent.ROWS + 1)),
    )
    initial = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    after_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    after_revealed[0, 0] = True
    after = windows_agent.ScreenBoard(
        revealed=after_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    reads = [initial, after]
    fake_grid = grid

    class FakeDesktop:
        grid = fake_grid
        capture_backend = "auto"

        def dialog_is_open(self) -> bool:
            return False

        def effective_click_method(self) -> str:
            return "mouse_event"

        def click_action(self, action):
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=24)

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, **kwargs):
            return windows_agent.action_to_index(
                windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
                windows_agent.ROWS,
                windows_agent.COLS,
            )

    def fake_read_stable_board(*args, **kwargs):
        return reads.pop(0)

    def fail_basic_filter(*args, **kwargs):
        raise AssertionError("basic safety filter should be off in the default pure RL path")

    def fail_solver_filter(*args, **kwargs):
        raise AssertionError("solver safety filter should be off in the default pure RL path")

    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent, "apply_basic_safety_filter", fail_basic_filter)
    monkeypatch.setattr(windows_agent, "apply_solver_safety_filter", fail_solver_filter)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="memory",
        no_persistent_reveals=False,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=1,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=FakeDesktop(),
        trainer=FakeTrainer(),
        timing={
            "capture_delay": 0.0,
            "action_delay": 0.0,
            "settle_reads": 1,
            "settle_read_delay": 0.0,
            "no_progress_reclicks": 0,
            "reclick_delay": 0.0,
            "click_pause": 0.0,
            "cursor_settle": 0.0,
            "post_click_settle": 0.0,
        },
    )

    trace = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert trace["audit_basic"] is False
    assert trace["basic_safety_filter"] == "none"
    assert trace["actions"][0]["action"] == {"kind": "open", "row": 0, "col": 0}
    assert "basic_safety_filter" not in trace["actions"][0]
    assert "solver_safety_filter" not in trace["actions"][0]
    assert "basic_audit" not in trace["actions"][0]


def test_play_game_quick_number_read_skips_full_readback(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=[index * 30 for index in range(windows_agent.COLS + 1)],
        y_lines=[index * 30 for index in range(windows_agent.ROWS + 1)],
    )
    initial = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
        pixels=np.zeros((windows_agent.ROWS * 30, windows_agent.COLS * 30, 3), dtype=np.uint8),
    )
    fake_grid = grid

    class FakeDesktop:
        capture_backend = "auto"
        grid = fake_grid

        def dialog_is_open(self) -> bool:
            return False

        def effective_click_method(self) -> str:
            return "mouse_event"

        def capture_region_array(self, left: int, top: int, right: int, bottom: int) -> np.ndarray:
            return np.full((bottom - top, right - left, 3), 220, dtype=np.uint8)

        def click_action(self, action):
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=24)

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, **kwargs):
            return windows_agent.action_to_index(
                windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
                windows_agent.ROWS,
                windows_agent.COLS,
            )

    reads = [initial]

    def fake_read_stable_board(*args, **kwargs):
        if not reads:
            raise AssertionError("quick number read should skip full readback")
        return reads.pop(0)

    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent, "classify_cell_fast", lambda array: {"kind": "revealed", "number": 2})
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="memory",
        no_persistent_reveals=True,
        quick_number_read=True,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        solver_assist="none",
        solver_exact_limit=None,
        solver_batch_size=1,
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=1,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=FakeDesktop(),
        trainer=FakeTrainer(),
        timing={
            "capture_delay": 0.0,
            "action_delay": 0.0,
            "settle_reads": 1,
            "settle_read_delay": 0.0,
            "no_progress_reclicks": 0,
            "click_confirm_retries": 1,
            "reclick_delay": 0.0,
            "click_pause": 0.0,
            "cursor_settle": 0.0,
            "post_click_settle": 0.0,
        },
    )

    action_record = json.loads(Path(result["path"]).read_text(encoding="utf-8"))["actions"][0]
    assert action_record["quick_number_read"]["accepted"]
    assert action_record["target_revealed_after_open"]
    assert result["summary"]["quick_number_read_actions"] == 1
    assert result["summary"]["quick_number_read_fallbacks"] == 0


def test_play_game_blocks_unconfirmed_open_in_simulated_desktop(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=list(range(windows_agent.COLS + 1)),
        y_lines=list(range(windows_agent.ROWS + 1)),
    )
    hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )

    class FakeDesktop:
        capture_backend = "auto"

        def __init__(self) -> None:
            self.grid = grid
            self.clicks: list[tuple[int, int]] = []

        def dialog_is_open(self) -> bool:
            return False

        def effective_click_method(self) -> str:
            return "mouse_event"

        def click_action(self, action):
            self.clicks.append((int(action.row), int(action.col)))
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=8)

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, *, action_mask, **kwargs):
            open_channel = windows_agent.action_channel(windows_agent.ActionType.OPEN)
            rows, cols = np.where(action_mask[open_channel])
            return windows_agent.action_to_index(
                windows_agent.Action(windows_agent.ActionType.OPEN, int(rows[0]), int(cols[0])),
                windows_agent.ROWS,
                windows_agent.COLS,
            )

    desktop = FakeDesktop()
    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", lambda *args, **kwargs: hidden)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="open-only",
        no_persistent_reveals=True,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        solver_assist="none",
        solver_exact_limit=None,
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=2,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )
    timing = {
        "capture_delay": 0.0,
        "action_delay": 0.0,
        "settle_reads": 1,
        "settle_read_delay": 0.0,
        "no_progress_reclicks": 0,
        "click_confirm_retries": 3,
        "reclick_delay": 0.0,
        "click_pause": 0.0,
        "cursor_settle": 0.0,
        "post_click_settle": 0.0,
    }

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=desktop,
        trainer=FakeTrainer(),
        timing=timing,
    )

    trace = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(desktop.clicks) == 4
    assert all(click == desktop.clicks[0] for click in desktop.clicks)
    assert len(trace["actions"]) == 1
    assert trace["actions"][0]["click_unconfirmed"] is True
    assert trace["actions"][0]["click_unconfirmed_reason"] == "target_still_hidden_after_readback"
    assert trace["summary"]["open_target_repeat_actions"] == 0


def test_play_game_retries_open_until_readback_confirms(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=list(range(windows_agent.COLS + 1)),
        y_lines=list(range(windows_agent.ROWS + 1)),
    )
    hidden = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    opened = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )

    class FakeDesktop:
        capture_backend = "auto"

        def __init__(self) -> None:
            self.grid = grid
            self.clicks: list[tuple[int, int]] = []

        def dialog_is_open(self) -> bool:
            return False

        def effective_click_method(self) -> str:
            return "mouse_event"

        def click_action(self, action):
            self.clicks.append((int(action.row), int(action.col)))
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=8)

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, *, action_mask, **kwargs):
            return windows_agent.action_to_index(
                windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
                windows_agent.ROWS,
                windows_agent.COLS,
            )

    desktop = FakeDesktop()
    reads = [hidden, hidden, hidden, opened]

    def fake_read_stable_board(*args, **kwargs):
        return reads.pop(0) if reads else opened

    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="open-only",
        no_persistent_reveals=True,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        solver_assist="none",
        solver_exact_limit=None,
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=1,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )
    timing = {
        "capture_delay": 0.0,
        "action_delay": 0.0,
        "settle_reads": 1,
        "settle_read_delay": 0.0,
        "no_progress_reclicks": 0,
        "click_confirm_retries": 3,
        "reclick_delay": 0.0,
        "click_pause": 0.0,
        "cursor_settle": 0.0,
        "post_click_settle": 0.0,
    }

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=desktop,
        trainer=FakeTrainer(),
        timing=timing,
    )

    trace = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(desktop.clicks) == 2
    assert trace["actions"][0]["open_confirm"]["confirmed"] is True
    assert trace["actions"][0]["reclicks"] == 1
    assert trace["actions"][0]["target_revealed_after_open"] is True


def test_play_game_defers_reads_inside_solver_safe_batch(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=list(range(windows_agent.COLS + 1)),
        y_lines=list(range(windows_agent.ROWS + 1)),
    )
    initial = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    revealed[0, 0] = True
    revealed[0, 1] = True
    revealed[0, 2] = True
    final = windows_agent.ScreenBoard(
        revealed=revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )

    class FakeDesktop:
        capture_backend = "auto"

        def __init__(self) -> None:
            self.grid = grid
            self.clicks: list[tuple[int, int]] = []

        def dialog_is_open(self) -> bool:
            return False

        def effective_click_method(self) -> str:
            return "mouse_event"

        def click_action(self, action):
            self.clicks.append((int(action.row), int(action.col)))
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=8)

    desktop = FakeDesktop()
    reads = [initial, final]

    def fake_read_stable_board(*args, **kwargs):
        if not reads:
            raise AssertionError("solver-safe batch should not read between queued clicks")
        return reads.pop(0)

    def fake_solver_assist(*args, **kwargs):
        return (
            windows_agent.Action(windows_agent.ActionType.OPEN, 0, 0),
            [],
            {
                "mode": "forced",
                "available": True,
                "applied": True,
                "virtual_flag_cells": [],
                "safe_batch_targets": [
                    {"row": 0, "col": 0},
                    {"row": 0, "col": 1},
                    {"row": 0, "col": 2},
                ],
                "forced_safe_count": 3,
                "forced_mine_count": 0,
                "decision": "forced_safe_open",
                "target": {"row": 0, "col": 0},
                "safe_batch_count": 3,
            },
        )

    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent, "select_solver_assist_action", fake_solver_assist)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="memory",
        no_persistent_reveals=True,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        solver_assist="forced",
        solver_exact_limit=8,
        solver_batch_size=3,
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=3,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )
    timing = {
        "capture_delay": 0.0,
        "action_delay": 0.0,
        "settle_reads": 1,
        "settle_read_delay": 0.0,
        "no_progress_reclicks": 0,
        "click_confirm_retries": 1,
        "reclick_delay": 0.0,
        "click_pause": 0.0,
        "cursor_settle": 0.0,
        "post_click_settle": 0.0,
    }

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=desktop,
        trainer=FakeTrainer(),
        timing=timing,
    )

    trace = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert desktop.clicks == [(0, 0), (0, 1), (0, 2)]
    assert reads == []
    assert trace["actions"][0]["deferred_read"] == "solver_batch"
    assert trace["actions"][1]["deferred_read"] == "solver_batch"
    assert "after_read_elapsed" not in trace["actions"][0]
    assert "after_read_elapsed" not in trace["actions"][1]
    assert trace["actions"][2]["target_revealed_after_open"] is True


def test_play_game_attributes_before_click_dialog_to_previous_action(monkeypatch, tmp_path: Path) -> None:
    grid = windows_agent.Grid(
        x_lines=list(range(windows_agent.COLS + 1)),
        y_lines=list(range(windows_agent.ROWS + 1)),
    )
    initial = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    after_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    after_revealed[0, 0] = True
    after = windows_agent.ScreenBoard(
        revealed=after_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=grid,
        screenshot=None,
    )
    reads = [initial, after]
    fake_grid = grid

    class FakeDesktop:
        grid = fake_grid
        capture_backend = "auto"
        clicks = 0

        def dialog_is_open(self) -> bool:
            return False

        def dialog_title(self) -> str:
            return "游戏失败"

        def effective_click_method(self) -> str:
            return "mouse_event"

        def click_action(self, action):
            self.clicks += 1
            if self.clicks == 2:
                raise RuntimeError("a Minesweeper dialog is open; handle it before clicking the board")
            return {
                "row": int(action.row),
                "col": int(action.col),
                "kind": action.kind.value,
                "issued": True,
                "cursor_ready": True,
                "method": "mouse_event",
            }

    class FakeTrainer:
        config = Namespace(exact_limit=24)

        def __init__(self) -> None:
            self.calls = 0

        def _decision_action_mask(self, action_mask):
            return action_mask

        def _select_action(self, **kwargs):
            action = windows_agent.Action(windows_agent.ActionType.OPEN, 0, self.calls)
            self.calls += 1
            return windows_agent.action_to_index(action, windows_agent.ROWS, windows_agent.COLS)

    def fake_read_stable_board(*args, **kwargs):
        return reads.pop(0)

    monkeypatch.setattr(windows_agent, "prepare_game_start", lambda desktop, start_mode: False)
    monkeypatch.setattr(windows_agent, "read_stable_board", fake_read_stable_board)
    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    args = Namespace(
        checkpoint=Path("artifacts/full_rlmix_20.pt"),
        capture_backend="auto",
        read_mode="fast",
        click_method="mouse_event",
        flag_mode="memory",
        no_persistent_reveals=False,
        audit_solver=False,
        audit_basic=False,
        basic_safety_filter="none",
        solver_safety_filter="none",
        clear_stop_on_start=False,
        start_mode="current",
        center_first_open=False,
        max_steps=2,
        record_frames="none",
        no_final_images=True,
        stall_limit=20,
    )

    result = windows_agent.play_game(
        args,
        game_index=1,
        output_dir=tmp_path,
        desktop=FakeDesktop(),
        trainer=FakeTrainer(),
        timing={
            "capture_delay": 0.0,
            "action_delay": 0.0,
            "settle_reads": 1,
            "settle_read_delay": 0.0,
            "no_progress_reclicks": 0,
            "reclick_delay": 0.0,
            "click_pause": 0.0,
            "cursor_settle": 0.0,
            "post_click_settle": 0.0,
        },
    )

    trace = json.loads(Path(result["path"]).read_text(encoding="utf-8"))
    assert len(trace["actions"]) == 1
    assert trace["summary"]["lost"]
    assert trace["actions"][0]["action"] == {"kind": "open", "row": 0, "col": 0}
    assert trace["actions"][0]["terminal_dialog"] == "游戏失败"
    assert trace["actions"][0]["terminal_detected_at"] == "before_next_click"


def test_analyze_game_trace_reports_terminal_known_mine(tmp_path: Path) -> None:
    trace_path = tmp_path / "game_001.json"
    trace_path.write_text(
        json.dumps(
            {
                "summary": {
                    "won": False,
                    "lost": True,
                    "done": True,
                    "agent_steps": 1,
                    "elapsed_seconds": 2.0,
                    "revealed_safe_cells": 100,
                    "open_zero_progress_actions": 0,
                    "open_target_miss_with_progress": 0,
                    "unconfirmed_open_actions": 0,
                },
                "actions": [
                    {
                        "step": 0,
                        "action": {"kind": "open", "row": 1, "col": 2},
                        "click": {"kind": "open", "row": 1, "col": 2, "method": "sendinput"},
                        "terminal_dialog": "游戏失败",
                        "terminal_detected_at": "after_click",
                        "solver_audit": {
                            "available": True,
                            "target": {"known_mine": True, "risk": 1.0},
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = windows_agent.analyze_game_trace(trace_path)

    assert report["lost"]
    assert report["click_methods"] == {"sendinput": 1}
    assert report["solver_known_mine_opens"] == 1
    assert "terminal_solver_known_mine" in report["signals"]


def test_analyze_game_trace_reports_terminal_basic_known_mine(tmp_path: Path) -> None:
    trace_path = tmp_path / "game_001.json"
    trace_path.write_text(
        json.dumps(
            {
                "summary": {
                    "won": False,
                    "lost": True,
                    "done": True,
                    "agent_steps": 1,
                    "elapsed_seconds": 2.0,
                    "revealed_safe_cells": 100,
                },
                "actions": [
                    {
                        "step": 0,
                        "action": {"kind": "open", "row": 1, "col": 2},
                        "click": {"kind": "open", "row": 1, "col": 2, "method": "mouse_event"},
                        "terminal_dialog": "游戏失败",
                        "terminal_detected_at": "after_click",
                        "basic_audit": {
                            "available": True,
                            "target": {
                                "known_mine": True,
                                "known_safe": False,
                                "conflict": False,
                                "witnesses": [{"row": 0, "col": 1, "clue": 1, "flags": 0, "hidden": 1}],
                            },
                        },
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = windows_agent.analyze_game_trace(trace_path)

    assert report["basic_known_mine_opens"] == 1
    assert report["terminal_action"]["basic_target"]["known_mine"]
    assert "terminal_basic_known_mine" in report["signals"]
    assert "basic_known_mine_opens" in report["signals"]


def test_analyze_game_trace_counts_repeated_open_targets(tmp_path: Path) -> None:
    trace_path = tmp_path / "game_002.json"
    trace_path.write_text(
        json.dumps(
            {
                "summary": {
                    "won": False,
                    "lost": True,
                    "done": True,
                    "agent_steps": 2,
                    "elapsed_seconds": 3.0,
                    "revealed_safe_cells": 120,
                },
                "actions": [
                    {
                        "step": 0,
                        "action": {"kind": "open", "row": 1, "col": 2},
                        "click": {"kind": "open", "row": 1, "col": 2, "method": "sendinput"},
                    },
                    {
                        "step": 1,
                        "action": {"kind": "open", "row": 1, "col": 2},
                        "click": {"kind": "open", "row": 1, "col": 2, "method": "sendinput"},
                    },
                ],
            }
        ),
        encoding="utf-8",
    )

    report = windows_agent.analyze_game_trace(trace_path)

    assert report["repeat_open_target_actions"] == 1
    assert "repeat_open_targets" in report["signals"]


def test_analyze_game_trace_reports_small_first_open(tmp_path: Path) -> None:
    trace_path = tmp_path / "game_003.json"
    trace_path.write_text(
        json.dumps(
            {
                "summary": {
                    "won": False,
                    "lost": True,
                    "done": True,
                    "agent_steps": 5,
                    "elapsed_seconds": 12.0,
                    "revealed_safe_cells": 20,
                    "first_open_action": {
                        "action": {"kind": "open", "row": 8, "col": 15},
                        "revealed_delta": 3,
                        "target_revealed_after_open": True,
                        "after_read_repairs": 0,
                        "after_read_restores": 0,
                    },
                },
                "actions": [
                    {
                        "step": 0,
                        "action": {"kind": "open", "row": 8, "col": 15},
                        "click": {"kind": "open", "row": 8, "col": 15, "method": "mouse_event"},
                    }
                ],
            }
        ),
        encoding="utf-8",
    )

    report = windows_agent.analyze_game_trace(trace_path)

    assert report["first_open_revealed_delta"] == 3
    assert not report["first_open_dirty_read"]
    assert "small_first_open" in report["signals"]


def test_analyze_log_diagnosis_prioritizes_read_instability() -> None:
    games = [
        {"won": False, "lost": True, "elapsed_seconds": 30.0},
        {"won": False, "lost": True, "elapsed_seconds": 40.0},
    ]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"read_restores": 2}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "board_read_instability"
    assert diagnosis["next_focus"] == "compare_read_mode"
    assert any("--read-mode accurate" in item for item in diagnosis["recommendations"])


def test_analyze_log_diagnosis_prioritizes_click_execution() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"open_zero_progress": 1, "read_restores": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "click_execution"
    assert diagnosis["next_focus"] == "compare_click_method"
    assert any("--click-method mouse_event" in item for item in diagnosis["recommendations"])


def test_analyze_log_diagnosis_reports_sendinput_fallback() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"sendinput_fallback": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "click_input_backend"
    assert diagnosis["next_focus"] == "force_mouse_event"
    assert diagnosis["sendinput_fallback_signals"] == 1
    assert any("--click-method mouse_event" in item for item in diagnosis["recommendations"])


def test_analyze_log_diagnosis_reports_read_recovery_dependency() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"read_recoveries": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "board_read_recovery_dependency"
    assert diagnosis["next_focus"] == "compare_read_mode"
    assert diagnosis["read_recovery_signals"] == 1
    assert any("--read-mode accurate" in item for item in diagnosis["recommendations"])


def test_analyze_log_diagnosis_reports_small_first_open() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"small_first_open": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "small_first_open"
    assert diagnosis["next_focus"] == "inspect_first_open"
    assert diagnosis["small_first_open_signals"] == 1


def test_analyze_log_diagnosis_reports_dirty_first_open_read() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"dirty_first_open_read": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "dirty_first_open_read"
    assert diagnosis["next_focus"] == "compare_read_mode"
    assert diagnosis["dirty_first_open_signals"] == 1


def test_analyze_log_diagnosis_reports_basic_known_mine() -> None:
    games = [{"won": False, "lost": True, "elapsed_seconds": 20.0}]
    diagnosis = windows_agent.analyze_log_diagnosis(
        games,
        Counter({"terminal_basic_known_mine": 1}),
        target_win_rate=0.4,
        target_avg_seconds=60.0,
    )

    assert diagnosis["primary_issue"] == "basic_logic_or_state"
    assert diagnosis["next_focus"] == "audit_terminal_board"
    assert diagnosis["basic_mine_signals"] == 1


def test_capture_client_array_parks_cursor_before_read(monkeypatch) -> None:
    events: list[str] = []

    class FakeDesktop:
        capture_delay = 0.0
        capture_backend = "window"

        def park_cursor(self) -> None:
            events.append("park")

        def client_bounds(self) -> tuple[int, int, int, int]:
            events.append("bounds")
            return (10, 20, 30, 40)

        def _capture_client_window_array(self) -> np.ndarray:
            events.append("capture")
            return np.zeros((4, 4, 3), dtype=np.uint8)

        def ensure_foreground(self) -> bool:
            raise AssertionError("not expected in window capture mode")

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)

    board, left, top = windows_agent.WindowsMinesweeper.capture_client_array(FakeDesktop())

    assert events[0] == "park"
    assert events.index("park") < events.index("capture")
    assert (left, top) == (10, 20)
    assert board.shape == (4, 4, 3)


def test_auto_capture_prefers_foreground_region_grab(monkeypatch) -> None:
    events: list[str] = []

    class FakeDesktop:
        hwnd = 321
        capture_delay = 0.0
        capture_backend = "auto"
        _mss = object()

        def park_cursor(self) -> None:
            events.append("park")

        def client_bounds(self) -> tuple[int, int, int, int]:
            events.append("bounds")
            return (10, 20, 30, 40)

        def capture_region_array(self, left: int, top: int, right: int, bottom: int) -> np.ndarray:
            events.append("region")
            assert (left, top, right, bottom) == (10, 20, 30, 40)
            return np.zeros((20, 20, 3), dtype=np.uint8)

        def _capture_client_window_array(self) -> np.ndarray:
            raise AssertionError("foreground auto capture should use the faster region path")

    monkeypatch.setattr(windows_agent.time, "sleep", lambda seconds: None)
    monkeypatch.setattr(windows_agent.win32gui, "GetForegroundWindow", lambda: 321)

    board, left, top = windows_agent.WindowsMinesweeper.capture_client_array(FakeDesktop())

    assert events == ["park", "bounds", "region"]
    assert (left, top) == (10, 20)
    assert board.shape == (20, 20, 3)


def test_read_board_from_array_reuses_unchanged_cells() -> None:
    cell_w = 32
    cell_h = 32
    board_array = np.full((windows_agent.ROWS * cell_h, windows_agent.COLS * cell_w, 3), [166, 212, 247], dtype=np.uint8)
    prev_array = board_array.copy()
    board_array[0:cell_h, 2 * cell_w : 3 * cell_w] = [190, 190, 190]
    board_array[10:22, 2 * cell_w + 14 : 2 * cell_w + 18] = [55, 75, 175]
    prev_board = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(
            x_lines=[i * cell_w for i in range(windows_agent.COLS + 1)],
            y_lines=[i * cell_h for i in range(windows_agent.ROWS + 1)],
        ),
        screenshot=Image.fromarray(prev_array, mode="RGB"),
        pixels=prev_array,
    )

    board = windows_agent.WindowsMinesweeper._read_board_from_array(
        object(),
        board_array,
        prev_board.grid,
        Image.fromarray(board_array, mode="RGB"),
        board_array,
        0,
        prev_board,
    )

    assert board.revealed[0, 2]
    assert board.adjacent[0, 2] == 1
    assert not board.revealed[0, 3]


def test_read_board_from_array_reuses_cells_with_small_pixel_noise() -> None:
    cell_w = 32
    cell_h = 32
    board_array = np.full((windows_agent.ROWS * cell_h, windows_agent.COLS * cell_w, 3), [166, 212, 247], dtype=np.uint8)
    prev_array = board_array.copy()
    prev_array[0:cell_h, 4 * cell_w : 5 * cell_w] = [190, 190, 190]
    prev_array[10:22, 4 * cell_w + 14 : 4 * cell_w + 18] = [55, 75, 175]
    board_array[0:cell_h, 2 * cell_w : 3 * cell_w] = [190, 190, 190]
    board_array[10:22, 2 * cell_w + 14 : 2 * cell_w + 18] = [55, 75, 175]
    board_array[0:cell_h, 4 * cell_w : 5 * cell_w] = np.clip(
        prev_array[0:cell_h, 4 * cell_w : 5 * cell_w].astype(np.int16) + 1,
        0,
        255,
    ).astype(np.uint8)
    prev_revealed = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool)
    prev_revealed[0, 4] = True
    prev_adjacent = np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8)
    prev_adjacent[0, 4] = 1
    prev_board = windows_agent.ScreenBoard(
        revealed=prev_revealed,
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=prev_adjacent,
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(
            x_lines=[i * cell_w for i in range(windows_agent.COLS + 1)],
            y_lines=[i * cell_h for i in range(windows_agent.ROWS + 1)],
        ),
        screenshot=Image.fromarray(prev_array, mode="RGB"),
        pixels=prev_array,
    )

    board = windows_agent.WindowsMinesweeper._read_board_from_array(
        object(),
        board_array,
        prev_board.grid,
        Image.fromarray(board_array, mode="RGB"),
        board_array,
        0,
        prev_board,
    )

    assert board.revealed[0, 4]
    assert board.adjacent[0, 4] == 1
    assert board.revealed[0, 2]


def test_read_board_from_array_reclassifies_unchanged_cells_previously_seen_as_hidden() -> None:
    cell_w = 32
    cell_h = 32
    board_array = np.full((windows_agent.ROWS * cell_h, windows_agent.COLS * cell_w, 3), [166, 212, 247], dtype=np.uint8)
    board_array[0:cell_h, 2 * cell_w : 3 * cell_w] = [190, 190, 190]
    board_array[10:22, 2 * cell_w + 14 : 2 * cell_w + 18] = [55, 75, 175]
    previous = windows_agent.ScreenBoard(
        revealed=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        flagged=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        adjacent=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=np.int8),
        mine_like=np.zeros((windows_agent.ROWS, windows_agent.COLS), dtype=bool),
        grid=windows_agent.Grid(
            x_lines=[i * cell_w for i in range(windows_agent.COLS + 1)],
            y_lines=[i * cell_h for i in range(windows_agent.ROWS + 1)],
        ),
        screenshot=Image.fromarray(board_array, mode="RGB"),
        pixels=board_array.copy(),
    )

    board = windows_agent.WindowsMinesweeper._read_board_from_array(
        object(),
        board_array,
        previous.grid,
        Image.fromarray(board_array, mode="RGB"),
        board_array,
        0,
        previous,
    )

    assert board.revealed[0, 2]
    assert board.adjacent[0, 2] == 1
