from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image, ImageDraw, ImageGrab
import win32api
import win32con
import win32gui
import win32process
import win32ui

try:
    from mss import MSS
except Exception:  # pragma: no cover - optional dependency
    MSS = None

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from minesweeper_rl.features import action_channel, action_to_index, decode_action_index, encode_state
from minesweeper_rl.solver import MinesweeperSolver
from minesweeper_rl.trainer import load_checkpoint
from minesweeper_rl.types import Action, ActionType


ROWS = 16
COLS = 30
MINES = 99
SAFE_CELLS = ROWS * COLS - MINES
STOP_FILE = ROOT / "artifacts" / "windows_agent" / "STOP"
HOTKEY_ID = 9919
HOTKEY_MODIFIERS = win32con.MOD_CONTROL | win32con.MOD_ALT | 0x4000
HOTKEY_VK = ord("Q")
PW_CLIENTONLY = 0x00000001
PW_RENDERFULLCONTENT = 0x00000002
MAX_REPAIRED_IMPOSSIBLE_NUMBERS = 8


@dataclass(frozen=True)
class Grid:
    x_lines: list[int]
    y_lines: list[int]

    @property
    def cell_width(self) -> float:
        return (self.x_lines[-1] - self.x_lines[0]) / COLS

    @property
    def cell_height(self) -> float:
        return (self.y_lines[-1] - self.y_lines[0]) / ROWS

    def center(self, row: int, col: int) -> tuple[int, int]:
        x = (self.x_lines[col] + self.x_lines[col + 1]) / 2.0
        y = (self.y_lines[row] + self.y_lines[row + 1]) / 2.0
        return int(round(x)), int(round(y))

    def crop_box(self, row: int, col: int, margin: int = 6) -> tuple[int, int, int, int]:
        return (
            self.x_lines[col] + margin,
            self.y_lines[row] + margin,
            self.x_lines[col + 1] - margin,
            self.y_lines[row + 1] - margin,
        )


@dataclass
class ScreenBoard:
    revealed: np.ndarray
    flagged: np.ndarray
    adjacent: np.ndarray
    mine_like: np.ndarray
    grid: Grid
    screenshot: Image.Image | None
    pixels: np.ndarray | None = None
    step_count: int = 0
    read_repairs: int = 0
    read_restores: int = 0

    rows: int = ROWS
    cols: int = COLS
    mine_count: int = MINES
    total_cells: int = ROWS * COLS
    total_safe_cells: int = SAFE_CELLS
    mines_placed: bool = False
    done: bool = False
    won: bool = False
    lost: bool = False

    def __post_init__(self) -> None:
        revealed_count = int(self.revealed.sum())
        self.mines_placed = revealed_count > 0 or bool(self.flagged.any())
        self.won = revealed_count == SAFE_CELLS and not bool(self.mine_like.any())
        self.lost = bool(self.mine_like.any()) and not self.won
        self.done = self.won or self.lost

    def hidden_mask(self) -> np.ndarray:
        return ~(self.revealed | self.flagged)

    def legal_open_mask(self) -> np.ndarray:
        return self.hidden_mask()

    def remaining_mines_estimate(self) -> int:
        return max(0, self.mine_count - int(self.flagged.sum()))

    def neighbors(self, row: int, col: int) -> list[tuple[int, int]]:
        cells: list[tuple[int, int]] = []
        for nr in range(max(0, row - 1), min(self.rows, row + 2)):
            for nc in range(max(0, col - 1), min(self.cols, col + 2)):
                if nr == row and nc == col:
                    continue
                cells.append((nr, nc))
        return cells


class WindowsMinesweeper:
    def __init__(
        self,
        window_class: str = "Minesweeper",
        capture_delay: float = 0.005,
        capture_backend: str = "auto",
        read_mode: str = "fast",
        click_pause: float = 0.01,
        cursor_settle: float = 0.005,
    ) -> None:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        self.window_class = window_class
        self.capture_delay = capture_delay
        self.capture_backend = capture_backend
        self.read_mode = read_mode
        self.click_pause = click_pause
        self.cursor_settle = cursor_settle
        self._mss = None
        if capture_backend not in {"auto", "pil", "mss", "window"}:
            raise ValueError("capture_backend must be one of: auto, pil, mss, window")
        if capture_backend in {"auto", "mss"} and MSS is not None:
            try:
                self._mss = MSS()
            except Exception:
                if capture_backend == "mss":
                    raise
                self._mss = None
        elif capture_backend == "mss":
            raise RuntimeError("mss capture backend requested but the package is not available")
        self.hwnd = self._find_window()
        self.grid: Grid | None = None

    def _find_window(self) -> int:
        matches: list[int] = []

        def callback(hwnd: int, _: object) -> None:
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetClassName(hwnd) == self.window_class:
                matches.append(hwnd)

        win32gui.EnumWindows(callback, None)
        if not matches:
            raise RuntimeError("Minesweeper window not found; start the desktop game first")
        return matches[0]

    def focus(self, timeout: float = 0.25) -> bool:
        try:
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
            flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            win32gui.BringWindowToTop(self.hwnd)
            win32gui.SetForegroundWindow(self.hwnd)
        except Exception:
            pass
        deadline = time.time() + timeout
        while time.time() < deadline:
            if win32gui.GetForegroundWindow() == self.hwnd:
                return True
            time.sleep(0.01)
        return win32gui.GetForegroundWindow() == self.hwnd

    def ensure_foreground(self) -> bool:
        if win32gui.GetForegroundWindow() == self.hwnd:
            return True
        return self.focus(timeout=0.75)

    def capture(self) -> Image.Image:
        array, _, _ = self.capture_client_array()
        return Image.fromarray(array, mode="RGB")

    def _capture_client_window_array(self) -> np.ndarray:
        _, _, width, height = win32gui.GetClientRect(self.hwnd)
        if width <= 0 or height <= 0:
            raise RuntimeError("Minesweeper window has an empty client area")

        hwnd_dc = win32gui.GetWindowDC(self.hwnd)
        source_dc = win32ui.CreateDCFromHandle(hwnd_dc)
        memory_dc = source_dc.CreateCompatibleDC()
        bitmap = win32ui.CreateBitmap()
        bitmap.CreateCompatibleBitmap(source_dc, width, height)
        old_object = memory_dc.SelectObject(bitmap)

        try:
            hdc = memory_dc.GetSafeHdc()
            rendered = ctypes.windll.user32.PrintWindow(
                self.hwnd,
                hdc,
                PW_CLIENTONLY | PW_RENDERFULLCONTENT,
            )
            if not rendered:
                rendered = ctypes.windll.user32.PrintWindow(self.hwnd, hdc, PW_CLIENTONLY)
            if not rendered:
                raise RuntimeError("PrintWindow could not render the Minesweeper client area")

            bits = bitmap.GetBitmapBits(True)
            array = np.frombuffer(bits, dtype=np.uint8).reshape((height, width, 4))
            return array[:, :, [2, 1, 0]].copy()
        finally:
            try:
                memory_dc.SelectObject(old_object)
            except Exception:
                pass
            win32gui.DeleteObject(bitmap.GetHandle())
            memory_dc.DeleteDC()
            source_dc.DeleteDC()
            win32gui.ReleaseDC(self.hwnd, hwnd_dc)

    def capture_region_array(self, left: int, top: int, right: int, bottom: int) -> np.ndarray:
        width = max(1, right - left)
        height = max(1, bottom - top)
        if self._mss is not None:
            screenshot = self._mss.grab({"left": left, "top": top, "width": width, "height": height})
            return np.asarray(screenshot)[:, :, [2, 1, 0]]
        return np.asarray(ImageGrab.grab(bbox=(left, top, right, bottom)).convert("RGB"))

    def capture_client_array(self) -> tuple[np.ndarray, int, int]:
        time.sleep(self.capture_delay)
        left, top, right, bottom = self.client_bounds()
        if self.capture_backend in {"auto", "window"}:
            try:
                return self._capture_client_window_array(), left, top
            except Exception:
                if self.capture_backend == "window":
                    raise
        if not self.ensure_foreground():
            raise RuntimeError("Minesweeper window is not foreground; refusing to read covered screen pixels")
        return self.capture_region_array(left, top, right, bottom), left, top

    def capture_client_grid(self) -> tuple[np.ndarray, Grid, Grid]:
        array, left, top = self.capture_client_array()
        image = Image.fromarray(array, mode="RGB")
        local_grid = detect_local_grid(image)
        screen_grid = offset_grid(local_grid, dx=left, dy=top)
        return array, local_grid, screen_grid

    def capture_board(self, grid: Grid) -> tuple[Image.Image, Grid]:
        array, local_grid = self.capture_board_array(grid)
        return Image.fromarray(array, mode="RGB"), local_grid

    def capture_board_array(self, grid: Grid) -> tuple[np.ndarray, Grid]:
        client_array, client_left, client_top = self.capture_client_array()
        board_left = max(0, grid.x_lines[0] - client_left)
        board_top = max(0, grid.y_lines[0] - client_top)
        board_right = min(client_array.shape[1], grid.x_lines[-1] - client_left + 1)
        board_bottom = min(client_array.shape[0], grid.y_lines[-1] - client_top + 1)
        if board_right <= board_left or board_bottom <= board_top:
            raise RuntimeError("detected grid lies outside the Minesweeper client area")
        array = client_array[board_top:board_bottom, board_left:board_right].copy()
        left = client_left + board_left
        top = client_top + board_top
        local_grid = Grid(
            x_lines=[x - left for x in grid.x_lines],
            y_lines=[y - top for y in grid.y_lines],
        )
        return array, local_grid

    def client_bounds(self) -> tuple[int, int, int, int]:
        _, _, width, height = win32gui.GetClientRect(self.hwnd)
        if width <= 0 or height <= 0:
            self.focus(timeout=1.0)
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        _, _, width, height = win32gui.GetClientRect(self.hwnd)
        if width <= 0 or height <= 0:
            raise RuntimeError("Minesweeper window has an empty client area; restore the game window first")
        return left, top, left + width, top + height

    def detect_grid_from_screenshot(self, screenshot: Image.Image) -> Grid:
        left, top, right, bottom = self.client_bounds()
        client_image = screenshot.crop((left, top, right, bottom))
        grid = detect_local_grid(client_image)
        return Grid(
            x_lines=[x + left for x in grid.x_lines],
            y_lines=[y + top for y in grid.y_lines],
        )

    def read_board(
        self,
        step_count: int = 0,
        keep_screenshot: bool = True,
        previous_board: ScreenBoard | None = None,
    ) -> ScreenBoard:
        if self.dialog_is_open():
            raise RuntimeError("a Minesweeper dialog is open; handle it before reading the board")
        if self.read_mode == "fast":
            return self._read_board_fast(
                step_count=step_count,
                keep_screenshot=keep_screenshot,
                previous_board=previous_board,
            )
        if self.grid is None:
            array, read_grid, screen_grid = self.capture_client_grid()
            self.grid = screen_grid
            source_image = Image.fromarray(array, mode="RGB")
        else:
            source_image, read_grid = self.capture_board(self.grid)
        screenshot = source_image.crop(grid_bbox(read_grid)) if keep_screenshot else None
        revealed = np.zeros((ROWS, COLS), dtype=bool)
        flagged = np.zeros((ROWS, COLS), dtype=bool)
        adjacent = np.zeros((ROWS, COLS), dtype=np.int8)
        mine_like = np.zeros((ROWS, COLS), dtype=bool)

        for row in range(ROWS):
            for col in range(COLS):
                crop = source_image.crop(read_grid.crop_box(row, col))
                cell = classify_cell(crop)
                if cell["kind"] == "flagged":
                    flagged[row, col] = True
                elif cell["kind"] == "mine":
                    revealed[row, col] = True
                    mine_like[row, col] = True
                elif cell["kind"] == "revealed":
                    revealed[row, col] = True
                    adjacent[row, col] = int(cell["number"])

        repairs = repair_impossible_numbers(revealed, adjacent, mine_like)
        board_pixels = np.asarray(source_image.crop(grid_bbox(read_grid)))

        return ScreenBoard(
            revealed=revealed,
            flagged=flagged,
            adjacent=adjacent,
            mine_like=mine_like,
            grid=read_grid,
            screenshot=screenshot,
            pixels=board_pixels,
            step_count=step_count,
            read_repairs=repairs,
        )

    def _read_board_fast(
        self,
        step_count: int = 0,
        keep_screenshot: bool = True,
        previous_board: ScreenBoard | None = None,
    ) -> ScreenBoard:
        if self.grid is None:
            board_array, grid_for_read, screen_grid = self.capture_client_grid()
            self.grid = screen_grid
            board_pixels = crop_grid_array(board_array, grid_for_read)
            screenshot = Image.fromarray(board_pixels, mode="RGB") if keep_screenshot else None
            return self._read_board_from_array(
                board_array,
                grid_for_read,
                screenshot,
                board_pixels,
                step_count,
                previous_board,
            )

        board_array, grid_for_read = self.capture_board_array(self.grid)
        board_pixels = board_array
        screenshot = Image.fromarray(board_pixels, mode="RGB") if keep_screenshot else None
        return self._read_board_from_array(
            board_array,
            grid_for_read,
            screenshot,
            board_pixels,
            step_count,
            previous_board,
        )

    def _read_board_from_array(
        self,
        board_array: np.ndarray,
        grid: Grid,
        screenshot: Image.Image | None,
        board_pixels: np.ndarray,
        step_count: int,
        previous_board: ScreenBoard | None = None,
    ) -> ScreenBoard:
        revealed = np.zeros((ROWS, COLS), dtype=bool)
        flagged = np.zeros((ROWS, COLS), dtype=bool)
        adjacent = np.zeros((ROWS, COLS), dtype=np.int8)
        mine_like = np.zeros((ROWS, COLS), dtype=bool)
        changed_mask = None
        if previous_board is not None and previous_board.pixels is not None and previous_board.pixels.shape == board_array.shape:
            changed_mask = np.any(board_array != previous_board.pixels, axis=2)

        for row in range(ROWS):
            for col in range(COLS):
                x0, y0, x1, y1 = grid.crop_box(row, col)
                if changed_mask is not None and not bool(changed_mask[y0:y1, x0:x1].any()):
                    revealed[row, col] = previous_board.revealed[row, col]
                    flagged[row, col] = previous_board.flagged[row, col]
                    adjacent[row, col] = previous_board.adjacent[row, col]
                    mine_like[row, col] = previous_board.mine_like[row, col]
                    continue
                cell = classify_cell_fast(board_array[y0:y1, x0:x1])
                if cell["kind"] == "flagged":
                    flagged[row, col] = True
                elif cell["kind"] == "mine":
                    revealed[row, col] = True
                    mine_like[row, col] = True
                elif cell["kind"] == "revealed":
                    revealed[row, col] = True
                    adjacent[row, col] = int(cell["number"])

        repairs = repair_impossible_numbers(revealed, adjacent, mine_like)

        return ScreenBoard(
            revealed=revealed,
            flagged=flagged,
            adjacent=adjacent,
            mine_like=mine_like,
            grid=grid,
            screenshot=screenshot,
            pixels=board_pixels,
            step_count=step_count,
            read_repairs=repairs,
        )

    def _full_monitor(self) -> dict[str, int]:
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        _, _, width, height = win32gui.GetClientRect(self.hwnd)
        return {"left": left, "top": top, "width": width, "height": height}

    def click_action(self, action: Action) -> None:
        if self.dialog_is_open():
            raise RuntimeError("a Minesweeper dialog is open; handle it before clicking the board")
        if action.kind != ActionType.OPEN:
            raise ValueError(f"open-only agent expected OPEN action, got {action.kind}")
        if self.grid is None:
            _, _, self.grid = self.capture_client_grid()
        x, y = self.grid.center(action.row, action.col)
        if not self.ensure_foreground():
            raise RuntimeError("Minesweeper window is not foreground; refusing to click")
        win32api.SetCursorPos((x, y))
        time.sleep(max(0.0, self.cursor_settle))
        click_mouse(
            win32con.MOUSEEVENTF_LEFTDOWN,
            win32con.MOUSEEVENTF_LEFTUP,
            x,
            y,
            pause=self.click_pause,
        )

    def new_game(self, option: str = "new") -> None:
        if option not in {"new", "restart", "continue"}:
            raise ValueError("option must be one of: new, restart, continue")
        if not self.focus(timeout=1.0):
            raise RuntimeError("could not focus Minesweeper before starting a new game")
        win32api.keybd_event(win32con.VK_F2, 0, 0, 0)
        time.sleep(0.03)
        win32api.keybd_event(win32con.VK_F2, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.2)
        dialog = self.find_new_game_dialog()
        if dialog is not None:
            self.choose_dialog_option(option=option)
        time.sleep(0.1)
        self.grid = None

    def dialog_is_open(self) -> bool:
        return self.find_new_game_dialog() is not None

    def dialog_title(self) -> str | None:
        dialog = self.find_new_game_dialog()
        return None if dialog is None else win32gui.GetWindowText(dialog)

    def find_new_game_dialog(self) -> int | None:
        _, target_pid = win32process.GetWindowThreadProcessId(self.hwnd)
        matches: list[int] = []

        def callback(hwnd: int, _: object) -> None:
            if not win32gui.IsWindowVisible(hwnd):
                return
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid == target_pid and win32gui.GetClassName(hwnd) == "#32770":
                matches.append(hwnd)

        win32gui.EnumWindows(callback, None)
        return matches[0] if matches else None

    def wait_until_no_dialog(self, timeout: float = 2.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            if not self.dialog_is_open():
                return True
            time.sleep(0.05)
        return not self.dialog_is_open()

    def choose_dialog_option(self, option: str, timeout: float = 2.0, attempts: int = 2) -> bool:
        for _ in range(attempts):
            dialog = self.find_new_game_dialog()
            if dialog is None:
                self.grid = None
                return True
            self.click_dialog_option(dialog, option=option)
            if self.wait_until_no_dialog(timeout=timeout):
                self.grid = None
                return True
            time.sleep(0.1)
        self.grid = None
        return not self.dialog_is_open()

    def click_dialog_option(self, dialog: int, option: str) -> None:
        title = win32gui.GetWindowText(dialog)
        button = self.find_dialog_button(dialog, option)
        if button is None and option == "restart":
            button = self.find_dialog_button(dialog, "new")
        if button is not None:
            left, top, right, bottom = win32gui.GetWindowRect(button)
            x = (left + right) // 2
            y = (top + bottom) // 2
            win32api.SetCursorPos((x, y))
            time.sleep(max(0.0, self.cursor_settle))
            click_mouse(
                win32con.MOUSEEVENTF_LEFTDOWN,
                win32con.MOUSEEVENTF_LEFTUP,
                x,
                y,
                pause=self.click_pause,
            )
            return

        origin = win32gui.ClientToScreen(dialog, (0, 0))
        if "新游戏" in title:
            # Client-space row centers for:
            #   1. exit and start new game
            #   2. restart this game
            #   3. continue game
            option_y = {"new": 135, "restart": 235, "continue": 335}[option]
            x = origin[0] + 110
            y = origin[1] + option_y
        else:
            # End-of-game dialogs use bottom buttons:
            #   left: exit, middle: restart this game, right: play another/new game.
            option_x = {"new": 575, "restart": 365, "continue": 575}[option]
            x = origin[0] + option_x
            y = origin[1] + 350
        win32api.SetCursorPos((x, y))
        time.sleep(max(0.0, self.cursor_settle))
        click_mouse(
            win32con.MOUSEEVENTF_LEFTDOWN,
            win32con.MOUSEEVENTF_LEFTUP,
            x,
            y,
            pause=self.click_pause,
        )

    def find_dialog_button(self, dialog: int, option: str) -> int | None:
        buttons: list[int] = []

        def callback(hwnd: int, _: object) -> None:
            if win32gui.IsWindowVisible(hwnd) and win32gui.GetClassName(hwnd) == "Button":
                buttons.append(hwnd)

        win32gui.EnumChildWindows(dialog, callback, None)
        for button in buttons:
            if dialog_button_matches(win32gui.GetWindowText(button), option):
                return button
        return None


def click_mouse(down: int, up: int, x: int, y: int, pause: float = 0.01) -> None:
    win32api.mouse_event(down, x, y, 0, 0)
    time.sleep(max(0.0, pause))
    win32api.mouse_event(up, x, y, 0, 0)


def dialog_button_matches(text: str, option: str) -> bool:
    normalized = text.lower()
    if option == "new":
        tokens = ("&p", "&n", "再玩", "开始新游戏")
    elif option == "restart":
        tokens = ("&r", "重新开始")
    elif option == "continue":
        tokens = ("&k", "继续")
    else:
        raise ValueError("option must be one of: new, restart, continue")
    return any(token in normalized for token in tokens)


def grid_bbox(grid: Grid) -> tuple[int, int, int, int]:
    return grid.x_lines[0], grid.y_lines[0], grid.x_lines[-1] + 1, grid.y_lines[-1] + 1


def crop_grid_array(array: np.ndarray, grid: Grid) -> np.ndarray:
    x0, y0, x1, y1 = grid_bbox(grid)
    return array[y0:y1, x0:x1].copy()


def offset_grid(grid: Grid, dx: int, dy: int) -> Grid:
    return Grid(x_lines=[x + dx for x in grid.x_lines], y_lines=[y + dy for y in grid.y_lines])


def localize_grid(grid: Grid) -> Grid:
    return offset_grid(grid, dx=-grid.x_lines[0], dy=-grid.y_lines[0])


def detect_local_grid(image: Image.Image) -> Grid:
    try:
        grid = detect_grid(image)
    except RuntimeError:
        grid = fallback_grid(image=image)
    if abs(grid.cell_width - grid.cell_height) > max(3.0, min(grid.cell_width, grid.cell_height) * 0.08):
        raise RuntimeError(
            f"detected non-square cells: width={grid.cell_width:.2f}, height={grid.cell_height:.2f}"
        )
    return grid


def detect_grid(image: Image.Image) -> Grid:
    array = np.asarray(image.convert("RGB"))
    dark = (array[:, :, 0] < 45) & (array[:, :, 1] < 55) & (array[:, :, 2] < 90)
    row_counts = dark.sum(axis=1)
    y_candidates = _line_centers(row_counts, min_count=1500, max_width=8, start=80, end=image.height - 120)
    y_lines = _best_grid_sequence(y_candidates, expected=ROWS + 1, min_spacing=45, max_spacing=130)

    col_counts = dark[y_lines[0] : y_lines[-1] + 1, :].sum(axis=0)
    x_candidates = _line_centers(col_counts, min_count=500, max_width=8, start=0, end=image.width)
    x_lines = _best_grid_sequence(x_candidates, expected=COLS + 1, min_spacing=45, max_spacing=130)

    return Grid(x_lines=x_lines, y_lines=y_lines)


def _line_centers(
    counts: np.ndarray,
    min_count: int,
    max_width: int,
    start: int,
    end: int,
) -> list[int]:
    mask = counts >= min_count
    mask[:start] = False
    mask[end:] = False
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []

    centers: list[int] = []
    run_start = int(indices[0])
    previous = int(indices[0])
    for index in indices[1:]:
        current = int(index)
        if current <= previous + 1:
            previous = current
            continue
        if previous - run_start + 1 <= max_width:
            centers.append((run_start + previous) // 2)
        run_start = previous = current
    if previous - run_start + 1 <= max_width:
        centers.append((run_start + previous) // 2)
    return centers


def _best_grid_sequence(
    candidates: list[int],
    expected: int,
    min_spacing: int,
    max_spacing: int,
) -> list[int]:
    candidates = sorted(candidates)
    best: tuple[float, list[int]] | None = None
    for start in range(0, max(0, len(candidates) - expected + 1)):
        seq = candidates[start : start + expected]
        if len(seq) != expected:
            continue
        diffs = np.diff(seq)
        if diffs.min() < min_spacing or diffs.max() > max_spacing:
            continue
        score = float(diffs.std())
        if best is None or score < best[0]:
            best = (score, seq)
    if best is None:
        raise RuntimeError(f"could not detect {expected} grid lines from candidates {candidates}")
    return best[1]


def fallback_grid(desktop: WindowsMinesweeper | None = None, image: Image.Image | None = None) -> Grid:
    if desktop is not None:
        left, top = win32gui.ClientToScreen(desktop.hwnd, (0, 0))
        _, _, width, height = win32gui.GetClientRect(desktop.hwnd)
        x0 = left + round(width * (310 / 2560))
        x1 = left + round(width * (2249 / 2560))
        y0 = top + round(height * (107 / 1260))
        y1 = top + round(height * (1141 / 1260))
    else:
        width = image.width if image is not None else 2560
        height = image.height if image is not None else 1440
        x0 = round(width * (310 / 2560))
        x1 = round(width * (2249 / 2560))
        y0 = round(height * (191 / 1440))
        y1 = round(height * (1225 / 1440))
    return Grid(x_lines=spaced_lines(x0, x1, COLS), y_lines=spaced_lines(y0, y1, ROWS))


def spaced_lines(start: int, end: int, cells: int) -> list[int]:
    return [int(round(start + (end - start) * index / cells)) for index in range(cells + 1)]


def _center_patch(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    y0 = max(0, height // 4)
    y1 = min(height, height - height // 4)
    x0 = max(0, width // 4)
    x1 = min(width, width - width // 4)
    patch = array[y0:y1, x0:x1]
    return patch if patch.size else array


def classify_cell(crop: Image.Image) -> dict[str, Any]:
    array = np.asarray(crop.convert("RGB"))
    saturation, value = _saturation_value(array)
    center = _center_patch(array)
    center_mean = center.reshape(-1, 3).mean(axis=0)
    center_saturation, center_value = _saturation_value(center)
    median_saturation = float(np.quantile(saturation, 0.5))
    value_mean = float(value.mean())
    revealed_background = median_saturation < 0.24 and value_mean > 0.62
    center_blue = center_mean[2] - center_mean[0] > 42 and center_mean[2] - center_mean[1] > 15 and center_mean[1] > center_mean[0]
    center_revealed = float(np.quantile(center_saturation, 0.5)) < 0.25 and float(center_value.mean()) > 0.45

    red_pixels = (array[:, :, 0] > 145) & (array[:, :, 1] < 110) & (array[:, :, 2] < 115)
    yellow_pixels = (array[:, :, 0] > 145) & (array[:, :, 1] > 100) & (array[:, :, 2] < 90)
    dark_pixels = (array[:, :, 0] < 45) & (array[:, :, 1] < 45) & (array[:, :, 2] < 55)
    if center_blue and not revealed_background and not center_revealed:
        if int(red_pixels.sum()) > 35 or int(yellow_pixels.sum()) > 35:
            return {"kind": "flagged", "number": 0}
        if int(dark_pixels.sum()) > 220 and int(red_pixels.sum()) > 20:
            return {"kind": "mine", "number": 0}
        return {"kind": "hidden", "number": 0}

    if int(dark_pixels.sum()) > 260 and int(red_pixels.sum()) > 20:
        return {"kind": "mine", "number": 0}

    number = classify_number(array)
    return {"kind": "revealed", "number": number}


def classify_cell_fast(array: np.ndarray) -> dict[str, Any]:
    if array.shape[0] < 14 or array.shape[1] < 14:
        return {"kind": "hidden", "number": 0}

    f = array.astype(np.int16, copy=False)
    center = _center_patch(f)
    center_mean = center.reshape(-1, 3).mean(axis=0)
    center_saturation, center_value = _saturation_value(center)
    center_blue = center_mean[2] - center_mean[0] > 42 and center_mean[2] - center_mean[1] > 15 and center_mean[1] > center_mean[0]
    center_revealed = float(np.quantile(center_saturation, 0.5)) < 0.25 and float(center_value.mean()) > 0.45

    red_pixels = (f[:, :, 0] > 145) & (f[:, :, 1] < 110) & (f[:, :, 2] < 115)
    yellow_pixels = (f[:, :, 0] > 145) & (f[:, :, 1] > 100) & (f[:, :, 2] < 90)
    dark_pixels = (f[:, :, 0] < 45) & (f[:, :, 1] < 45) & (f[:, :, 2] < 55)

    if center_blue and not center_revealed:
        red_count = int(red_pixels.sum())
        yellow_count = int(yellow_pixels.sum())
        if red_count > 35 or yellow_count > 35:
            return {"kind": "flagged", "number": 0}
        if int(dark_pixels.sum()) > 220 and red_count > 20:
            return {"kind": "mine", "number": 0}
        return {"kind": "hidden", "number": 0}

    # Ambiguous cells are cheaper to classify with the full heuristic than to
    # let a false "revealed" guess poison the board state.
    return classify_cell(Image.fromarray(array, mode="RGB"))


def classify_number(array: np.ndarray) -> int:
    hsv_mask = _digit_mask(array)
    if int(hsv_mask.sum()) < 25:
        return 0
    pixels = array[hsv_mask].astype(np.float32)
    color = np.median(pixels, axis=0)
    prototypes = {
        1: np.array([55, 75, 175], dtype=np.float32),
        2: np.array([55, 125, 45], dtype=np.float32),
        3: np.array([175, 35, 35], dtype=np.float32),
        4: np.array([45, 45, 125], dtype=np.float32),
        5: np.array([120, 35, 35], dtype=np.float32),
        6: np.array([35, 125, 125], dtype=np.float32),
        7: np.array([35, 35, 35], dtype=np.float32),
        8: np.array([105, 105, 105], dtype=np.float32),
    }
    distances = {number: float(np.linalg.norm(color - proto)) for number, proto in prototypes.items()}
    return min(distances, key=distances.get)


def max_neighbor_count(row: int, col: int, rows: int = ROWS, cols: int = COLS) -> int:
    return (1 + min(row, 1) + min(rows - 1 - row, 1)) * (1 + min(col, 1) + min(cols - 1 - col, 1)) - 1


def repair_impossible_numbers(
    revealed: np.ndarray,
    adjacent: np.ndarray,
    mine_like: np.ndarray,
    max_repairs: int = MAX_REPAIRED_IMPOSSIBLE_NUMBERS,
) -> int:
    impossible: list[tuple[int, int, int, int]] = []
    for row in range(revealed.shape[0]):
        for col in range(revealed.shape[1]):
            if not revealed[row, col] or mine_like[row, col]:
                continue
            value = int(adjacent[row, col])
            limit = max_neighbor_count(row, col, rows=revealed.shape[0], cols=revealed.shape[1])
            if value > limit:
                impossible.append((row, col, value, limit))

    if not impossible:
        return 0
    if len(impossible) > max_repairs:
        sample = ", ".join(f"r{row}c{col}:{value}>{limit}" for row, col, value, limit in impossible[:8])
        raise RuntimeError(f"invalid board read; impossible numbers detected ({len(impossible)}): {sample}")

    for row, col, _, _ in impossible:
        revealed[row, col] = False
        adjacent[row, col] = 0
        mine_like[row, col] = False
    return len(impossible)


def restore_revealed_cells(board: ScreenBoard, previous: ScreenBoard | None) -> ScreenBoard:
    if previous is None or previous.done or board.done:
        return board
    restored = previous.revealed & ~board.revealed
    if not bool(restored.any()):
        return board

    revealed = board.revealed.copy()
    flagged = board.flagged.copy()
    adjacent = board.adjacent.copy()
    mine_like = board.mine_like.copy()
    revealed[restored] = True
    flagged[restored] = previous.flagged[restored]
    adjacent[restored] = previous.adjacent[restored]
    mine_like[restored] = previous.mine_like[restored]

    return ScreenBoard(
        revealed=revealed,
        flagged=flagged,
        adjacent=adjacent,
        mine_like=mine_like,
        grid=board.grid,
        screenshot=board.screenshot,
        pixels=board.pixels,
        step_count=board.step_count,
        read_repairs=board.read_repairs,
        read_restores=int(restored.sum()),
    )


def _digit_mask(array: np.ndarray) -> np.ndarray:
    saturation, value = _saturation_value(array)
    colorful = (saturation > 0.42) & (value < 0.92)
    dark = value < 0.25
    return colorful | dark


def _saturation_value(array: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    arr = array.astype(np.float32) / 255.0
    maxc = arr.max(axis=2)
    minc = arr.min(axis=2)
    saturation = np.zeros_like(maxc)
    nonzero = maxc > 1e-6
    saturation[nonzero] = (maxc[nonzero] - minc[nonzero]) / maxc[nonzero]
    return saturation, maxc


def board_to_text(board: ScreenBoard) -> str:
    lines: list[str] = []
    for row in range(ROWS):
        chars: list[str] = []
        for col in range(COLS):
            if board.flagged[row, col]:
                chars.append("F")
            elif board.mine_like[row, col]:
                chars.append("*")
            elif not board.revealed[row, col]:
                chars.append("#")
            else:
                value = int(board.adjacent[row, col])
                chars.append("." if value == 0 else str(value))
        lines.append("".join(chars))
    return "\n".join(lines)


def cell_snapshot(board: ScreenBoard, row: int, col: int) -> dict[str, Any]:
    return {
        "row": row,
        "col": col,
        "revealed": bool(board.revealed[row, col]),
        "flagged": bool(board.flagged[row, col]),
        "mine_like": bool(board.mine_like[row, col]),
        "adjacent": int(board.adjacent[row, col]),
    }


def solver_audit(solver: MinesweeperSolver, board: ScreenBoard, action: Action) -> dict[str, Any]:
    if not board.mines_placed:
        return {"available": False, "reason": "before_first_open"}
    if board.done:
        return {"available": False, "reason": "terminal_board"}
    try:
        snapshot = solver.analyze(board)
    except Exception as exc:
        return {"available": False, "error": type(exc).__name__, "message": str(exc)}

    target: dict[str, Any] = {}
    if 0 <= action.row < ROWS and 0 <= action.col < COLS:
        target = {
            "known_safe": bool(snapshot.safe_mask[action.row, action.col]),
            "known_mine": bool(snapshot.mine_mask[action.row, action.col]),
            "risk": float(snapshot.risk_map[action.row, action.col]),
            "frontier": bool(snapshot.frontier_mask[action.row, action.col]),
            "is_best_guess": snapshot.best_guess == (action.row, action.col),
        }

    return {
        "available": True,
        "has_forced_moves": bool(snapshot.has_forced_moves),
        "forced_safe_count": int(snapshot.safe_mask.sum()),
        "forced_mine_count": int(snapshot.mine_mask.sum()),
        "best_guess": None
        if snapshot.best_guess is None
        else {"row": int(snapshot.best_guess[0]), "col": int(snapshot.best_guess[1])},
        "best_guess_risk": None if snapshot.best_guess_risk is None else float(snapshot.best_guess_risk),
        "target": target,
    }


def apply_solver_safety_filter(
    solver: MinesweeperSolver,
    board: ScreenBoard,
    action_mask: np.ndarray,
    mode: str,
) -> dict[str, Any]:
    record: dict[str, Any] = {"mode": mode, "applied": False}
    if mode != "avoid-known-mines":
        raise ValueError("solver_safety_filter must be one of: none, avoid-known-mines")
    if not board.mines_placed or board.done:
        record["reason"] = "inactive_board"
        return record

    try:
        snapshot = solver.analyze(board)
    except Exception as exc:
        record.update({"error": type(exc).__name__, "message": str(exc)})
        return record

    open_channel = action_channel(ActionType.OPEN)
    open_mask = action_mask[open_channel]
    blocked = open_mask & snapshot.mine_mask
    if blocked.any():
        action_mask[open_channel] = open_mask & ~snapshot.mine_mask
        record["applied"] = True

    record.update(
        {
            "blocked_known_mine_opens": int(blocked.sum()),
            "forced_safe_count": int(snapshot.safe_mask.sum()),
            "forced_mine_count": int(snapshot.mine_mask.sum()),
            "open_actions_before": int(open_mask.sum()),
            "open_actions_after": int(action_mask[open_channel].sum()),
        }
    )
    return record


def draw_overlay(board: ScreenBoard, path: Path) -> None:
    if board.screenshot is not None:
        image = board.screenshot.copy()
    elif board.pixels is not None:
        image = Image.fromarray(board.pixels, mode="RGB")
    else:
        raise RuntimeError("board has no image data for overlay")
    draw = ImageDraw.Draw(image)
    for row in range(ROWS):
        for col in range(COLS):
            x, y = board.grid.center(row, col)
            if board.flagged[row, col]:
                text = "F"
                color = "red"
            elif board.mine_like[row, col]:
                text = "*"
                color = "black"
            elif not board.revealed[row, col]:
                text = "#"
                color = "white"
            else:
                number = int(board.adjacent[row, col])
                text = "." if number == 0 else str(number)
                color = "yellow"
            draw.text((x - 5, y - 7), text, fill=color)
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)


def request_stop(reason: str = "manual") -> None:
    STOP_FILE.parent.mkdir(parents=True, exist_ok=True)
    STOP_FILE.write_text(json.dumps({"reason": reason, "time": datetime.now().isoformat()}), encoding="utf-8")


def read_stop_request() -> dict[str, Any] | None:
    if not STOP_FILE.exists():
        return None
    try:
        return json.loads(STOP_FILE.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return {"reason": "unknown", "time": None}


def clear_stop_request() -> None:
    try:
        STOP_FILE.unlink()
    except FileNotFoundError:
        pass


def stop_requested() -> bool:
    return STOP_FILE.exists() or is_stop_hotkey_pressed()


def is_stop_hotkey_pressed() -> bool:
    return (
        key_is_down(win32con.VK_CONTROL)
        and key_is_down(win32con.VK_MENU)
        and key_is_down(HOTKEY_VK)
    )


def key_is_down(vk: int) -> bool:
    return bool(win32api.GetAsyncKeyState(vk) & 0x8000)


class StopHotkey:
    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled
        self.registered = False

    def __enter__(self) -> "StopHotkey":
        if not self.enabled:
            return self
        try:
            self.registered = bool(win32gui.RegisterHotKey(None, HOTKEY_ID, HOTKEY_MODIFIERS, HOTKEY_VK))
        except Exception:
            self.registered = False
        return self

    def __exit__(self, exc_type: object, exc: object, tb: object) -> None:
        if self.registered:
            try:
                win32gui.UnregisterHotKey(None, HOTKEY_ID)
            except Exception:
                pass

    def poll(self) -> bool:
        if not self.enabled:
            return False
        if is_stop_hotkey_pressed():
            request_stop(reason="hotkey")
            return True
        return False


def decision_actions_for_flag_mode(flag_mode: str) -> str:
    if flag_mode == "open-only":
        return "open"
    if flag_mode == "memory":
        return "full"
    raise ValueError("flag_mode must be one of: memory, open-only")


def sync_virtual_flags(raw_board: ScreenBoard, virtual_flags: np.ndarray) -> None:
    virtual_flags &= ~(raw_board.revealed | raw_board.mine_like)


def board_with_virtual_flags(raw_board: ScreenBoard, virtual_flags: np.ndarray) -> ScreenBoard:
    visible_virtual_flags = virtual_flags & ~raw_board.revealed & ~raw_board.mine_like
    return ScreenBoard(
        revealed=raw_board.revealed.copy(),
        flagged=(raw_board.flagged | visible_virtual_flags).copy(),
        adjacent=raw_board.adjacent.copy(),
        mine_like=raw_board.mine_like.copy(),
        grid=raw_board.grid,
        screenshot=raw_board.screenshot,
        pixels=raw_board.pixels,
        step_count=raw_board.step_count,
        read_repairs=raw_board.read_repairs,
        read_restores=raw_board.read_restores,
    )


def memory_board(raw_board: ScreenBoard, virtual_flags: np.ndarray, enabled: bool) -> ScreenBoard:
    if not enabled:
        return raw_board
    sync_virtual_flags(raw_board, virtual_flags)
    return board_with_virtual_flags(raw_board, virtual_flags)


def neighbors(row: int, col: int) -> list[tuple[int, int]]:
    cells: list[tuple[int, int]] = []
    for nr in range(max(0, row - 1), min(ROWS, row + 2)):
        for nc in range(max(0, col - 1), min(COLS, col + 2)):
            if nr == row and nc == col:
                continue
            cells.append((nr, nc))
    return cells


def chord_open_targets(board: ScreenBoard, action: Action) -> list[Action]:
    if action.kind != ActionType.CHORD or not board.revealed[action.row, action.col]:
        return []
    clue = int(board.adjacent[action.row, action.col])
    if clue <= 0:
        return []
    around = neighbors(action.row, action.col)
    flag_count = sum(1 for row, col in around if board.flagged[row, col])
    if flag_count != clue:
        return []
    targets: list[Action] = []
    for row, col in around:
        if not board.revealed[row, col] and not board.flagged[row, col]:
            targets.append(Action(ActionType.OPEN, row, col))
    return targets


def pop_legal_pending_open(
    pending_opens: list[Action],
    board: ScreenBoard,
    blocked_open_cells: set[tuple[int, int]] | None = None,
) -> Action | None:
    while pending_opens:
        action = pending_opens.pop(0)
        if (
            action.kind == ActionType.OPEN
            and 0 <= action.row < ROWS
            and 0 <= action.col < COLS
            and not board.revealed[action.row, action.col]
            and not board.flagged[action.row, action.col]
            and (blocked_open_cells is None or (action.row, action.col) not in blocked_open_cells)
        ):
            return action
    return None


def load_configured_trainer(args: argparse.Namespace) -> Any:
    trainer = load_checkpoint(args.checkpoint, device=args.device)
    trainer.config.rows = ROWS
    trainer.config.cols = COLS
    trainer.config.mines = MINES
    trainer.config.safe_radius = 1
    trainer.config.max_steps = args.max_steps
    trainer.config.decision_actions = decision_actions_for_flag_mode(args.flag_mode)
    trainer.config.inference_augment_flips = args.inference_flips
    trainer.config.inference_ensemble = args.inference_ensemble
    trainer.model.eval()
    return trainer


def live_timing_settings(args: argparse.Namespace) -> dict[str, Any]:
    profile = getattr(args, "speed_profile", "safe")
    capture_delay = max(0.0, float(getattr(args, "capture_delay", 0.0)))
    action_delay = max(0.0, float(getattr(args, "action_delay", 0.0)))
    stable_reads = max(1, int(getattr(args, "stable_reads", 1)))
    stable_read_delay = max(0.0, float(getattr(args, "stable_read_delay", 0.0)))
    no_progress_reclicks = max(0, int(getattr(args, "no_progress_reclicks", 0)))
    reclick_delay = max(0.0, float(getattr(args, "reclick_delay", 0.0)))

    if profile == "safe":
        return {
            "profile": profile,
            "capture_delay": max(0.001, capture_delay),
            "action_delay": max(0.01, action_delay),
            "settle_reads": max(3, stable_reads),
            "settle_read_delay": max(0.03, stable_read_delay),
            "reclick_delay": max(0.01, reclick_delay),
            "no_progress_reclicks": no_progress_reclicks,
            "click_pause": 0.01,
            "cursor_settle": 0.005,
        }
    if profile == "fast":
        return {
            "profile": profile,
            "capture_delay": min(capture_delay, 0.001),
            "action_delay": min(action_delay, 0.005),
            "settle_reads": min(stable_reads, 2),
            "settle_read_delay": min(stable_read_delay, 0.01),
            "reclick_delay": min(reclick_delay, 0.005),
            "no_progress_reclicks": max(1, no_progress_reclicks),
            "click_pause": 0.005,
            "cursor_settle": 0.002,
        }
    if profile == "turbo":
        return {
            "profile": profile,
            "capture_delay": 0.0,
            "action_delay": 0.0,
            "settle_reads": 1,
            "settle_read_delay": 0.0,
            "reclick_delay": 0.0,
            "no_progress_reclicks": 0,
            "click_pause": 0.001,
            "cursor_settle": 0.0,
        }
    if profile == "custom":
        return {
            "profile": profile,
            "capture_delay": capture_delay,
            "action_delay": action_delay,
            "settle_reads": stable_reads,
            "settle_read_delay": stable_read_delay,
            "reclick_delay": reclick_delay,
            "no_progress_reclicks": no_progress_reclicks,
            "click_pause": 0.005,
            "cursor_settle": 0.002,
        }
    raise ValueError("speed_profile must be one of: safe, fast, turbo, custom")


def play_game(
    args: argparse.Namespace,
    game_index: int,
    output_dir: Path,
    desktop: WindowsMinesweeper | None = None,
    trainer: Any | None = None,
    start_mode: str | None = None,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing = timing or live_timing_settings(args)
    desktop = desktop or WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
    )
    trainer = trainer or load_configured_trainer(args)
    if getattr(args, "clear_stop_on_start", False):
        clear_stop_request()
    effective_start_mode = args.start_mode if start_mode is None else start_mode
    prepare_game_start(desktop, effective_start_mode)
    if desktop.dialog_is_open():
        title = desktop.dialog_title() or "unknown dialog"
        raise RuntimeError(f"could not start Minesweeper game; dialog is still open: {title}")
    keep_screenshot = False
    action_delay = float(timing["action_delay"])
    settle_reads = int(timing["settle_reads"])
    settle_read_delay = float(timing["settle_read_delay"])
    if desktop.capture_backend == "window":
        settle_reads = max(1, settle_reads)
        settle_read_delay = max(0.0, settle_read_delay)
    frames: list[dict[str, Any]] = []
    actions: list[dict[str, Any]] = []
    use_memory_flags = args.flag_mode == "memory"
    virtual_flags = np.zeros((ROWS, COLS), dtype=bool)
    pending_open_actions: list[Action] = []
    blocked_open_cells: set[tuple[int, int]] = set()
    no_progress_streak = 0
    terminal_dialog: str | None = None
    latest_raw_board: ScreenBoard | None = None
    last_board: ScreenBoard | None = None
    started_at = time.time()

    for step in range(args.max_steps):
        if stop_requested():
            break
        if latest_raw_board is None:
            try:
                latest_raw_board = read_stable_board(
                    desktop,
                    step_count=step,
                    keep_screenshot=keep_screenshot,
                    reads=settle_reads,
                    delay=settle_read_delay,
                    previous_board=last_board,
                )
            except RuntimeError:
                terminal_dialog = desktop.dialog_title()
                break
        board = memory_board(latest_raw_board, virtual_flags, use_memory_flags)
        last_board = board
        frame_index: int | None = None
        if args.record_frames == "all":
            frames.append(frame_from_board(board, step=step, action=None))
            frame_index = len(frames) - 1
        if board.done:
            break

        queued_open = pop_legal_pending_open(pending_open_actions, board, blocked_open_cells) if use_memory_flags else None
        if queued_open is not None:
            action = queued_open
            action_index = action_to_index(action, ROWS, COLS)
        else:
            encoded_board, global_features, action_mask = encode_state(board)
            action_mask = trainer._decision_action_mask(action_mask)
            if blocked_open_cells:
                action_mask = action_mask.copy()
                open_mask = action_mask[action_channel(ActionType.OPEN)]
                for row, col in blocked_open_cells:
                    if 0 <= row < ROWS and 0 <= col < COLS:
                        open_mask[row, col] = False
                action_mask[action_channel(ActionType.OPEN)] = open_mask
            if not action_mask.any():
                if int(board.revealed.sum()) == 0:
                    action = Action(ActionType.OPEN, ROWS // 2, COLS // 2)
                    action_index = action_to_index(action, ROWS, COLS)
                else:
                    break
            else:
                action_index = trainer._select_action(
                    board=encoded_board,
                    global_features=global_features,
                    action_mask=action_mask,
                    game=board,
                    deterministic=True,
                    mode="rl",
                    risk_weight=0.0,
                )
                action = decode_action_index(action_index, ROWS, COLS)
                if action.kind != ActionType.OPEN and not use_memory_flags:
                    action = Action(ActionType.OPEN, action.row, action.col)
                    action_index = action_to_index(action, ROWS, COLS)

        before_signature = board_signature(board)
        before_revealed = int(board.revealed.sum())
        action_record = {"step": step, "action_index": int(action_index), "action": action_to_dict(action)}
        if board.read_repairs:
            action_record["before_read_repairs"] = int(board.read_repairs)
        if board.read_restores:
            action_record["before_read_restores"] = int(board.read_restores)
        if queued_open is not None:
            action_record["queued_from_chord"] = True
        if action.kind == ActionType.OPEN and desktop.grid is not None:
            action_record["screen_target"] = {
                "x": int(desktop.grid.center(action.row, action.col)[0]),
                "y": int(desktop.grid.center(action.row, action.col)[1]),
            }
        action_record["before_target"] = cell_snapshot(board, action.row, action.col)
        if frame_index is not None:
            frames[frame_index]["action"] = action_record
        if use_memory_flags and action.kind in {ActionType.FLAG, ActionType.UNFLAG, ActionType.CHORD}:
            if action.kind == ActionType.FLAG:
                if not board.revealed[action.row, action.col] and not board.flagged[action.row, action.col]:
                    virtual_flags[action.row, action.col] = True
                    action_record["virtual_only"] = True
                    action_record["virtual_change"] = "flag"
                    no_progress_streak = 0
                    last_board = memory_board(latest_raw_board, virtual_flags, use_memory_flags)
                else:
                    action_record["no_progress"] = True
                    no_progress_streak += 1
                actions.append(action_record)
                if no_progress_streak >= args.stall_limit:
                    break
                continue
            if action.kind == ActionType.UNFLAG:
                if virtual_flags[action.row, action.col]:
                    virtual_flags[action.row, action.col] = False
                    action_record["virtual_only"] = True
                    action_record["virtual_change"] = "unflag"
                    no_progress_streak = 0
                    last_board = memory_board(latest_raw_board, virtual_flags, use_memory_flags)
                else:
                    action_record["no_progress"] = True
                    no_progress_streak += 1
                actions.append(action_record)
                if no_progress_streak >= args.stall_limit:
                    break
                continue
            queued_targets = chord_open_targets(board, action)
            queued_targets = [
                target
                for target in queued_targets
                if (target.row, target.col) not in blocked_open_cells
            ]
            action_record["virtual_only"] = True
            if queued_targets:
                action_record["queued_opens"] = [action_to_dict(target) for target in queued_targets]
                pending_open_actions.extend(queued_targets)
                no_progress_streak = 0
                last_board = board
            else:
                action_record["no_progress"] = True
                no_progress_streak += 1
            actions.append(action_record)
            if no_progress_streak >= args.stall_limit:
                break
            continue
        try:
            desktop.click_action(action)
        except RuntimeError:
            terminal_dialog = desktop.dialog_title()
            action_record["terminal_dialog"] = terminal_dialog
            actions.append(action_record)
            break
        time.sleep(action_delay)

        try:
            after_raw_board = read_stable_board(
                desktop,
                step_count=step + 1,
                keep_screenshot=keep_screenshot,
                reads=settle_reads,
                delay=settle_read_delay,
                previous_board=board,
            )
        except RuntimeError:
            terminal_dialog = desktop.dialog_title()
            action_record["terminal_dialog"] = terminal_dialog
            actions.append(action_record)
            break
        latest_raw_board = after_raw_board
        after_board = memory_board(after_raw_board, virtual_flags, use_memory_flags)
        if action.kind == ActionType.OPEN and (
            after_board.read_repairs > 0 or not after_board.revealed[action.row, action.col]
        ):
            retry_board = confirm_open_read(
                desktop=desktop,
                action=action,
                previous_board=board,
                step_count=step + 1,
                keep_screenshot=keep_screenshot,
                settle_reads=settle_reads,
                settle_read_delay=settle_read_delay,
                reclicks=int(timing["no_progress_reclicks"]),
                reclick_delay=float(timing["reclick_delay"]),
                virtual_flags=virtual_flags,
                use_memory_flags=use_memory_flags,
            )
            if retry_board is not None:
                after_board = retry_board
        last_board = after_board
        changed = board_signature(after_board) != before_signature
        progress = int(after_board.revealed.sum()) > before_revealed or after_board.done
        action_record["after_target"] = cell_snapshot(after_board, action.row, action.col)
        if after_board.read_repairs:
            action_record["after_read_repairs"] = int(after_board.read_repairs)
        if after_board.read_restores:
            action_record["after_read_restores"] = int(after_board.read_restores)
        action_record["after_grid"] = {
            "x0": int(after_board.grid.x_lines[0]),
            "x1": int(after_board.grid.x_lines[-1]),
            "y0": int(after_board.grid.y_lines[0]),
            "y1": int(after_board.grid.y_lines[-1]),
            "cell_width": float(after_board.grid.cell_width),
            "cell_height": float(after_board.grid.cell_height),
            "cell_square_error": abs(float(after_board.grid.cell_width) - float(after_board.grid.cell_height)),
        }
        if not changed or not progress:
            action_record["no_progress"] = True
            no_progress_streak += 1
            if action.kind == ActionType.OPEN:
                blocked_open_cells.add((action.row, action.col))
                action_record["blocked_repeat_open"] = True
        else:
            no_progress_streak = 0
        actions.append(action_record)
        if no_progress_streak >= args.stall_limit:
            break

    if desktop.dialog_is_open():
        terminal_dialog = terminal_dialog or desktop.dialog_title()
        final_board = last_board if last_board is not None else empty_board(desktop, keep_screenshot=keep_screenshot)
    elif last_board is not None:
        final_board = last_board
    else:
        try:
            final_raw_board = read_stable_board(
                desktop,
                step_count=len(actions),
                keep_screenshot=keep_screenshot,
                reads=settle_reads,
                delay=settle_read_delay,
                previous_board=last_board,
            )
        except RuntimeError:
            final_board = empty_board(desktop, keep_screenshot=keep_screenshot)
        else:
            final_board = memory_board(final_raw_board, virtual_flags, use_memory_flags)
    if args.record_frames in {"all", "final"}:
        frames.append(frame_from_board(final_board, step=len(actions), action=None))
    trace = {
        "version": 1,
        "source": "windows_minesweeper",
        "checkpoint": str(args.checkpoint),
        "game_index": game_index,
        "timing": timing,
        "summary": summary_from_board(
            final_board,
            actions=actions,
            elapsed=time.time() - started_at,
            terminal_dialog=terminal_dialog,
        ),
        "actions": actions,
        "frames": frames,
    }
    path = output_dir / f"game_{game_index:03d}.json"
    path.write_text(json.dumps(trace, separators=(",", ":")), encoding="utf-8")
    if not args.no_final_images:
        final_image = desktop.capture()
        final_image.save(output_dir / f"game_{game_index:03d}_final.png")
    return {"path": str(path), "summary": trace["summary"]}


def frame_from_board(board: ScreenBoard, step: int, action: dict[str, Any] | None) -> dict[str, Any]:
    return {
        "step": step,
        "action": action,
        "won": bool(board.won),
        "lost": bool(board.lost),
        "done": bool(board.done),
        "revealed_safe_cells": int(board.revealed.sum()),
        "flags": int(board.flagged.sum()),
        "read_repairs": int(board.read_repairs),
        "read_restores": int(board.read_restores),
        "board": {
            "revealed": board.revealed.astype(np.int8).tolist(),
            "flagged": board.flagged.astype(np.int8).tolist(),
            "numbers": np.where(board.revealed, board.adjacent, -1).astype(int).tolist(),
            "mine_like": board.mine_like.astype(np.int8).tolist(),
        },
    }


def empty_board(desktop: WindowsMinesweeper, keep_screenshot: bool = True) -> ScreenBoard:
    if desktop.grid is not None:
        if keep_screenshot:
            array, grid = desktop.capture_board_array(desktop.grid)
            screenshot = Image.fromarray(array, mode="RGB")
        else:
            grid = localize_grid(desktop.grid)
            screenshot = None
    else:
        array, grid, screen_grid = desktop.capture_client_grid()
        desktop.grid = screen_grid
        screenshot = Image.fromarray(array, mode="RGB") if keep_screenshot else None
    return ScreenBoard(
        revealed=np.zeros((ROWS, COLS), dtype=bool),
        flagged=np.zeros((ROWS, COLS), dtype=bool),
        adjacent=np.zeros((ROWS, COLS), dtype=np.int8),
        mine_like=np.zeros((ROWS, COLS), dtype=bool),
        grid=grid,
        screenshot=screenshot,
    )


def read_stable_board(
    desktop: WindowsMinesweeper,
    step_count: int,
    keep_screenshot: bool,
    reads: int = 2,
    delay: float = 0.015,
    previous_board: ScreenBoard | None = None,
) -> ScreenBoard:
    board = restore_revealed_cells(
        desktop.read_board(step_count=step_count, keep_screenshot=keep_screenshot, previous_board=previous_board),
        previous_board,
    )
    reads = max(1, int(reads))
    for _ in range(reads - 1):
        previous_signature = board_signature(board)
        time.sleep(max(0.0, delay))
        next_board = restore_revealed_cells(
            desktop.read_board(step_count=step_count, keep_screenshot=keep_screenshot, previous_board=board),
            board,
        )
        if next_board.read_repairs == 0 and board.read_repairs == 0 and board_signature(next_board) == previous_signature:
            return next_board
        if next_board.read_repairs < board.read_repairs:
            board = next_board
            continue
        board = next_board
    return board


def board_signature(board: ScreenBoard) -> tuple[bytes, bytes, bytes, bool, bool, int]:
    return (
        board.revealed.tobytes(),
        board.flagged.tobytes(),
        board.adjacent.tobytes(),
        bool(board.won),
        bool(board.lost),
        int(board.read_repairs),
    )


def confirm_open_read(
    desktop: WindowsMinesweeper,
    action: Action,
    previous_board: ScreenBoard | None,
    step_count: int,
    keep_screenshot: bool,
    settle_reads: int,
    settle_read_delay: float,
    reclicks: int,
    reclick_delay: float,
    virtual_flags: np.ndarray,
    use_memory_flags: bool,
) -> ScreenBoard | None:
    best_board: ScreenBoard | None = None
    attempts = max(1, min(3, 1 + max(0, int(reclicks))))
    for attempt in range(attempts):
        if attempt > 0:
            time.sleep(max(0.0, reclick_delay))
            try:
                desktop.click_action(action)
            except RuntimeError:
                return None
        try:
            retry_raw_board = read_stable_board(
                desktop,
                step_count=step_count,
                keep_screenshot=keep_screenshot,
                reads=max(2, settle_reads),
                delay=settle_read_delay,
                previous_board=previous_board,
            )
        except RuntimeError:
            return None
        retry_board = memory_board(retry_raw_board, virtual_flags, use_memory_flags)
        if best_board is None:
            best_board = retry_board
        else:
            best_score = int(best_board.revealed.sum()) - best_board.read_repairs
            retry_score = int(retry_board.revealed.sum()) - retry_board.read_repairs
            if retry_score > best_score:
                best_board = retry_board
        if retry_board.revealed[action.row, action.col] and retry_board.read_repairs == 0:
            return retry_board
    return best_board


def summary_from_board(
    board: ScreenBoard,
    actions: list[dict[str, Any]],
    elapsed: float,
    terminal_dialog: str | None = None,
) -> dict[str, Any]:
    dialog_lost = terminal_dialog is not None and "失败" in terminal_dialog
    dialog_won = terminal_dialog is not None and ("获胜" in terminal_dialog or "胜利" in terminal_dialog)
    dialog_done = terminal_dialog is not None and is_terminal_dialog_title(terminal_dialog)
    audited_opens = [
        action
        for action in actions
        if action.get("action", {}).get("kind") == "open"
        and action.get("solver_audit", {}).get("available")
        and action.get("solver_audit", {}).get("target")
    ]
    safety_records = [action.get("solver_safety_filter") for action in actions if action.get("solver_safety_filter")]
    action_count = len(actions)
    return {
        "won": bool(board.won or dialog_won) and not dialog_lost,
        "lost": bool(board.lost or dialog_lost),
        "done": bool(board.done or dialog_done),
        "agent_steps": action_count,
        "no_progress_actions": sum(1 for action in actions if action.get("no_progress")),
        "reclicks": sum(int(action.get("reclicks", 0)) for action in actions),
        "revealed_safe_cells": int(board.revealed.sum()),
        "flags": int(board.flagged.sum()),
        "read_repairs": int(board.read_repairs),
        "read_restores": int(board.read_restores),
        "blocked_repeat_open_actions": sum(1 for action in actions if action.get("blocked_repeat_open")),
        "terminal_dialog": terminal_dialog,
        "elapsed_seconds": elapsed,
        "seconds_per_action": (elapsed / action_count) if action_count else None,
        "actions_per_second": (action_count / elapsed) if elapsed > 0 else None,
        "solver_audited_opens": len(audited_opens),
        "solver_known_mine_opens": sum(1 for action in audited_opens if action["solver_audit"]["target"].get("known_mine")),
        "solver_known_safe_opens": sum(1 for action in audited_opens if action["solver_audit"]["target"].get("known_safe")),
        "solver_forced_available_opens": sum(1 for action in audited_opens if action["solver_audit"].get("has_forced_moves")),
        "solver_high_risk_opens": sum(1 for action in audited_opens if float(action["solver_audit"]["target"].get("risk", 0.0)) >= 0.5),
        "solver_safety_filter_actions": sum(1 for record in safety_records if record.get("applied")),
        "solver_safety_blocked_opens": sum(int(record.get("blocked_known_mine_opens", 0)) for record in safety_records),
    }


def action_to_dict(action: Action) -> dict[str, Any]:
    return {"kind": action.kind.value, "row": int(action.row), "col": int(action.col)}


def is_terminal_dialog_title(title: str) -> bool:
    return "失败" in title or "获胜" in title or "胜利" in title


def board_is_fresh(board: ScreenBoard) -> bool:
    return (
        not board.done
        and int(board.revealed.sum()) == 0
        and int(board.flagged.sum()) == 0
        and not bool(board.mine_like.any())
    )


def wait_for_fresh_board(desktop: WindowsMinesweeper, timeout: float = 1.0) -> bool:
    deadline = time.time() + timeout
    while time.time() < deadline:
        if desktop.dialog_is_open():
            time.sleep(0.05)
            continue
        try:
            board = desktop.read_board(keep_screenshot=False)
        except RuntimeError:
            time.sleep(0.05)
            continue
        if board_is_fresh(board):
            return True
        time.sleep(0.05)
    return False


def prepare_game_start(desktop: WindowsMinesweeper, start_mode: str) -> bool:
    if start_mode == "current":
        if desktop.dialog_is_open():
            dialog = desktop.find_new_game_dialog()
            if dialog is not None:
                title = desktop.dialog_title() or ""
                option = "new" if is_terminal_dialog_title(title) else "continue"
                desktop.choose_dialog_option(option=option)
                if option == "new":
                    wait_for_fresh_board(desktop)
                return option == "new"
        try:
            board = desktop.read_board(keep_screenshot=False)
        except RuntimeError:
            return False
        if board.done:
            desktop.new_game(option="restart")
            wait_for_fresh_board(desktop)
            return True
        return False

    if start_mode in {"new", "restart"}:
        desktop.new_game(option=start_mode)
        wait_for_fresh_board(desktop)
        return True

    if start_mode != "auto":
        raise ValueError("start_mode must be one of: auto, current, new, restart")

    if desktop.dialog_is_open():
        dialog = desktop.find_new_game_dialog()
        if dialog is not None:
            title = desktop.dialog_title() or ""
            desktop.choose_dialog_option(option="new" if is_terminal_dialog_title(title) else "restart")
            wait_for_fresh_board(desktop)
            return True
        return False

    try:
        board = desktop.read_board(keep_screenshot=False)
    except RuntimeError:
        dialog = desktop.find_new_game_dialog()
        if dialog is not None:
            title = desktop.dialog_title() or ""
            desktop.choose_dialog_option(option="new" if is_terminal_dialog_title(title) else "restart")
            return True
        return False

    if int(board.revealed.sum()) == 0 and int(board.flagged.sum()) == 0:
        return False
    desktop.new_game(option="restart")
    wait_for_fresh_board(desktop)
    return True


def run_streak(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stop_request()
    args.clear_stop_on_start = False
    if getattr(args, "record_frames", "all") == "all":
        args.record_frames = "final"
    if not getattr(args, "no_final_images", False):
        args.no_final_images = True
    streak = 0
    results: list[dict[str, Any]] = []
    timing = live_timing_settings(args)
    desktop = WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
    )
    trainer = load_configured_trainer(args)
    manifest: dict[str, Any] = {}
    with StopHotkey(enabled=not args.no_hotkey) as hotkey:
        for game_index in range(1, args.max_games + 1):
            hotkey.poll()
            if stop_requested():
                manifest = streak_manifest(args, output_dir, results, streak, game_index - 1, "stopped", timing)
                write_manifest(output_dir, manifest)
                print(json.dumps({k: manifest[k] for k in ("status", "streak", "games_played")}, separators=(",", ":")), flush=True)
                return
            game_start_mode = args.start_mode if game_index == 1 else "new"
            result = play_game(
                args,
                game_index=game_index,
                output_dir=output_dir,
                desktop=desktop,
                trainer=trainer,
                start_mode=game_start_mode,
                timing=timing,
            )
            results.append(result)
            streak = streak + 1 if result["summary"]["won"] else 0
            status = "found" if streak >= args.streak_length else "running"
            if stop_requested():
                status = "stopped"
            manifest = streak_manifest(args, output_dir, results, streak, game_index, status, timing)
            write_manifest(output_dir, manifest)
            print(json.dumps({k: manifest[k] for k in ("status", "streak", "games_played")}, separators=(",", ":")), flush=True)
            if status in {"found", "stopped"}:
                print(json.dumps(manifest, indent=2))
                return


def run_benchmark(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stop_request()
    args.clear_stop_on_start = False
    args.record_frames = "none"
    args.no_final_images = True

    timing = live_timing_settings(args)
    desktop = WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
    )
    trainer = load_configured_trainer(args)
    results: list[dict[str, Any]] = []
    started_at = time.time()
    first_start_mode = "restart" if args.start_mode == "auto" else args.start_mode

    for game_index in range(1, args.games + 1):
        if stop_requested():
            break
        game_start_mode = first_start_mode if game_index == 1 else "new"
        result = play_game(
            args,
            game_index=game_index,
            output_dir=output_dir,
            desktop=desktop,
            trainer=trainer,
            start_mode=game_start_mode,
            timing=timing,
        )
        results.append(result)

    games_played = len(results)
    summaries = [result["summary"] for result in results]
    wins = sum(1 for summary in summaries if summary.get("won"))
    avg_elapsed = (sum(float(summary.get("elapsed_seconds", 0.0)) for summary in summaries) / games_played) if games_played else 0.0
    avg_steps = (sum(float(summary.get("agent_steps", 0.0)) for summary in summaries) / games_played) if games_played else 0.0
    avg_actions_per_second = (
        sum(float(summary.get("actions_per_second", 0.0) or 0.0) for summary in summaries) / games_played
        if games_played
        else 0.0
    )
    target_passed = (
        games_played > 0
        and wins / games_played >= float(args.target_win_rate)
        and avg_elapsed <= float(args.target_avg_seconds)
    )
    status = "stopped" if stop_requested() and games_played < int(args.games) else "completed"
    manifest = {
        "status": status,
        "output_dir": str(output_dir),
        "games_requested": int(args.games),
        "games_played": games_played,
        "checkpoint": str(args.checkpoint),
        "timing": timing,
        "final_decision_mode": "rl",
        "solver_allowed_during_final_decision": False,
        "decision_action_mode": args.flag_mode,
        "no_progress_reclicks": timing["no_progress_reclicks"],
        "reclick_delay": timing["reclick_delay"],
        "speed_profile": getattr(args, "speed_profile", "safe"),
        "model_flip_ensemble": args.inference_flips,
        "model_ensemble_method": args.inference_ensemble,
        "target_win_rate": float(args.target_win_rate),
        "target_avg_seconds": float(args.target_avg_seconds),
        "target_passed": target_passed,
        "win_rate": wins / games_played if games_played else 0.0,
        "avg_elapsed_seconds": avg_elapsed,
        "avg_agent_steps": avg_steps,
        "avg_actions_per_second": avg_actions_per_second,
        "results": results,
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def streak_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    results: list[dict[str, Any]],
    streak: int,
    games_played: int,
    status: str,
    timing: dict[str, Any] | None = None,
) -> dict[str, Any]:
    timing = timing or live_timing_settings(args)
    return {
        "status": status,
        "output_dir": str(output_dir),
        "streak": streak,
        "required_streak": args.streak_length,
        "games_played": games_played,
        "checkpoint": str(args.checkpoint),
        "timing": timing,
        "final_decision_mode": "rl",
        "solver_allowed_during_final_decision": False,
        "decision_action_mode": args.flag_mode,
        "no_progress_reclicks": timing["no_progress_reclicks"],
        "reclick_delay": timing["reclick_delay"],
        "speed_profile": getattr(args, "speed_profile", "safe"),
        "model_flip_ensemble": args.inference_flips,
        "model_ensemble_method": args.inference_ensemble,
        "solver_audit_enabled": bool(getattr(args, "audit_solver", False)),
        "solver_safety_filter": getattr(args, "solver_safety_filter", "none"),
        "streak_start_mode": args.start_mode,
        "streak_followup_start_mode": "new",
        "streak_advances_only_after_settlement": False,
        "stop_file": str(STOP_FILE),
        "stop_hotkey": None if args.no_hotkey else "Ctrl+Alt+Q",
        "results": results,
    }


def write_manifest(output_dir: Path, manifest: dict[str, Any]) -> None:
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def default_output_dir() -> Path:
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return ROOT / "artifacts" / "windows_agent" / f"run_{stamp}"


def add_subcommand_output_dir(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=argparse.SUPPRESS,
        help="write this command's logs under the given directory",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the desktop Windows Minesweeper with the RL agent.")
    parser.add_argument("--checkpoint", type=Path, default=ROOT / "artifacts" / "full_rlmix_20_refine.pt")
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--capture-backend", choices=["auto", "pil", "mss", "window"], default="auto")
    parser.add_argument("--read-mode", choices=["fast", "accurate"], default="fast")
    parser.add_argument("--speed-profile", choices=["safe", "fast", "turbo", "custom"], default="safe")
    parser.add_argument("--inference-flips", action="store_true")
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default="probs")
    parser.add_argument("--max-steps", type=int, default=2000)
    parser.add_argument("--action-delay", type=float, default=0.01)
    parser.add_argument("--capture-delay", type=float, default=0.001)
    parser.add_argument("--stable-reads", type=int, default=1)
    parser.add_argument("--stable-read-delay", type=float, default=0.005)
    parser.add_argument("--no-progress-reclicks", type=int, default=0)
    parser.add_argument("--reclick-delay", type=float, default=0.01)
    parser.add_argument("--stall-limit", type=int, default=20)
    parser.add_argument("--start-mode", choices=["auto", "current", "new", "restart"], default="auto")
    parser.add_argument("--flag-mode", choices=["memory", "open-only"], default="memory")
    parser.add_argument("--record-frames", choices=["all", "final", "none"], default="all")
    parser.add_argument("--no-final-images", action="store_true")
    parser.add_argument("--no-hotkey", action="store_true")
    parser.add_argument("--audit-solver", action="store_true")
    parser.add_argument("--solver-safety-filter", choices=["none", "avoid-known-mines"], default="none")
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser("read", help="read and print the current board")
    read_parser.add_argument("--overlay", type=Path, default=ROOT / "artifacts" / "windows_agent" / "overlay.png")

    play_once_parser = subparsers.add_parser("play-once", help="play one fresh desktop Minesweeper game")
    add_subcommand_output_dir(play_once_parser)
    play_once_parser.set_defaults(
        capture_backend="window",
        read_mode="fast",
        speed_profile="fast",
        start_mode="restart",
        inference_flips=True,
        record_frames="final",
        no_final_images=True,
    )

    streak_parser = subparsers.add_parser("run-streak", help="play until the requested winning streak is reached")
    add_subcommand_output_dir(streak_parser)
    streak_parser.add_argument("--streak-length", type=int, default=10)
    streak_parser.add_argument("--max-games", type=int, default=100)
    streak_parser.set_defaults(
        capture_backend="window",
        read_mode="fast",
        speed_profile="fast",
        start_mode="restart",
        inference_flips=True,
        record_frames="final",
        no_final_images=True,
    )

    subparsers.add_parser("stop", help="request any running desktop agent to stop")
    subparsers.add_parser("clear-stop", help="clear a stale stop request file")
    benchmark_parser = subparsers.add_parser("benchmark", help="run a fixed number of desktop games and report aggregate metrics")
    add_subcommand_output_dir(benchmark_parser)
    benchmark_parser.add_argument("--games", type=int, default=10)
    benchmark_parser.add_argument("--target-win-rate", type=float, default=0.4)
    benchmark_parser.add_argument("--target-avg-seconds", type=float, default=60.0)
    benchmark_parser.set_defaults(
        capture_backend="window",
        read_mode="fast",
        speed_profile="fast",
        start_mode="restart",
        inference_flips=True,
        record_frames="none",
        no_final_images=True,
    )

    args = parser.parse_args()
    if args.command == "read":
        desktop = WindowsMinesweeper(
            capture_backend=args.capture_backend,
            read_mode=args.read_mode,
        )
        board = desktop.read_board(keep_screenshot=True)
        draw_overlay(board, args.overlay)
        print(board_to_text(board))
        print(
            json.dumps(
                {
                    "revealed": int(board.revealed.sum()),
                    "flags": int(board.flagged.sum()),
                    "won": board.won,
                    "lost": board.lost,
                    "overlay": str(args.overlay),
                    "grid": {
                        "x0": board.grid.x_lines[0],
                        "x1": board.grid.x_lines[-1],
                        "y0": board.grid.y_lines[0],
                        "y1": board.grid.y_lines[-1],
                        "cell_width": board.grid.cell_width,
                        "cell_height": board.grid.cell_height,
                        "cell_square_error": abs(board.grid.cell_width - board.grid.cell_height),
                    },
                    "screen_grid": None
                    if desktop.grid is None
                    else {
                        "x0": desktop.grid.x_lines[0],
                        "x1": desktop.grid.x_lines[-1],
                        "y0": desktop.grid.y_lines[0],
                        "y1": desktop.grid.y_lines[-1],
                        "cell_width": desktop.grid.cell_width,
                        "cell_height": desktop.grid.cell_height,
                    },
                },
                indent=2,
            )
        )
    elif args.command == "play-once":
        clear_stop_request()
        args.clear_stop_on_start = False
        output_dir = args.output_dir or default_output_dir()
        output_dir.mkdir(parents=True, exist_ok=True)
        result = play_game(args, game_index=1, output_dir=output_dir)
        print(json.dumps({"output_dir": str(output_dir), "result": result}, indent=2))
    elif args.command == "run-streak":
        run_streak(args)
    elif args.command == "benchmark":
        run_benchmark(args)
    elif args.command == "stop":
        previous = read_stop_request()
        request_stop(reason="command")
        print(
            json.dumps(
                {
                    "status": "already_stop_requested" if previous is not None else "stop_requested",
                    "stop_file": str(STOP_FILE),
                    "previous_request": previous,
                    "note": "This writes a stop signal; if no agent is running, there is nothing else to stop.",
                },
                indent=2,
            )
        )
    elif args.command == "clear-stop":
        previous = read_stop_request()
        clear_stop_request()
        print(
            json.dumps(
                {
                    "status": "cleared" if previous is not None else "no_stop_file",
                    "stop_file": str(STOP_FILE),
                    "previous_request": previous,
                },
                indent=2,
            )
        )


if __name__ == "__main__":
    main()
