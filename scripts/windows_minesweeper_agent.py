from __future__ import annotations

import argparse
import ctypes
import json
import os
import sys
import time
from collections import Counter
from ctypes import wintypes
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
REPAIRED_READ_RETRY_DELAYS = (0.025, 0.05, 0.09)
GRID_REFRESH_READ_INTERVAL = 64
INPUT_MOUSE = 0
MOUSEEVENTF_MOVE = 0x0001
MOUSEEVENTF_ABSOLUTE = 0x8000
MOUSEEVENTF_VIRTUALDESK = 0x4000
SM_XVIRTUALSCREEN = 76
SM_YVIRTUALSCREEN = 77
SM_CXVIRTUALSCREEN = 78
SM_CYVIRTUALSCREEN = 79
READ_PIXEL_DIFF_TOLERANCE = 2
CONFIRM_ACCURATE_MIN_MEAN_DELTA = 8.0
CONFIRM_ACCURATE_MIN_CHANGED_RATIO = 0.08
RETRY_CLICK_HOLD_SECONDS = 0.05
RETRY_CURSOR_SETTLE_SECONDS = 0.02
RETRY_POST_CLICK_SETTLE_SECONDS = 0.10
RETRY_PRECLICK_DELAY_SECONDS = 0.08
RETRY_READ_DELAY_SECONDS = 0.06
BACKUP_SENDINPUT_DELAY_SECONDS = 0.012
BACKUP_SENDINPUT_HOLD_SECONDS = 0.018
RETRY_CLICK_FRACTIONS = (
    (0.75, 0.25),
    (0.25, 0.75),
    (0.75, 0.75),
    (0.50, 0.50),
)
NUMBER_LABELS = np.array([1, 2, 3, 4, 5, 6, 7, 8], dtype=np.int8)
NUMBER_PROTOTYPES = np.array(
    [
        [55, 75, 175],
        [55, 125, 45],
        [175, 35, 35],
        [45, 45, 125],
        [120, 35, 35],
        [35, 125, 125],
        [35, 35, 35],
        [105, 105, 105],
    ],
    dtype=np.int16,
)

ULONG_PTR = ctypes.c_ulonglong if ctypes.sizeof(ctypes.c_void_p) == 8 else ctypes.c_ulong


class _MouseInput(ctypes.Structure):
    _fields_ = [
        ("dx", wintypes.LONG),
        ("dy", wintypes.LONG),
        ("mouseData", wintypes.DWORD),
        ("dwFlags", wintypes.DWORD),
        ("time", wintypes.DWORD),
        ("dwExtraInfo", ULONG_PTR),
    ]


class _InputUnion(ctypes.Union):
    _fields_ = [("mi", _MouseInput)]


class _Input(ctypes.Structure):
    _fields_ = [
        ("type", wintypes.DWORD),
        ("u", _InputUnion),
    ]


_USER32 = ctypes.WinDLL("user32", use_last_error=True)
_SEND_INPUT = _USER32.SendInput
_SEND_INPUT.argtypes = (wintypes.UINT, ctypes.POINTER(_Input), ctypes.c_int)
_SEND_INPUT.restype = wintypes.UINT

BASE_DEFAULTS: dict[str, Any] = {
    "checkpoint": ROOT / "artifacts" / "full_rlmix_20.pt",
    "device": "cuda",
    "output_dir": None,
    "capture_backend": "auto",
    "read_mode": "fast",
    "click_method": "auto",
    "speed_profile": "safe",
    "inference_flips": False,
    "inference_ensemble": "probs",
    "max_steps": 2000,
    "action_delay": 0.01,
    "capture_delay": 0.001,
    "stable_reads": 1,
    "stable_read_delay": 0.005,
    "no_progress_reclicks": 0,
    "reclick_delay": 0.01,
    "click_confirm_retries": 4,
    "post_click_settle": 0.05,
    "stall_limit": 20,
    "start_mode": "new",
    "flag_mode": "memory",
    "record_frames": "all",
    "no_final_images": False,
    "no_hotkey": False,
    "no_persistent_reveals": False,
    "audit_solver": False,
    "audit_basic": False,
    "basic_safety_filter": "none",
    "solver_safety_filter": "none",
    "solver_assist": "none",
    "solver_exact_limit": None,
    "solver_batch_size": 1,
    "quick_number_read": False,
}

COMMAND_DEFAULTS: dict[str, dict[str, Any]] = {
    "read-benchmark": {
        "capture_backend": "auto",
        "read_mode": "fast",
        "click_method": "auto",
        "speed_profile": "fast",
        "capture_delay": 0.001,
        "stable_reads": 1,
        "stable_read_delay": 0.008,
        "record_frames": "none",
        "no_final_images": True,
    },
    "play-once": {
        "capture_backend": "auto",
        "read_mode": "fast",
        "click_method": "mouse_event",
        "speed_profile": "fast",
        "max_steps": 600,
        "action_delay": 0.006,
        "capture_delay": 0.001,
        "stable_reads": 1,
        "stable_read_delay": 0.008,
        "no_progress_reclicks": 1,
        "reclick_delay": 0.015,
        "post_click_settle": 0.05,
        "stall_limit": 20,
        "start_mode": "new",
        "inference_flips": True,
        "record_frames": "final",
        "no_final_images": True,
    },
    "run-streak": {
        "capture_backend": "auto",
        "read_mode": "fast",
        "click_method": "mouse_event",
        "speed_profile": "fast",
        "max_steps": 600,
        "action_delay": 0.006,
        "capture_delay": 0.001,
        "stable_reads": 1,
        "stable_read_delay": 0.008,
        "no_progress_reclicks": 1,
        "reclick_delay": 0.015,
        "post_click_settle": 0.05,
        "stall_limit": 20,
        "start_mode": "new",
        "inference_flips": True,
        "record_frames": "final",
        "no_final_images": True,
    },
    "benchmark": {
        "capture_backend": "auto",
        "read_mode": "fast",
        "click_method": "mouse_event",
        "speed_profile": "fast",
        "max_steps": 600,
        "action_delay": 0.006,
        "capture_delay": 0.001,
        "stable_reads": 1,
        "stable_read_delay": 0.008,
        "no_progress_reclicks": 1,
        "reclick_delay": 0.015,
        "post_click_settle": 0.05,
        "stall_limit": 20,
        "start_mode": "new",
        "inference_flips": True,
        "record_frames": "none",
        "no_final_images": True,
    },
}


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

    def point(
        self,
        row: int,
        col: int,
        fraction: float = 0.5,
        y_fraction: float | None = None,
    ) -> tuple[int, int]:
        x_fraction = max(0.1, min(0.9, float(fraction)))
        y_fraction = x_fraction if y_fraction is None else max(0.1, min(0.9, float(y_fraction)))
        x0, x1 = self.x_lines[col], self.x_lines[col + 1]
        y0, y1 = self.y_lines[row], self.y_lines[row + 1]
        return (
            int(round(x0 + (x1 - x0) * x_fraction)),
            int(round(y0 + (y1 - y0) * y_fraction)),
        )

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
    read_recoveries: int = 0
    read_timing: dict[str, Any] | None = None

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
        click_method: str = "auto",
        click_pause: float = 0.01,
        cursor_settle: float = 0.005,
        post_click_settle: float = 0.05,
    ) -> None:
        try:
            ctypes.windll.shcore.SetProcessDpiAwareness(2)
        except Exception:
            pass
        self.window_class = window_class
        self.capture_delay = capture_delay
        self.capture_backend = capture_backend
        self.read_mode = read_mode
        if click_method not in {"auto", "sendinput", "sendinput_absolute", "mouse_event"}:
            raise ValueError("click_method must be one of: auto, sendinput, sendinput_absolute, mouse_event")
        self.click_method = click_method
        self.click_pause = click_pause
        self.cursor_settle = cursor_settle
        self.post_click_settle = post_click_settle
        # The classic Windows tile has a bright center highlight that can
        # swallow synthetic clicks on some cells. Use a stable interior point
        # instead of the exact geometric center.
        self.click_fraction: tuple[float, float] = (0.25, 0.25)
        self._mss = None
        self._resolved_click_method: str | None = None
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
        self._grid_client_origin: tuple[int, int] | None = None
        self._grid_client_size: tuple[int, int] | None = None
        self._reads_since_grid_refresh = 0
        self.last_click_report: dict[str, Any] | None = None
        self.last_capture_report: dict[str, Any] | None = None

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
        attached_threads: list[int] = []
        try:
            current_thread = win32api.GetCurrentThreadId()
            foreground = win32gui.GetForegroundWindow()
            foreground_thread = win32process.GetWindowThreadProcessId(foreground)[0] if foreground else 0
            target_thread = win32process.GetWindowThreadProcessId(self.hwnd)[0]
            for thread_id in {foreground_thread, target_thread}:
                if thread_id and thread_id != current_thread:
                    try:
                        win32process.AttachThreadInput(current_thread, thread_id, True)
                        attached_threads.append(thread_id)
                    except Exception:
                        pass
            if win32gui.IsIconic(self.hwnd):
                win32gui.ShowWindow(self.hwnd, win32con.SW_RESTORE)
            else:
                win32gui.ShowWindow(self.hwnd, win32con.SW_SHOW)
            flags = win32con.SWP_NOMOVE | win32con.SWP_NOSIZE | win32con.SWP_SHOWWINDOW
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_TOPMOST, 0, 0, 0, 0, flags)
            win32gui.SetWindowPos(self.hwnd, win32con.HWND_NOTOPMOST, 0, 0, 0, 0, flags)
            win32gui.BringWindowToTop(self.hwnd)
            win32gui.SetForegroundWindow(self.hwnd)
            try:
                win32gui.SetActiveWindow(self.hwnd)
                win32gui.SetFocus(self.hwnd)
            except Exception:
                pass
        except Exception:
            pass
        finally:
            try:
                current_thread = win32api.GetCurrentThreadId()
                for thread_id in attached_threads:
                    win32process.AttachThreadInput(current_thread, thread_id, False)
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
        self.park_cursor()
        time.sleep(self.capture_delay)
        left, top, right, bottom = self.client_bounds()
        mss_backend = getattr(self, "_mss", None)
        desktop_hwnd = int(getattr(self, "hwnd", 0) or 0)
        foreground = bool(desktop_hwnd and win32gui.GetForegroundWindow() == desktop_hwnd)
        if not foreground and self.capture_backend == "auto":
            # Keep the fast screen-capture path available after the pointer
            # parks outside the board. PrintWindow is much slower and can
            # return a stale frame on this classic game.
            foreground = self.ensure_foreground()
        if mss_backend is not None and self.capture_backend in {"auto", "mss"}:
            if foreground:
                try:
                    self.last_capture_report = {"path": "mss_client", "foreground": True}
                    return self.capture_region_array(left, top, right, bottom), left, top
                except Exception:
                    if self.capture_backend == "mss":
                        raise
            elif self.capture_backend == "mss":
                raise RuntimeError("Minesweeper window is not foreground; refusing to read covered screen pixels")
        if self.capture_backend in {"auto", "window"}:
            try:
                self.last_capture_report = {"path": "print_window_client", "foreground": False}
                return self._capture_client_window_array(), left, top
            except Exception:
                if self.capture_backend == "window":
                    raise
        if not self.ensure_foreground():
            raise RuntimeError("Minesweeper window is not foreground; refusing to read covered screen pixels")
        self.last_capture_report = {"path": "pil_client", "foreground": True}
        return self.capture_region_array(left, top, right, bottom), left, top

    def capture_client_grid(self) -> tuple[np.ndarray, Grid, Grid]:
        array, left, top = self.capture_client_array()
        local_grid, screen_grid = self._detect_client_grid(array, left, top)
        return array, local_grid, screen_grid

    def _detect_client_grid(self, array: np.ndarray, left: int, top: int) -> tuple[Grid, Grid]:
        image = Image.fromarray(array, mode="RGB")
        local_grid = detect_local_grid(image)
        screen_grid = offset_grid(local_grid, dx=left, dy=top)
        self._grid_client_origin = (int(left), int(top))
        self._grid_client_size = (int(array.shape[1]), int(array.shape[0]))
        self._reads_since_grid_refresh = 0
        return local_grid, screen_grid

    def _capture_board_region(self, grid: Grid) -> np.ndarray:
        """Capture only the board when the cached screen-space grid is valid."""
        left = int(grid.x_lines[0])
        top = int(grid.y_lines[0])
        right = int(grid.x_lines[-1] + 1)
        bottom = int(grid.y_lines[-1] + 1)
        if right <= left or bottom <= top:
            raise RuntimeError("detected grid has an empty board region")

        self.park_cursor()
        time.sleep(self.capture_delay)
        desktop_hwnd = int(getattr(self, "hwnd", 0) or 0)
        foreground = bool(desktop_hwnd and win32gui.GetForegroundWindow() == desktop_hwnd)
        if not foreground and self.capture_backend == "auto":
            foreground = self.ensure_foreground()
        if self._mss is not None and self.capture_backend in {"auto", "mss"}:
            if foreground:
                try:
                    self.last_capture_report = {"path": "mss_board", "foreground": True}
                    return self.capture_region_array(left, top, right, bottom)
                except Exception:
                    if self.capture_backend == "mss":
                        raise
            elif self.capture_backend == "mss":
                raise RuntimeError(
                    "Minesweeper window is not foreground; refusing to read covered screen pixels"
                )

        if self.capture_backend == "window" or (self.capture_backend == "auto" and not foreground):
            try:
                client_array = self._capture_client_window_array()
                client_left, client_top, _, _ = self.client_bounds()
                local_grid = offset_grid(grid, dx=-client_left, dy=-client_top)
                self.last_capture_report = {"path": "print_window_board", "foreground": False}
                return crop_grid_array(client_array, local_grid)
            except Exception:
                if self.capture_backend == "window":
                    raise

        if not self.ensure_foreground():
            raise RuntimeError(
                "Minesweeper window is not foreground; refusing to read covered screen pixels"
            )
        self.last_capture_report = {"path": "pil_board", "foreground": True}
        return self.capture_region_array(left, top, right, bottom)

    def capture_client_board(self) -> tuple[np.ndarray, Grid, Grid]:
        """Capture the board while reusing stable geometry between reads."""
        if self.grid is not None:
            left, top, right, bottom = self.client_bounds()
            origin = (int(left), int(top))
            size = (int(right - left), int(bottom - top))
            local_grid = offset_grid(self.grid, dx=-left, dy=-top)
            geometry_valid = (
                local_grid.x_lines[0] >= 0
                and local_grid.y_lines[0] >= 0
                and local_grid.x_lines[-1] < size[0]
                and local_grid.y_lines[-1] < size[1]
            )
            needs_detection = (
                self._grid_client_origin != origin
                or self._grid_client_size != size
                or self._reads_since_grid_refresh >= GRID_REFRESH_READ_INTERVAL
                or not geometry_valid
            )
            if not needs_detection:
                try:
                    board_array = self._capture_board_region(self.grid)
                except Exception:
                    board_array = None
                if board_array is not None:
                    self._reads_since_grid_refresh += 1
                    return board_array, localize_grid(local_grid), self.grid

        array, left, top = self.capture_client_array()
        local_grid, screen_grid = self._detect_client_grid(array, left, top)
        self.grid = screen_grid
        return array, local_grid, screen_grid

    def refresh_grid(self) -> Grid:
        """Re-detect the screen-space grid after a missed click or layout change."""
        _, _, screen_grid = self.capture_client_grid()
        self.grid = screen_grid
        return screen_grid

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
        return self._read_board_accurate(step_count=step_count, keep_screenshot=keep_screenshot)

    def _read_board_accurate(self, step_count: int = 0, keep_screenshot: bool = True) -> ScreenBoard:
        started_at = time.perf_counter()
        capture_started_at = time.perf_counter()
        array, read_grid, screen_grid = self.capture_client_grid()
        capture_elapsed = time.perf_counter() - capture_started_at
        self.grid = screen_grid
        source_image = Image.fromarray(array, mode="RGB")
        screenshot = source_image.crop(grid_bbox(read_grid)) if keep_screenshot else None
        revealed = np.zeros((ROWS, COLS), dtype=bool)
        flagged = np.zeros((ROWS, COLS), dtype=bool)
        adjacent = np.zeros((ROWS, COLS), dtype=np.int8)
        mine_like = np.zeros((ROWS, COLS), dtype=bool)

        classification_started_at = time.perf_counter()
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
        classification_elapsed = time.perf_counter() - classification_started_at

        repairs = repair_impossible_numbers(revealed, adjacent, mine_like)
        board_pixels = np.asarray(source_image.crop(grid_bbox(read_grid)))
        output_grid = localize_grid(read_grid)

        board = ScreenBoard(
            revealed=revealed,
            flagged=flagged,
            adjacent=adjacent,
            mine_like=mine_like,
            grid=output_grid,
            screenshot=screenshot,
            pixels=board_pixels,
            step_count=step_count,
            read_repairs=repairs,
        )
        board.read_timing = {
            "mode": "accurate",
            "capture_path": (self.last_capture_report or {}).get("path"),
            "capture_seconds": capture_elapsed,
            "classification_seconds": classification_elapsed,
            "total_seconds": time.perf_counter() - started_at,
        }
        return board

    def _read_board_fast(
        self,
        step_count: int = 0,
        keep_screenshot: bool = True,
        previous_board: ScreenBoard | None = None,
    ) -> ScreenBoard:
        started_at = time.perf_counter()
        capture_started_at = time.perf_counter()
        board_array, grid_for_read, _ = self.capture_client_board()
        capture_elapsed = time.perf_counter() - capture_started_at
        classification_started_at = time.perf_counter()
        board_pixels = crop_grid_array(board_array, grid_for_read)
        local_grid = localize_grid(grid_for_read)
        screenshot = Image.fromarray(board_pixels, mode="RGB") if keep_screenshot else None
        board = self._read_board_from_array(
            board_pixels,
            local_grid,
            screenshot,
            board_pixels,
            step_count,
            previous_board,
            output_grid=local_grid,
        )
        board.read_timing = {
            "mode": "fast",
            "capture_path": (self.last_capture_report or {}).get("path"),
            "capture_seconds": capture_elapsed,
            "classification_seconds": time.perf_counter() - classification_started_at,
            "total_seconds": time.perf_counter() - started_at,
        }
        return self._accurate_fallback_for_repaired_fast_board(board, step_count, keep_screenshot)

    def _accurate_fallback_for_repaired_fast_board(
        self,
        board: ScreenBoard,
        step_count: int,
        keep_screenshot: bool,
    ) -> ScreenBoard:
        if board.read_repairs <= 0:
            return board
        best_board = board
        for retry_delay in REPAIRED_READ_RETRY_DELAYS:
            time.sleep(retry_delay)
            try:
                accurate = self._read_board_accurate(step_count=step_count, keep_screenshot=keep_screenshot)
            except RuntimeError:
                if self.dialog_is_open():
                    raise
                continue
            if read_quality_key(accurate) > read_quality_key(best_board):
                best_board = accurate
            if accurate.read_repairs == 0 or accurate.done:
                accurate.read_recoveries = board.read_recoveries + 1
                return accurate
        if best_board is not board:
            best_board.read_recoveries = board.read_recoveries + 1
        return best_board

    def _read_board_from_array(
        self,
        board_array: np.ndarray,
        grid: Grid,
        screenshot: Image.Image | None,
        board_pixels: np.ndarray,
        step_count: int,
        previous_board: ScreenBoard | None = None,
        output_grid: Grid | None = None,
    ) -> ScreenBoard:
        revealed = np.zeros((ROWS, COLS), dtype=bool)
        flagged = np.zeros((ROWS, COLS), dtype=bool)
        adjacent = np.zeros((ROWS, COLS), dtype=np.int8)
        mine_like = np.zeros((ROWS, COLS), dtype=bool)
        changed_mask = None
        refresh_hidden_cells = previous_board is not None and previous_board.step_count == step_count
        if previous_board is not None and previous_board.pixels is not None and previous_board.pixels.shape == board_array.shape:
            changed_mask = np.any(board_array != previous_board.pixels, axis=2)
            if bool(changed_mask.any()):
                changed_rows, changed_cols = np.where(changed_mask)
                pixel_delta = np.abs(
                    board_array[changed_rows, changed_cols].astype(np.int16, copy=False)
                    - previous_board.pixels[changed_rows, changed_cols].astype(np.int16, copy=False)
                )
                changed_mask = np.zeros_like(changed_mask)
                changed_mask[changed_rows, changed_cols] = np.any(pixel_delta > READ_PIXEL_DIFF_TOLERANCE, axis=1)

        for row in range(ROWS):
            for col in range(COLS):
                x0, y0, x1, y1 = grid.crop_box(row, col)
                if changed_mask is not None and not bool(changed_mask[y0:y1, x0:x1].any()):
                    if (
                        not refresh_hidden_cells
                        or previous_board.revealed[row, col]
                        or previous_board.flagged[row, col]
                        or previous_board.mine_like[row, col]
                    ):
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
            grid=output_grid or grid,
            screenshot=screenshot,
            pixels=board_pixels,
            step_count=step_count,
            read_repairs=repairs,
        )

    def _full_monitor(self) -> dict[str, int]:
        left, top = win32gui.ClientToScreen(self.hwnd, (0, 0))
        _, _, width, height = win32gui.GetClientRect(self.hwnd)
        return {"left": left, "top": top, "width": width, "height": height}

    def effective_click_method(self) -> str:
        return self._resolved_click_method or self.click_method

    def click_point(
        self,
        row: int,
        col: int,
        fraction_override: tuple[float, float] | None = None,
    ) -> tuple[int, int]:
        if self.grid is None:
            raise RuntimeError("Minesweeper grid has not been detected")
        fraction = self.click_fraction if fraction_override is None else fraction_override
        if isinstance(fraction, tuple):
            return self.grid.point(row, col, fraction=fraction[0], y_fraction=fraction[1])
        return self.grid.point(row, col, fraction=float(fraction))

    def click_action(
        self,
        action: Action,
        method_override: str | None = None,
        click_pause_override: float | None = None,
        cursor_settle_override: float | None = None,
        post_click_settle_override: float | None = None,
        click_fraction_override: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        started_at = time.time()
        requested_method = method_override or effective_click_method(self)
        if requested_method not in {
            "auto",
            "sendinput",
            "sendinput_absolute",
            "mouse_event",
        }:
            raise ValueError(
                "method_override must be one of: auto, sendinput, sendinput_absolute, mouse_event"
            )
        click_pause = max(
            0.0,
            float(getattr(self, "click_pause", 0.01) if click_pause_override is None else click_pause_override),
        )
        cursor_settle = max(
            0.0,
            float(
                getattr(self, "cursor_settle", 0.005)
                if cursor_settle_override is None
                else cursor_settle_override
            ),
        )
        post_click_settle = max(
            0.0,
            float(
                getattr(self, "post_click_settle", 0.05)
                if post_click_settle_override is None
                else post_click_settle_override
            ),
        )
        report: dict[str, Any] = {
            "row": int(action.row),
            "col": int(action.col),
            "kind": action.kind.value,
            "issued": False,
            "cursor_ready": False,
            "cursor_samples": [],
            "method": "pending",
            "requested_method": requested_method,
            "button_down_issued": False,
            "button_up_issued": False,
        }
        self.last_click_report = report
        if self.dialog_is_open():
            raise RuntimeError("a Minesweeper dialog is open; handle it before clicking the board")
        if action.kind != ActionType.OPEN:
            raise ValueError(f"open-only agent expected OPEN action, got {action.kind}")
        if self.grid is None:
            refresh_grid = getattr(self, "refresh_grid", None)
            if not callable(refresh_grid):
                raise RuntimeError("Minesweeper grid has not been detected")
            refresh_grid()
        if not self.ensure_foreground():
            self.last_click_report = report
            raise RuntimeError("Minesweeper window is not foreground; refusing to click")
        report["foreground_after_focus"] = foreground_snapshot()

        def target_window_info(target_x: int, target_y: int) -> dict[str, Any]:
            target_hwnd = int(win32gui.WindowFromPoint((target_x, target_y)) or 0)
            parent_chain: list[int] = []
            current = target_hwnd
            desktop_hwnd = int(getattr(self, "hwnd", 0) or 0)
            for _ in range(8):
                if not current:
                    break
                parent_chain.append(int(current))
                if desktop_hwnd and current == desktop_hwnd:
                    break
                try:
                    current = int(win32gui.GetParent(current) or 0)
                except Exception:
                    current = 0
            return {
                "hwnd": target_hwnd,
                "class": win32gui.GetClassName(target_hwnd) if target_hwnd else None,
                "inside_minesweeper": bool(not desktop_hwnd or desktop_hwnd in parent_chain),
                "parent_chain": parent_chain,
            }

        target_window: dict[str, Any] = {}
        x = y = 0
        click_point = getattr(self, "click_point", None)
        target_fraction = (
            tuple(float(value) for value in click_fraction_override)
            if click_fraction_override is not None
            else getattr(self, "click_fraction", (0.25, 0.25))
        )
        if not isinstance(target_fraction, tuple):
            target_fraction = (float(target_fraction), float(target_fraction))
        for target_attempt in range(2):
            # Stay inside the tile, away from both the bevel and the bright
            # center highlight that can swallow synthetic clicks.
            if callable(click_point):
                x, y = click_point(action.row, action.col, fraction_override=target_fraction)
            else:
                x, y = self.grid.center(action.row, action.col)
            target_window = target_window_info(x, y)
            if not getattr(self, "hwnd", None) or target_window["inside_minesweeper"]:
                break
            # A DPI/layout change can leave the cached screen grid inside the
            # window but no longer aligned with the board.
            self.refresh_grid()
        report["target_screen"] = {"x": int(x), "y": int(y)}
        report["target_fraction"] = {"x": float(target_fraction[0]), "y": float(target_fraction[1])}
        report["target_window_before_click"] = target_window
        if getattr(self, "hwnd", None) and not target_window["inside_minesweeper"]:
            self.last_click_report = report
            raise RuntimeError(
                f"click target is outside Minesweeper window at ({x}, {y}); "
                f"window_info={target_window}"
            )
        cursor_ready = False
        current_x, current_y = x, y

        def is_cursor_in_target_cell(cursor_x: int, cursor_y: int) -> bool:
            grid_for_cursor = getattr(self, "grid", None)
            return bool(
                grid_for_cursor is not None
                and grid_for_cursor.x_lines[action.col] <= cursor_x < grid_for_cursor.x_lines[action.col + 1]
                and grid_for_cursor.y_lines[action.row] <= cursor_y < grid_for_cursor.y_lines[action.row + 1]
            )

        for _ in range(3):
            win32api.SetCursorPos((x, y))
            time.sleep(cursor_settle)
            try:
                current_x, current_y = win32api.GetCursorPos()
            except Exception:
                cursor_ready = True
                break
            report["cursor_samples"].append({"x": int(current_x), "y": int(current_y)})
            if is_cursor_in_target_cell(int(current_x), int(current_y)) or (
                abs(current_x - x) <= 3 and abs(current_y - y) <= 3
            ):
                cursor_ready = True
                break
        report["cursor_ready"] = bool(cursor_ready)
        report["cursor_target_error"] = None
        if not cursor_ready:
            self.last_click_report = report
            raise RuntimeError(f"cursor did not reach target cell ({action.row}, {action.col}) at ({x}, {y})")
        current_x = int(current_x)
        current_y = int(current_y)
        grid_for_cursor = getattr(self, "grid", None)
        cursor_in_target_cell = is_cursor_in_target_cell(current_x, current_y)
        report["cursor_cell_at_target"] = {
            "row": int(action.row) if cursor_in_target_cell else None,
            "col": int(action.col) if cursor_in_target_cell else None,
            "same_cell": cursor_in_target_cell,
        }
        actual_target = target_window_info(int(current_x), int(current_y))
        report["target_window_at_cursor"] = actual_target
        if getattr(self, "hwnd", None) and not actual_target["inside_minesweeper"]:
            report["cursor_target_error"] = "cursor_not_inside_minesweeper"
            self.last_click_report = report
            raise RuntimeError(
                f"cursor reached ({current_x}, {current_y}) but is outside Minesweeper"
            )
        if grid_for_cursor is not None and not cursor_in_target_cell:
            report["cursor_target_error"] = "cursor_outside_target_tolerance"
            self.last_click_report = report
            raise RuntimeError(
                f"cursor reached ({current_x}, {current_y}) in a different cell from "
                f"({action.row}, {action.col})"
            )
        if grid_for_cursor is None and (abs(current_x - x) > 8 or abs(current_y - y) > 8):
            report["cursor_target_error"] = "cursor_outside_target_tolerance"
            self.last_click_report = report
            raise RuntimeError(
                f"cursor reached ({current_x}, {current_y}) outside target tolerance for "
                f"({action.row}, {action.col})"
            )
        if not self.ensure_foreground():
            report["foreground_before_click"] = foreground_snapshot()
            self.last_click_report = report
            raise RuntimeError("Minesweeper window lost foreground before clicking")
        report["foreground_before_click"] = foreground_snapshot()
        edge_click = action.col in {0, COLS - 1}
        click_backend = requested_method
        report["edge_click"] = bool(edge_click)
        if click_backend != requested_method:
            report["edge_click_backend"] = click_backend
        try:
            click_method = click_mouse(
                win32con.MOUSEEVENTF_LEFTDOWN,
                win32con.MOUSEEVENTF_LEFTUP,
                x,
                y,
                pause=click_pause,
                method=click_backend,
                report=report,
            )
        except Exception as exc:
            report["click_error"] = f"{type(exc).__name__}: {exc}"
            report["elapsed_seconds"] = time.time() - started_at
            self.last_click_report = report
            raise
        if method_override is None and self.click_method == "auto" and click_method not in {"sendinput", "sendinput_absolute"}:
            self._resolved_click_method = "mouse_event"
        report["resolved_method"] = effective_click_method(self)
        report["method"] = click_method
        report["issued"] = True
        try:
            released_x, released_y = win32api.GetCursorPos()
            report["cursor_at_release"] = {"x": int(released_x), "y": int(released_y)}
        except Exception:
            pass
        report["click_pause"] = click_pause
        report["cursor_settle"] = cursor_settle
        report["post_click_settle"] = post_click_settle
        report["release_settle_after_up"] = post_click_settle
        time.sleep(post_click_settle)
        self.park_cursor()
        try:
            parked_x, parked_y = win32api.GetCursorPos()
            report["parked_at"] = {"x": int(parked_x), "y": int(parked_y)}
        except Exception:
            pass
        report["elapsed_seconds"] = time.time() - started_at
        self.last_click_report = report
        return report

    def click_action_with_method(
        self,
        action: Action,
        method: str,
        click_pause: float | None = None,
        cursor_settle: float | None = None,
        post_click_settle: float | None = None,
        click_fraction: tuple[float, float] | None = None,
    ) -> dict[str, Any]:
        return self.click_action(
            action,
            method_override=method,
            click_pause_override=click_pause,
            cursor_settle_override=cursor_settle,
            post_click_settle_override=post_click_settle,
            click_fraction_override=click_fraction,
        )

    def park_cursor(self) -> None:
        try:
            left, top, _, _ = self.client_bounds()
        except Exception:
            try:
                win32api.SetCursorPos((0, 0))
            except Exception:
                pass
            return
        x = max(0, left - 32)
        y = max(0, top - 32)
        try:
            win32api.SetCursorPos((x, y))
        except Exception:
            return
        time.sleep(max(0.0, self.cursor_settle))

    def new_game(self, option: str = "new") -> None:
        if option not in {"new", "restart", "continue"}:
            raise ValueError("option must be one of: new, restart, continue")
        if not self.focus(timeout=1.0):
            raise RuntimeError("could not focus Minesweeper before starting a new game")
        if self.dialog_is_open() and is_statistics_dialog_title(self.dialog_title() or ""):
            self.choose_dialog_option(option="close", timeout=1.0, attempts=3)
        win32api.keybd_event(win32con.VK_F2, 0, 0, 0)
        time.sleep(0.03)
        win32api.keybd_event(win32con.VK_F2, 0, win32con.KEYEVENTF_KEYUP, 0)
        time.sleep(0.2)
        dialog = self.find_new_game_dialog()
        if dialog is not None:
            self.choose_dialog_option(option=option)
        time.sleep(0.1)
        self.grid = None
        self._grid_client_origin = None
        self._grid_client_size = None
        self._reads_since_grid_refresh = 0

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
        else:
            dialog_rect = win32gui.GetWindowRect(dialog)
            x, y = dialog_fallback_point(title, dialog_rect, option)
        win32api.SetCursorPos((x, y))
        time.sleep(max(0.0, self.cursor_settle))
        requested_click_method = effective_click_method(self)
        click_method = click_mouse(
            win32con.MOUSEEVENTF_LEFTDOWN,
            win32con.MOUSEEVENTF_LEFTUP,
            x,
            y,
            pause=self.click_pause,
            method=requested_click_method,
        )
        if self.click_method == "auto" and click_method not in {"sendinput", "sendinput_absolute"}:
            self._resolved_click_method = "mouse_event"
        self.park_cursor()

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


def send_mouse_input(dx: int, dy: int, flags: int) -> None:
    event = _Input()
    event.type = INPUT_MOUSE
    event.u.mi = _MouseInput(dx, dy, 0, flags, 0, 0)
    sent = _SEND_INPUT(1, ctypes.byref(event), ctypes.sizeof(_Input))
    if sent != 1:
        error_code = ctypes.get_last_error()
        if error_code:
            raise ctypes.WinError(error_code)
        raise OSError("SendInput returned 0 without a Windows error code")


def send_mouse_button(flags: int) -> None:
    send_mouse_input(0, 0, flags)


def send_mouse_absolute_move(x: int, y: int) -> None:
    left = int(win32api.GetSystemMetrics(SM_XVIRTUALSCREEN))
    top = int(win32api.GetSystemMetrics(SM_YVIRTUALSCREEN))
    width = max(1, int(win32api.GetSystemMetrics(SM_CXVIRTUALSCREEN)))
    height = max(1, int(win32api.GetSystemMetrics(SM_CYVIRTUALSCREEN)))
    absolute_x = int(round((int(x) - left) * 65535 / max(1, width - 1)))
    absolute_y = int(round((int(y) - top) * 65535 / max(1, height - 1)))
    send_mouse_input(
        max(0, min(65535, absolute_x)),
        max(0, min(65535, absolute_y)),
        MOUSEEVENTF_MOVE | MOUSEEVENTF_ABSOLUTE | MOUSEEVENTF_VIRTUALDESK,
    )


def click_mouse(
    down: int,
    up: int,
    x: int,
    y: int,
    pause: float = 0.01,
    method: str = "auto",
    report: dict[str, Any] | None = None,
) -> str:
    if method not in {"auto", "sendinput", "sendinput_absolute", "mouse_event"}:
        raise ValueError("method must be one of: auto, sendinput, sendinput_absolute, mouse_event")
    if method == "sendinput_absolute":
        sendinput_down = False
        if report is not None:
            report["sendinput_attempted"] = True
            report["sendinput_absolute_attempted"] = True
            report["click_hold_seconds"] = float(max(0.0, pause))
        try:
            send_mouse_absolute_move(x, y)
            time.sleep(0.005)
            send_mouse_button(down)
            sendinput_down = True
            if report is not None:
                report["button_down_issued"] = True
            time.sleep(max(0.0, pause))
            send_mouse_button(up)
            if report is not None:
                report["button_up_issued"] = True
            return "sendinput_absolute"
        except Exception as exc:
            if report is not None:
                report["sendinput_error"] = f"{type(exc).__name__}: {exc}"
            if sendinput_down:
                try:
                    win32api.mouse_event(up, x, y, 0, 0)
                    if report is not None:
                        report["sendinput_fallback"] = "mouse_event_up"
                except Exception as release_exc:
                    if report is not None:
                        report["sendinput_release_error"] = f"{type(release_exc).__name__}: {release_exc}"
            try:
                win32api.mouse_event(down, x, y, 0, 0)
                if report is not None:
                    report["button_down_issued"] = True
                time.sleep(max(0.0, pause))
                win32api.mouse_event(up, x, y, 0, 0)
                if report is not None:
                    report["button_up_issued"] = True
                    report["sendinput_fallback"] = "mouse_event"
                return "mouse_event"
            except Exception:
                raise RuntimeError("absolute SendInput click failed") from exc
    if method in {"auto", "sendinput"}:
        sendinput_down = False
        if report is not None:
            report["sendinput_attempted"] = True
            report["click_hold_seconds"] = float(max(0.0, pause))
        try:
            send_mouse_button(down)
            sendinput_down = True
            if report is not None:
                report["button_down_issued"] = True
            time.sleep(max(0.0, pause))
            send_mouse_button(up)
            if report is not None:
                report["button_up_issued"] = True
            return "sendinput"
        except Exception as exc:
            if report is not None:
                report["sendinput_error"] = f"{type(exc).__name__}: {exc}"
            if sendinput_down:
                try:
                    win32api.mouse_event(up, x, y, 0, 0)
                    if report is not None:
                        report["sendinput_fallback"] = "mouse_event_up"
                except Exception as release_exc:
                    if report is not None:
                        report["sendinput_release_error"] = f"{type(release_exc).__name__}: {release_exc}"
                if method == "auto":
                    return "sendinput_mouse_event_up"
            if method == "sendinput":
                raise RuntimeError("SendInput click failed") from exc
    if report is not None and method == "auto":
        report["sendinput_fallback"] = "mouse_event"
        report["click_hold_seconds"] = float(max(0.0, pause))
    # Re-assert the physical pointer position immediately before the button
    # event. The board can briefly repaint between the outer cursor check and
    # this call, and mouse_event uses the current pointer position.
    win32api.SetCursorPos((int(x), int(y)))
    time.sleep(0.01)
    if report is not None:
        report["click_cursor_repositioned"] = True
        try:
            cursor_x, cursor_y = win32api.GetCursorPos()
            report["cursor_before_down"] = {"x": int(cursor_x), "y": int(cursor_y)}
            report["target_window_before_down"] = window_snapshot(
                int(win32gui.WindowFromPoint((int(cursor_x), int(cursor_y))) or 0)
            )
        except Exception:
            pass
        report["foreground_before_down"] = foreground_snapshot()
    win32api.mouse_event(down, 0, 0, 0, 0)
    if report is not None:
        report["button_down_issued"] = True
        try:
            cursor_x, cursor_y = win32api.GetCursorPos()
            report["cursor_after_down"] = {"x": int(cursor_x), "y": int(cursor_y)}
        except Exception:
            pass
        report["foreground_after_down"] = foreground_snapshot()
    time.sleep(max(0.0, pause))
    if report is not None:
        try:
            cursor_x, cursor_y = win32api.GetCursorPos()
            report["cursor_before_up"] = {"x": int(cursor_x), "y": int(cursor_y)}
        except Exception:
            pass
    win32api.mouse_event(up, 0, 0, 0, 0)
    if report is not None:
        report["button_up_issued"] = True
        try:
            cursor_x, cursor_y = win32api.GetCursorPos()
            report["cursor_after_up"] = {"x": int(cursor_x), "y": int(cursor_y)}
        except Exception:
            pass
        report["foreground_after_up"] = foreground_snapshot()
        if method == "mouse_event":
            report["backup_sendinput_attempted"] = True
            try:
                time.sleep(BACKUP_SENDINPUT_DELAY_SECONDS)
                try:
                    cursor_x, cursor_y = win32api.GetCursorPos()
                    report["cursor_before_backup_down"] = {"x": int(cursor_x), "y": int(cursor_y)}
                except Exception:
                    pass
                report["foreground_before_backup_down"] = foreground_snapshot()
                send_mouse_button(down)
                report["backup_button_down_issued"] = True
                time.sleep(BACKUP_SENDINPUT_HOLD_SECONDS)
                send_mouse_button(up)
                report["backup_button_up_issued"] = True
                try:
                    cursor_x, cursor_y = win32api.GetCursorPos()
                    report["cursor_after_backup_up"] = {"x": int(cursor_x), "y": int(cursor_y)}
                except Exception:
                    pass
                report["foreground_after_backup_up"] = foreground_snapshot()
            except Exception as exc:
                report["backup_sendinput_error"] = f"{type(exc).__name__}: {exc}"
    return "mouse_event"


def effective_click_method(desktop: Any) -> str:
    method = getattr(desktop, "effective_click_method", None)
    if callable(method):
        return str(method())
    return str(getattr(desktop, "_resolved_click_method", None) or getattr(desktop, "click_method", "auto"))


def click_report_matches_action(click_report: dict[str, Any], action: Action) -> bool:
    if not all(key in click_report for key in ("row", "col", "kind")):
        return True
    return (
        int(click_report["row"]) == int(action.row)
        and int(click_report["col"]) == int(action.col)
        and click_report["kind"] == action.kind.value
    )


def window_snapshot(hwnd: int) -> dict[str, Any] | None:
    hwnd = int(hwnd or 0)
    if not hwnd:
        return None
    try:
        rect = win32gui.GetWindowRect(hwnd)
    except Exception:
        rect = None
    try:
        class_name = win32gui.GetClassName(hwnd)
    except Exception:
        class_name = None
    try:
        title = win32gui.GetWindowText(hwnd)
    except Exception:
        title = None
    return {
        "hwnd": hwnd,
        "class": class_name,
        "title": title,
        "rect": None if rect is None else [int(value) for value in rect],
    }


def foreground_snapshot() -> dict[str, Any] | None:
    try:
        return window_snapshot(int(win32gui.GetForegroundWindow() or 0))
    except Exception:
        return None


def dialog_button_matches(text: str, option: str) -> bool:
    normalized = text.lower()
    if option == "new":
        tokens = ("&p", "&n", "再玩", "开始新游戏")
    elif option == "restart":
        tokens = ("&r", "重新开始")
    elif option == "continue":
        tokens = ("&k", "继续")
    elif option == "close":
        tokens = ("&c", "close", "\u5173\u95ed")
    else:
        raise ValueError("option must be one of: new, restart, continue, close")
    return any(token in normalized for token in tokens)


def dialog_fallback_point(
    title: str,
    dialog_rect: tuple[int, int, int, int],
    option: str,
) -> tuple[int, int]:
    """Return a proportional click point when a dialog has no child buttons."""
    if option not in {"new", "restart", "continue", "close"}:
        raise ValueError("option must be one of: new, restart, continue, close")
    left, top, right, bottom = (int(value) for value in dialog_rect)
    width = max(1, right - left)
    height = max(1, bottom - top)
    normalized = title.lower()
    is_new_game_dialog = "new game" in normalized or "新游戏" in title
    if is_new_game_dialog:
        row_fraction = {"new": 0.48, "restart": 0.68, "continue": 0.88}[option]
        return left + width // 2, top + int(round(height * row_fraction))

    button_fraction = {"new": 0.83, "restart": 0.50, "continue": 0.83, "close": 0.40}[option]
    return left + int(round(width * button_fraction)), top + int(round(height * 0.90))


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
    y_lines = [max(0, min(image.height, line)) for line in y_lines]

    col_counts = dark[y_lines[0] : y_lines[-1] + 1, :].sum(axis=0)
    x_candidates = _line_centers(col_counts, min_count=500, max_width=8, start=0, end=image.width)
    x_lines = _best_grid_sequence(x_candidates, expected=COLS + 1, min_spacing=45, max_spacing=130)
    x_lines = [max(0, min(image.width, line)) for line in x_lines]

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
    if best is not None:
        return best[1]

    # A cropped board image often omits the outermost dark borders because
    # they are clipped or fall outside the row-count search window. Recover
    # those lines from the remaining nearly uniform sequence instead of
    # falling back to a layout-specific desktop ratio.
    missing = expected - len(candidates)
    if 1 <= missing <= 4 and len(candidates) >= 2:
        candidate_values = [float(value) for value in candidates]
        recovered: tuple[float, list[int]] | None = None
        for positions in _sequence_position_choices(expected, len(candidates)):
            if missing >= 2 and (positions[0] == 0 or positions[-1] == expected - 1):
                # The row/column search intentionally clips the outer borders,
                # so prefer a sequence with at least one missing line at each
                # edge over an equally good sequence shifted by one cell.
                continue
            count = len(positions)
            sum_x = float(sum(positions))
            sum_y = float(sum(candidate_values))
            sum_xx = float(sum(index * index for index in positions))
            sum_xy = float(sum(index * value for index, value in zip(positions, candidate_values)))
            denominator = count * sum_xx - sum_x * sum_x
            if denominator <= 0.0:
                continue
            spacing = (count * sum_xy - sum_x * sum_y) / denominator
            intercept = (sum_y - spacing * sum_x) / count
            if int(round(intercept)) < 0:
                continue
            if spacing < min_spacing or spacing > max_spacing:
                continue
            residuals = [
                abs(intercept + spacing * index - value)
                for index, value in zip(positions, candidate_values)
            ]
            if max(residuals) > max(4.0, spacing * 0.08):
                continue
            lines = [int(round(intercept + spacing * index)) for index in range(expected)]
            diffs = np.diff(lines)
            if diffs.min() < min_spacing or diffs.max() > max_spacing:
                continue
            score = float(sum(residuals) / len(residuals) + diffs.std())
            if recovered is None or score < recovered[0]:
                recovered = (score, lines)
        if recovered is not None:
            return recovered[1]

    raise RuntimeError(f"could not detect {expected} grid lines from candidates {candidates}")


def _sequence_position_choices(expected: int, observed: int) -> list[tuple[int, ...]]:
    """Return possible grid indices for an incomplete candidate sequence."""
    if observed <= 0 or observed > expected:
        return []
    if expected - observed > 4:
        return []

    # The grid is small (17 horizontal or 31 vertical lines), so enumerating
    # the possible missing positions is cheap and handles clipped borders as
    # well as an occasional missing interior line.
    choices: list[tuple[int, ...]] = []

    def visit(next_index: int, selected: list[int]) -> None:
        if len(selected) == observed:
            choices.append(tuple(selected))
            return
        remaining = observed - len(selected)
        last_start = expected - remaining
        for index in range(next_index, last_start + 1):
            selected.append(index)
            visit(index + 1, selected)
            selected.pop()

    visit(0, [])
    return choices


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


def _corner_background_patch(array: np.ndarray) -> np.ndarray:
    height, width = array.shape[:2]
    inset = max(2, min(height, width) // 6)
    size = max(3, min(height, width) // 7)
    y0 = inset
    y1 = min(height, inset + size)
    y2 = max(0, height - inset - size)
    y3 = max(0, height - inset)
    x0 = inset
    x1 = min(width, inset + size)
    x2 = max(0, width - inset - size)
    x3 = max(0, width - inset)
    patches = [
        array[y0:y1, x0:x1],
        array[y0:y1, x2:x3],
        array[y2:y3, x0:x1],
        array[y2:y3, x2:x3],
    ]
    return np.concatenate([patch.reshape(-1, 3) for patch in patches if patch.size], axis=0)


def _looks_like_hidden_blue_cell(array: np.ndarray) -> bool:
    if array.shape[0] < 14 or array.shape[1] < 14:
        return False
    f = array.astype(np.int16, copy=False)
    height, width = f.shape[:2]
    inset = max(1, min(height, width) // 12)
    interior = f[inset : height - inset, inset : width - inset]
    if interior.size == 0:
        interior = f
    center = _center_patch(f)
    background = _corner_background_patch(f)
    background_mean = background.mean(axis=0)
    center_mean = center.reshape(-1, 3).mean(axis=0)

    blue_pixels = (
        (interior[:, :, 2] - interior[:, :, 0] > 35)
        & (interior[:, :, 2] - interior[:, :, 1] > 8)
        & (interior[:, :, 1] - interior[:, :, 0] > -8)
        & (interior[:, :, 2] > 95)
    )
    center_blue_pixels = (
        (center[:, :, 2] - center[:, :, 0] > 30)
        & (center[:, :, 2] - center[:, :, 1] > 8)
        & (center[:, :, 1] - center[:, :, 0] > -8)
        & (center[:, :, 2] > 95)
    )
    blue_ratio = float(blue_pixels.mean())
    center_blue_ratio = float(center_blue_pixels.mean())
    background_blue = float(background_mean[2] - background_mean[0])
    background_green = float(background_mean[1] - background_mean[0])
    center_blue = float(center_mean[2] - center_mean[0])

    classic_hidden = background_blue > 74 and background_green > 12 and center_blue > 35
    broad_hidden = blue_ratio > 0.42 and center_blue_ratio > 0.32 and background_green > 12
    gradient_hidden = blue_ratio > 0.30 and center_blue > 24 and background_mean[2] > 112 and background_green > 12
    return bool(classic_hidden or broad_hidden or gradient_hidden)


def _looks_like_revealed_background(array: np.ndarray) -> bool:
    f = array.astype(np.int16, copy=False)
    maxc = f.max(axis=2)
    minc = f.min(axis=2)
    low_saturation = ((maxc - minc) * 100 < maxc * 24) & (maxc > 80)
    return bool(float(low_saturation.mean()) > 0.55 and float(maxc.mean()) > 158.0)


def _looks_like_revealed_center(array: np.ndarray) -> bool:
    center = _center_patch(array).astype(np.int16, copy=False)
    maxc = center.max(axis=2)
    minc = center.min(axis=2)
    low_saturation = ((maxc - minc) * 100 < maxc * 25) & (maxc > 80)
    return bool(float(low_saturation.mean()) > 0.50 and float(maxc.mean()) > 115.0)


def classify_cell(crop: Image.Image) -> dict[str, Any]:
    array = np.asarray(crop.convert("RGB"))
    mean = array.reshape(-1, 3).mean(axis=0)
    saturation, value = _saturation_value(array)
    median_saturation = float(np.quantile(saturation, 0.5))
    value_mean = float(value.mean())
    revealed_background = median_saturation < 0.24 and value_mean > 0.62
    blue_background = mean[2] - mean[0] > 42 and mean[2] - mean[1] > 15 and mean[1] > mean[0]

    red_pixels = (array[:, :, 0] > 145) & (array[:, :, 1] < 110) & (array[:, :, 2] < 115)
    yellow_pixels = (array[:, :, 0] > 145) & (array[:, :, 1] > 100) & (array[:, :, 2] < 90)
    dark_pixels = (array[:, :, 0] < 45) & (array[:, :, 1] < 45) & (array[:, :, 2] < 55)
    center_revealed = _looks_like_revealed_center(array)
    hidden_blue = _looks_like_hidden_blue_cell(array)
    digit_count = int(_digit_mask(array).sum())
    digit_is_number = (
        digit_count >= 25
        and digit_count < array.shape[0] * array.shape[1] * 0.45
        and number_color_distance(array) <= 68.0
    )
    if hidden_blue and not revealed_background and not center_revealed and not digit_is_number:
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
    corners = (
        f[2:12, 2:12],
        f[2:12, -12:-2],
        f[-12:-2, 2:12],
        f[-12:-2, -12:-2],
    )
    blue_votes = 0
    for patch in corners:
        mean = patch.reshape(-1, 3).mean(axis=0)
        if mean[2] - mean[0] > 42 and mean[2] - mean[1] > 15 and mean[1] > mean[0]:
            blue_votes += 1

    red_pixels = (f[:, :, 0] > 145) & (f[:, :, 1] < 110) & (f[:, :, 2] < 115)
    yellow_pixels = (f[:, :, 0] > 145) & (f[:, :, 1] > 100) & (f[:, :, 2] < 90)
    dark_pixels = (f[:, :, 0] < 45) & (f[:, :, 1] < 45) & (f[:, :, 2] < 55)

    if blue_votes >= 3:
        center_revealed = _looks_like_revealed_center(array)
        digit_count = int(_digit_mask(array).sum())
        digit_is_number = (
            digit_count >= 25
            and digit_count < array.shape[0] * array.shape[1] * 0.45
            and number_color_distance(array) <= 68.0
        )
        # The blue bevel can remain visible around a freshly revealed zero or
        # dark-blue 4. Let the full classifier win when the center already
        # looks like a revealed tile.
        if _looks_like_revealed_background(array) or center_revealed or digit_is_number:
            return {
                "kind": "revealed",
                "number": classify_number(array),
            }
        red_count = int(red_pixels.sum())
        yellow_count = int(yellow_pixels.sum())
        if red_count > 35 or yellow_count > 35:
            return {"kind": "flagged", "number": 0}
        if int(dark_pixels.sum()) > 220 and red_count > 20:
            return {"kind": "mine", "number": 0}
        return {"kind": "hidden", "number": 0}

    # Ambiguous cells use the slower classifier. It is safer to spend a few
    # microseconds here than to poison the board with a false hidden/revealed
    # classification near the bright top-left border.
    return classify_cell(Image.fromarray(array, mode="RGB"))


def classify_number(array: np.ndarray) -> int:
    digit_mask = _digit_mask(array)
    if int(digit_mask.sum()) < 25:
        return 0
    color = np.median(array[digit_mask].astype(np.float32), axis=0)
    delta = NUMBER_PROTOTYPES.astype(np.float32) - color
    distances = np.sum(delta * delta, axis=1)
    return int(NUMBER_LABELS[int(np.argmin(distances))])


def number_color_distance(array: np.ndarray) -> float:
    digit_mask = _digit_mask(array)
    if int(digit_mask.sum()) < 25:
        return float("inf")
    color = np.median(array[digit_mask].astype(np.float32), axis=0)
    delta = NUMBER_PROTOTYPES.astype(np.float32) - color
    return float(np.sqrt(np.min(np.sum(delta * delta, axis=1))))


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


def repair_impossible_zero_reveals(
    revealed: np.ndarray,
    flagged: np.ndarray,
    adjacent: np.ndarray,
    mine_like: np.ndarray,
) -> int:
    impossible: list[tuple[int, int]] = []
    rows, cols = revealed.shape
    for row, col in zip(*np.where(revealed & ~mine_like & (adjacent == 0))):
        row = int(row)
        col = int(col)
        for nr in range(max(0, row - 1), min(rows, row + 2)):
            for nc in range(max(0, col - 1), min(cols, col + 2)):
                if nr == row and nc == col:
                    continue
                if not revealed[nr, nc] or flagged[nr, nc]:
                    impossible.append((row, col))
                    break
            else:
                continue
            break

    for row, col in impossible:
        revealed[row, col] = False
        flagged[row, col] = False
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
        read_recoveries=board.read_recoveries,
        read_timing=board.read_timing,
    )


def _digit_mask(array: np.ndarray) -> np.ndarray:
    f = array.astype(np.int16, copy=False)
    maxc = f.max(axis=2)
    minc = f.min(axis=2)
    spread = maxc - minc
    colorful = (spread * 100 > maxc * 42) & (maxc < 235)
    dark = maxc < 64
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


def cell_pixel_change(
    before: ScreenBoard | None,
    after: ScreenBoard | None,
    row: int,
    col: int,
    tolerance: int = READ_PIXEL_DIFF_TOLERANCE,
) -> dict[str, Any]:
    """Measure whether a cell visibly changed between two board captures."""
    if before is None or after is None or before.pixels is None or after.pixels is None:
        return {"available": False, "changed": False}
    if before.pixels.ndim != 3 or after.pixels.ndim != 3:
        return {"available": False, "changed": False}

    before_grid = localize_grid(before.grid)
    after_grid = localize_grid(after.grid)
    bx0, by0, bx1, by1 = before_grid.crop_box(row, col)
    ax0, ay0, ax1, ay1 = after_grid.crop_box(row, col)
    width = min(bx1 - bx0, ax1 - ax0)
    height = min(by1 - by0, ay1 - ay0)
    if width < 4 or height < 4:
        return {"available": False, "changed": False}

    before_cell = before.pixels[by0 : by0 + height, bx0 : bx0 + width]
    after_cell = after.pixels[ay0 : ay0 + height, ax0 : ax0 + width]
    delta = np.abs(
        before_cell.astype(np.int16, copy=False) - after_cell.astype(np.int16, copy=False)
    )
    changed_pixels = np.any(delta > int(tolerance), axis=2)
    changed_ratio = float(changed_pixels.mean())
    mean_delta = float(delta.mean())
    return {
        "available": True,
        "changed": bool(changed_ratio >= 0.03 or mean_delta >= 5.0),
        "changed_ratio": changed_ratio,
        "mean_delta": mean_delta,
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


def basic_action_audit(board: ScreenBoard, action: Action) -> dict[str, Any]:
    if action.kind != ActionType.OPEN:
        return {"available": False, "reason": "non_open_action"}
    if not board.mines_placed:
        return {"available": False, "reason": "before_first_open"}
    if board.done:
        return {"available": False, "reason": "terminal_board"}
    target = (int(action.row), int(action.col))
    if not (0 <= target[0] < ROWS and 0 <= target[1] < COLS):
        return {"available": False, "reason": "target_out_of_bounds"}

    snapshot = basic_inference(board)
    return basic_action_audit_from_snapshot(snapshot, action)


def basic_action_audit_from_snapshot(snapshot: dict[str, Any], action: Action) -> dict[str, Any]:
    if action.kind != ActionType.OPEN:
        return {"available": False, "reason": "non_open_action"}
    target = (int(action.row), int(action.col))
    safe_mask = snapshot["safe_mask"]
    mine_mask = snapshot["mine_mask"]
    witnesses = snapshot["witnesses_by_cell"].get(target, [])
    target_known_safe = bool(safe_mask[target])
    target_known_mine = bool(mine_mask[target])
    return {
        "available": True,
        "known_safe_count": int(safe_mask.sum()),
        "known_mine_count": int(mine_mask.sum()),
        "contradictions": int(snapshot["contradictions"]),
        "target": {
            "known_safe": bool(target_known_safe),
            "known_mine": bool(target_known_mine),
            "conflict": bool(target_known_safe and target_known_mine),
            "witnesses": witnesses[:4],
        },
    }


def basic_inference(board: ScreenBoard) -> dict[str, Any]:
    safe_mask = np.zeros((ROWS, COLS), dtype=bool)
    mine_mask = np.zeros((ROWS, COLS), dtype=bool)
    witnesses_by_cell: dict[tuple[int, int], list[dict[str, Any]]] = {}
    contradictions = 0

    for row, col in zip(*np.where(board.revealed & ~board.mine_like)):
        row = int(row)
        col = int(col)
        clue = int(board.adjacent[row, col])
        around = neighbors(row, col)
        flag_count = sum(1 for nr, nc in around if board.flagged[nr, nc])
        hidden_cells = [
            (nr, nc)
            for nr, nc in around
            if not board.revealed[nr, nc] and not board.flagged[nr, nc] and not board.mine_like[nr, nc]
        ]
        if flag_count > clue or flag_count + len(hidden_cells) < clue:
            contradictions += 1
            continue
        conclusion: str | None = None
        if hidden_cells and flag_count == clue:
            for cell in hidden_cells:
                safe_mask[cell] = True
            conclusion = "safe"
        elif hidden_cells and flag_count + len(hidden_cells) == clue:
            for cell in hidden_cells:
                mine_mask[cell] = True
            conclusion = "mine"
        if conclusion:
            witness = {
                "row": row,
                "col": col,
                "clue": clue,
                "flags": int(flag_count),
                "hidden": int(len(hidden_cells)),
                "conclusion": conclusion,
            }
            for cell in hidden_cells:
                bucket = witnesses_by_cell.setdefault(cell, [])
                if len(bucket) < 4:
                    bucket.append(witness)

    return {
        "safe_mask": safe_mask,
        "mine_mask": mine_mask,
        "contradictions": contradictions,
        "witnesses_by_cell": witnesses_by_cell,
    }


def apply_basic_safety_filter(
    board: ScreenBoard,
    action_mask: np.ndarray,
    mode: str,
    snapshot: dict[str, Any] | None = None,
) -> dict[str, Any]:
    record: dict[str, Any] = {"mode": mode, "applied": False}
    if mode == "none":
        return record
    if mode != "avoid-known-mines":
        raise ValueError("basic_safety_filter must be one of: none, avoid-known-mines")
    if not board.mines_placed or board.done:
        record["reason"] = "inactive_board"
        return record

    snapshot = snapshot or basic_inference(board)
    open_channel = action_channel(ActionType.OPEN)
    open_mask = action_mask[open_channel]
    mine_mask = snapshot["mine_mask"]
    safe_mask = snapshot["safe_mask"]
    conflict_mask = mine_mask & safe_mask
    blocked = open_mask & mine_mask & ~conflict_mask
    if blocked.any():
        action_mask[open_channel] = open_mask & ~blocked
        record["applied"] = True

    record.update(
        {
            "blocked_known_mine_opens": int(blocked.sum()),
            "known_safe_count": int(safe_mask.sum()),
            "known_mine_count": int(mine_mask.sum()),
            "conflict_count": int(conflict_mask.sum()),
            "contradictions": int(snapshot["contradictions"]),
            "open_actions_before": int(open_mask.sum()),
            "open_actions_after": int(action_mask[open_channel].sum()),
        }
    )
    return record


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


def select_solver_assist_action(
    solver: MinesweeperSolver,
    board: ScreenBoard,
    forbidden_open_cells: set[tuple[int, int]],
    use_memory_flags: bool,
    batch_size: int = 1,
) -> tuple[Action | None, list[tuple[int, int]], dict[str, Any]]:
    """Return only solver-proven actions; leave all guesses to the RL policy."""
    record: dict[str, Any] = {
        "mode": "forced",
        "available": False,
        "applied": False,
        "virtual_flag_cells": [],
        "safe_batch_targets": [],
    }
    if not board.mines_placed or board.done:
        record["reason"] = "inactive_board"
        return None, [], record

    try:
        snapshot = solver.analyze(board)
    except Exception as exc:
        record.update({"error": type(exc).__name__, "message": str(exc)})
        return None, [], record

    hidden = board.legal_open_mask()
    safe_mask = snapshot.safe_mask & hidden
    mine_mask = snapshot.mine_mask & hidden
    safe_mask &= ~np.array(
        [[(row, col) in forbidden_open_cells for col in range(COLS)] for row in range(ROWS)],
        dtype=bool,
    )
    safe_cells = [(int(row), int(col)) for row, col in zip(*np.where(safe_mask))]
    mine_cells = [(int(row), int(col)) for row, col in zip(*np.where(mine_mask))]
    record.update(
        {
            "available": True,
            "forced_safe_count": len(safe_cells),
            "forced_mine_count": len(mine_cells),
            "best_guess": None
            if snapshot.best_guess is None
            else {"row": int(snapshot.best_guess[0]), "col": int(snapshot.best_guess[1])},
            "best_guess_risk": None
            if snapshot.best_guess_risk is None
            else float(snapshot.best_guess_risk),
        }
    )

    virtual_flag_cells = mine_cells if use_memory_flags else []
    if virtual_flag_cells:
        record["virtual_flag_cells"] = [
            {"row": row, "col": col} for row, col in virtual_flag_cells
        ]

    if safe_cells:
        # Prefer a safe frontier cell with more constraints: it tends to open
        # a larger useful region while remaining completely solver-proven.
        safe_cells.sort(
            key=lambda cell: (
                float(snapshot.frontier_degree_map[cell]),
                -cell[0],
                -cell[1],
            ),
            reverse=True,
        )
        batch_size = max(1, int(batch_size))
        batch_cells = safe_cells[:batch_size]
        row, col = batch_cells[0]
        record.update(
            {
                "applied": True,
                "decision": "forced_safe_open",
                "target": {"row": row, "col": col},
                "safe_batch_count": len(batch_cells),
                "safe_batch_targets": [
                    {"row": cell_row, "col": cell_col}
                    for cell_row, cell_col in batch_cells
                ],
            }
        )
        return Action(ActionType.OPEN, row, col), virtual_flag_cells, record

    if virtual_flag_cells:
        row, col = virtual_flag_cells[0]
        record.update(
            {
                "applied": True,
                "decision": "virtual_flag_batch",
                "target": {"row": row, "col": col},
            }
        )
        return Action(ActionType.FLAG, row, col), virtual_flag_cells, record

    record["decision"] = "rl_fallback"
    return None, [], record


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


def attribute_terminal_to_previous_action(actions: list[dict[str, Any]], title: str | None, detected_at: str) -> None:
    if title is None or not actions:
        return
    previous = actions[-1]
    if not previous.get("terminal_dialog"):
        previous["terminal_dialog"] = title
    if not previous.get("terminal_detected_at"):
        previous["terminal_detected_at"] = detected_at


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


def sync_virtual_flags(
    raw_board: ScreenBoard,
    virtual_flags: np.ndarray,
    persistent_revealed_mask: np.ndarray | None = None,
) -> None:
    unavailable = raw_board.revealed | raw_board.mine_like
    if persistent_revealed_mask is not None:
        unavailable = unavailable | persistent_revealed_mask
    virtual_flags &= ~unavailable


def clear_virtual_flags_for_cells(virtual_flags: np.ndarray, cells: set[tuple[int, int]]) -> int:
    if not cells:
        return 0
    cleared = 0
    for row, col in cells:
        if 0 <= row < ROWS and 0 <= col < COLS and bool(virtual_flags[row, col]):
            virtual_flags[row, col] = False
            cleared += 1
    return cleared


def clear_blocked_open_cells_for_confirmed(
    blocked_open_cells: set[tuple[int, int]],
    blocked_open_attempts: dict[tuple[int, int], int],
    confirmed_open_cells: set[tuple[int, int]],
) -> int:
    if not confirmed_open_cells:
        return 0
    cleared = 0
    for cell in confirmed_open_cells:
        if cell in blocked_open_cells:
            blocked_open_cells.discard(cell)
            cleared += 1
        blocked_open_attempts.pop(cell, None)
    return cleared


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
        read_recoveries=raw_board.read_recoveries,
    )


def board_with_persistent_reveals(
    raw_board: ScreenBoard,
    persistent_revealed_mask: np.ndarray,
    persistent_revealed_adjacent: np.ndarray,
) -> ScreenBoard:
    if not bool(persistent_revealed_mask.any()):
        return raw_board

    visible_persistent = persistent_revealed_mask & ~raw_board.revealed & ~raw_board.mine_like
    if not bool(visible_persistent.any()):
        return raw_board

    revealed = raw_board.revealed.copy()
    flagged = raw_board.flagged.copy()
    adjacent = raw_board.adjacent.copy()
    mine_like = raw_board.mine_like.copy()
    revealed[visible_persistent] = True
    flagged[visible_persistent] = False
    adjacent[visible_persistent] = persistent_revealed_adjacent[visible_persistent]
    mine_like[visible_persistent] = False

    return ScreenBoard(
        revealed=revealed,
        flagged=flagged,
        adjacent=adjacent,
        mine_like=mine_like,
        grid=raw_board.grid,
        screenshot=raw_board.screenshot,
        pixels=raw_board.pixels,
        step_count=raw_board.step_count,
        read_repairs=raw_board.read_repairs,
        read_restores=raw_board.read_restores,
        read_recoveries=raw_board.read_recoveries,
    )


def memory_board(
    raw_board: ScreenBoard,
    virtual_flags: np.ndarray,
    enabled: bool,
    persistent_revealed_mask: np.ndarray | None = None,
    persistent_revealed_adjacent: np.ndarray | None = None,
) -> ScreenBoard:
    board = raw_board
    if enabled:
        sync_virtual_flags(raw_board, virtual_flags, persistent_revealed_mask)
        board = board_with_virtual_flags(raw_board, virtual_flags)
    if persistent_revealed_mask is not None and persistent_revealed_adjacent is not None:
        board = board_with_persistent_reveals(board, persistent_revealed_mask, persistent_revealed_adjacent)
    return board


def remember_revealed_cells(
    board: ScreenBoard,
    persistent_revealed_mask_or_cells: np.ndarray | set[tuple[int, int]],
    persistent_revealed_adjacent: np.ndarray | None = None,
    observations: np.ndarray | None = None,
    min_observations: int = 1,
) -> int:
    if board.done or board.read_repairs > 0 or board.read_restores > 0:
        return 0
    min_observations = max(1, int(min_observations))
    revealed_mask = board.revealed & ~board.mine_like
    if isinstance(persistent_revealed_mask_or_cells, set):
        cells = persistent_revealed_mask_or_cells
        before = len(cells)
        if observations is not None:
            remembered_mask = np.zeros_like(revealed_mask, dtype=bool)
            for row, col in cells:
                if 0 <= row < ROWS and 0 <= col < COLS:
                    remembered_mask[row, col] = True
            observations[~revealed_mask & ~remembered_mask] = 0
        for row, col in zip(*np.where(revealed_mask)):
            row = int(row)
            col = int(col)
            if observations is not None:
                observations[row, col] = min(255, int(observations[row, col]) + 1)
                if int(observations[row, col]) < min_observations:
                    continue
            cells.add((row, col))
        return len(cells) - before

    persistent_revealed_mask = persistent_revealed_mask_or_cells
    if persistent_revealed_adjacent is None:
        raise ValueError("persistent_revealed_adjacent is required when using mask memory")
    if observations is not None:
        observations[~revealed_mask & ~persistent_revealed_mask] = 0
    before = int(persistent_revealed_mask.sum())
    for row, col in zip(*np.where(revealed_mask)):
        row = int(row)
        col = int(col)
        if observations is not None:
            observations[row, col] = min(255, int(observations[row, col]) + 1)
            if int(observations[row, col]) < min_observations:
                continue
        persistent_revealed_mask[row, col] = True
        persistent_revealed_adjacent[row, col] = int(board.adjacent[row, col])
    return int(persistent_revealed_mask.sum()) - before


def remember_confirmed_open_cells(
    board: ScreenBoard,
    confirmed_open_cells: set[tuple[int, int]],
    target: tuple[int, int] | None = None,
    force_target: bool = False,
) -> int:
    before = len(confirmed_open_cells)
    if board.done and board.lost:
        return 0
    if board.read_repairs == 0 and board.read_restores == 0:
        for row, col in zip(*np.where(board.revealed & ~board.mine_like)):
            confirmed_open_cells.add((int(row), int(col)))
    if target is not None:
        row, col = target
        if 0 <= row < ROWS and 0 <= col < COLS and (
            force_target or (board.revealed[row, col] and not board.mine_like[row, col])
        ):
            confirmed_open_cells.add((int(row), int(col)))
    return len(confirmed_open_cells) - before


def apply_confirmed_open_action_mask(
    action_mask: np.ndarray,
    open_forbidden_cells: set[tuple[int, int]],
    confirmed_open_cells: set[tuple[int, int]],
) -> np.ndarray:
    if not open_forbidden_cells and not confirmed_open_cells:
        return action_mask
    masked = action_mask.copy()
    open_mask = masked[action_channel(ActionType.OPEN)]
    for row, col in open_forbidden_cells | confirmed_open_cells:
        if 0 <= row < ROWS and 0 <= col < COLS:
            open_mask[row, col] = False
    for kind in (ActionType.FLAG, ActionType.UNFLAG):
        channel = action_channel(kind)
        if channel >= masked.shape[0]:
            continue
        kind_mask = masked[channel]
        for row, col in confirmed_open_cells:
            if 0 <= row < ROWS and 0 <= col < COLS:
                kind_mask[row, col] = False
    return masked


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
    click_confirm_retries = max(1, min(4, int(getattr(args, "click_confirm_retries", 3))))
    post_click_settle = max(0.0, float(getattr(args, "post_click_settle", 0.0)))
    requested_click_hold = getattr(args, "click_hold", None)
    requested_cursor_settle = getattr(args, "cursor_settle", None)

    if profile == "safe":
        click_pause = 0.01 if requested_click_hold is None else max(0.001, float(requested_click_hold))
        cursor_settle = 0.005 if requested_cursor_settle is None else max(0.0, float(requested_cursor_settle))
        return {
            "profile": profile,
            "capture_delay": max(0.001, capture_delay),
            "action_delay": max(0.01, action_delay),
            "settle_reads": max(3, stable_reads),
            "settle_read_delay": max(0.03, stable_read_delay),
            "reclick_delay": max(0.01, reclick_delay),
            "no_progress_reclicks": no_progress_reclicks,
            "click_confirm_retries": click_confirm_retries,
            "click_pause": click_pause,
            "click_hold": click_pause,
            "cursor_settle": cursor_settle,
            "post_click_settle": max(0.05, post_click_settle),
        }
    if profile == "fast":
        click_pause = 0.008 if requested_click_hold is None else max(0.001, float(requested_click_hold))
        cursor_settle = 0.003 if requested_cursor_settle is None else max(0.0, float(requested_cursor_settle))
        return {
            "profile": profile,
            "capture_delay": max(0.0005, capture_delay),
            "action_delay": max(0.004, action_delay),
            "settle_reads": max(1, stable_reads),
            "settle_read_delay": max(0.02, stable_read_delay),
            "reclick_delay": max(0.08, reclick_delay),
            "no_progress_reclicks": max(1, no_progress_reclicks),
            "click_confirm_retries": click_confirm_retries,
            "click_pause": click_pause,
            "click_hold": click_pause,
            "cursor_settle": cursor_settle,
            "post_click_settle": max(0.08, post_click_settle),
        }
    if profile == "turbo":
        click_pause = 0.001 if requested_click_hold is None else max(0.001, float(requested_click_hold))
        cursor_settle = 0.0 if requested_cursor_settle is None else max(0.0, float(requested_cursor_settle))
        return {
            "profile": profile,
            "capture_delay": 0.0,
            "action_delay": 0.0,
            "settle_reads": 1,
            "settle_read_delay": 0.0,
            "reclick_delay": 0.0,
            "no_progress_reclicks": 0,
            "click_confirm_retries": click_confirm_retries,
            "click_pause": click_pause,
            "click_hold": click_pause,
            "cursor_settle": cursor_settle,
            "post_click_settle": 0.0,
        }
    if profile == "custom":
        click_pause = 0.04 if requested_click_hold is None else max(0.001, float(requested_click_hold))
        cursor_settle = 0.015 if requested_cursor_settle is None else max(0.0, float(requested_cursor_settle))
        return {
            "profile": profile,
            "capture_delay": capture_delay,
            "action_delay": action_delay,
            "settle_reads": stable_reads,
            "settle_read_delay": stable_read_delay,
            "reclick_delay": reclick_delay,
            "no_progress_reclicks": no_progress_reclicks,
            "click_confirm_retries": click_confirm_retries,
            "click_pause": click_pause,
            "click_hold": click_pause,
            "cursor_settle": cursor_settle,
            "post_click_settle": post_click_settle,
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
        click_method=args.click_method,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
        post_click_settle=float(timing["post_click_settle"]),
    )
    trainer = trainer or load_configured_trainer(args)
    audit_solver_enabled = bool(getattr(args, "audit_solver", False))
    audit_basic_enabled = bool(getattr(args, "audit_basic", False))
    basic_filter = getattr(args, "basic_safety_filter", "none")
    solver_filter = getattr(args, "solver_safety_filter", "none")
    solver_assist = getattr(args, "solver_assist", "none")
    solver_exact_limit = getattr(args, "solver_exact_limit", None)
    solver: MinesweeperSolver | None = None
    if audit_solver_enabled or solver_filter != "none" or solver_assist != "none":
        configured_limit = (
            int(solver_exact_limit)
            if solver_exact_limit is not None
            else int(getattr(trainer.config, "exact_limit", 24))
        )
        solver = MinesweeperSolver(exact_limit=max(1, configured_limit))
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
    use_persistent_reveals = not bool(getattr(args, "no_persistent_reveals", False))
    virtual_flags = np.zeros((ROWS, COLS), dtype=bool)
    pending_open_actions: list[Action] = []
    solver_batch_cells: set[tuple[int, int]] = set()
    blocked_open_cells: set[tuple[int, int]] = set()
    blocked_open_attempts: dict[tuple[int, int], int] = {}
    confirmed_open_cells: set[tuple[int, int]] = set()
    persistent_revealed_mask = np.zeros((ROWS, COLS), dtype=bool)
    persistent_revealed_adjacent = np.zeros((ROWS, COLS), dtype=np.int8)
    persistent_revealed_observations = np.zeros((ROWS, COLS), dtype=np.uint8)
    persistent_memory_kwargs = (
        {
            "persistent_revealed_mask": persistent_revealed_mask,
            "persistent_revealed_adjacent": persistent_revealed_adjacent,
        }
        if use_persistent_reveals
        else {}
    )
    no_progress_streak = 0
    terminal_dialog: str | None = None
    latest_raw_board: ScreenBoard | None = None
    last_board: ScreenBoard | None = None
    started_at = time.time()

    for step in range(args.max_steps):
        if stop_requested():
            break
        if desktop.dialog_is_open():
            terminal_dialog = desktop.dialog_title()
            if is_statistics_dialog_title(terminal_dialog or ""):
                desktop.choose_dialog_option(option="close", timeout=1.0, attempts=3)
                terminal_dialog = None
                latest_raw_board = None
                continue
            attribute_terminal_to_previous_action(actions, terminal_dialog, "before_next_action")
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
                attribute_terminal_to_previous_action(actions, terminal_dialog, "initial_read")
                break
        cleared_confirmed_flags = clear_virtual_flags_for_cells(virtual_flags, confirmed_open_cells)
        cleared_blocked_confirms = clear_blocked_open_cells_for_confirmed(
            blocked_open_cells,
            blocked_open_attempts,
            confirmed_open_cells,
        )
        board = memory_board(
            latest_raw_board,
            virtual_flags,
            use_memory_flags,
            **persistent_memory_kwargs,
        )
        confirmed_before_action = remember_confirmed_open_cells(board, confirmed_open_cells)
        cleared_confirmed_flags += clear_virtual_flags_for_cells(virtual_flags, confirmed_open_cells)
        cleared_blocked_confirms += clear_blocked_open_cells_for_confirmed(
            blocked_open_cells,
            blocked_open_attempts,
            confirmed_open_cells,
        )
        remembered_before_action = (
            remember_revealed_cells(
                board,
                persistent_revealed_mask,
                persistent_revealed_adjacent,
                observations=persistent_revealed_observations,
                min_observations=2,
            )
            if use_persistent_reveals
            else 0
        )
        last_board = board
        frame_index: int | None = None
        if args.record_frames == "all":
            frames.append(frame_from_board(board, step=step, action=None))
            frame_index = len(frames) - 1
        if board.done:
            break

        persistent_forbidden_cells = set(zip(*np.where(persistent_revealed_mask))) if use_persistent_reveals else set()
        forbidden_open_cells = persistent_forbidden_cells | confirmed_open_cells | blocked_open_cells
        queued_open = pop_legal_pending_open(pending_open_actions, board, forbidden_open_cells)
        queued_from_solver_batch = False
        if queued_open is not None:
            queued_from_solver_batch = (queued_open.row, queued_open.col) in solver_batch_cells
            solver_batch_cells.discard((queued_open.row, queued_open.col))
        selection_safety_record: dict[str, Any] | None = None
        basic_safety_record: dict[str, Any] | None = None
        basic_snapshot: dict[str, Any] | None = None
        solver_assist_record: dict[str, Any] | None = None
        solver_assist_flag_cells: list[tuple[int, int]] = []
        selection_started_at = time.perf_counter()
        if queued_open is not None:
            action = queued_open
            action_index = action_to_index(action, ROWS, COLS)
        else:
            if solver is not None and solver_assist != "none":
                action, solver_assist_flag_cells, solver_assist_record = select_solver_assist_action(
                    solver,
                    board,
                    forbidden_open_cells,
                    use_memory_flags=use_memory_flags,
                    batch_size=max(1, int(getattr(args, "solver_batch_size", 4))),
                )
                if solver_assist_flag_cells:
                    for row, col in solver_assist_flag_cells:
                        virtual_flags[row, col] = True
                    board = memory_board(
                        latest_raw_board,
                        virtual_flags,
                        use_memory_flags,
                        **persistent_memory_kwargs,
                    )
                    last_board = board
                if action is not None:
                    action_index = action_to_index(action, ROWS, COLS)
                    if (
                        solver_assist_record is not None
                        and solver_assist_record.get("decision") == "forced_safe_open"
                    ):
                        queued_targets = [
                            Action(
                                ActionType.OPEN,
                                int(target["row"]),
                                int(target["col"]),
                            )
                            for target in solver_assist_record.get("safe_batch_targets", [])
                        ]
                        queued_keys = {
                            (pending.row, pending.col)
                            for pending in pending_open_actions
                        }
                        current_key = (action.row, action.col)
                        added_batch = 0
                        for queued_target in queued_targets:
                            target_key = (queued_target.row, queued_target.col)
                            if target_key == current_key or target_key in queued_keys:
                                continue
                            pending_open_actions.append(queued_target)
                            solver_batch_cells.add(target_key)
                            queued_keys.add(target_key)
                            added_batch += 1
                        if added_batch:
                            solver_assist_record["queued_safe_count"] = added_batch
                else:
                    action = None
            else:
                action = None

            if action is None:
                encoded_board, global_features, action_mask = encode_state(board)
                action_mask = trainer._decision_action_mask(action_mask)
                action_mask = apply_confirmed_open_action_mask(
                    action_mask,
                    open_forbidden_cells=forbidden_open_cells,
                    confirmed_open_cells=confirmed_open_cells,
                )
                if basic_filter != "none" or audit_basic_enabled:
                    basic_snapshot = basic_inference(board)
                if basic_filter != "none":
                    basic_safety_record = apply_basic_safety_filter(board, action_mask, basic_filter, snapshot=basic_snapshot)
                if solver is not None and solver_filter != "none":
                    selection_safety_record = apply_solver_safety_filter(solver, board, action_mask, solver_filter)
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

        selection_elapsed = time.perf_counter() - selection_started_at
        before_signature = board_signature(board)
        before_revealed = int(board.revealed.sum())
        action_record = {"step": step, "action_index": int(action_index), "action": action_to_dict(action)}
        action_record["selection_elapsed"] = selection_elapsed
        if board.read_repairs:
            action_record["before_read_repairs"] = int(board.read_repairs)
        if board.read_restores:
            action_record["before_read_restores"] = int(board.read_restores)
        if board.read_recoveries:
            action_record["before_read_recoveries"] = int(board.read_recoveries)
        if remembered_before_action:
            action_record["remembered_revealed_before"] = int(remembered_before_action)
        if confirmed_before_action:
            action_record["confirmed_open_added_before"] = int(confirmed_before_action)
        if cleared_confirmed_flags:
            action_record["cleared_confirmed_virtual_flags"] = int(cleared_confirmed_flags)
        if cleared_blocked_confirms:
            action_record["cleared_confirmed_blocked_opens"] = int(cleared_blocked_confirms)
        if confirmed_open_cells:
            action_record["confirmed_open_cells"] = len(confirmed_open_cells)
        persistent_revealed_count = int(persistent_revealed_mask.sum())
        if persistent_revealed_count:
            action_record["persistent_revealed_cells"] = persistent_revealed_count
        if queued_open is not None and not queued_from_solver_batch:
            action_record["queued_from_chord"] = True
        if queued_from_solver_batch:
            action_record["queued_from_solver_batch"] = True
        if solver_assist_record is not None:
            action_record["solver_assist"] = solver_assist_record
            if solver_assist_flag_cells:
                action_record["solver_assist_virtual_flags"] = [
                    {"row": row, "col": col} for row, col in solver_assist_flag_cells
                ]
        if selection_safety_record is not None:
            action_record["solver_safety_filter"] = selection_safety_record
        if basic_safety_record is not None:
            action_record["basic_safety_filter"] = basic_safety_record
        if audit_basic_enabled and action.kind == ActionType.OPEN:
            action_record["basic_audit"] = (
                basic_action_audit_from_snapshot(basic_snapshot, action)
                if basic_snapshot is not None
                else basic_action_audit(board, action)
            )
        if audit_solver_enabled and solver is not None:
            action_record["solver_audit"] = solver_audit(solver, board, action)
        if action.kind == ActionType.OPEN and desktop.grid is not None:
            click_point = getattr(desktop, "click_point", None)
            target_x, target_y = (
                click_point(action.row, action.col)
                if callable(click_point)
                else desktop.grid.center(action.row, action.col)
            )
            action_record["screen_target"] = {
                "x": int(target_x),
                "y": int(target_y),
            }
        action_record["before_target"] = cell_snapshot(board, action.row, action.col)
        if frame_index is not None:
            frames[frame_index]["action"] = action_record
        if use_memory_flags and action.kind in {ActionType.FLAG, ActionType.UNFLAG, ActionType.CHORD}:
            if action.kind == ActionType.FLAG:
                if solver_assist_flag_cells:
                    action_record["virtual_only"] = True
                    action_record["virtual_change"] = "flag_batch"
                    action_record["no_progress"] = False
                    action_record["virtual_flag_batch_count"] = len(solver_assist_flag_cells)
                    no_progress_streak = 0
                    last_board = board
                elif not board.revealed[action.row, action.col] and not board.flagged[action.row, action.col]:
                    virtual_flags[action.row, action.col] = True
                    action_record["virtual_only"] = True
                    action_record["virtual_change"] = "flag"
                    no_progress_streak = 0
                    last_board = memory_board(
                        latest_raw_board,
                        virtual_flags,
                        use_memory_flags,
                        **persistent_memory_kwargs,
                    )
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
                    last_board = memory_board(
                        latest_raw_board,
                        virtual_flags,
                        use_memory_flags,
                        **persistent_memory_kwargs,
                    )
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
                if (target.row, target.col) not in forbidden_open_cells
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
            click_report = desktop.click_action(action)
        except RuntimeError as exc:
            terminal_dialog = desktop.dialog_title()
            if terminal_dialog and actions:
                attribute_terminal_to_previous_action(actions, terminal_dialog, "before_next_click")
                action_record["skipped"] = True
                action_record["skip_reason"] = "terminal_dialog_before_click"
                action_record["terminal_dialog"] = terminal_dialog
                action_record["terminal_detected_at"] = "before_click"
                action_record["click_error"] = str(exc)
                break
            action_record["terminal_dialog"] = terminal_dialog
            action_record["terminal_detected_at"] = "before_click" if terminal_dialog else "click_error"
            action_record["click_error"] = str(exc)
            last_click = getattr(desktop, "last_click_report", None)
            if last_click is not None:
                action_record["last_click_report"] = last_click
            actions.append(action_record)
            break
        has_report_target = all(key in click_report for key in ("row", "col", "kind"))
        if has_report_target and not click_report_matches_action(click_report, action):
            action_record["click_target_mismatch"] = {
                "requested": action_to_dict(action),
                "reported": click_report,
            }
            action_record["click_error"] = "click report target did not match requested action"
            actions.append(action_record)
            break
        action_record["click"] = click_report
        action_delay_started_at = time.perf_counter()
        time.sleep(action_delay)
        action_record["action_delay_elapsed"] = time.perf_counter() - action_delay_started_at
        if desktop.dialog_is_open():
            terminal_dialog = desktop.dialog_title()
            action_record["terminal_dialog"] = terminal_dialog
            action_record["terminal_detected_at"] = "after_click"
            actions.append(action_record)
            break

        pending_solver_batch_count = sum(
            1
            for pending in pending_open_actions
            if (pending.row, pending.col) in solver_batch_cells
        )
        if action.kind == ActionType.OPEN and (
            (
                solver_assist_record is not None
                and solver_assist_record.get("decision") == "forced_safe_open"
                and int(solver_assist_record.get("queued_safe_count", 0) or 0) > 0
            )
            or (queued_from_solver_batch and pending_solver_batch_count > 0)
        ):
            action_record["deferred_read"] = "solver_batch"
            action_record["deferred_pending_solver_opens"] = int(pending_solver_batch_count)
            no_progress_streak = 0
            actions.append(action_record)
            last_board = board
            continue

        before_raw_board = latest_raw_board
        read_started_at = time.time()
        quick_read_info: dict[str, Any] | None = None
        after_raw_board: ScreenBoard | None = None
        if bool(getattr(args, "quick_number_read", False)) and action.kind == ActionType.OPEN:
            after_raw_board, quick_read_info = quick_number_board_after_open(
                desktop=desktop,
                previous_board=before_raw_board,
                action=action,
                step_count=step + 1,
                keep_screenshot=keep_screenshot,
            )
            action_record["quick_number_read"] = quick_read_info
        if after_raw_board is None:
            try:
                after_raw_board = read_stable_board(
                    desktop,
                    step_count=step + 1,
                    keep_screenshot=keep_screenshot,
                    reads=settle_reads,
                    delay=settle_read_delay,
                    previous_board=before_raw_board,
                )
            except RuntimeError:
                terminal_dialog = desktop.dialog_title()
                action_record["terminal_dialog"] = terminal_dialog
                action_record["terminal_detected_at"] = "after_click_read"
                actions.append(action_record)
                break
        action_record["after_read_elapsed"] = time.time() - read_started_at
        if after_raw_board.read_timing is not None:
            action_record["read_timing"] = after_raw_board.read_timing
        latest_raw_board = after_raw_board
        after_board = memory_board(
            after_raw_board,
            virtual_flags,
            use_memory_flags,
            **persistent_memory_kwargs,
        )
        terminal_after_read = False
        if desktop.dialog_is_open():
            terminal_dialog = desktop.dialog_title()
            action_record["terminal_dialog"] = terminal_dialog
            action_record["terminal_detected_at"] = "after_click_read"
            terminal_after_read = True
        if not terminal_after_read and open_needs_confirmation(action, after_board, before_revealed):
            confirm_started_at = time.time()
            confirmed_read = confirm_open_read(
                desktop=desktop,
                action=action,
                previous_board=after_raw_board,
                before_click_board=before_raw_board,
                step_count=step + 1,
                keep_screenshot=keep_screenshot,
                settle_reads=settle_reads,
                settle_read_delay=settle_read_delay,
                reclicks=max(
                    1,
                    int(timing.get("click_confirm_retries", 3)),
                    int(timing.get("no_progress_reclicks", 0)),
                ),
                reclick_delay=float(timing["reclick_delay"]),
                virtual_flags=virtual_flags,
                use_memory_flags=use_memory_flags,
                **persistent_memory_kwargs,
            )
            if confirmed_read is not None:
                latest_raw_board, after_board, confirm_info = confirmed_read
                action_record["open_confirm"] = confirm_info
                action_record["reclicks"] = int(confirm_info.get("reclicks", 0))
                action_record["open_confirm_elapsed"] = time.time() - confirm_started_at
        last_board = after_board
        changed = board_signature(after_board) != before_signature
        after_revealed = int(after_board.revealed.sum())
        revealed_delta = after_revealed - before_revealed
        progress = revealed_delta > 0 or after_board.done
        action_record["after_target"] = cell_snapshot(after_board, action.row, action.col)
        action_record["changed"] = bool(changed)
        action_record["progress"] = bool(progress)
        action_record["revealed_delta"] = int(revealed_delta)
        if action.kind == ActionType.OPEN:
            open_effect = open_effect_summary(board, after_board, action)
            action_record["open_effect"] = open_effect
            action_record["target_revealed_after_open"] = bool(open_effect["target_revealed"])
            clean_after_read = after_board.read_repairs == 0 and after_board.read_restores == 0
            clean_confirm = (
                action_record.get("open_confirm", {}).get("target_revealed")
                and int(action_record.get("open_confirm", {}).get("final_read_repairs", 0) or 0) == 0
                and int(action_record.get("open_confirm", {}).get("final_read_restores", 0) or 0) == 0
            )
            if use_persistent_reveals and clean_after_read and after_board.revealed[action.row, action.col]:
                persistent_revealed_mask[action.row, action.col] = True
                persistent_revealed_adjacent[action.row, action.col] = int(after_board.adjacent[action.row, action.col])
                persistent_revealed_observations[action.row, action.col] = max(
                    persistent_revealed_observations[action.row, action.col],
                    2,
                )
            if use_persistent_reveals and clean_confirm:
                persistent_revealed_mask[action.row, action.col] = True
                persistent_revealed_adjacent[action.row, action.col] = int(after_board.adjacent[action.row, action.col])
                persistent_revealed_observations[action.row, action.col] = max(
                    persistent_revealed_observations[action.row, action.col],
                    2,
                )
            remembered_after_action = 0
            if (
                use_persistent_reveals
                and revealed_delta > 0
                and after_board.read_repairs == 0
                and after_board.read_restores == 0
            ):
                remembered_after_action = remember_revealed_cells(
                    after_board,
                    persistent_revealed_mask,
                    persistent_revealed_adjacent,
                    observations=persistent_revealed_observations,
                    min_observations=1,
                )
            if remembered_after_action:
                action_record["remembered_revealed_after"] = int(remembered_after_action)
            confirmed_after_action = remember_confirmed_open_cells(
                after_board,
                confirmed_open_cells,
                target=(action.row, action.col) if (open_effect["target_revealed"] or revealed_delta > 0) else None,
                force_target=revealed_delta > 0,
            )
            if confirmed_after_action:
                action_record["confirmed_open_added_after"] = int(confirmed_after_action)
            cleared_after_confirm = clear_virtual_flags_for_cells(virtual_flags, confirmed_open_cells)
            if cleared_after_confirm:
                action_record["cleared_confirmed_virtual_flags_after"] = int(cleared_after_confirm)
            cleared_blocked_after_confirm = clear_blocked_open_cells_for_confirmed(
                blocked_open_cells,
                blocked_open_attempts,
                confirmed_open_cells,
            )
            if cleared_blocked_after_confirm:
                action_record["cleared_confirmed_blocked_opens_after"] = int(cleared_blocked_after_confirm)
            if confirmed_open_cells:
                action_record["confirmed_open_cells"] = len(confirmed_open_cells)
        if persistent_revealed_mask.any():
            action_record["persistent_revealed_cells"] = int(persistent_revealed_mask.sum())
        if after_board.read_repairs:
            action_record["after_read_repairs"] = int(after_board.read_repairs)
        if after_board.read_restores:
            action_record["after_read_restores"] = int(after_board.read_restores)
        if after_board.read_recoveries:
            action_record["after_read_recoveries"] = int(after_board.read_recoveries)
        action_record["after_grid"] = {
            "x0": int(after_board.grid.x_lines[0]),
            "x1": int(after_board.grid.x_lines[-1]),
            "y0": int(after_board.grid.y_lines[0]),
            "y1": int(after_board.grid.y_lines[-1]),
            "cell_width": float(after_board.grid.cell_width),
            "cell_height": float(after_board.grid.cell_height),
            "cell_square_error": abs(float(after_board.grid.cell_width) - float(after_board.grid.cell_height)),
        }
        target_open_confirmed = bool(after_board.revealed[action.row, action.col]) or bool(after_board.done)
        if action.kind == ActionType.OPEN and not target_open_confirmed:
            action_record["click_unconfirmed"] = True
            action_record["click_unconfirmed_reason"] = "target_still_hidden_after_readback"
            blocked_open_cells.add((action.row, action.col))
            blocked_open_attempts[(action.row, action.col)] = max(
                2,
                blocked_open_attempts.get((action.row, action.col), 0),
            )
            action_record["halted_after_unconfirmed_open"] = True
            actions.append(action_record)
            break
        if not changed or not progress:
            action_record["no_progress"] = True
            no_progress_streak += 1
            if action.kind == ActionType.OPEN:
                cell_key = (action.row, action.col)
                attempts = blocked_open_attempts.get(cell_key, 0) + 1
                blocked_open_attempts[cell_key] = attempts
                if attempts >= 2:
                    blocked_open_cells.add(cell_key)
                    action_record["blocked_repeat_open"] = True
                else:
                    action_record["blocked_repeat_open_pending"] = True
        else:
            no_progress_streak = 0
            if action.kind == ActionType.OPEN:
                blocked_open_attempts.pop((action.row, action.col), None)
        actions.append(action_record)
        if terminal_after_read:
            break
        if no_progress_streak >= args.stall_limit:
            break

    if desktop.dialog_is_open():
        terminal_dialog = terminal_dialog or desktop.dialog_title()
        attribute_terminal_to_previous_action(actions, terminal_dialog, "final_check")
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
            final_board = memory_board(
                final_raw_board,
                virtual_flags,
                use_memory_flags,
                **persistent_memory_kwargs,
            )
    if args.record_frames in {"all", "final"}:
        frames.append(frame_from_board(final_board, step=len(actions), action=None))
    trace = {
        "version": 1,
        "source": "windows_minesweeper",
        "checkpoint": str(args.checkpoint),
        "game_index": game_index,
        "click_method": args.click_method,
        "resolved_click_method": desktop.effective_click_method(),
        "persistent_reveals": use_persistent_reveals,
        "audit_basic": audit_basic_enabled,
        "basic_safety_filter": basic_filter,
        "solver_assist": solver_assist,
        "solver_exact_limit": None if solver_exact_limit is None else int(solver_exact_limit),
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
    return {"path": str(path), "summary": trace["summary"], "resolved_click_method": desktop.effective_click_method()}


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
        "read_recoveries": int(board.read_recoveries),
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
    timing: dict[str, Any] = {
        "reads_requested": max(1, int(reads)),
        "read_delay": max(0.0, float(delay)),
    }
    started_at = time.perf_counter()
    first_read_started = time.perf_counter()
    board = recover_unstable_read(
        desktop,
        restore_revealed_cells(
            desktop.read_board(step_count=step_count, keep_screenshot=keep_screenshot, previous_board=previous_board),
            previous_board,
        ),
        previous_board=previous_board,
        step_count=step_count,
        keep_screenshot=keep_screenshot,
    )
    timing["first_read_seconds"] = time.perf_counter() - first_read_started
    timing["first_read"] = board.read_timing
    reads = max(1, int(reads))
    best_board = board
    for _ in range(reads - 1):
        previous_signature = board_signature(board)
        sleep_started = time.perf_counter()
        time.sleep(max(0.0, delay))
        timing["between_read_sleep_seconds"] = timing.get("between_read_sleep_seconds", 0.0) + (
            time.perf_counter() - sleep_started
        )
        repeat_read_started = time.perf_counter()
        next_board = recover_unstable_read(
            desktop,
            restore_revealed_cells(
                desktop.read_board(step_count=step_count, keep_screenshot=keep_screenshot, previous_board=board),
                board,
            ),
            previous_board=board,
            step_count=step_count,
            keep_screenshot=keep_screenshot,
        )
        timing["repeat_read_seconds"] = timing.get("repeat_read_seconds", 0.0) + (
            time.perf_counter() - repeat_read_started
        )
        timing["repeat_read"] = next_board.read_timing
        if next_board.done:
            timing["total_seconds"] = time.perf_counter() - started_at
            next_board.read_timing = timing
            return next_board
        if board_signature(next_board) == previous_signature:
            timing["total_seconds"] = time.perf_counter() - started_at
            result = next_board if read_quality_key(next_board) >= read_quality_key(board) else board
            result.read_timing = timing
            return result
        if read_quality_key(next_board) > read_quality_key(board):
            board = next_board
            if read_quality_key(board) > read_quality_key(best_board):
                best_board = board
            continue
        if read_quality_key(next_board) > read_quality_key(best_board):
            best_board = next_board
        board = best_board
    timing["total_seconds"] = time.perf_counter() - started_at
    result = best_board if read_quality_key(best_board) >= read_quality_key(board) else board
    result.read_timing = timing
    return result


def quick_number_board_after_open(
    desktop: WindowsMinesweeper,
    previous_board: ScreenBoard | None,
    action: Action,
    step_count: int,
    keep_screenshot: bool,
) -> tuple[ScreenBoard | None, dict[str, Any]]:
    info: dict[str, Any] = {"attempted": True, "mode": "quick_number"}
    if action.kind != ActionType.OPEN:
        info["reason"] = "not_open_action"
        return None, info
    if previous_board is None:
        info["reason"] = "missing_previous_board"
        return None, info
    if previous_board.pixels is None:
        info["reason"] = "missing_previous_pixels"
        return None, info
    if previous_board.revealed[action.row, action.col] or previous_board.flagged[action.row, action.col]:
        info["reason"] = "target_not_hidden"
        return None, info
    screen_grid = getattr(desktop, "grid", None)
    if screen_grid is None:
        info["reason"] = "missing_screen_grid"
        return None, info

    try:
        screen_box = screen_grid.crop_box(action.row, action.col)
        local_box = previous_board.grid.crop_box(action.row, action.col)
        capture_started_at = time.perf_counter()
        crop = desktop.capture_region_array(*screen_box)
        capture_elapsed = time.perf_counter() - capture_started_at
        classify_started_at = time.perf_counter()
        cell = classify_cell_fast(crop)
        classify_elapsed = time.perf_counter() - classify_started_at
    except Exception as exc:
        info["reason"] = "capture_or_classify_error"
        info["error"] = f"{type(exc).__name__}: {exc}"
        return None, info

    info["screen_box"] = [int(value) for value in screen_box]
    info["cell_kind"] = cell.get("kind")
    info["cell_number"] = int(cell.get("number", 0) or 0)
    info["capture_seconds"] = capture_elapsed
    info["classification_seconds"] = classify_elapsed
    if cell.get("kind") != "revealed":
        info["reason"] = "target_not_revealed"
        return None, info
    number = int(cell.get("number", 0) or 0)
    if number <= 0:
        info["reason"] = "zero_or_ambiguous_reveal"
        return None, info
    if number > max_neighbor_count(action.row, action.col):
        info["reason"] = "impossible_number"
        return None, info

    revealed = previous_board.revealed.copy()
    flagged = previous_board.flagged.copy()
    adjacent = previous_board.adjacent.copy()
    mine_like = previous_board.mine_like.copy()
    revealed[action.row, action.col] = True
    flagged[action.row, action.col] = False
    adjacent[action.row, action.col] = number
    mine_like[action.row, action.col] = False

    pixels = previous_board.pixels.copy()
    lx0, ly0, lx1, ly1 = (int(value) for value in local_box)
    target_h = max(0, ly1 - ly0)
    target_w = max(0, lx1 - lx0)
    if (
        target_h > 0
        and target_w > 0
        and 0 <= ly0 < ly1 <= pixels.shape[0]
        and 0 <= lx0 < lx1 <= pixels.shape[1]
        and crop.shape[0] >= target_h
        and crop.shape[1] >= target_w
    ):
        pixels[ly0:ly1, lx0:lx1] = crop[:target_h, :target_w]
        info["pixels_updated"] = True
    else:
        info["pixels_updated"] = False

    repairs = repair_impossible_numbers(revealed, adjacent, mine_like)
    if repairs:
        info["reason"] = "repair_needed"
        info["read_repairs"] = int(repairs)
        return None, info

    board = ScreenBoard(
        revealed=revealed,
        flagged=flagged,
        adjacent=adjacent,
        mine_like=mine_like,
        grid=previous_board.grid,
        screenshot=previous_board.screenshot if keep_screenshot else None,
        pixels=pixels,
        step_count=step_count,
        read_repairs=0,
        read_restores=0,
        read_recoveries=0,
    )
    board.read_timing = {
        "mode": "quick_number",
        "capture_path": "cell_region",
        "capture_seconds": capture_elapsed,
        "classification_seconds": classify_elapsed,
        "total_seconds": capture_elapsed + classify_elapsed,
        "target_number": number,
    }
    info["accepted"] = True
    info["reason"] = "accepted"
    return board, info


def read_quality_key(board: ScreenBoard) -> tuple[int, int, int, int]:
    return (
        -int(board.read_repairs),
        -int(board.read_restores),
        -int(board.read_recoveries),
        int(board.revealed.sum()),
    )


def update_confirm_read_info(info: dict[str, Any], board: ScreenBoard | None) -> None:
    if board is None:
        info["best_revealed"] = 0
        info["final_read_repairs"] = 0
        info["final_read_restores"] = 0
        info["final_read_recoveries"] = 0
        return
    info["best_revealed"] = int(board.revealed.sum())
    info["final_read_repairs"] = int(board.read_repairs)
    info["final_read_restores"] = int(board.read_restores)
    info["final_read_recoveries"] = int(board.read_recoveries)


def confirm_read_quality_key(board: ScreenBoard, action: Action) -> tuple[int, int, int, int, int, int]:
    return (
        int(board.done),
        int(board.revealed[action.row, action.col]),
        -int(board.read_repairs),
        -int(board.read_restores),
        -int(board.read_recoveries),
        int(board.revealed.sum()),
    )


def recover_unstable_read(
    desktop: WindowsMinesweeper,
    board: ScreenBoard,
    previous_board: ScreenBoard | None,
    step_count: int,
    keep_screenshot: bool,
) -> ScreenBoard:
    if previous_board is None or board.read_restores <= 0:
        return board
    if getattr(desktop, "read_mode", None) != "fast" or not hasattr(desktop, "_read_board_accurate"):
        return board
    try:
        accurate = desktop._read_board_accurate(step_count=step_count, keep_screenshot=keep_screenshot)
    except RuntimeError:
        return board
    accurate = restore_revealed_cells(accurate, previous_board)
    if read_quality_key(accurate) > read_quality_key(board):
        accurate.read_recoveries = board.read_recoveries + 1
        return accurate
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


def board_difference_counts(board: ScreenBoard, previous: ScreenBoard | None) -> dict[str, int | bool]:
    if previous is None:
        return {
            "available": False,
            "revealed_changed": 0,
            "flagged_changed": 0,
            "adjacent_changed": 0,
            "mine_like_changed": 0,
            "signature_changed": False,
        }
    return {
        "available": True,
        "revealed_changed": int(np.logical_xor(board.revealed, previous.revealed).sum()),
        "flagged_changed": int(np.logical_xor(board.flagged, previous.flagged).sum()),
        "adjacent_changed": int((board.adjacent != previous.adjacent).sum()),
        "mine_like_changed": int(np.logical_xor(board.mine_like, previous.mine_like).sum()),
        "signature_changed": board_signature(board) != board_signature(previous),
    }


def board_compare_counts(fast_board: ScreenBoard, accurate_board: ScreenBoard) -> dict[str, int | bool]:
    both_revealed = fast_board.revealed & accurate_board.revealed
    return {
        "available": True,
        "fast_hidden_accurate_revealed": int((~fast_board.revealed & accurate_board.revealed).sum()),
        "fast_revealed_accurate_hidden": int((fast_board.revealed & ~accurate_board.revealed).sum()),
        "revealed_mismatch": int(np.logical_xor(fast_board.revealed, accurate_board.revealed).sum()),
        "flag_mismatch": int(np.logical_xor(fast_board.flagged, accurate_board.flagged).sum()),
        "mine_like_mismatch": int(np.logical_xor(fast_board.mine_like, accurate_board.mine_like).sum()),
        "number_mismatch": int((both_revealed & (fast_board.adjacent != accurate_board.adjacent)).sum()),
        "fast_revealed": int(fast_board.revealed.sum()),
        "accurate_revealed": int(accurate_board.revealed.sum()),
        "fast_read_repairs": int(fast_board.read_repairs),
        "accurate_read_repairs": int(accurate_board.read_repairs),
        "fast_read_restores": int(fast_board.read_restores),
        "accurate_read_restores": int(accurate_board.read_restores),
        "same_signature": board_signature(fast_board) == board_signature(accurate_board),
    }


def summarize_accurate_compare_records(records: list[dict[str, Any]]) -> dict[str, Any] | None:
    available_records = [record for record in records if record.get("available")]
    if not available_records:
        return None
    return {
        "records": len(available_records),
        "fast_hidden_accurate_revealed": int(
            sum(int(record.get("fast_hidden_accurate_revealed", 0)) for record in available_records)
        ),
        "fast_revealed_accurate_hidden": int(
            sum(int(record.get("fast_revealed_accurate_hidden", 0)) for record in available_records)
        ),
        "number_mismatch": int(sum(int(record.get("number_mismatch", 0)) for record in available_records)),
        "all_same_signature": all(bool(record.get("same_signature")) for record in available_records),
        "avg_accurate_seconds": sum(float(record.get("accurate_elapsed_seconds", 0.0)) for record in available_records)
        / len(available_records),
    }


def accurate_compare_for_board(
    desktop: WindowsMinesweeper,
    board: ScreenBoard,
    previous_board: ScreenBoard | None,
    step_count: int,
) -> dict[str, Any]:
    if getattr(desktop, "read_mode", None) != "fast":
        return {"available": False, "reason": "read_mode_is_not_fast"}
    accurate_started_at = time.time()
    try:
        accurate_raw = desktop._read_board_accurate(step_count=step_count, keep_screenshot=False)
        accurate_board = restore_revealed_cells(accurate_raw, previous_board)
    except RuntimeError as exc:
        return {
            "available": False,
            "error": type(exc).__name__,
            "message": str(exc),
        }
    compare = board_compare_counts(board, accurate_board)
    compare["accurate_elapsed_seconds"] = time.time() - accurate_started_at
    return compare


def read_compare_preflight(
    desktop: WindowsMinesweeper,
    timing: dict[str, Any],
    reads: int = 3,
    interval: float = 0.02,
) -> dict[str, Any]:
    reads = max(1, int(reads))
    interval = max(0.0, float(interval))
    settle_reads = int(timing["settle_reads"])
    settle_read_delay = float(timing["settle_read_delay"])
    previous_board: ScreenBoard | None = None
    records: list[dict[str, Any]] = []
    started_at = time.time()

    for index in range(reads):
        read_started_at = time.time()
        try:
            board = read_stable_board(
                desktop,
                step_count=index,
                keep_screenshot=False,
                reads=settle_reads,
                delay=settle_read_delay,
                previous_board=previous_board,
            )
        except RuntimeError as exc:
            records.append(
                {
                    "index": index,
                    "available": False,
                    "error": type(exc).__name__,
                    "message": str(exc),
                }
            )
            break
        compare = accurate_compare_for_board(desktop, board, previous_board, step_count=index)
        records.append(
            {
                "index": index,
                "available": True,
                "elapsed_seconds": time.time() - read_started_at,
                "revealed": int(board.revealed.sum()),
                "flags": int(board.flagged.sum()),
                "won": bool(board.won),
                "lost": bool(board.lost),
                "read_repairs": int(board.read_repairs),
                "read_restores": int(board.read_restores),
                "read_recoveries": int(board.read_recoveries),
                "accurate_compare": compare,
            }
        )
        previous_board = board
        if interval > 0 and index < reads - 1:
            time.sleep(interval)

    compare_records = [
        record["accurate_compare"]
        for record in records
        if record.get("available") and record.get("accurate_compare", {}).get("available")
    ]
    compare_summary = summarize_accurate_compare_records(compare_records)
    read_elapsed = [float(record.get("elapsed_seconds", 0.0)) for record in records if record.get("available")]
    passed = bool(compare_summary and compare_summary.get("all_same_signature"))
    return {
        "enabled": True,
        "status": "passed" if passed else "warning",
        "reads": reads,
        "avg_read_seconds": sum(read_elapsed) / len(read_elapsed) if read_elapsed else 0.0,
        "accurate_compare_summary": compare_summary,
        "records": records,
        "elapsed_seconds": time.time() - started_at,
    }


def open_effect_summary(before: ScreenBoard, after: ScreenBoard, action: Action) -> dict[str, Any]:
    newly_revealed = after.revealed & ~before.revealed
    lost_revealed = before.revealed & ~after.revealed
    target_revealed = bool(after.revealed[action.row, action.col])
    target_newly_revealed = bool(newly_revealed[action.row, action.col])
    added_count = int(newly_revealed.sum())
    summary: dict[str, Any] = {
        "target_revealed": target_revealed,
        "target_newly_revealed": target_newly_revealed,
        "new_revealed_cells": added_count,
        "lost_revealed_cells": int(lost_revealed.sum()),
        "target_missed_with_progress": bool(added_count > 0 and not target_revealed),
    }
    if added_count:
        rows, cols = np.where(newly_revealed)
        dr = rows.astype(np.int16) - int(action.row)
        dc = cols.astype(np.int16) - int(action.col)
        chebyshev = np.maximum(np.abs(dr), np.abs(dc))
        manhattan = np.abs(dr) + np.abs(dc)
        order = np.lexsort((manhattan, chebyshev))
        best = int(order[0])
        nearest_row = int(rows[best])
        nearest_col = int(cols[best])
        summary["nearest_new_revealed"] = {
            "row": nearest_row,
            "col": nearest_col,
            "dr": int(nearest_row - action.row),
            "dc": int(nearest_col - action.col),
            "chebyshev": int(chebyshev[best]),
            "manhattan": int(manhattan[best]),
        }
    return summary


def open_needs_confirmation(action: Action, board: ScreenBoard, before_revealed: int) -> bool:
    if action.kind != ActionType.OPEN or board.done:
        return False
    if board.revealed[action.row, action.col] and board.read_repairs == 0:
        return False
    if board.revealed[action.row, action.col] and int(board.revealed.sum()) > before_revealed:
        return False
    return bool(board.read_repairs > 0 or not board.revealed[action.row, action.col])


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
    before_click_board: ScreenBoard | None = None,
    persistent_revealed_mask: np.ndarray | None = None,
    persistent_revealed_adjacent: np.ndarray | None = None,
) -> tuple[ScreenBoard, ScreenBoard, dict[str, Any]] | None:
    best_raw_board = previous_board
    best_board = (
        memory_board(
            previous_board,
            virtual_flags,
            use_memory_flags,
            persistent_revealed_mask=persistent_revealed_mask,
            persistent_revealed_adjacent=persistent_revealed_adjacent,
        )
        if previous_board is not None
        else None
    )
    read_previous = previous_board
    max_reclicks = max(0, min(4, int(reclicks)))
    read_count = max(1, int(settle_reads))
    read_delay = max(0.0, settle_read_delay)
    info: dict[str, Any] = {
        "read_attempts": 0,
        "passive_reads": 0,
        "reclicks": 0,
        "accurate_fallback_reads": 0,
        "target_pixel_changed": False,
        "confirmed": False,
        "target_revealed": bool(best_board is not None and best_board.revealed[action.row, action.col]),
        "required_reclicks": max_reclicks,
        "clicks": [],
    }
    update_confirm_read_info(info, best_board)
    if best_board is not None and (best_board.done or (info["target_revealed"] and best_board.read_repairs == 0)):
        info["confirmed"] = True
        return best_raw_board, best_board, info

    def observe_after_delay(delay_seconds: float) -> tuple[ScreenBoard, ScreenBoard] | None:
        nonlocal best_raw_board, best_board, read_previous
        time.sleep(max(0.0, delay_seconds))
        try:
            retry_raw_board = read_stable_board(
                desktop,
                step_count=step_count,
                keep_screenshot=keep_screenshot,
                reads=read_count,
                delay=read_delay,
                previous_board=read_previous,
            )
        except RuntimeError:
            return None
        info["read_attempts"] += 1
        retry_board = memory_board(
            retry_raw_board,
            virtual_flags,
            use_memory_flags,
            persistent_revealed_mask=persistent_revealed_mask,
            persistent_revealed_adjacent=persistent_revealed_adjacent,
        )
        read_previous = retry_raw_board
        pixel_change = cell_pixel_change(
            before_click_board,
            retry_raw_board,
            action.row,
            action.col,
        )
        if pixel_change.get("available"):
            info["target_pixel_changed"] = bool(
                info["target_pixel_changed"] or pixel_change.get("changed", False)
            )
            info["target_pixel_changed_ratio"] = float(pixel_change.get("changed_ratio", 0.0))
            info["target_pixel_mean_delta"] = float(pixel_change.get("mean_delta", 0.0))

        # The fast classifier can lag on the blue-to-gray transition, most
        # noticeably near the top-left where hidden cells are brighter. If the
        # target pixels changed, use one accurate pass before re-clicking.
        if (
            pixel_change.get("available")
            and pixel_change.get("changed")
            and not retry_board.revealed[action.row, action.col]
            and getattr(desktop, "read_mode", "fast") == "fast"
            and info["accurate_fallback_reads"] == 0
            and (
                float(pixel_change.get("mean_delta", 0.0)) >= CONFIRM_ACCURATE_MIN_MEAN_DELTA
                and float(pixel_change.get("changed_ratio", 0.0)) >= CONFIRM_ACCURATE_MIN_CHANGED_RATIO
            )
        ):
            accurate_reader = getattr(desktop, "_read_board_accurate", None)
            if callable(accurate_reader):
                try:
                    accurate_raw_board = accurate_reader(
                        step_count=step_count,
                        keep_screenshot=keep_screenshot,
                    )
                except RuntimeError:
                    accurate_raw_board = None
                if accurate_raw_board is not None:
                    info["accurate_fallback_reads"] += 1
                    accurate_board = memory_board(
                        accurate_raw_board,
                        virtual_flags,
                        use_memory_flags,
                        persistent_revealed_mask=persistent_revealed_mask,
                        persistent_revealed_adjacent=persistent_revealed_adjacent,
                    )
                    if confirm_read_quality_key(accurate_board, action) > confirm_read_quality_key(
                        retry_board,
                        action,
                    ):
                        retry_raw_board = accurate_raw_board
                        retry_board = accurate_board
                        read_previous = accurate_raw_board
        if best_board is None:
            best_raw_board = retry_raw_board
            best_board = retry_board
        else:
            if confirm_read_quality_key(retry_board, action) > confirm_read_quality_key(best_board, action):
                best_raw_board = retry_raw_board
                best_board = retry_board
        target_revealed = bool(retry_board.revealed[action.row, action.col])
        info["target_revealed"] = target_revealed
        update_confirm_read_info(info, best_board)
        if retry_board.done or (target_revealed and retry_board.read_repairs == 0):
            info["confirmed"] = True
        return retry_raw_board, retry_board

    if max_reclicks > 0:
        observed = observe_after_delay(reclick_delay)
        if observed is None:
            info["confirmation_error"] = "readback_failed"
            return None
        info["passive_reads"] += 1
        if info["confirmed"]:
            return observed[0], observed[1], info

    for reclick_index in range(max_reclicks):
        retry_method = None
        click_with_method = getattr(desktop, "click_action_with_method", None)
        use_hardened_retry = (
            getattr(desktop, "click_method", None) == "mouse_event"
            and callable(click_with_method)
        )
        time.sleep(
            max(
                0.0,
                RETRY_PRECLICK_DELAY_SECONDS if use_hardened_retry else reclick_delay,
            )
        )
        refresh_grid = getattr(desktop, "refresh_grid", None)
        if reclick_index == 0 and callable(refresh_grid):
            try:
                refresh_grid()
            except Exception:
                # The existing board read remains usable if a transient capture
                # prevents a geometry refresh; the click itself will still be
                # validated by WindowsMinesweeper.click_action.
                pass
        try:
            if use_hardened_retry:
                # Keep the backend that worked for the primary click.
                # This classic window is less reliable when retries switch
                # from mouse_event to SendInput.
                retry_method = "mouse_event"
            if retry_method is not None and callable(click_with_method):
                retry_fraction = RETRY_CLICK_FRACTIONS[
                    min(reclick_index, len(RETRY_CLICK_FRACTIONS) - 1)
                ]
                click_report = click_with_method(
                    action,
                    retry_method,
                    click_pause=RETRY_CLICK_HOLD_SECONDS,
                    cursor_settle=RETRY_CURSOR_SETTLE_SECONDS,
                    post_click_settle=RETRY_POST_CLICK_SETTLE_SECONDS,
                    click_fraction=retry_fraction,
                )
                click_report["retry_method_switch"] = False
                click_report["retry_same_backend"] = True
                click_report["retry_hardened_timing"] = True
                click_report["retry_click_fraction"] = {
                    "x": float(retry_fraction[0]),
                    "y": float(retry_fraction[1]),
                }
            else:
                click_report = desktop.click_action(action)
        except RuntimeError:
            info["confirmation_error"] = "reclick_failed"
            return None
        has_report_target = all(key in click_report for key in ("row", "col", "kind"))
        if has_report_target and not click_report_matches_action(click_report, action):
            info["click_target_mismatch"] = {
                "requested": action_to_dict(action),
                "reported": click_report,
            }
            info["confirmation_error"] = "reclick_target_mismatch"
            return None
        info["reclicks"] += 1
        info["clicks"].append(click_report)
        observed = observe_after_delay(
            max(read_delay, RETRY_READ_DELAY_SECONDS) if use_hardened_retry else read_delay
        )
        if observed is None:
            info["confirmation_error"] = "readback_failed"
            return None
        if info["confirmed"]:
            return observed[0], observed[1], info
        # The desktop game can finish painting a newly opened tile after the
        # first post-click capture. Give the UI one extra passive read before
        # treating a safe click as a failed hit and moving on.
        observed = observe_after_delay(max(read_delay, 0.12))
        if observed is None:
            info["confirmation_error"] = "readback_failed"
            return None
        info["passive_reads"] += 1
        if info["confirmed"]:
            return observed[0], observed[1], info
    if best_raw_board is None or best_board is None:
        info["confirmation_error"] = "no_board_read"
        return None
    update_confirm_read_info(info, best_board)
    if not info["confirmed"]:
        info["confirmation_error"] = "target_not_revealed_after_retries"
    return best_raw_board, best_board, info


def summary_from_board(
    board: ScreenBoard,
    actions: list[dict[str, Any]],
    elapsed: float,
    terminal_dialog: str | None = None,
) -> dict[str, Any]:
    dialog_lost = terminal_dialog is not None and is_lost_dialog_title(terminal_dialog)
    dialog_won = terminal_dialog is not None and is_won_dialog_title(terminal_dialog)
    dialog_done = terminal_dialog is not None and is_terminal_dialog_title(terminal_dialog)
    audited_opens = [
        action
        for action in actions
        if action.get("action", {}).get("kind") == "open"
        and action.get("solver_audit", {}).get("available")
        and action.get("solver_audit", {}).get("target")
    ]
    basic_audited_opens = [
        action
        for action in actions
        if action.get("action", {}).get("kind") == "open"
        and action.get("basic_audit", {}).get("available")
        and action.get("basic_audit", {}).get("target")
    ]
    basic_safety_records = [action.get("basic_safety_filter") for action in actions if action.get("basic_safety_filter")]
    safety_records = [action.get("solver_safety_filter") for action in actions if action.get("solver_safety_filter")]
    click_records = [action.get("click") for action in actions if action.get("click")]
    confirm_records = [action.get("open_confirm") for action in actions if action.get("open_confirm")]
    physical_open_actions = [
        action
        for action in actions
        if action.get("click") and action.get("action", {}).get("kind") == ActionType.OPEN.value
    ]
    open_target_repeats = 0
    seen_open_targets: set[tuple[int, int]] = set()
    for action in physical_open_actions:
        target = action.get("action", {})
        cell = (int(target.get("row", -1)), int(target.get("col", -1)))
        if cell in seen_open_targets:
            open_target_repeats += 1
        else:
            seen_open_targets.add(cell)
    click_elapsed = [float(click.get("elapsed_seconds", 0.0)) for click in click_records if click]
    read_elapsed = [float(action.get("after_read_elapsed", 0.0)) for action in actions if action.get("after_read_elapsed") is not None]
    confirm_elapsed = [float(action.get("open_confirm_elapsed", 0.0)) for action in actions if action.get("open_confirm_elapsed") is not None]
    quick_number_records = [
        action.get("quick_number_read", {})
        for action in actions
        if action.get("quick_number_read")
    ]
    open_revealed_delta = [int(action.get("revealed_delta", 0)) for action in physical_open_actions]
    open_effects = [action.get("open_effect", {}) for action in physical_open_actions if action.get("open_effect")]
    first_open_action = physical_open_actions[0] if physical_open_actions else None
    read_recovery_actions = [
        action
        for action in actions
        if int(action.get("before_read_recoveries", 0) or 0) > 0
        or int(action.get("after_read_recoveries", 0) or 0) > 0
    ]
    missed_offsets = [
        effect.get("nearest_new_revealed", {})
        for effect in open_effects
        if effect.get("target_missed_with_progress") and effect.get("nearest_new_revealed")
    ]
    persistent_counts = [int(action.get("persistent_revealed_cells", 0)) for action in actions]
    confirmed_open_counts = [int(action.get("confirmed_open_cells", 0)) for action in actions]
    cleared_confirmed_flags = [int(action.get("cleared_confirmed_virtual_flags", 0)) for action in actions]
    cleared_confirmed_flags += [int(action.get("cleared_confirmed_virtual_flags_after", 0)) for action in actions]
    cleared_confirmed_blocked = [int(action.get("cleared_confirmed_blocked_opens", 0)) for action in actions]
    cleared_confirmed_blocked += [int(action.get("cleared_confirmed_blocked_opens_after", 0)) for action in actions]
    action_count = len(actions)
    return {
        "won": bool(board.won or dialog_won) and not dialog_lost,
        "lost": bool(board.lost or dialog_lost),
        "done": bool(board.done or dialog_done),
        "agent_steps": action_count,
        "no_progress_actions": sum(1 for action in actions if action.get("no_progress")),
        "reclicks": sum(int(action.get("reclicks", 0)) for action in actions),
        "physical_open_actions": len(click_records),
        "click_issued_actions": sum(1 for click in click_records if click.get("issued")),
        "click_unready_actions": sum(1 for click in click_records if not click.get("cursor_ready")),
        "sendinput_fallback_actions": sum(1 for click in click_records if click.get("sendinput_fallback")),
        "sendinput_error_actions": sum(1 for click in click_records if click.get("sendinput_error")),
        "avg_click_seconds": (sum(click_elapsed) / len(click_elapsed)) if click_elapsed else 0.0,
        "avg_after_read_seconds": (sum(read_elapsed) / len(read_elapsed)) if read_elapsed else 0.0,
        "avg_confirm_seconds": (sum(confirm_elapsed) / len(confirm_elapsed)) if confirm_elapsed else 0.0,
        "quick_number_read_actions": sum(1 for record in quick_number_records if record.get("accepted")),
        "quick_number_read_fallbacks": sum(1 for record in quick_number_records if not record.get("accepted")),
        "open_confirm_reads": sum(int(confirm.get("read_attempts", 0)) for confirm in confirm_records),
        "open_confirm_passive_reads": sum(int(confirm.get("passive_reads", 0)) for confirm in confirm_records),
        "open_confirm_reclicks": sum(int(confirm.get("reclicks", 0)) for confirm in confirm_records),
        "unconfirmed_open_actions": sum(1 for confirm in confirm_records if not confirm.get("confirmed")),
        "open_target_repeat_actions": open_target_repeats,
        "max_confirmed_open_cells": max(confirmed_open_counts) if confirmed_open_counts else 0,
        "confirmed_open_remembered_cells": sum(int(action.get("confirmed_open_added_after", 0)) for action in actions)
        + sum(int(action.get("confirmed_open_added_before", 0)) for action in actions),
        "cleared_confirmed_virtual_flags": sum(cleared_confirmed_flags),
        "cleared_confirmed_blocked_opens": sum(cleared_confirmed_blocked),
        "max_persistent_revealed_cells": max(persistent_counts) if persistent_counts else 0,
        "remembered_revealed_cells": sum(int(action.get("remembered_revealed_after", 0)) for action in actions)
        + sum(int(action.get("remembered_revealed_before", 0)) for action in actions),
        "open_target_miss_with_progress": sum(
            1
            for action in physical_open_actions
            if int(action.get("revealed_delta", 0)) > 0 and not action.get("target_revealed_after_open")
        ),
        "open_target_miss_nearest_avg_manhattan": (
            sum(int(offset.get("manhattan", 0)) for offset in missed_offsets) / len(missed_offsets)
        )
        if missed_offsets
        else 0.0,
        "open_target_miss_nearest_max_manhattan": (
            max(int(offset.get("manhattan", 0)) for offset in missed_offsets) if missed_offsets else 0
        ),
        "open_zero_progress_actions": sum(
            1
            for action in physical_open_actions
            if (
                not action.get("terminal_dialog")
                and action.get("after_target") is not None
                and int(action.get("revealed_delta", 0)) <= 0
                and not action.get("after_target", {}).get("mine_like")
            )
        ),
        "avg_revealed_delta_per_open": (sum(open_revealed_delta) / len(open_revealed_delta)) if open_revealed_delta else 0.0,
        "revealed_safe_cells": int(board.revealed.sum()),
        "flags": int(board.flagged.sum()),
        "read_repairs": int(board.read_repairs),
        "read_restores": int(board.read_restores),
        "read_recoveries": int(board.read_recoveries),
        "read_recovery_actions": len(read_recovery_actions),
        "open_confirm_read_recoveries": sum(int(confirm.get("final_read_recoveries", 0)) for confirm in confirm_records),
        "blocked_repeat_open_actions": sum(1 for action in actions if action.get("blocked_repeat_open")),
        "terminal_dialog": terminal_dialog,
        "elapsed_seconds": elapsed,
        "seconds_per_action": (elapsed / action_count) if action_count else None,
        "actions_per_second": (action_count / elapsed) if elapsed > 0 else None,
        "first_open_action": None
        if first_open_action is None
        else {
            "action": first_open_action.get("action"),
            "revealed_delta": int(first_open_action.get("revealed_delta", 0)),
            "target_revealed_after_open": first_open_action.get("target_revealed_after_open"),
            "after_read_repairs": int(first_open_action.get("after_read_repairs", 0)),
            "after_read_restores": int(first_open_action.get("after_read_restores", 0)),
            "after_grid": first_open_action.get("after_grid"),
            "open_effect": first_open_action.get("open_effect"),
        },
        "solver_audited_opens": len(audited_opens),
        "basic_audited_opens": len(basic_audited_opens),
        "basic_known_mine_opens": sum(1 for action in basic_audited_opens if action["basic_audit"]["target"].get("known_mine")),
        "basic_known_safe_opens": sum(1 for action in basic_audited_opens if action["basic_audit"]["target"].get("known_safe")),
        "basic_conflict_opens": sum(1 for action in basic_audited_opens if action["basic_audit"]["target"].get("conflict")),
        "basic_safety_filter_actions": sum(1 for record in basic_safety_records if record.get("applied")),
        "basic_safety_blocked_opens": sum(int(record.get("blocked_known_mine_opens", 0)) for record in basic_safety_records),
        "solver_known_mine_opens": sum(1 for action in audited_opens if action["solver_audit"]["target"].get("known_mine")),
        "solver_known_safe_opens": sum(1 for action in audited_opens if action["solver_audit"]["target"].get("known_safe")),
        "solver_forced_available_opens": sum(1 for action in audited_opens if action["solver_audit"].get("has_forced_moves")),
        "solver_high_risk_opens": sum(1 for action in audited_opens if float(action["solver_audit"]["target"].get("risk", 0.0)) >= 0.5),
        "solver_safety_filter_actions": sum(1 for record in safety_records if record.get("applied")),
        "solver_safety_blocked_opens": sum(int(record.get("blocked_known_mine_opens", 0)) for record in safety_records),
        "solver_assist_actions": sum(
            1 for action in actions if action.get("solver_assist", {}).get("applied")
        ),
        "solver_assist_safe_opens": sum(
            1
            for action in actions
            if action.get("solver_assist", {}).get("decision") == "forced_safe_open"
        ),
        "solver_assist_virtual_flag_batches": sum(
            1
            for action in actions
            if action.get("solver_assist", {}).get("decision") == "virtual_flag_batch"
        ),
        "solver_assist_virtual_flags": sum(
            int(action.get("virtual_flag_batch_count", 0)) for action in actions
        ),
        "solver_assist_rl_fallbacks": sum(
            1
            for action in actions
            if action.get("solver_assist", {}).get("decision") == "rl_fallback"
        ),
    }


def _resolve_trace_path(raw_path: str | Path, base: Path) -> Path | None:
    path = Path(raw_path)
    candidates = [path] if path.is_absolute() else [Path.cwd() / path, ROOT / path, base / path]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def _manifest_trace_paths(manifest_path: Path) -> list[Path]:
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = []
    for result in manifest.get("results", []):
        resolved = _resolve_trace_path(result.get("path", ""), manifest_path.parent)
        if resolved is not None:
            paths.append(resolved)
    return paths


def game_trace_paths(input_path: Path) -> list[Path]:
    if input_path.is_dir():
        manifest_path = input_path / "manifest.json"
        if manifest_path.exists():
            paths = _manifest_trace_paths(manifest_path)
            if paths:
                return paths
        return sorted(input_path.glob("game_*.json"))
    if input_path.name == "manifest.json":
        paths = _manifest_trace_paths(input_path)
        if paths:
            return paths
    return [input_path]


def analyze_game_trace(path: Path) -> dict[str, Any]:
    trace = json.loads(path.read_text(encoding="utf-8"))
    actions = trace.get("actions", [])
    summary = trace.get("summary", {})
    open_actions = [action for action in actions if action.get("action", {}).get("kind") == ActionType.OPEN.value]
    physical_open_actions = [action for action in open_actions if action.get("click")]
    click_records = [action.get("click") for action in open_actions if action.get("click")]
    click_methods = Counter(str(click.get("method", "unknown")) for click in click_records)
    click_mismatches = [
        action
        for action in open_actions
        if action.get("click")
        and (
            int(action["click"].get("row", -1)) != int(action.get("action", {}).get("row", -2))
            or int(action["click"].get("col", -1)) != int(action.get("action", {}).get("col", -2))
            or action["click"].get("kind") != action.get("action", {}).get("kind")
        )
    ]
    terminal_actions = [action for action in actions if action.get("terminal_dialog")]
    terminal_action = terminal_actions[-1] if terminal_actions else None
    terminal_target = {}
    if terminal_action is not None:
        terminal_target = terminal_action.get("solver_audit", {}).get("target", {}) or {}
    terminal_basic_target = {}
    if terminal_action is not None:
        terminal_basic_target = terminal_action.get("basic_audit", {}).get("target", {}) or {}
    sendinput_fallback_actions = sum(1 for click in click_records if click.get("sendinput_fallback"))
    sendinput_error_actions = sum(1 for click in click_records if click.get("sendinput_error"))
    basic_safety_records = [action.get("basic_safety_filter") for action in actions if action.get("basic_safety_filter")]
    first_open = summary.get("first_open_action") or {}
    first_open_delta = int(first_open.get("revealed_delta", 0) or 0) if first_open else 0
    first_open_dirty = bool(
        first_open
        and (
            int(first_open.get("after_read_repairs", 0) or 0) > 0
            or int(first_open.get("after_read_restores", 0) or 0) > 0
        )
    )

    read_repair_actions = sum(
        1
        for action in actions
        if int(action.get("before_read_repairs", 0) or 0) > 0
        or int(action.get("after_read_repairs", 0) or 0) > 0
    )
    read_restore_actions = sum(
        1
        for action in actions
        if int(action.get("before_read_restores", 0) or 0) > 0
        or int(action.get("after_read_restores", 0) or 0) > 0
    )
    read_recovery_actions = sum(
        1
        for action in actions
        if int(action.get("before_read_recoveries", 0) or 0) > 0
        or int(action.get("after_read_recoveries", 0) or 0) > 0
    )
    repeat_open_targets = 0
    seen_open_targets: set[tuple[int, int]] = set()
    for action in physical_open_actions:
        target = action.get("action", {})
        cell = (int(target.get("row", -1)), int(target.get("col", -1)))
        if cell in seen_open_targets:
            repeat_open_targets += 1
        else:
            seen_open_targets.add(cell)
    solver_audited_opens = [action for action in open_actions if action.get("solver_audit", {}).get("available")]
    basic_audited_opens = [action for action in open_actions if action.get("basic_audit", {}).get("available")]
    basic_known_mine_opens = [
        action
        for action in basic_audited_opens
        if action.get("basic_audit", {}).get("target", {}).get("known_mine")
    ]
    basic_known_safe_opens = [
        action
        for action in basic_audited_opens
        if action.get("basic_audit", {}).get("target", {}).get("known_safe")
    ]
    known_mine_opens = [
        action
        for action in solver_audited_opens
        if action.get("solver_audit", {}).get("target", {}).get("known_mine")
    ]
    high_risk_opens = [
        action
        for action in solver_audited_opens
        if float(action.get("solver_audit", {}).get("target", {}).get("risk", 0.0) or 0.0) >= 0.5
    ]

    signals: list[str] = []
    if click_mismatches:
        signals.append("click_log_mismatch")
    if sendinput_fallback_actions:
        signals.append("sendinput_fallback")
    basic_safety_blocked_opens = sum(int(record.get("blocked_known_mine_opens", 0)) for record in basic_safety_records)
    basic_safety_filter_actions = sum(1 for record in basic_safety_records if record.get("applied"))
    if basic_safety_blocked_opens:
        signals.append("basic_safety_filter")
    if int(summary.get("open_zero_progress_actions", 0) or 0) > 0:
        signals.append("open_zero_progress")
    if int(summary.get("open_target_miss_with_progress", 0) or 0) > 0:
        signals.append("open_target_miss_with_progress")
    if read_repair_actions:
        signals.append("read_repairs")
    if read_restore_actions:
        signals.append("read_restores")
    if read_recovery_actions:
        signals.append("read_recoveries")
    if first_open and first_open_delta < 9:
        signals.append("small_first_open")
    if first_open_dirty:
        signals.append("dirty_first_open_read")
    if repeat_open_targets:
        signals.append("repeat_open_targets")
    if terminal_basic_target.get("known_mine"):
        signals.append("terminal_basic_known_mine")
    if terminal_target.get("known_mine"):
        signals.append("terminal_solver_known_mine")
    elif terminal_target and float(terminal_target.get("risk", 0.0) or 0.0) >= 0.5:
        signals.append("terminal_high_risk")
    if known_mine_opens:
        signals.append("solver_known_mine_opens")
    if basic_known_mine_opens:
        signals.append("basic_known_mine_opens")
    if not signals and summary.get("lost"):
        signals.append("unclassified_loss")

    terminal_action_summary = None
    if terminal_action is not None:
        action = terminal_action.get("action", {})
        terminal_action_summary = {
            "step": terminal_action.get("step"),
            "action": action,
            "terminal_dialog": terminal_action.get("terminal_dialog"),
            "terminal_detected_at": terminal_action.get("terminal_detected_at"),
            "before_target": terminal_action.get("before_target"),
            "after_target": terminal_action.get("after_target"),
            "revealed_delta": terminal_action.get("revealed_delta"),
            "target_revealed_after_open": terminal_action.get("target_revealed_after_open"),
            "open_effect": terminal_action.get("open_effect"),
            "basic_target": terminal_basic_target or None,
            "solver_target": terminal_target or None,
        }

    return {
        "path": str(path),
        "won": bool(summary.get("won")),
        "lost": bool(summary.get("lost")),
        "done": bool(summary.get("done")),
        "agent_steps": int(summary.get("agent_steps", len(actions)) or 0),
        "elapsed_seconds": float(summary.get("elapsed_seconds", 0.0) or 0.0),
        "revealed_safe_cells": int(summary.get("revealed_safe_cells", 0) or 0),
        "first_open_action": first_open or None,
        "first_open_revealed_delta": first_open_delta,
        "first_open_dirty_read": first_open_dirty,
        "no_progress_actions": int(summary.get("no_progress_actions", 0) or 0),
        "open_zero_progress_actions": int(summary.get("open_zero_progress_actions", 0) or 0),
        "open_target_miss_with_progress": int(summary.get("open_target_miss_with_progress", 0) or 0),
        "open_target_miss_nearest_avg_manhattan": float(
            summary.get("open_target_miss_nearest_avg_manhattan", 0.0) or 0.0
        ),
        "open_target_miss_nearest_max_manhattan": int(
            summary.get("open_target_miss_nearest_max_manhattan", 0) or 0
        ),
        "unconfirmed_open_actions": int(summary.get("unconfirmed_open_actions", 0) or 0),
        "open_confirm_passive_reads": int(summary.get("open_confirm_passive_reads", 0) or 0),
        "max_confirmed_open_cells": int(summary.get("max_confirmed_open_cells", 0) or 0),
        "confirmed_open_remembered_cells": int(summary.get("confirmed_open_remembered_cells", 0) or 0),
        "cleared_confirmed_virtual_flags": int(summary.get("cleared_confirmed_virtual_flags", 0) or 0),
        "cleared_confirmed_blocked_opens": int(summary.get("cleared_confirmed_blocked_opens", 0) or 0),
        "max_persistent_revealed_cells": int(summary.get("max_persistent_revealed_cells", 0) or 0),
        "remembered_revealed_cells": int(summary.get("remembered_revealed_cells", 0) or 0),
        "read_repair_actions": read_repair_actions,
        "read_restore_actions": read_restore_actions,
        "read_recovery_actions": int(summary.get("read_recovery_actions", read_recovery_actions) or 0),
        "open_confirm_read_recoveries": int(summary.get("open_confirm_read_recoveries", 0) or 0),
        "repeat_open_target_actions": repeat_open_targets,
        "click_mismatch_actions": len(click_mismatches),
        "sendinput_fallback_actions": int(summary.get("sendinput_fallback_actions", sendinput_fallback_actions) or 0),
        "sendinput_error_actions": int(summary.get("sendinput_error_actions", sendinput_error_actions) or 0),
        "click_methods": dict(click_methods),
        "basic_audited_opens": len(basic_audited_opens),
        "basic_known_mine_opens": len(basic_known_mine_opens),
        "basic_known_safe_opens": len(basic_known_safe_opens),
        "basic_safety_filter_actions": int(summary.get("basic_safety_filter_actions", basic_safety_filter_actions) or 0),
        "basic_safety_blocked_opens": int(summary.get("basic_safety_blocked_opens", basic_safety_blocked_opens) or 0),
        "solver_audited_opens": len(solver_audited_opens),
        "solver_known_mine_opens": len(known_mine_opens),
        "solver_high_risk_opens": len(high_risk_opens),
        "terminal_action": terminal_action_summary,
        "signals": signals,
    }


def analyze_log_diagnosis(
    games: list[dict[str, Any]],
    signal_counts: Counter[str],
    target_win_rate: float = 0.4,
    target_avg_seconds: float = 60.0,
) -> dict[str, Any]:
    if not games:
        return {
            "primary_issue": "no_game_logs",
            "next_focus": "collect_logs",
            "target_win_rate": target_win_rate,
            "target_avg_seconds": target_avg_seconds,
            "target_passed": False,
            "recommendations": ["Run benchmark first; no game_*.json traces were found."],
        }

    total = len(games)
    wins = sum(1 for game in games if game["won"])
    win_rate = wins / total if total else 0.0
    elapsed_values = [float(game["elapsed_seconds"]) for game in games if float(game["elapsed_seconds"]) > 0]
    avg_elapsed = sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0

    click_signals = signal_counts.get("open_zero_progress", 0) + signal_counts.get("open_target_miss_with_progress", 0)
    read_signals = signal_counts.get("read_repairs", 0) + signal_counts.get("read_restores", 0)
    read_recovery_signals = signal_counts.get("read_recoveries", 0)
    repeat_signals = signal_counts.get("repeat_open_targets", 0)
    solver_mine_signals = signal_counts.get("terminal_solver_known_mine", 0) + signal_counts.get("solver_known_mine_opens", 0)
    basic_mine_signals = signal_counts.get("terminal_basic_known_mine", 0) + signal_counts.get("basic_known_mine_opens", 0)
    basic_safety_filter_signals = signal_counts.get("basic_safety_filter", 0)
    click_log_mismatches = signal_counts.get("click_log_mismatch", 0)
    sendinput_fallback_signals = signal_counts.get("sendinput_fallback", 0)
    small_first_open_signals = signal_counts.get("small_first_open", 0)
    dirty_first_open_signals = signal_counts.get("dirty_first_open_read", 0)

    primary_issue = "none"
    next_focus = "continue_benchmark"
    recommendations: list[str] = []

    if click_signals:
        primary_issue = "click_execution"
        next_focus = "compare_click_method"
        recommendations.append("Compare with --click-method mouse_event; OPEN actions are not consistently hitting their target.")
        recommendations.append("If mouse_event helps, keep that click method and avoid changing model settings first.")
    elif read_signals:
        primary_issue = "board_read_instability"
        next_focus = "compare_read_mode"
        recommendations.append("Run a short --read-mode accurate benchmark; board reads are being repaired/restored.")
        recommendations.append("If accurate mode fixes the deaths, the current fast read path still needs more work.")
    elif read_recovery_signals and win_rate < target_win_rate:
        primary_issue = "board_read_recovery_dependency"
        next_focus = "compare_read_mode"
        recommendations.append("Fast reads are being recovered by accurate reads; compare with --read-mode accurate.")
        recommendations.append("If accurate mode wins more but is too slow, tune fast read classification rather than the model.")
    elif small_first_open_signals:
        primary_issue = "small_first_open"
        next_focus = "inspect_first_open"
        recommendations.append("First open is revealing very few cells; inspect whether the desktop game's first-click rule differs from training.")
        recommendations.append("If this persists without read signals, compare model checkpoints or train for this first-open distribution.")
    elif dirty_first_open_signals:
        primary_issue = "dirty_first_open_read"
        next_focus = "compare_read_mode"
        recommendations.append("The first board read after opening is dirty; compare --read-mode accurate before changing the model.")
        recommendations.append("If accurate reads clean this up, tune fast classification around the first opened region.")
    elif repeat_signals:
        primary_issue = "repeat_open_targets"
        next_focus = "persistent_reveals"
        recommendations.append("Keep persistent reveals enabled; repeated physical OPEN targets are still present.")
        recommendations.append("If this came from an old batch, rerun with current code before tuning the model.")
    elif solver_mine_signals:
        primary_issue = "visible_logic_or_state"
        next_focus = "audit_terminal_board"
        recommendations.append("Inspect the terminal board; solver audit marked at least one chosen target as a known mine.")
        recommendations.append("If read signals are absent, this points more toward policy quality than click timing.")
    elif basic_mine_signals:
        primary_issue = "basic_logic_or_state"
        next_focus = "audit_terminal_board"
        recommendations.append("Inspect the terminal action; basic local rules marked at least one chosen target as a known mine.")
        recommendations.append("If read signals are absent, this points toward policy quality or virtual flag state.")
    elif click_log_mismatches:
        primary_issue = "stale_or_dirty_click_logs"
        next_focus = "rerun_current_code"
        recommendations.append("Rerun with current code; old logs contained mismatched click/action records and are unreliable.")
    elif sendinput_fallback_signals:
        primary_issue = "click_input_backend"
        next_focus = "force_mouse_event"
        recommendations.append("SendInput is falling back; force --click-method mouse_event to avoid failed click attempts.")
        recommendations.append("If click signals appear after forcing mouse_event, compare with --click-method sendinput separately.")
    elif win_rate < target_win_rate:
        primary_issue = "low_win_rate_without_clear_execution_signal"
        next_focus = "increase_sample_or_model"
        recommendations.append("Run more games with --audit-solver; no single execution-layer signal dominates these logs.")
    elif avg_elapsed > target_avg_seconds:
        primary_issue = "speed"
        next_focus = "speed_profile"
        recommendations.append("Win rate is okay, but average time is too high; reduce recording and compare fast/turbo timings.")
    else:
        primary_issue = "target_met_in_logs"
        next_focus = "longer_validation"
        recommendations.append("This batch meets the target; validate with a larger run before moving to streak mode.")

    if not recommendations:
        recommendations.append("No clear execution-layer issue is dominating these logs.")

    return {
        "primary_issue": primary_issue,
        "next_focus": next_focus,
        "target_win_rate": target_win_rate,
        "target_avg_seconds": target_avg_seconds,
        "target_passed": win_rate >= target_win_rate and avg_elapsed <= target_avg_seconds,
        "win_rate": win_rate,
        "avg_elapsed_seconds": avg_elapsed,
        "click_signals": int(click_signals),
        "read_signals": int(read_signals),
        "read_recovery_signals": int(read_recovery_signals),
        "repeat_signals": int(repeat_signals),
        "solver_mine_signals": int(solver_mine_signals),
        "basic_mine_signals": int(basic_mine_signals),
        "basic_safety_filter_signals": int(basic_safety_filter_signals),
        "click_log_mismatches": int(click_log_mismatches),
        "sendinput_fallback_signals": int(sendinput_fallback_signals),
        "small_first_open_signals": int(small_first_open_signals),
        "dirty_first_open_signals": int(dirty_first_open_signals),
        "recommendations": recommendations,
    }


def build_analyze_log_report(input_path: Path, target_win_rate: float = 0.4, target_avg_seconds: float = 60.0) -> dict[str, Any]:
    paths = game_trace_paths(input_path)
    games = [analyze_game_trace(path) for path in paths]
    wins = sum(1 for game in games if game["won"])
    losses = sum(1 for game in games if game["lost"])
    elapsed_values = [float(game["elapsed_seconds"]) for game in games if float(game["elapsed_seconds"]) > 0]
    signal_counts = Counter(signal for game in games for signal in game["signals"])
    diagnosis = analyze_log_diagnosis(
        games,
        signal_counts,
        target_win_rate=target_win_rate,
        target_avg_seconds=target_avg_seconds,
    )
    return {
        "input": str(input_path),
        "games": len(games),
        "wins": wins,
        "losses": losses,
        "win_rate": wins / len(games) if games else 0.0,
        "avg_elapsed_seconds": sum(elapsed_values) / len(elapsed_values) if elapsed_values else 0.0,
        "avg_revealed_safe_cells": (
            sum(int(game["revealed_safe_cells"]) for game in games) / len(games)
            if games
            else 0.0
        ),
        "signal_counts": dict(signal_counts),
        "diagnosis": diagnosis,
        "games_detail": games,
    }


def run_analyze_log(args: argparse.Namespace) -> None:
    aggregate = build_analyze_log_report(
        args.input,
        target_win_rate=float(getattr(args, "target_win_rate", 0.4)),
        target_avg_seconds=float(getattr(args, "target_avg_seconds", 60.0)),
    )
    print(json.dumps(aggregate, indent=2, ensure_ascii=False))


def action_to_dict(action: Action) -> dict[str, Any]:
    return {"kind": action.kind.value, "row": int(action.row), "col": int(action.col)}


def is_lost_dialog_title(title: str) -> bool:
    normalized = title.lower()
    return any(token in normalized for token in ("失败", "失敗", "lost", "lose", "fail"))


def is_won_dialog_title(title: str) -> bool:
    normalized = title.lower()
    return any(token in normalized for token in ("胜利", "勝利", "获胜", "獲勝", "win", "won"))


def is_terminal_dialog_title(title: str) -> bool:
    return is_lost_dialog_title(title) or is_won_dialog_title(title)


def is_statistics_dialog_title(title: str) -> bool:
    normalized = title.lower()
    return "\u7edf\u8ba1" in title or "statistics" in normalized or "stats" in normalized


def board_is_fresh(board: ScreenBoard) -> bool:
    return (
        not board.done
        and int(board.revealed.sum()) == 0
        and int(board.flagged.sum()) == 0
        and not bool(board.mine_like.any())
    )


def wait_for_fresh_board(desktop: WindowsMinesweeper, timeout: float = 1.0, reads: int = 2) -> bool:
    deadline = time.time() + timeout
    fresh_streak = 0
    reads = max(1, int(reads))
    while time.time() < deadline:
        if desktop.dialog_is_open():
            if is_statistics_dialog_title(desktop.dialog_title() or ""):
                desktop.choose_dialog_option(option="close", timeout=1.0, attempts=3)
            fresh_streak = 0
            time.sleep(0.05)
            continue
        try:
            board = desktop.read_board(keep_screenshot=False)
        except RuntimeError:
            fresh_streak = 0
            time.sleep(0.05)
            continue
        if board_is_fresh(board):
            fresh_streak += 1
            if fresh_streak >= reads:
                return True
        else:
            fresh_streak = 0
        time.sleep(0.05)
    return False


def prepare_game_start(desktop: WindowsMinesweeper, start_mode: str) -> bool:
    def confirm_fresh_board(retries: int = 3) -> bool:
        for _ in range(max(1, retries)):
            if wait_for_fresh_board(desktop, timeout=1.5, reads=2):
                return True
            time.sleep(0.1)
        return False

    if desktop.dialog_is_open() and is_statistics_dialog_title(desktop.dialog_title() or ""):
        desktop.choose_dialog_option(option="close", timeout=1.0, attempts=3)

    if start_mode == "current":
        if desktop.dialog_is_open():
            dialog = desktop.find_new_game_dialog()
            if dialog is not None:
                title = desktop.dialog_title() or ""
                option = "new" if is_terminal_dialog_title(title) else "continue"
                desktop.choose_dialog_option(option=option)
                if option == "new":
                    if not confirm_fresh_board():
                        raise RuntimeError("could not confirm a fresh board after closing the dialog")
                return option == "new"
        try:
            board = desktop.read_board(keep_screenshot=False)
        except RuntimeError:
            return False
        if board.done:
            desktop.new_game(option="new")
            if not confirm_fresh_board():
                raise RuntimeError("could not confirm a fresh board after starting a new game")
            return True
        return False

    if start_mode in {"new", "restart"}:
        desktop.new_game(option=start_mode)
        if not confirm_fresh_board():
            raise RuntimeError("could not confirm a fresh board after starting a new game")
        return True

    if start_mode != "auto":
        raise ValueError("start_mode must be one of: auto, current, new, restart")

    if desktop.dialog_is_open():
        dialog = desktop.find_new_game_dialog()
        if dialog is not None:
            title = desktop.dialog_title() or ""
            desktop.choose_dialog_option(option="new")
            if not confirm_fresh_board():
                raise RuntimeError("could not confirm a fresh board after closing the dialog")
            return True
        return False

    try:
        board = desktop.read_board(keep_screenshot=False)
    except RuntimeError:
        dialog = desktop.find_new_game_dialog()
        if dialog is not None:
            title = desktop.dialog_title() or ""
            desktop.choose_dialog_option(option="new")
            if not confirm_fresh_board():
                raise RuntimeError("could not confirm a fresh board after closing the dialog")
            return True
        return False

    if int(board.revealed.sum()) == 0 and int(board.flagged.sum()) == 0:
        return False
    desktop.new_game(option="new")
    if not confirm_fresh_board():
        raise RuntimeError("could not confirm a fresh board after starting a new game")
    return True


def run_streak(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    clear_stop_request()
    args.clear_stop_on_start = False
    streak = 0
    results: list[dict[str, Any]] = []
    timing = live_timing_settings(args)
    desktop = WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_method=args.click_method,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
        post_click_settle=float(timing["post_click_settle"]),
    )
    preflight_read_compare = None
    if bool(getattr(args, "preflight_read_compare", False)):
        preflight_read_compare = read_compare_preflight(
            desktop,
            timing,
            reads=int(getattr(args, "preflight_reads", 3)),
            interval=float(getattr(args, "preflight_interval", 0.02)),
        )
    trainer = load_configured_trainer(args)
    manifest: dict[str, Any] = {}
    with StopHotkey(enabled=not args.no_hotkey) as hotkey:
        for game_index in range(1, args.max_games + 1):
            hotkey.poll()
            if stop_requested():
                manifest = streak_manifest(
                    args,
                    output_dir,
                    results,
                    streak,
                    game_index - 1,
                    "stopped",
                    timing,
                    desktop.effective_click_method(),
                )
                write_manifest(output_dir, manifest)
                print(json.dumps({k: manifest[k] for k in ("status", "streak", "games_played")}, separators=(",", ":")), flush=True)
                return
            game_start_mode = "new" if game_index == 1 and args.start_mode == "auto" else (args.start_mode if game_index == 1 else "new")
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
            manifest = streak_manifest(
                args,
                output_dir,
                results,
                streak,
                game_index,
                status,
                timing,
                desktop.effective_click_method(),
            )
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

    timing = live_timing_settings(args)
    desktop = WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_method=args.click_method,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
        post_click_settle=float(timing["post_click_settle"]),
    )
    trainer = load_configured_trainer(args)
    results: list[dict[str, Any]] = []
    started_at = time.time()
    first_start_mode = "new" if args.start_mode == "auto" else args.start_mode
    preflight_read_compare = None
    if bool(getattr(args, "preflight_read_compare", False)):
        preflight_read_compare = read_compare_preflight(
            desktop,
            timing,
            reads=int(getattr(args, "preflight_reads", 3)),
            interval=float(getattr(args, "preflight_interval", 0.02)),
        )
    preflight_read_compare = None
    if bool(getattr(args, "preflight_read_compare", False)):
        preflight_read_compare = read_compare_preflight(
            desktop,
            timing,
            reads=int(getattr(args, "preflight_reads", 3)),
            interval=float(getattr(args, "preflight_interval", 0.02)),
        )

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
        "click_method": args.click_method,
        "resolved_click_method": desktop.effective_click_method(),
        "persistent_reveals": not bool(getattr(args, "no_persistent_reveals", False)),
        "audit_basic": bool(getattr(args, "audit_basic", False)),
        "basic_safety_filter": getattr(args, "basic_safety_filter", "none"),
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
        "preflight_read_compare": preflight_read_compare,
        "results": results,
        "elapsed_seconds": time.time() - started_at,
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    analysis = build_analyze_log_report(
        output_dir,
        target_win_rate=float(args.target_win_rate),
        target_avg_seconds=float(args.target_avg_seconds),
    )
    (output_dir / "analysis.json").write_text(json.dumps(analysis, indent=2, ensure_ascii=False), encoding="utf-8")
    manifest["analysis"] = {
        "path": str(output_dir / "analysis.json"),
        "primary_issue": analysis["diagnosis"]["primary_issue"],
        "next_focus": analysis["diagnosis"]["next_focus"],
        "signal_counts": analysis["signal_counts"],
    }
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


def run_read_benchmark(args: argparse.Namespace) -> None:
    output_dir = args.output_dir or default_output_dir()
    output_dir.mkdir(parents=True, exist_ok=True)
    timing = live_timing_settings(args)
    desktop = WindowsMinesweeper(
        capture_delay=float(timing["capture_delay"]),
        capture_backend=args.capture_backend,
        read_mode=args.read_mode,
        click_method=args.click_method,
        click_pause=float(timing["click_pause"]),
        cursor_settle=float(timing["cursor_settle"]),
        post_click_settle=float(timing["post_click_settle"]),
    )
    reads = max(1, int(args.reads))
    interval = max(0.0, float(args.interval))
    settle_reads = int(timing["settle_reads"])
    settle_read_delay = float(timing["settle_read_delay"])
    previous_board: ScreenBoard | None = None
    records: list[dict[str, Any]] = []
    started_at = time.time()

    for index in range(reads):
        keep_screenshot = index == reads - 1
        read_started_at = time.time()
        board = read_stable_board(
            desktop,
            step_count=index,
            keep_screenshot=keep_screenshot,
            reads=settle_reads,
            delay=settle_read_delay,
            previous_board=previous_board,
        )
        elapsed = time.time() - read_started_at
        diff = board_difference_counts(board, previous_board)
        accurate_compare: dict[str, Any] | None = None
        if bool(getattr(args, "compare_accurate", False)):
            accurate_compare = accurate_compare_for_board(desktop, board, previous_board, step_count=index)
        record = {
            "index": index,
            "elapsed_seconds": elapsed,
            "revealed": int(board.revealed.sum()),
            "flags": int(board.flagged.sum()),
            "won": bool(board.won),
            "lost": bool(board.lost),
            "read_repairs": int(board.read_repairs),
            "read_restores": int(board.read_restores),
            "read_recoveries": int(board.read_recoveries),
            "grid": {
                "x0": int(board.grid.x_lines[0]),
                "x1": int(board.grid.x_lines[-1]),
                "y0": int(board.grid.y_lines[0]),
                "y1": int(board.grid.y_lines[-1]),
                "cell_width": float(board.grid.cell_width),
                "cell_height": float(board.grid.cell_height),
                "cell_square_error": abs(float(board.grid.cell_width) - float(board.grid.cell_height)),
            },
            "diff_from_previous": diff,
        }
        if accurate_compare is not None:
            record["accurate_compare"] = accurate_compare
        records.append(record)
        previous_board = board
        if interval > 0 and index < reads - 1:
            time.sleep(interval)

    elapsed_values = [float(record["elapsed_seconds"]) for record in records]
    unstable_reads = sum(
        1
        for record in records[1:]
        if record["diff_from_previous"].get("signature_changed")
    )
    accurate_compare_records = [record["accurate_compare"] for record in records if record.get("accurate_compare", {}).get("available")]
    accurate_compare_summary = summarize_accurate_compare_records(accurate_compare_records)
    final_board = previous_board
    overlay_path = output_dir / "read_benchmark_overlay.png"
    if final_board is not None and final_board.screenshot is not None:
        draw_overlay(final_board, overlay_path)
    manifest = {
        "status": "completed",
        "output_dir": str(output_dir),
        "reads": reads,
        "timing": timing,
        "click_method": args.click_method,
        "persistent_reveals": not bool(getattr(args, "no_persistent_reveals", False)),
        "audit_basic": bool(getattr(args, "audit_basic", False)),
        "basic_safety_filter": getattr(args, "basic_safety_filter", "none"),
        "capture_backend": args.capture_backend,
        "read_mode": args.read_mode,
        "avg_read_seconds": sum(elapsed_values) / len(elapsed_values),
        "max_read_seconds": max(elapsed_values),
        "min_read_seconds": min(elapsed_values),
        "unstable_reads": unstable_reads,
        "stable_after_first": unstable_reads == 0,
        "compare_accurate": bool(getattr(args, "compare_accurate", False)),
        "accurate_compare_summary": accurate_compare_summary,
        "final_board": None
        if final_board is None
        else {
            "revealed": int(final_board.revealed.sum()),
            "flags": int(final_board.flagged.sum()),
            "won": bool(final_board.won),
            "lost": bool(final_board.lost),
            "read_repairs": int(final_board.read_repairs),
            "read_restores": int(final_board.read_restores),
            "read_recoveries": int(final_board.read_recoveries),
            "text": board_to_text(final_board),
        },
        "overlay": str(overlay_path) if final_board is not None and final_board.screenshot is not None else None,
        "records": records,
        "elapsed_seconds": time.time() - started_at,
    }
    write_manifest(output_dir, manifest)
    print(json.dumps(manifest, indent=2))


def streak_manifest(
    args: argparse.Namespace,
    output_dir: Path,
    results: list[dict[str, Any]],
    streak: int,
    games_played: int,
    status: str,
    timing: dict[str, Any] | None = None,
    resolved_click_method: str | None = None,
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
        "audit_basic": bool(getattr(args, "audit_basic", False)),
        "no_progress_reclicks": timing["no_progress_reclicks"],
        "reclick_delay": timing["reclick_delay"],
        "speed_profile": getattr(args, "speed_profile", "safe"),
        "model_flip_ensemble": args.inference_flips,
        "model_ensemble_method": args.inference_ensemble,
        "resolved_click_method": resolved_click_method or args.click_method,
        "solver_audit_enabled": bool(getattr(args, "audit_solver", False)),
        "basic_safety_filter": getattr(args, "basic_safety_filter", "none"),
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


def apply_missing_defaults(args: argparse.Namespace, defaults: dict[str, Any]) -> None:
    for key, value in defaults.items():
        if not hasattr(args, key):
            setattr(args, key, value)


def main() -> None:
    parser = argparse.ArgumentParser(description="Drive the desktop Windows Minesweeper with the RL agent.")
    parser.add_argument("--checkpoint", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--device", type=str, default=argparse.SUPPRESS)
    parser.add_argument("--output-dir", type=Path, default=argparse.SUPPRESS)
    parser.add_argument("--capture-backend", choices=["auto", "pil", "mss", "window"], default=argparse.SUPPRESS)
    parser.add_argument("--read-mode", choices=["fast", "accurate"], default=argparse.SUPPRESS)
    parser.add_argument(
        "--click-method",
        choices=["auto", "sendinput", "sendinput_absolute", "mouse_event"],
        default=argparse.SUPPRESS,
    )
    parser.add_argument("--speed-profile", choices=["safe", "fast", "turbo", "custom"], default=argparse.SUPPRESS)
    parser.add_argument("--inference-flips", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--inference-ensemble", choices=["logits", "probs"], default=argparse.SUPPRESS)
    parser.add_argument("--max-steps", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--action-delay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--capture-delay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--stable-reads", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--stable-read-delay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--no-progress-reclicks", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--reclick-delay", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--click-confirm-retries", type=int, default=argparse.SUPPRESS)
    parser.add_argument(
        "--click-hold",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds between mouse-button down and up; raise this when clicks are missed",
    )
    parser.add_argument(
        "--cursor-settle",
        type=float,
        default=argparse.SUPPRESS,
        help="seconds to wait after moving to the cell center before clicking",
    )
    parser.add_argument("--post-click-settle", type=float, default=argparse.SUPPRESS)
    parser.add_argument("--stall-limit", type=int, default=argparse.SUPPRESS)
    parser.add_argument("--start-mode", choices=["auto", "current", "new", "restart"], default=argparse.SUPPRESS)
    parser.add_argument("--flag-mode", choices=["memory", "open-only"], default=argparse.SUPPRESS)
    parser.add_argument("--record-frames", choices=["all", "final", "none"], default=argparse.SUPPRESS)
    parser.add_argument("--no-final-images", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-hotkey", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--no-persistent-reveals", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument(
        "--quick-number-read",
        action="store_true",
        default=argparse.SUPPRESS,
        help="after an open, read only the target cell when it becomes a nonzero number",
    )
    parser.add_argument("--audit-solver", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--audit-basic", action="store_true", default=argparse.SUPPRESS)
    parser.add_argument("--basic-safety-filter", choices=["none", "avoid-known-mines"], default=argparse.SUPPRESS)
    parser.add_argument("--solver-safety-filter", choices=["none", "avoid-known-mines"], default=argparse.SUPPRESS)
    parser.add_argument(
        "--solver-assist",
        choices=["none", "forced"],
        default=argparse.SUPPRESS,
        help="use solver-proven safe opens and virtual mine flags before RL guesses",
    )
    parser.add_argument(
        "--solver-exact-limit",
        type=int,
        default=argparse.SUPPRESS,
        help="maximum frontier component size for live solver assistance",
    )
    parser.add_argument(
        "--solver-batch-size",
        type=int,
        default=argparse.SUPPRESS,
        help="maximum number of solver-proven safe cells to queue and open consecutively",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    read_parser = subparsers.add_parser("read", help="read and print the current board")
    read_parser.add_argument("--overlay", type=Path, default=ROOT / "artifacts" / "windows_agent" / "overlay.png")

    read_benchmark_parser = subparsers.add_parser("read-benchmark", help="measure repeated board reads without clicking")
    add_subcommand_output_dir(read_benchmark_parser)
    read_benchmark_parser.add_argument("--reads", type=int, default=20)
    read_benchmark_parser.add_argument("--interval", type=float, default=0.02)
    read_benchmark_parser.add_argument("--compare-accurate", action="store_true")

    analyze_parser = subparsers.add_parser("analyze-log", help="summarize desktop agent game logs")
    analyze_parser.add_argument("input", type=Path, help="a game json, manifest json, or output directory")
    analyze_parser.add_argument("--target-win-rate", type=float, default=0.4)
    analyze_parser.add_argument("--target-avg-seconds", type=float, default=60.0)

    play_once_parser = subparsers.add_parser("play-once", help="play one fresh desktop Minesweeper game")
    add_subcommand_output_dir(play_once_parser)

    streak_parser = subparsers.add_parser("run-streak", help="play until the requested winning streak is reached")
    add_subcommand_output_dir(streak_parser)
    streak_parser.add_argument("--streak-length", type=int, default=10)
    streak_parser.add_argument("--max-games", type=int, default=100)

    subparsers.add_parser("stop", help="request any running desktop agent to stop")
    subparsers.add_parser("clear-stop", help="clear a stale stop request file")
    benchmark_parser = subparsers.add_parser("benchmark", help="run a fixed number of desktop games and report aggregate metrics")
    add_subcommand_output_dir(benchmark_parser)
    benchmark_parser.add_argument("--games", type=int, default=10)
    benchmark_parser.add_argument("--target-win-rate", type=float, default=0.4)
    benchmark_parser.add_argument("--target-avg-seconds", type=float, default=60.0)
    benchmark_parser.add_argument("--preflight-read-compare", action="store_true")
    benchmark_parser.add_argument("--preflight-reads", type=int, default=3)
    benchmark_parser.add_argument("--preflight-interval", type=float, default=0.02)

    args = parser.parse_args()
    apply_missing_defaults(args, COMMAND_DEFAULTS.get(args.command, {}))
    apply_missing_defaults(args, BASE_DEFAULTS)
    if args.command == "read":
        desktop = WindowsMinesweeper(
            capture_backend=args.capture_backend,
            read_mode=args.read_mode,
            click_method=args.click_method,
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
    elif args.command == "read-benchmark":
        run_read_benchmark(args)
    elif args.command == "analyze-log":
        run_analyze_log(args)
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
