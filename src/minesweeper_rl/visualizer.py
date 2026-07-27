from __future__ import annotations

from pathlib import Path
from typing import Any, Iterator

from minesweeper_rl.replay import ReplayFrame, ReplayTrace, iter_episode_frames, save_trace
from minesweeper_rl.trainer import MinesweeperTrainer


class MinesweeperViewer:
    def __init__(
        self,
        trace: ReplayTrace | None = None,
        frame_source: Iterator[ReplayFrame] | None = None,
        metadata: dict[str, Any] | None = None,
        save_path: str | Path | None = None,
        speed_ms: int = 350,
        autostart: bool = True,
        title: str = "Minesweeper RL Viewer",
    ) -> None:
        import tkinter as tk
        from tkinter import ttk

        self.tk = tk
        self.ttk = ttk
        self.trace = trace
        self.metadata = _trace_metadata(trace) if trace is not None else dict(metadata or {})
        self.frames: list[ReplayFrame] = list(trace.get("frames", [])) if trace is not None else []
        self.frame_source = frame_source
        self.source_exhausted = trace is not None
        self.save_path = Path(save_path) if save_path is not None else None
        self.autosaved = False
        self.current_index = -1
        self.playing = False
        self.cell_size = int(self.metadata.get("cell_size", 26))

        self.root = tk.Tk()
        self.root.title(title)
        self.root.minsize(980, 560)

        self.status_var = tk.StringVar()
        self.action_var = tk.StringVar()
        self.decision_var = tk.StringVar()
        self.totals_var = tk.StringVar()
        self.play_text = tk.StringVar(value="Pause" if autostart else "Play")
        self.speed_var = tk.IntVar(value=max(50, int(speed_ms)))
        self.show_risk_var = tk.BooleanVar(value=True)
        self.show_mines_var = tk.BooleanVar(value=False)

        self._build_layout()
        self._ensure_frame()
        self._render_current()
        if autostart:
            self._toggle_play()

    def run(self) -> None:
        self.root.mainloop()

    def _build_layout(self) -> None:
        tk = self.tk
        ttk = self.ttk

        root_frame = ttk.Frame(self.root, padding=12)
        root_frame.grid(row=0, column=0, sticky="nsew")
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(0, weight=1)
        root_frame.columnconfigure(0, weight=1)
        root_frame.rowconfigure(0, weight=1)

        self.canvas = tk.Canvas(root_frame, background="#1f252b", highlightthickness=0)
        self.canvas.grid(row=0, column=0, sticky="nsew")

        panel = ttk.Frame(root_frame, padding=(12, 0, 0, 0), width=330)
        panel.grid(row=0, column=1, sticky="ns")
        panel.grid_propagate(False)

        ttk.Label(panel, textvariable=self.status_var, anchor="w", justify="left").grid(row=0, column=0, sticky="ew")
        ttk.Separator(panel).grid(row=1, column=0, sticky="ew", pady=10)
        ttk.Label(panel, textvariable=self.action_var, anchor="w", justify="left").grid(row=2, column=0, sticky="ew")
        ttk.Label(panel, textvariable=self.decision_var, anchor="w", justify="left", wraplength=300).grid(
            row=3,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )
        ttk.Label(panel, textvariable=self.totals_var, anchor="w", justify="left").grid(
            row=4,
            column=0,
            sticky="ew",
            pady=(10, 0),
        )
        ttk.Separator(panel).grid(row=5, column=0, sticky="ew", pady=10)

        controls = ttk.Frame(panel)
        controls.grid(row=6, column=0, sticky="ew")
        ttk.Button(controls, textvariable=self.play_text, command=self._toggle_play).grid(row=0, column=0, padx=(0, 6))
        ttk.Button(controls, text="Back", command=self._previous).grid(row=0, column=1, padx=(0, 6))
        ttk.Button(controls, text="Step", command=self._next).grid(row=0, column=2, padx=(0, 6))
        ttk.Button(controls, text="Save", command=self._save_dialog).grid(row=0, column=3)

        ttk.Checkbutton(panel, text="Risk", variable=self.show_risk_var, command=self._render_current).grid(
            row=7,
            column=0,
            sticky="w",
            pady=(14, 0),
        )
        ttk.Checkbutton(panel, text="Mines", variable=self.show_mines_var, command=self._render_current).grid(
            row=8,
            column=0,
            sticky="w",
        )
        ttk.Scale(panel, from_=50, to=1500, orient="horizontal", variable=self.speed_var).grid(
            row=9,
            column=0,
            sticky="ew",
            pady=(12, 0),
        )

        panel.columnconfigure(0, weight=1)

    def _ensure_frame(self) -> bool:
        if self.frames and self.current_index >= 0:
            return True
        return self._pull_frame()

    def _pull_frame(self) -> bool:
        if self.frame_source is None or self.source_exhausted:
            if self.frames and self.current_index < 0:
                self.current_index = 0
                return True
            return False
        try:
            frame = next(self.frame_source)
        except StopIteration:
            self.source_exhausted = True
            self._autosave()
            return False
        self.frames.append(frame)
        self.current_index = len(self.frames) - 1
        return True

    def _toggle_play(self) -> None:
        self.playing = not self.playing
        self.play_text.set("Pause" if self.playing else "Play")
        if self.playing:
            self._schedule_tick()

    def _schedule_tick(self) -> None:
        if self.playing:
            self.root.after(int(self.speed_var.get()), self._tick)

    def _tick(self) -> None:
        if not self.playing:
            return
        if not self._next():
            self.playing = False
            self.play_text.set("Play")
            return
        self._schedule_tick()

    def _next(self) -> bool:
        if self.current_index + 1 < len(self.frames):
            self.current_index += 1
            self._render_current()
            return True
        if self._pull_frame():
            self._render_current()
            return True
        return False

    def _previous(self) -> None:
        if self.current_index > 0:
            self.current_index -= 1
            self._render_current()

    def _render_current(self) -> None:
        if self.current_index < 0 or not self.frames:
            return
        frame = self.frames[self.current_index]
        self._render_board(frame)
        self._render_panel(frame)

    def _render_board(self, frame: ReplayFrame) -> None:
        canvas = self.canvas
        canvas.delete("all")
        board = frame["board"]
        revealed = board["revealed"]
        rows = len(revealed)
        cols = len(revealed[0]) if rows else 0
        cell = self.cell_size
        canvas.configure(width=cols * cell, height=rows * cell)

        action = frame.get("action")
        action_cell = None if action is None else (action["row"], action["col"])

        for row in range(rows):
            for col in range(cols):
                x0 = col * cell
                y0 = row * cell
                x1 = x0 + cell
                y1 = y0 + cell
                is_revealed = bool(board["revealed"][row][col])
                is_flagged = bool(board["flagged"][row][col])
                is_mine = bool(board["mines"][row][col])
                number = int(board["numbers"][row][col])
                risk = float(board["risk"][row][col])

                if is_revealed:
                    fill = "#e3e7eb"
                    if is_mine:
                        fill = "#cf4f4a"
                else:
                    fill = "#303943"

                canvas.create_rectangle(x0, y0, x1, y1, fill=fill, outline="#111820", width=1)

                if not is_revealed and self.show_risk_var.get() and not is_flagged:
                    bar_width = max(2, int((cell - 4) * min(max(risk, 0.0), 1.0)))
                    canvas.create_rectangle(
                        x0 + 2,
                        y1 - 5,
                        x0 + 2 + bar_width,
                        y1 - 2,
                        fill=_risk_color(risk),
                        outline="",
                    )

                if not is_revealed and bool(board["frontier"][row][col]):
                    canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="#516579", width=1)

                if not is_revealed and bool(board["safe"][row][col]):
                    canvas.create_oval(x0 + 5, y0 + 5, x0 + 10, y0 + 10, fill="#69b36d", outline="")

                if not is_revealed and bool(board["mine"][row][col]):
                    canvas.create_oval(x1 - 10, y0 + 5, x1 - 5, y0 + 10, fill="#d96b65", outline="")

                if is_flagged:
                    canvas.create_polygon(
                        x0 + cell * 0.34,
                        y0 + cell * 0.22,
                        x0 + cell * 0.72,
                        y0 + cell * 0.36,
                        x0 + cell * 0.34,
                        y0 + cell * 0.52,
                        fill="#e4564f",
                        outline="#7a2420",
                    )
                    canvas.create_line(x0 + cell * 0.34, y0 + cell * 0.2, x0 + cell * 0.34, y0 + cell * 0.76, fill="#1a1a1a", width=2)
                elif is_revealed and is_mine:
                    canvas.create_oval(x0 + 6, y0 + 6, x1 - 6, y1 - 6, fill="#1c1f23", outline="#0a0c0e")
                elif is_revealed and number > 0:
                    canvas.create_text(
                        x0 + cell / 2,
                        y0 + cell / 2,
                        text=str(number),
                        fill=_number_color(number),
                        font=("Segoe UI", max(9, int(cell * 0.52)), "bold"),
                    )

                if action_cell == (row, col):
                    canvas.create_rectangle(x0 + 2, y0 + 2, x1 - 2, y1 - 2, outline="#ffd24c", width=3)

    def _render_panel(self, frame: ReplayFrame) -> None:
        summary = frame["summary"]
        known_total = len(self.frames) if self.source_exhausted else f"{len(self.frames)}+"
        status = "won" if summary["won"] else "lost" if summary["lost"] else "running"
        self.status_var.set(
            f"Frame {self.current_index + 1}/{known_total}\n"
            f"Seed {summary['seed']} | {status}\n"
            f"Board {self.metadata['config']['rows']} x {self.metadata['config']['cols']} | "
            f"Mines {self.metadata['config']['mines']}"
        )

        action = frame.get("action")
        if action is None:
            action_text = "Action: none"
        else:
            action_text = (
                f"Action: {action['kind']} "
                f"({action['row'] + 1}, {action['col'] + 1})\n"
                f"Event: {frame['event']}\n"
                f"Reward: {frame['reward']:.4f}"
            )
        self.action_var.set(action_text)

        decision = frame.get("decision", {})
        risk = decision.get("risk")
        best = decision.get("best_guess")
        baseline = decision.get("baseline", decision.get("teacher"))
        lines = [f"Decision: {decision.get('type', 'unknown')}", f"Mode: {decision.get('mode', self.metadata.get('mode', 'unknown'))}"]
        if risk is not None:
            lines.append(f"Risk: {float(risk):.2%}")
        if best is not None:
            lines.append(f"Best: ({best[0] + 1}, {best[1] + 1})")
        if baseline is not None:
            lines.append(f"Baseline: ({baseline[0] + 1}, {baseline[1] + 1})")
        action_index = decision.get("action_index")
        if action_index is not None:
            lines.append(f"Action index: {int(action_index)}")
        note = decision.get("note")
        if note:
            lines.append(str(note))
        self.decision_var.set("\n".join(lines))

        self.totals_var.set(
            f"Reward total: {frame['cumulative_reward']:.3f}\n"
            f"Game steps: {summary['game_steps']}\n"
            f"Agent: {summary.get('agent_steps', summary['guess_steps'])} | Forced: {summary['forced_steps']}\n"
            f"Revealed safe: {summary['revealed_safe_cells']}"
        )

    def _current_trace(self) -> ReplayTrace:
        trace = dict(self.metadata)
        trace.setdefault("version", 1)
        trace["frames"] = self.frames
        trace["summary"] = self.frames[-1]["summary"] if self.frames else {}
        return trace

    def _save_dialog(self) -> None:
        from tkinter import filedialog

        path = self.save_path
        if path is None:
            selected = filedialog.asksaveasfilename(
                title="Save replay",
                defaultextension=".json",
                filetypes=[("Replay JSON", "*.json"), ("All files", "*.*")],
            )
            if not selected:
                return
            path = Path(selected)
        save_trace(path, self._current_trace())
        self.save_path = path

    def _autosave(self) -> None:
        if self.save_path is not None and not self.autosaved and self.frames:
            save_trace(self.save_path, self._current_trace())
            self.autosaved = True


def show_trace(trace: ReplayTrace, speed_ms: int = 350, paused: bool = False) -> None:
    viewer = MinesweeperViewer(trace=trace, speed_ms=speed_ms, autostart=not paused)
    viewer.run()


def show_live_episode(
    trainer: MinesweeperTrainer,
    seed: int,
    mode: str = "rl",
    risk_weight: float = 1.0,
    speed_ms: int = 350,
    save_path: str | Path | None = None,
    paused: bool = False,
) -> None:
    metadata = {
        "version": 1,
        "seed": seed,
        "mode": mode,
        "risk_weight": risk_weight,
        "config": {
            "rows": trainer.config.rows,
            "cols": trainer.config.cols,
            "mines": trainer.config.mines,
            "safe_radius": trainer.config.safe_radius,
        },
    }
    viewer = MinesweeperViewer(
        frame_source=iter_episode_frames(trainer=trainer, seed=seed, mode=mode, risk_weight=risk_weight),
        metadata=metadata,
        save_path=save_path,
        speed_ms=speed_ms,
        autostart=not paused,
        title=f"Minesweeper RL Live - seed {seed}",
    )
    viewer.run()


def _trace_metadata(trace: ReplayTrace) -> dict[str, Any]:
    return {key: value for key, value in trace.items() if key != "frames"}


def _risk_color(risk: float) -> str:
    if risk < 0.15:
        return "#69b36d"
    if risk < 0.30:
        return "#e5b84b"
    if risk < 0.50:
        return "#d98945"
    return "#d96b65"


def _number_color(number: int) -> str:
    return {
        1: "#2457b3",
        2: "#2e7d32",
        3: "#c0392b",
        4: "#5b3f9c",
        5: "#8a4a2f",
        6: "#168a8a",
        7: "#222222",
        8: "#666666",
    }.get(number, "#222222")
