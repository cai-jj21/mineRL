# Windows Minesweeper Agent Results

## Pure RL Fast2 500-Game Run

Date: 2026-08-10

Local trace directory:

```text
artifacts/windows_agent/pure_rl_fast2_500
```

The full per-game JSON traces are intentionally not committed because
`artifacts/` is ignored.

## Configuration

```text
board: 16 x 30, 99 mines
checkpoint: artifacts/full_rlmix_20.pt
final decision mode: rl
solver assist: none
solver safety filter: none
decision action mode: memory
inference flips: enabled
inference ensemble: probs
click method: mouse_event
read mode: fast
```

Timing:

```text
capture_delay: 0.0005
action_delay: 0.0
stable_reads: 1
stable_read_delay: 0.001
click_hold: 0.03
cursor_settle: 0.008
post_click_settle: 0.025
click_confirm_retries: 2
no_progress_reclicks: 0
```

## Results

```text
files: 500
terminal_games: 495
wins: 200
losses: 295
incomplete: 5
win_rate_completed: 40.40%
longest_streak: 7
avg_elapsed_seconds: 55.826
avg_agent_steps: 260.541
avg_actions_per_second: 4.577
avg_click_seconds: 0.1230
avg_after_read_seconds: 0.1713
total_reclicks: 0
total_unconfirmed_open_actions: 0
total_read_recoveries: 22
fastest_win_seconds: 46.088
slowest_win_seconds: 85.278
```

The benchmark target passed:

```text
target_win_rate: 40%
target_avg_seconds: 60.0
target_passed_by_completed_games: true
```

This block is recomputed from the available `game_*.json` files with
`scripts/summarize_windows_games.py`; it replaces the earlier handwritten
`212 / 500` summary, which is not supported by the current local per-game
artifact set.

## Reproduction Command

```powershell
cd D:\python\mineRL

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py clear-stop

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py `
  --checkpoint artifacts\full_rlmix_20.pt `
  --device cuda `
  --capture-backend auto `
  --read-mode fast `
  --click-method mouse_event `
  --solver-assist none `
  --solver-safety-filter none `
  --start-mode new `
  --flag-mode memory `
  --speed-profile custom `
  --inference-flips `
  --inference-ensemble probs `
  --action-delay 0 `
  --capture-delay 0.0005 `
  --stable-reads 1 `
  --stable-read-delay 0.001 `
  --click-hold 0.03 `
  --cursor-settle 0.008 `
  --post-click-settle 0.025 `
  --click-confirm-retries 2 `
  --no-progress-reclicks 0 `
  --max-steps 600 `
  --stall-limit 20 `
  --record-frames final `
  --no-final-images `
  --output-dir artifacts\windows_agent\pure_rl_fast2_500 `
  benchmark --games 500
```

## Latest Desktop Ensemble Run

Date: 2026-08-15

Local trace directory:

```text
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000
```

The directory contains 954 per-game files. Two trailing files were incomplete,
so the completed-game result is:

```text
completed_games: 952
wins: 377
losses: 575
win_rate: 39.60%
avg_elapsed_seconds: 27.212
avg_actions_per_second: 9.341
```

A verified ten-win streak occurred in:

```text
game_747.json through game_756.json
```

All ten games reported `won: true`, `done: true`, and the terminal dialog
`游戏胜利`. They also had zero click-unready actions, zero unconfirmed opens,
zero target misses, zero reclicks, and zero read recoveries. Individual game
times ranged from 26.143 to 33.615 seconds.

Ten-win replay summary:

```text
range: game_747.json through game_756.json
wins: 10 / 10
avg_elapsed_seconds: 30.09
avg_agent_steps: 291.9
avg_physical_open_actions: 193.3
avg_virtual_flags: 98.6
avg_actions_per_second: 9.71
avg_click_seconds: 0.0906
avg_after_read_seconds: 0.0294
quick_number_reads: 1787
quick_number_fallbacks: 136
execution_anomalies: 0
solver_assist: none
```

Each `game_*.json` in the streak keeps action-level evidence: chosen action,
target row and column, screen coordinate, foreground window, cursor-cell match,
mouse down/up delivery, post-click target state, quick-number read result, and
terminal detection. This makes the streak auditable as a real Windows execution
trace rather than just a terminal counter.

Note: the directory manifest reports only 252 games and `status: stopped`
because it is an earlier interruption snapshot. The per-game JSON files are
the source used for the recomputed 952-game result above.

Recompute command:

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json
```

## Center First Open Quickread

Date: 2026-08-11

Local trace directory:

```text
artifacts/windows_agent/pure_rl_quickread_centerfirst_30
```

Configuration:

```text
board: 16 x 30, 99 mines
checkpoint: artifacts/full_rlmix_20.pt
final decision mode: rl
solver assist: none
solver safety filter: none
decision action mode: memory
inference flips: enabled
inference ensemble: probs
quick number read: enabled
center first open: enabled
click method: mouse_event
read mode: fast
```

Results:

```text
games: 21
wins: 11
losses: 10
incomplete: 0
win_rate: 52.38%
avg_elapsed_seconds: 27.142
avg_agent_steps: 248.95
avg_actions_per_second: 9.094
target_passed: true
```

Reproduction command:

```powershell
cd D:\python\mineRL

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py `
  --checkpoint artifacts\full_rlmix_20.pt `
  --device cuda `
  --capture-backend auto `
  --read-mode fast `
  --click-method mouse_event `
  --solver-assist none `
  --solver-safety-filter none `
  --quick-number-read `
  --center-first-open `
  --start-mode new `
  --flag-mode memory `
  --speed-profile custom `
  --inference-flips `
  --inference-ensemble probs `
  --action-delay 0 `
  --capture-delay 0.0005 `
  --stable-reads 1 `
  --stable-read-delay 0.0005 `
  --click-hold 0.02 `
  --cursor-settle 0.004 `
  --post-click-settle 0.014 `
  --click-confirm-retries 2 `
  --no-progress-reclicks 0 `
  --max-steps 600 `
  --stall-limit 20 `
  --record-frames final `
  --no-final-images `
  --output-dir artifacts\windows_agent\pure_rl_quickread_centerfirst_30 `
  benchmark --games 30
```

## Pure RL Checkpoint Ensemble

Date: 2026-08-11

Internal simulator validation:

```text
primary checkpoint: artifacts/full_rlmix_20.pt
ensemble checkpoint: artifacts/full_rlmix_100_best.pt
final decision mode: rl
solver decision: false
inference flips: enabled
inference ensemble: probs
games: 1000
wins: 430
win_rate: 43.0%
longest_streak: 9
avg_agent_steps: 261.505
avg_revealed_safe_cells: 334.835
```

The matching internal result is saved locally at:

```text
artifacts/ensemble_20_100best_eval_1000_seed0.json
```

Recommended Windows command:

```powershell
cd D:\python\mineRL

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py clear-stop

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py `
  --checkpoint artifacts\full_rlmix_20.pt `
  --ensemble-checkpoint artifacts\full_rlmix_100_best.pt `
  --device cuda `
  --capture-backend auto `
  --read-mode fast `
  --click-method mouse_event `
  --solver-assist none `
  --solver-safety-filter none `
  --quick-number-read `
  --center-first-open `
  --start-mode new `
  --flag-mode memory `
  --speed-profile custom `
  --inference-flips `
  --inference-ensemble probs `
  --action-delay 0 `
  --capture-delay 0.0005 `
  --stable-reads 1 `
  --stable-read-delay 0.0005 `
  --click-hold 0.02 `
  --cursor-settle 0.004 `
  --post-click-settle 0.014 `
  --click-confirm-retries 2 `
  --no-progress-reclicks 0 `
  --max-steps 600 `
  --stall-limit 20 `
  --record-frames final `
  --no-final-images `
  --output-dir artifacts\windows_agent\pure_rl_ensemble_20_100best_500 `
  benchmark --games 500
```
