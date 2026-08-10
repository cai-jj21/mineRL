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
games: 500
wins: 212
losses: 288
incomplete: 0
win_rate: 42.4%
longest_streak: 7
avg_elapsed_seconds: 55.753
avg_agent_steps: 260.816
avg_actions_per_second: 4.580
avg_click_seconds: 0.1212
avg_after_read_seconds: 0.1711
total_reclicks: 0
total_unconfirmed_open_actions: 0
total_read_recoveries: 0
fastest_win_seconds: 47.377
slowest_win_seconds: 79.513
```

The benchmark target passed:

```text
target_win_rate: 40%
target_avg_seconds: 60.0
target_passed: true
```

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
