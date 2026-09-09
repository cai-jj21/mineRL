-- Minesweeper RL experiment warehouse analysis queries.
-- Run after:
--   python scripts/build_experiment_database.py --include-actions all
--   python scripts/analyze_experiment_database.py
--
-- Example:
--   sqlite3 artifacts/report_assets/minesweeper_experiments.sqlite ".read sql/warehouse_analysis.sql"

.headers on
.mode column

-- 1. ETL batch audit: one row per local warehouse build.
SELECT
  batch_id,
  status,
  started_at,
  finished_at,
  include_actions,
  source_file_count,
  total_row_count
FROM etl_batch
ORDER BY started_at DESC;

-- 2. ODS/source inventory: file lineage, source volume, and freshness.
SELECT
  layer,
  source_type,
  file_count,
  total_size_bytes,
  total_row_count,
  first_loaded_at,
  last_loaded_at
FROM v_ods_source_inventory
ORDER BY layer, source_type;

-- 3. ADS dashboard: experiment leaderboard with target gap.
SELECT
  name,
  source,
  games,
  wins,
  ROUND(win_rate * 100.0, 2) AS win_rate_pct,
  ROUND(target_win_rate * 100.0, 2) AS target_win_rate_pct,
  ROUND(target_gap * 100.0, 2) AS target_gap_pct,
  point_target_pass
FROM v_ads_experiment_dashboard
ORDER BY win_rate DESC, games DESC;

-- 4. DWS run KPI: separates model outcome from execution-layer quality.
SELECT
  run_id,
  games,
  wins,
  losses,
  incomplete_games,
  ROUND(win_rate_completed * 100.0, 2) AS win_rate_completed_pct,
  ROUND(avg_elapsed_seconds, 2) AS avg_elapsed_seconds,
  ROUND(avg_actions_per_second, 2) AS avg_actions_per_second,
  total_reclicks,
  total_unconfirmed_open_actions,
  total_read_recoveries
FROM v_dws_run_kpi
ORDER BY run_id;

-- 5. Outcome distribution for Windows desktop runs.
SELECT
  run_id,
  outcome,
  COUNT(*) AS games,
  ROUND(AVG(elapsed_seconds), 2) AS avg_elapsed_seconds,
  ROUND(AVG(agent_steps), 2) AS avg_agent_steps,
  ROUND(AVG(revealed_safe_cells), 2) AS avg_revealed_safe_cells
FROM v_dwd_game_session
WHERE run_type = 'windows_desktop'
GROUP BY run_id, outcome
ORDER BY run_id, outcome;

-- 6. Late-loss samples: losses with high revealed-safe-cell count.
SELECT
  run_id,
  game_index,
  revealed_safe_cells,
  agent_steps,
  physical_open_actions,
  flags,
  ROUND(elapsed_seconds, 2) AS elapsed_seconds,
  ROUND(actions_per_second, 2) AS actions_per_second
FROM v_dwd_game_session
WHERE outcome = 'loss'
ORDER BY revealed_safe_cells DESC, agent_steps DESC
LIMIT 20;

-- 7. Action mix: open/flag/chord behavior and average board progress.
SELECT
  run_id,
  kind,
  COUNT(*) AS actions,
  SUM(virtual_action) AS virtual_actions,
  ROUND(AVG(revealed_delta), 2) AS avg_revealed_delta
FROM v_dwd_action_event
GROUP BY run_id, kind
ORDER BY run_id, actions DESC;

-- 8. Execution anomaly audit.
SELECT
  run_id,
  games,
  total_reclicks,
  total_click_unready_actions,
  total_unconfirmed_open_actions,
  total_read_recoveries,
  total_execution_anomalies
FROM v_execution_anomalies
ORDER BY total_execution_anomalies DESC, run_id;

-- 9. Verified ten-streak detail.
SELECT
  game_index,
  outcome,
  ROUND(elapsed_seconds, 2) AS elapsed_seconds,
  agent_steps,
  physical_open_actions,
  flags,
  revealed_safe_cells,
  ROUND(actions_per_second, 2) AS actions_per_second
FROM v_ten_streak_games
ORDER BY game_index;

-- 10. Diagnostic signal profile: what the latest loss review thinks matters.
SELECT
  diagnostic_run_id,
  signal_name,
  signal_count
FROM v_ads_diagnostic_signal_profile
ORDER BY signal_count DESC, signal_name;

-- 11. Training feedback signal: warehouse metrics that drive hard-loss refinement.
SELECT
  failure_analysis_id,
  ROUND(wrong_flag_loss_rate * 100.0, 2) AS wrong_flag_loss_rate_pct,
  ROUND(terminal_forced_rate * 100.0, 2) AS terminal_forced_rate_pct,
  ROUND(avg_risk_gap_to_best_guess, 4) AS avg_risk_gap_to_best_guess,
  selected_endgame_safe_left,
  ROUND(selected_endgame_loss_rate * 100.0, 2) AS selected_endgame_loss_rate_pct,
  recommended_exact_limit
FROM v_ads_failure_training_signal;

-- 12. Endgame buckets used by the feedback plan.
SELECT
  failure_analysis_id,
  bucket,
  safe_left_threshold,
  loss_count,
  ROUND(loss_rate * 100.0, 2) AS loss_rate_pct,
  has_wrong_flags,
  forced_available,
  target_known_mine,
  ROUND(avg_target_risk, 4) AS avg_target_risk,
  ROUND(avg_best_guess_risk, 4) AS avg_best_guess_risk
FROM failure_endgame_bucket
ORDER BY failure_analysis_id, safe_left_threshold;

-- 13. Training feedback profiles: data assets available for model improvement.
SELECT
  profile_name,
  record_count,
  counterfactual_candidate_labels,
  ROUND(negative_counterfactual_rate * 100.0, 2) AS negative_candidate_rate_pct,
  ROUND(avg_behavior_regret, 3) AS avg_behavior_regret,
  ROUND(high_regret_rate * 100.0, 2) AS high_regret_rate_pct,
  ROUND(behavior_mine_rate * 100.0, 2) AS behavior_mine_rate_pct
FROM v_ads_training_dataset_profile
ORDER BY avg_behavior_regret DESC, record_count DESC;

-- 14. Slice-level shortfall: use this to choose the next sampling/training slice.
SELECT
  region,
  safe_left_bucket,
  open_candidate_bucket,
  behavior_regret_bucket,
  records,
  ROUND(avg_safe_left, 1) AS avg_safe_left,
  ROUND(avg_open_candidate_count, 1) AS avg_open_candidate_count,
  ROUND(avg_negative_counterfactual_rate * 100.0, 2) AS negative_candidate_rate_pct,
  ROUND(avg_behavior_regret, 3) AS avg_behavior_regret,
  ROUND(max_behavior_regret, 3) AS max_behavior_regret,
  behavior_mine_records
FROM v_ads_training_feedback_slice
ORDER BY avg_behavior_regret DESC, records DESC;

-- 15. Highest-priority replay records.
SELECT
  profile_id,
  record_index,
  family,
  region,
  safe_left,
  open_candidate_count,
  ROUND(behavior_regret, 3) AS behavior_regret,
  behavior_mine,
  priority
FROM v_ads_training_feedback_priority
ORDER BY
  CASE priority WHEN 'critical' THEN 0 WHEN 'high' THEN 1 WHEN 'medium' THEN 2 ELSE 3 END,
  behavior_regret DESC
LIMIT 100;
