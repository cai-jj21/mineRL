# 数据开发案例：用数据资产反哺扫雷 RL 训练

本文档把扫雷 RL 项目整理成真实数据开发工作流：数据采集、ETL、质量校验、指标汇总、可复现报告、训练反馈。它补充 `PROJECT_REPORT.md` 和 `REPRODUCIBILITY.md`，重点说明数据仓库和分析资产如何服务模型优化。

## 数据采集

Windows 执行层和内部评估会产出逐局 JSON、逐步 action log、失败诊断和统计摘要。核心入口包括：

- `scripts/summarize_windows_games.py` 汇总 `game_*.json`，生成 `per_game_summary.json`。
- `scripts/generate_report_assets.py` 生成报告图表和 `analysis.json`。
- `scripts/generate_statistical_report.py` 生成 Wilson 区间和统计口径。
- `scripts/summarize_failure_analysis.py` 生成 `failure_analysis_summary.json`。

## ETL 与仓库分层

`scripts/build_experiment_database.py` 将实验资产落到 `minesweeper_experiments.sqlite`。仓库采用 ODS/DWD/DWS/ADS 分层，保留批次审计、源文件血缘、逐局明细和面向分析的宽表。

- ODS: `etl_batch`、`source_file` 和 `v_ods_source_inventory` 记录 ETL 批次、源文件清单和文件血缘。
- DWD: `diagnostic_runs`、`diagnostic_game_detail`、`diagnostic_signal`、`diagnostic_recommendation`、`failure_analysis_run`、`failure_endgame_bucket`、`failure_exact_limit`、`v_dwd_game_session`、`v_dwd_action_event` 记录标准化事实。
- DWS: `v_dws_run_kpi` 做指标汇总。
- ADS: `v_ads_experiment_dashboard`、`v_ads_diagnostic_signal_profile`、`v_ads_failure_training_signal` 面向展示、诊断和训练反馈。

SQL 查询沉淀在 `sql/warehouse_analysis.sql`，用于复查源文件、运行 KPI、失败 bucket、诊断信号和训练建议。

## 质量校验

发布前的质量校验由 `scripts/validate_project_evidence.py`、`scripts/validate_claims.py`、`scripts/validate_release.py` 串联完成。它们检查报告数字、十连胜区间、文档声明、测试基线、产物清单和可复现入口，避免把偶然结果或过期数字写进 README。

关键报告资产包括 `database_analysis.md`、`database_analysis.json`、`claim_audit.json`、`artifact_manifest.json`、`training_feedback_plan.json`、`training_feedback_plan.md`、`training_feedback_dataset.json`、`training_feedback_dataset.jsonl`、`training_feedback_dataset.md`。

## 指标汇总

当前结果口径：

- 内部仿真 checkpoint 概率集成：`43.0%`。
- Windows 单模型批次：`40.40%`，`495 完成局`。
- Windows 高速集成批次：`39.60%`，`952 完成局`。
- 十连胜区间：`game_747` 到 `game_756`。

这些数字不是稳定超过 40% 的 Windows 胜率声明，而是实验批次证据。`analysis.json` 和 `per_game_summary.json` 用于复盘输赢、耗时、执行异常和最长连胜。

## 训练反馈

`scripts/analyze_experiment_database.py` 输出 `database_analysis.md` / `database_analysis.json`，`scripts/generate_training_feedback_plan.py` 将错旗率、终局 forced move、风险 gap、残局 bucket 转成训练计划，`scripts/export_training_feedback_dataset.py` 导出极端局面数据集。目标不是单纯堆局数，而是把边角、残局、无 forced move 的高风险 guess 等短板场景抽出来，形成可定向训练的数据资产。
