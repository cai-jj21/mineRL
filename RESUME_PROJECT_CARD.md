# 简历项目卡：高级扫雷强化学习智能体

## 一句话版本

独立构建高级扫雷 `16 x 30 / 99` 强化学习智能体，覆盖仿真环境、Actor-Critic 策略网络、solver 辅助训练、真实 Windows GUI 自动执行、逐局日志、数据仓库和训练反馈数据资产；内部 1000 局纯 RL 集成胜率 `43.0%`，真实 Windows 扫雷完成 `10` 连胜。

## 量化结果

- Windows 单模型：`40.40%`，`495 完成局`。
- Windows 高速集成：`39.60%`，`952 完成局`。
- 发布质量：`160` 项测试，证据校验 `25 / 25`。

## 简历 Bullet

- 实现高级扫雷 RL 平台，支持 `open / flag / unflag / chord` 动作、可见状态编码、Actor-Critic 训练、checkpoint ensemble 和失败复盘。
- 构建 Windows GUI 自动执行层，完成读盘、点击、终局检测、逐步 JSON 日志和十连胜复盘，执行证据可由 `scripts/validate_project_evidence.py` 校验。
- 搭建数据开发链路：`scripts/build_experiment_database.py`、`scripts/analyze_experiment_database.py`、`scripts/generate_training_feedback_plan.py`、`scripts/export_training_feedback_dataset.py`、`scripts/build_artifact_manifest.py` 和 `sql/warehouse_analysis.sql` 将实验日志沉淀为 SQLite 仓库、SQL 分析、产物哈希清单和训练反馈数据集。
- 完成 GitHub 展示材料：`ARCHITECTURE.md`、`ABLATION_STUDY.md`、`PROJECT_COMPLETION_AUDIT.md`、`DEMO_GUIDE.md`、`INTERVIEW_QA.md`、`DATA_DEVELOPMENT_CASE.md`、`EVALUATION_PROTOCOL.md`、`MODEL_CARD.md`、`ARTIFACTS.md`、`FAILURE_ANALYSIS.md`、`REPRODUCIBILITY.md`、`RELEASE_CHECKLIST.md`。

## 面试讲法

先讲 `ARCHITECTURE.md` 中的系统边界，再用 `DATA_DEVELOPMENT_CASE.md` 展示数据采集、ETL、质量校验、指标汇总、可复现报告和训练反馈，最后用 `PROJECT_REPORT.md`、`INTERVIEW_QA.md` 和 `RELEASE_CHECKLIST.md` 收束到结果、风险和发布质量。

当前 `160` 项测试通过，项目可以作为强化学习、自动化执行和数据开发结合的完整案例。
