# 高级扫雷强化学习智能体项目报告

## 摘要

本项目研究经典高级扫雷 `16 x 30 / 99` 下的强化学习决策问题，从零实现仿真环境、可见状态编码、Actor-Critic 策略网络、solver 辅助训练、真实 Windows 扫雷执行层、逐局日志和数据分析仓库。最终验证路径中 solver 不参与动作选择。

当前最佳内部仿真结果为两个纯 RL checkpoint 概率集成：1000 局胜率 `43.0%`。真实 Windows 桌面中，单模型日志有 `495 个完成局`、`40.40%` 胜率；最新高速集成批次有 `952 个完成局`、`39.60%` 胜率，并记录完整 `10 连胜`。

## 系统结构

项目分层见 `ARCHITECTURE.md`：`src/minesweeper_rl/game.py` 提供环境，`src/minesweeper_rl/features.py` 提供可见特征，`src/minesweeper_rl/model.py` 提供策略网络，`src/minesweeper_rl/trainer.py` 负责训练，`src/minesweeper_rl/solver.py` 只用于训练、基线和诊断。Windows 真实执行由 `scripts/windows_minesweeper_agent.py` 完成。

模型卡见 `MODEL_CARD.md`，checkpoint 和大体积产物策略见 `ARTIFACTS.md`。评估口径见 `EVALUATION_PROTOCOL.md`，演示路线见 `DEMO_GUIDE.md`，答辩边界见 `INTERVIEW_QA.md`，简历表达见 `RESUME_PROJECT_CARD.md`。

## 实验与证据

对照实验和风险讨论见 `ABLATION_STUDY.md`，完成度审计见 `PROJECT_COMPLETION_AUDIT.md`，失败归因见 `FAILURE_ANALYSIS.md`。关键证据命令包括：

- `scripts/summarize_failure_analysis.py`
- `scripts/generate_statistical_report.py`
- `scripts/validate_claims.py`
- `scripts/build_experiment_database.py`
- `scripts/analyze_experiment_database.py`
- `scripts/generate_training_feedback_plan.py`
- `scripts/export_training_feedback_dataset.py`

证据校验当前覆盖 `25 / 25` 项，文档和发布门禁使用 `160` 测试基线。数据开发链路见 `DATA_DEVELOPMENT_CASE.md`，它将逐局日志、失败诊断、产物 manifest、SQLite 仓库、SQL 分析和训练反馈数据集串成闭环。

## 结论

项目已经完成从 RL 训练、Windows 自动执行、十连胜记录、失败分析到 GitHub 发布材料的闭环。下一步优化重点不是继续调点击层，而是利用数据仓库抽取边角、残局、无 forced move 的猜测局面，定向提升模型在极端局面下的风险排序。
