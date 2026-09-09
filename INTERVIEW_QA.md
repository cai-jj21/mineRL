# 面试答辩问答：高级扫雷 RL 智能体

本文档用于面试、答辩和 GitHub 展示时快速说明项目边界。完整报告见 `PROJECT_REPORT.md`，简历表达见 `RESUME_PROJECT_CARD.md`，数据开发链路见 `DATA_DEVELOPMENT_CASE.md`。

## 一分钟版本

这个项目不是 solver 代打，而是 solver-guided RL。Solver 在训练和诊断阶段提供 forced move、风险标签、残局对照和失败归因；最终验证命令使用 `--solver-assist none --solver-safety-filter none`，动作来自 RL checkpoint 或 checkpoint 概率集成，solver 不参与动作选择。

当前可引用结果：内部仿真 `full_rlmix_20 + full_rlmix_100_best` 概率集成 1000 局胜率 `43.0%`；Windows 单模型日志 `495 完成局` 达到 `40.40%`；最新 Windows 高速集成 `952 完成局` 达到 `39.60%`，并记录 `game_747` 到 `game_756` 的十连胜。准确说法是“不是稳定超过 40% 的 Windows 胜率声明”，而是内部策略已达到 43.0%，真实桌面执行链路完成过 10 连胜。

## 高频追问

**Q: 这是不是 solver 在代打？**

A: 不是。最终验证关闭 solver assist 和 solver safety filter；solver 只做 teacher、baseline 和 diagnostic tool。训练时它提供可见逻辑样本、mine/risk 标签和 hard-state 线索，失败后用它判断模型是否错过 forced move 或选择了高风险 guess。

**Q: teacher 是不是单纯蒸馏？**

A: 不是。Teacher 信号帮助探索和标注可见逻辑，但模型仍通过奖励、终局胜负、DAgger replay、risk auxiliary loss 和 checkpoint 评估继续优化。最终策略只看可见棋盘，不读取真实雷图，也不在推理时调用 solver 决策。

**Q: 为什么 Windows 最新结果略低于 40%？**

A: 内部仿真没有屏幕识别、窗口焦点、点击确认和新局切换噪声。最新桌面批次的执行指标已经稳定，瓶颈主要转向策略：错旗污染、残局剩余雷数计数、低风险 guess 排序。不能把十连胜等同于总体胜率。

**Q: 数据开发方向怎么讲？**

A: 可以从 `DATA_DEVELOPMENT_CASE.md` 切入：`scripts/summarize_windows_games.py`、`scripts/generate_report_assets.py`、`scripts/generate_statistical_report.py` 负责数据采集和指标汇总；`scripts/build_experiment_database.py`、`scripts/analyze_experiment_database.py` 和 `sql/warehouse_analysis.sql` 把实验日志落到 SQLite/SQL 分析层；`scripts/generate_training_feedback_plan.py`、`scripts/export_training_feedback_dataset.py` 再把失败信号转成训练反馈数据资产。

## 现场证据命令

```powershell
python scripts/validate_project_evidence.py --output artifacts/report_assets/evidence_validation.json
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
python scripts/validate_release.py --expected-tests 160
```
