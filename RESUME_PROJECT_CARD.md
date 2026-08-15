# 简历项目卡：高级扫雷强化学习智能体

## 一句话版本

独立构建高级扫雷 `16 x 30 / 99` 强化学习智能体，覆盖仿真环境、Actor-Critic 策略网络、solver 辅助训练、真实 Windows GUI 自动化与逐局实验日志；内部 1000 局纯 RL 集成胜率 `43.0%`，真实 Windows 扫雷完成 `10` 连胜。

## 推荐标题

**高级扫雷强化学习智能体：solver-guided RL 与 Windows GUI 自动执行系统**

## 技术栈

- Python, NumPy, PyTorch, pytest
- Actor-Critic, DAgger, imitation learning, risk supervision
- Constraint solver, component enumeration, global mine-count probability
- Windows GUI automation, screen capture, mouse event execution
- Experiment logging, replay analysis, checkpoint ensemble

## 量化成果

| 指标 | 结果 |
| --- | ---: |
| 高级图规模 | `16 x 30 / 99` |
| 动作空间 | `4 x 16 x 30 = 1920` |
| 内部仿真最佳胜率 | `43.0%` / 1000 局 |
| Windows 单模型稳定运行 | `40.40%` / 495 完成局 |
| Windows 最新高速运行 | `39.60%` / 952 完成局 |
| Windows 最长连胜 | `10` |
| 十连胜平均耗时 | `30.09` 秒/局 |
| 最新运行平均耗时 | `27.21` 秒/局 |
| 执行层异常 | 重复点击、未确认打开、读盘恢复均为 `0` |

## 简历 Bullet 版本

- 独立实现高级扫雷 RL 平台，支持 `open / flag / unflag / chord` 四类动作和 `1920` 维动作空间，完成环境、特征编码、训练器、可视化回放与 checkpoint 管理。
- 设计全卷积 Actor-Critic 网络，结合 solver imitation、DAgger replay、mine auxiliary loss 和 risk supervision，在最终纯 RL 推理路径下达到内部 1000 局 `43.0%` 胜率。
- 构建可见信息 solver 基线，使用约束集合、组件枚举和全局剩余雷数概率进行训练期标注与失败诊断，定位错旗污染和残局计数不足问题。
- 开发 Windows 扫雷 GUI 自动执行层，实现网格检测、快速读盘、鼠标点击确认、终局弹窗检测和逐步 JSON 日志，真实桌面完成 `10` 连胜。
- 建立实验复现工具链，提供 checkpoint ensemble 评估、逐局日志汇总、十连胜复盘、统计置信区间、失败分析、证据校验、声明校验、文档校验、发布校验、产物哈希清单和 CI 测试，当前 `147` 项测试通过。

## 面试讲述结构

### 背景

扫雷高级图是一个高风险、稀疏奖励、部分可观测的决策问题。一次错误 open 会直接失败，同时后期局面需要综合局部数字和全局剩余雷数。

### 方法

我先实现了可控仿真环境和可见状态编码，再训练 Actor-Critic 策略网络。Solver 不直接代打，而是作为训练期 teacher 和诊断器，提供 forced move、低风险 guess、mine/risk 辅助标签。最终推理时只使用神经网络策略和 checkpoint 概率集成。

### 工程落地

为了验证模型不只是在仿真里工作，我写了 Windows 扫雷执行层：截图读盘、识别格子、移动鼠标、点击、移出棋盘、检测胜负弹窗，并保存每一步 action log。后续用逐局汇总脚本证明十连胜和执行稳定性。

### 结果

内部仿真中，`full_rlmix_20 + full_rlmix_100_best` 概率集成 1000 局达到 `43.0%`。Windows 桌面中，可验证单模型日志 495 完成局达到 `40.40%`，最新高速集成运行 952 完成局达到 `39.60%`，并在 `game_747` 到 `game_756` 完成十连胜。

### 反思

主要瓶颈已经从点击/识图转移到模型策略。失败复盘显示错旗会污染剩余雷数估计，残局组合推理仍弱于 exact solver。下一步会做残局课程学习、错旗长期惩罚和更强全局上下文模型。

500 局内部输局 solver 对照中，291 个输局里有 67 局存在错旗，92 局在终局仍有 solver 可见 forced move；这说明优化方向不是盲目调点击，而是加强 flag 校准、残局计数和风险排序。

## 可追问点与回答要点

**Q: 这是不是 solver 在代打？**
A: 不是。Solver 用于训练标签、基线和诊断。最终验证命令使用 `--solver-assist none --solver-safety-filter none`，动作来自 `rl` 策略网络或多个 RL checkpoint 的概率平均。

**Q: 为什么要让模型学 flag，而不是只 open？**
A: 这个 checkpoint 的训练分布包含 flag/unflag/chord。强行 open-only 会明显降胜率；flag 也给模型提供剩余雷数和 chord 条件。问题在于错旗代价，需要改进长期惩罚和残局修正。

**Q: 为什么内部 43%，Windows 最新只有 39.60%？**
A: 内部仿真没有屏幕识别和 GUI 状态切换分布差异。最新 Windows 运行速度更快，且 952 局样本里略低于 40%。执行层指标显示没有漏点和读盘恢复，下一步重点是模型策略。

**Q: 十连胜怎么证明？**
A: `scripts/summarize_windows_games.py` 按 `game_*.json` 逐局重算，输出 `longest_streak = 10`，区间是 `game_747` 到 `game_756`。十局均为 `won: true`、`done: true`、终局弹窗 `游戏胜利`。

**Q: 项目最有技术含量的部分是什么？**
A: 研究上是 solver-guided RL 与失败诊断；工程上是真实 Windows 扫雷的稳定执行层和逐步可审计日志；实验上是把内部策略评估、真实桌面执行和报告复现串成闭环。

## 仓库证据入口

- 项目入口：`README.md`
- 架构总览：`ARCHITECTURE.md`
- 演示指南：`DEMO_GUIDE.md`
- 消融对照：`ABLATION_STUDY.md`
- 评估协议：`EVALUATION_PROTOCOL.md`
- 科研报告：`PROJECT_REPORT.md`
- 模型卡：`MODEL_CARD.md`
- 产物说明：`ARTIFACTS.md`
- 失败分析附录：`FAILURE_ANALYSIS.md`
- 复现实验说明：`REPRODUCIBILITY.md`
- 发布检查清单：`RELEASE_CHECKLIST.md`
- 实验产物清单：`EXPERIMENT_MANIFEST.md`
- Windows 实验记录：`WINDOWS_AGENT_RESULTS.md`
- 十连胜汇总：`artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`
- 报告资产：`artifacts/report_assets/experiment_summary.md`
- 图表生成脚本：`scripts/generate_report_assets.py`
- 失败分析脚本：`scripts/summarize_failure_analysis.py`
- 证据校验脚本：`scripts/validate_project_evidence.py`
- 文档校验脚本：`scripts/validate_documentation.py`
- 产物清单脚本：`scripts/build_artifact_manifest.py`
- 逐局汇总脚本：`scripts/summarize_windows_games.py`
- 集成评估脚本：`scripts/evaluate_ensemble.py`
