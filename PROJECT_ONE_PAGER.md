# 一页式项目总览：高级扫雷强化学习智能体

## 一句话

这是一个面向经典高级扫雷 `16 x 30 / 99` 的强化学习项目：从仿真环境、Actor-Critic 策略网络、solver-guided 训练信号，到真实 Windows 扫雷执行层和可审计实验报告，形成了一套可以放进简历和 GitHub 展示的研究闭环。

## 当前结果

| 场景 | 策略 | 样本 | 结果 | 证据 |
| --- | --- | ---: | ---: | --- |
| 内部仿真 | `full_rlmix_20 + full_rlmix_100_best` 概率集成 | 1000 局 | 43.0% 胜率 | `artifacts/ensemble_20_100best_eval_1000_seed0.json` |
| Windows 桌面 | `full_rlmix_20` 单模型 | 495 完成局 | 40.40% 胜率 | `artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json` |
| Windows 桌面 | `full_rlmix_20 + full_rlmix_100_best` 概率集成 | 952 完成局 | 39.60% 胜率，平均 27.21 秒/局 | `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json` |
| Windows 十连胜 | `full_rlmix_20 + full_rlmix_100_best` 概率集成 | `game_747` 到 `game_756` | 10 / 10 胜 | `artifacts/report_assets/ten_streak_review.md` |

十连胜区间记录了每步动作、目标格、屏幕坐标、鼠标按下/抬起、读盘结果和终局弹窗。该区间 `reclicks = 0`、`unconfirmed_open_actions = 0`、`read_recoveries = 0`、`open_target_miss_with_progress = 0`。

## 技术路线

1. 自研高级扫雷环境，支持延迟布雷、首开安全、`open / flag / unflag / chord` 和胜负奖励。
2. 用可见棋盘、邻域统计、坐标先验、剩余雷数估计构造模型输入，不读取真实雷图。
3. 训练 Actor-Critic 网络，策略头覆盖 `4 x 16 x 30 = 1920` 维动作空间。
4. 使用 solver 作为训练 teacher、风险监督、基线和失败诊断器；最终验证中 solver 不参与动作选择。
5. 将策略接入真实 Windows 扫雷窗口，完成截图读盘、点击执行、终局检测和逐局 JSON 日志。
6. 用证据校验、声明审计、统计置信区间和 artifact 哈希清单约束报告数字。

## 关键贡献

- **研究侧**：把扫雷中的局部逻辑、全局剩余雷数约束和高风险稀疏奖励问题放进 RL 框架中，比较纯 RL、solver baseline、checkpoint ensemble 和 hard-loss refine。
- **工程侧**：实现真实 Windows 扫雷执行层，并用 action-level 日志证明点击、读盘和终局检测不是黑盒。
- **实验侧**：将内部仿真、真实桌面运行、十连胜复盘、失败分析和报告资产生成串成可复现闭环。
- **交付侧**：提供 `PROJECT_REPORT.md`、`ABLATION_STUDY.md`、`PROJECT_COMPLETION_AUDIT.md`、`MODEL_CARD.md`、`EVALUATION_PROTOCOL.md`、`REPRODUCIBILITY.md`、`RESUME_PROJECT_CARD.md`、CI 和发布门禁。

## 可信边界

- 最终动作来自 RL checkpoint 或 checkpoint 概率集成，solver 不在最终验证路径中代打。
- Windows 最新高速集成结果为 `39.60%`，略低于 40%；可验证单模型桌面日志为 `40.40%`。
- 内部 `43.0%` 和 Windows 桌面结果存在执行分布差异，应同时报告样本量和 Wilson 置信区间。
- 十连胜证明模型和执行层能连续完成高质量桌面局面，不代表总体胜率 100%。

## 面试讲法

> 我做的是一个完整扫雷 RL 研究平台，而不是单纯规则 solver。模型只看可见棋盘，训练时借助 solver 提供逻辑先验和风险标签，最终验证时由 RL 策略独立决策。我还把模型接到真实 Windows 扫雷窗口上，记录每步点击和读盘证据，最终在桌面环境中完成 952 局评估和十连胜复盘。项目亮点在于把算法、真实 GUI 执行、实验统计和可复现报告都做成了闭环。

## 证据入口

- 完整报告：`PROJECT_REPORT.md`
- 架构总览：`ARCHITECTURE.md`
- 演示路线：`DEMO_GUIDE.md`
- 消融对照：`ABLATION_STUDY.md`
- 完成度审计：`PROJECT_COMPLETION_AUDIT.md`
- 评估协议：`EVALUATION_PROTOCOL.md`
- 模型卡：`MODEL_CARD.md`
- 复现流程：`REPRODUCIBILITY.md`
- 简历卡：`RESUME_PROJECT_CARD.md`
- Windows 结果：`WINDOWS_AGENT_RESULTS.md`
- 产物哈希：`EXPERIMENT_MANIFEST.md`
- 发布门禁：`python scripts/validate_release.py --expected-tests 147`
