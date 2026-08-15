# 面试答辩问答：高级扫雷 RL 智能体

本文档用于简历投递、面试追问或项目答辩时快速防守关键问题。它不替代 `PROJECT_REPORT.md`、`DEMO_GUIDE.md` 和 `RESUME_PROJECT_CARD.md`，而是把容易被追问的边界、数字和证据入口集中在一处。

## 一分钟答辩版

这个项目不是传统规则 solver 代打，而是 solver-guided RL 研究系统。Solver 在训练和诊断阶段提供 forced move、风险排序、残局对照和失败归因，最终验证命令使用 `--solver-assist none --solver-safety-filter none`，动作来自 RL checkpoint 或多个 RL checkpoint 的概率集成，solver 不参与动作选择。

当前可引用结果是：内部仿真 `full_rlmix_20 + full_rlmix_100_best` 概率集成 1000 局达到 `43.0%` 胜率；Windows 单模型桌面日志 `495 完成局` 达到 `40.40%`；最新 Windows 高速集成 `952 完成局` 达到 `39.60%`，平均 27.21 秒/局，并记录 `game_747` 到 `game_756` 的十连胜。

答辩时不要说“稳定超过 40% Windows 胜率”。更准确的说法是：它不是稳定超过 40% 的 Windows 胜率声明，而是内部集成评估达到 43.0%，Windows 单模型批次达到 40.40%，最新高速集成批次为 39.60%，但执行层异常为 0，十连胜证明模型和真实桌面执行链路可以连续完成高质量局面。

## 追问与回答

**Q: 这是不是 solver 在代打？**

A: 不是。最终验证路径禁用 solver assist 和 solver safety filter，动作由 RL 策略网络或 checkpoint 概率集成输出。Solver 的角色是 teacher、baseline 和 diagnostic tool：训练时提供模仿样本、risk label 和 hard-state 线索，输局后用来判断哪里仍有可见 forced move 或明显风险差距。

**Q: teacher 是不是等于单纯蒸馏？**

A: 不是单纯蒸馏。项目里 teacher 信号用于引导探索和标注可见逻辑，但模型仍在扫雷环境里通过奖励、终局胜负、mine/risk auxiliary loss、DAgger replay 和 checkpoint 评估持续优化。最终策略只看可见棋盘特征，不读取真实雷图，也不在推理时调用 solver 决策。

**Q: 为什么要让模型学 flag，而不是只 open？**

A: 当前 checkpoint 的训练分布包含 `open / flag / unflag / chord` 四类动作，flag 会影响剩余雷数估计和 chord 条件。强行 open-only 会改变策略分布并降低胜率。真正的问题不是“是否使用 flag”，而是错旗会长期污染局面，所以后续优化重点是错旗惩罚、残局计数和风险校准。

**Q: 为什么内部 43.0%，Windows 最新只有 39.60%？**

A: 内部仿真没有屏幕识别、窗口前台状态、真实点击和局面切换分布差异。最新 Windows 高速集成批次有 952 个完成局，点估计略低于 40%，但逐局汇总显示 `reclicks = 0`、`unconfirmed_open_actions = 0`、`read_recoveries = 0`、`open_target_miss_with_progress = 0`。这说明最新瓶颈主要在模型策略，而不是执行层。

**Q: 十连胜证明了什么，不证明什么？**

A: 十连胜证明模型和 Windows 执行层能在真实桌面上连续完成十个高级盘，并且每局都有逐步 JSON 日志。它不证明总体胜率 100%，也不能替代 952 完成局的总体统计。正确讲法是把十连胜作为执行链路和策略上限样本，把总体胜率、Wilson 区间和失败分析作为全局证据。

**Q: 失败主要在哪里？**

A: 当前失败分析指向三类瓶颈：错旗污染、残局剩余雷数/组合计数不足、低风险 guess 的排序不够稳。500 局内部输局对照中，291 个输局里有 67 局存在错旗，92 局在终局仍有 solver 可见 forced move；失败动作的平均目标风险也高于 solver 候选。这些结果支持后续做残局课程学习和更强全局上下文模型。

**Q: 项目最值得讲的技术贡献是什么？**

A: 研究侧是把 solver-guided signal 和 RL 决策边界拆清楚；工程侧是真实 Windows 扫雷执行层、快速读盘、点击证据和逐步日志；实验侧是把内部仿真、真实桌面、十连胜复盘、失败分析、claim audit、artifact manifest 和 CI release gate 串成可复现闭环。

## 现场证据命令

复盘十连胜：

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json `
  --streak-report-output artifacts/report_assets/ten_streak_review.md
```

校验报告数字和十连胜区间：

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json
```

校验简历、README 和报告中的结果声明：

```powershell
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
```

发布前总门禁：

```powershell
python scripts/validate_release.py --expected-tests 147
```

## 推荐讲述顺序

1. 用 `PROJECT_ONE_PAGER.md` 开场，先给目标、方法和结果。
2. 用 `ARCHITECTURE.md` 解释环境、模型、solver 和 Windows 执行层边界。
3. 用 `EVALUATION_PROTOCOL.md` 解释胜率、速度、连胜和样本量口径。
4. 用 `ABLATION_STUDY.md` 回答为什么集成、为什么还输、下一步怎么做。
5. 用 `FAILURE_ANALYSIS.md` 展示失败不是简单点击问题，而是策略和残局推理问题。
6. 用 `RESUME_PROJECT_CARD.md` 收束成可复制的简历 bullet。

## 回答边界

- 可以说内部 1000 局集成胜率为 `43.0%`。
- 可以说 Windows 单模型日志 `495 完成局` 达到 `40.40%`。
- 可以说最新 Windows 高速集成 `952 完成局` 为 `39.60%`，并完成十连胜。
- 可以说十连胜区间是 `game_747` 到 `game_756`，执行异常为 0。
- 不说 Windows 胜率已经稳定超过 40%。
- 不说 solver 在最终验证中参与动作选择。
- 不说十连胜等价于总体胜率。
