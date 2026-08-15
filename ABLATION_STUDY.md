# 消融与对照研究：高级扫雷 RL 智能体

## 目的

本文档回答一个面试和答辩中很常见的问题：这个项目里哪些设计真的带来了收益，哪些只是工程迭代中的观察？所有数字都来自本仓库已有 artifact，不额外声明未验证结果。

## 证据等级

| 等级 | 含义 | 本项目示例 |
| --- | --- | --- |
| 严格评估 | 固定评估口径、有明确样本量、可由脚本复现汇总 | 内部 1000 局集成评估、Windows 952 完成局汇总 |
| 对照证据 | 样本量或配置不同，但能支持趋势判断 | 单模型 vs checkpoint 概率集成、Windows 单模型慢速 vs 高速集成 |
| 诊断证据 | 用 solver 或失败日志解释输局原因，不直接等同于胜率提升 | 错旗率、terminal forced move、exact limit 对比 |

## 已完成对照

| 问题 | 对照 | 样本 | 结果 | 结论边界 |
| --- | --- | ---: | ---: | --- |
| checkpoint 概率集成是否有帮助？ | 内部单模型 `full_rlmix_20` | 200 局 | 41.50% | 单模型候选筛选结果，样本较小 |
| checkpoint 概率集成是否有帮助？ | 内部 `full_rlmix_20 + full_rlmix_100_best` 概率集成 | 1000 局 | 43.0% | 同为纯 RL 决策，样本更大；支持集成带来更稳的内部评估 |
| ensemble 方式是否敏感？ | `logits` 集成 | 500 局 | 41.60% | 与 `probs` 1000 局样本量不同，只能作为弱对照 |
| ensemble 方式是否敏感？ | `probs` 集成 | 1000 局 | 43.0% | 当前报告采用 `probs`，但不能单独证明概率集成必然优于 logits |
| Windows 执行层加速是否破坏策略？ | 单模型 Windows 慢速确认 | 495 完成局 | 40.40%，55.83 秒/局 | 通过 40% 点估计，但速度慢 |
| Windows 执行层加速是否破坏策略？ | 高速 Windows 集成 | 952 完成局 | 39.60%，27.21 秒/局 | 速度约翻倍，胜率略低于 40%；可展示速度工程收益，但不应声称稳定超过 40% |

## 失败诊断对照

输局分析来自 `artifacts/report_assets/failure_analysis_summary.json`，基于 500 局内部对局，其中 209 胜、291 负，胜率 41.80%。

| 诊断项 | 观测值 | 含义 |
| --- | ---: | --- |
| 错旗输局占比 | 23.02% | 一旦内存标雷错误，会污染剩余雷数估计和后续 frontier 判断 |
| 终局仍有 solver forced move | 31.62% | 模型在残局仍会漏掉一些确定安全格或确定雷 |
| 终局目标是 solver 可见安全格 | 21.99% | 部分失败不是不可避免猜雷，而是策略没有利用足够的确定信息 |
| 平均目标风险 | 28.75% | 失败动作整体风险偏高 |
| 相对 best guess 平均风险差 | 6.18 个百分点 | 即使需要猜，策略也常没有选到最低风险候选 |

## exact solver 深度对照

同一批 117 个 terminal states 上，提升 exact limit 能发现更多 forced move，并降低 best-guess 风险估计：

| exact limit | forced available | target known mine | target known safe | avg target risk | avg best guess risk |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 35 | 1 | 30 | 28.90% | 25.84% |
| 24 | 41 | 2 | 30 | 27.85% | 22.14% |
| 32 | 45 | 2 | 30 | 27.75% | 21.67% |

这个对照说明：残局并不是单纯靠局部数字就能解决，更多组合约束会暴露出模型没有学透的全局计数结构。因此后续训练应该加强残局课程、错旗长期惩罚和全局剩余雷数建模。

## hard-loss refine 结果

当前 hard-loss refine 的两个 200 局候选结果是 42.00% 和 41.50%，最长连胜分别为 6 和 7。它们没有显著超过内部概率集成的 43.0%，因此报告中不把 hard-loss refine 作为最终最优策略，只把它作为失败诊断和后续优化方向。

## 可放进报告的结论

1. checkpoint 概率集成是当前内部评估最好的纯 RL 推理配置，1000 局达到 43.0%。
2. Windows 执行层已经从主要瓶颈转为可控变量：高速运行达到 27.21 秒/局，并完成 `game_747` 到 `game_756` 的十连胜。
3. 当前主要研究瓶颈是策略本身，尤其是错旗污染、残局 forced move 漏检和最低风险猜测选择。
4. solver 的价值主要在训练监督、baseline 和诊断，不应在最终动作路径中代替 RL 决策。

## 证据入口

- 内部单模型：`artifacts/candidate_eval_200_seed0.json`
- 内部概率集成：`artifacts/ensemble_20_100best_eval_1000_seed0.json`
- logits 集成：`artifacts/ensemble_20_100best_logits_eval_500_seed0.json`
- Windows 单模型：`artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json`
- Windows 高速集成：`artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`
- 失败诊断汇总：`artifacts/report_assets/failure_analysis_summary.json`
- 统计置信区间：`artifacts/report_assets/statistical_summary.md`
