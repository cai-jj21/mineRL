# 高级扫雷强化学习智能体项目报告

## 摘要

本项目研究经典高级扫雷 `16 x 30 / 99` 场景下的强化学习决策问题。项目从零实现扫雷环境、可见状态编码、Actor-Critic 策略网络、可见信息 solver 基线、训练与评估流程，并进一步接入真实 Windows 扫雷窗口，实现读盘、点击、终局检测和逐步日志记录。

当前最好的内部仿真结果为两个纯 RL checkpoint 的概率集成：1000 局胜率 `43.0%`，最长 `9` 连胜。真实 Windows 桌面执行中，可验证单模型日志为 495 个完成局 `40.40%` 胜率；最新集成版本在 952 个完成局中达到 `39.60%`，平均 `27.21` 秒/局，并成功记录完整 `10` 连胜。

项目定位不是传统规则 solver，而是“solver 辅助训练、RL 独立决策”的强化学习系统。最终验证路径中 solver 不参与动作选择。

最终模型/集成策略的模型卡见 `MODEL_CARD.md`，消融与对照证据见 `ABLATION_STUDY.md`，评估口径见 `EVALUATION_PROTOCOL.md`，checkpoint 与本地实验产物说明见 `ARTIFACTS.md`，系统分层和证据链总览见 `ARCHITECTURE.md`。

## 研究目标

高级扫雷有三个困难点：

1. **局部逻辑与全局雷数耦合**：许多局面不能只靠局部数字判断，需要结合剩余雷数和边界组件概率。
2. **动作空间稀疏且风险高**：一次错误 open 即失败，flag/unflag/chord 又会改变后续可见状态。
3. **真实桌面执行误差**：Windows 扫雷的高亮、缩放、窗口焦点、点击确认和读盘误差都会放大策略错误。

项目目标：

- 在高级图 `16 x 30 / 99` 上实现可运行 RL agent。
- 最终决策路径只由模型策略输出，solver 不直接代打。
- 达到可展示的胜率、速度和连续胜利记录。
- 形成可复现的实验日志与简历级科研项目报告。

## 系统设计

### 环境

环境实现位于 `src/minesweeper_rl/game.py`，包含：

- 延迟布雷和首点安全半径。
- `open / flag / unflag / chord` 四类动作。
- 安全格展开、踩雷失败、全安全格揭示胜利。
- 奖励设计：打开安全格小奖励、胜利大奖励、踩雷大惩罚、每步轻微惩罚。

### 状态编码

状态编码位于 `src/minesweeper_rl/features.py`。模型只接收可见信息：

- hidden mask 和 flagged mask。
- `0` 到 `8` 的已揭示数字通道。
- frontier、邻近 flag 数、邻近 hidden 数、邻近 revealed 数。
- chord-ready mask。
- 行列坐标、边缘距离、中心先验。
- 全局特征：步数比例、已揭示比例、flag 比例、hidden 比例、剩余雷数估计、covered 比例。

输入通道数为 `20`，全局特征数为 `6`。

### 策略网络

模型位于 `src/minesweeper_rl/model.py`。核心结构：

- 全卷积 residual backbone。
- policy head 输出四类动作的棋盘 logits。
- value head 估计状态价值。
- risk head 用于训练期风险监督和实验性推理偏置。

动作空间为：

```text
4 x 16 x 30 = 1920
```

其中四类动作是 `open / flag / unflag / chord`。

### Solver 的角色

Solver 位于 `src/minesweeper_rl/solver.py`，使用可见棋盘约束、局部集合推理、组件枚举和全局剩余雷数概率估计。

Solver 用途：

- 训练期 imitation / DAgger 标签。
- 风险监督和 mine/risk 辅助标签。
- 内部基线评估。
- 输局复盘，判断模型是否错过 forced move 或残局计数。

最终 `mode=rl` 或 Windows `--solver-assist none --solver-safety-filter none` 时，solver 不参与动作决策。

## 训练与评估流程

### 训练信号

训练器位于 `src/minesweeper_rl/trainer.py`。核心信号包括：

- 强化学习回报：胜负、开格收益和步数惩罚。
- solver imitation loss：训练期对 solver 可证明动作和低风险猜测进行约束。
- DAgger replay：收集模型访问到的状态，再用 solver 重新标注。
- mine auxiliary loss：学习真实雷分布的辅助监督。
- risk supervision loss：学习 solver 风险图上的低风险排序。

重要边界：这些信号只用于训练或诊断，最终评估的动作来自模型策略。

### 推理增强

当前效果最好的推理组合：

- 棋盘翻转增强：对垂直/水平翻转做 test-time augmentation。
- 概率集成：对多个 checkpoint 的 masked policy probability 取平均。
- 中心首开：统一第一步为 `row=8, col=15`，降低首步随机性。

内部集成脚本：

```powershell
python scripts/evaluate_ensemble.py `
  --checkpoint artifacts/full_rlmix_20.pt `
  --ensemble-checkpoint artifacts/full_rlmix_100_best.pt `
  --games 1000 `
  --batch-size 64 `
  --device cuda
```

## 真实 Windows 执行层

桌面 agent 位于 `scripts/windows_minesweeper_agent.py`。它完成：

- 捕获 Windows 扫雷棋盘。
- 自动检测网格、识别数字、空格、未开格和旗子。
- 将模型动作映射到屏幕坐标。
- 使用 `mouse_event` 点击，并记录鼠标按下/抬起、焦点窗口和目标格。
- 点击后移出棋盘，避免鼠标悬停高亮影响识别。
- 支持快速单格读数 `--quick-number-read`。
- 检测胜利/失败弹窗并自动下一局。
- 每局保存逐步 JSON 日志。

执行层结果可通过逐局汇总脚本复现：

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json `
  --streak-report-output artifacts/report_assets/ten_streak_review.md
```

## 实验结果

### 内部仿真

| 策略 | 局数 | 胜率 | 最长连胜 | 说明 |
| --- | ---: | ---: | ---: | --- |
| `full_rlmix_20.pt` | 200 | 约 41% 到 42% | 约 7 | 单模型主力 checkpoint |
| `full_rlmix_100_best.pt` | 200 | 约 41% | 约 6 | 单模型候选 |
| `full_rlmix_20 + full_rlmix_100_best` | 1000 | 43.0% | 9 | 纯 RL 概率集成 |
| exact32 solver baseline | 200 | 约 44% 到 46% | 约 5 | 仅作为上界参考 |

内部集成结果保存在：

```text
artifacts/ensemble_20_100best_eval_1000_seed0.json
```

### Windows 桌面

| 运行 | 局数 | 胜率 | 平均耗时 | 备注 |
| --- | ---: | ---: | ---: | --- |
| `pure_rl_fast2_500` | 495 完成局 | 40.40% | 55.83 秒 | 单模型，目标通过 |
| `pure_rl_ensemble_20_100best_1000` | 952 完成局 | 39.60% | 27.21 秒 | 集成，高速，出现 10 连胜 |

桌面执行层信号：

- `total_reclicks = 0`
- `total_click_unready_actions = 0`
- `total_unconfirmed_open_actions = 0`
- `total_read_recoveries = 0`

这说明最新 952 个完成局中，主要瓶颈已从执行层转移到模型策略本身。

### 报告资产

可用以下命令从实验 JSON 重新生成报告表格和 SVG 图表：

```powershell
python scripts/generate_report_assets.py `
  --output-dir artifacts/report_assets
```

统计置信区间可用以下命令生成：

```powershell
python scripts/generate_statistical_report.py
```

报告声明可用以下命令校验，防止旧数字或过度胜率表述重新进入文档：

```powershell
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
```

输出文件：

- `artifacts/report_assets/experiment_summary.md`
- `artifacts/report_assets/experiment_summary.json`
- `artifacts/report_assets/statistical_summary.md`
- `artifacts/report_assets/statistical_summary.json`
- `artifacts/report_assets/claim_audit.json`
- `artifacts/report_assets/win_rate_comparison.svg`
- `artifacts/report_assets/longest_streak_comparison.svg`
- `artifacts/report_assets/ten_streak_times.svg`
- `artifacts/report_assets/ten_streak_review.md`

报告数字可用以下命令校验：

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json
```

当前本地校验结果为 `25 / 25` 项通过，覆盖内部仿真、Windows 桌面汇总、十连胜区间、执行层异常和报告资产一致性。

由于 `artifacts/` 默认不提交到 git，可用以下命令生成可提交的产物哈希索引：

```powershell
python scripts/build_artifact_manifest.py
```

产物索引见 `EXPERIMENT_MANIFEST.md`，记录关键 JSON、SVG 和十连胜逐局日志的 SHA-256。

## 十连胜复盘

十连胜区间：

```text
game_747.json ~ game_756.json
```

汇总：

| 指标 | 数值 |
| --- | ---: |
| 胜利局数 | 10 / 10 |
| 平均步数 | 291.9 |
| 平均实际 open 动作 | 193.3 |
| 平均耗时 | 30.09 秒 |
| 平均速度 | 9.71 actions/s |
| 最快胜局 | 26.14 秒 |
| 最慢胜局 | 33.62 秒 |
| 重复点击 | 0 |
| 未确认打开 | 0 |
| 读盘恢复 | 0 |

十局均以中心首开 `row=8, col=15` 开始，终局弹窗均为 `游戏胜利`，每局最终揭示 `380` 个安全格。

## 失败分析

对 `full_rlmix_20.pt` 的输局前态进行 solver 对照后，发现两个主要问题：

详细附录见 `FAILURE_ANALYSIS.md`，可由以下命令重新生成：

```powershell
python scripts/summarize_failure_analysis.py
```

500 局 batched 输局分析显示：291 个输局中，67 局存在错旗，92 局在终局仍有 solver 可见 forced move；模型终局目标平均风险为 `0.287`，高于 solver 最优可见猜测风险 `0.226`。

1. **错旗污染剩余雷数估计**
   当模型插错旗后，`remaining_mines_estimate = mine_count - flags` 会偏离真实局面。残局时模型可能把真雷当成“剩余雷数已为 0 后的安全格”。

2. **残局全局计数弱于 solver**
   在无错旗的输局里，solver 仍能在部分状态中推出 forced safe / forced mine，说明模型对全局组件组合和剩余雷数约束的表达仍不足。

尝试过 hard-loss refine：

- 挖掘模型自己的输局尾部状态。
- 使用 exact32 solver 对错旗、forced move、末盘状态重新标注。
- 小规模微调后未稳定超过原 checkpoint，说明直接强灌输局尾部会破坏原策略分布。

当前更稳的策略是 checkpoint 概率集成，而不是继续硬蒸馏。

## 贡献与工程价值

项目完整覆盖了从研究到落地的链条：

- 自研扫雷环境与高级图配置。
- 强化学习训练、专家辅助标签、风险监督和 checkpoint 评估。
- 可见信息 solver 基线与失败诊断。
- Windows 桌面读盘和执行层自动化。
- 逐局 JSON 日志、连胜证明和可复现实验命令。
- 报告数字、图表资产和本地实验 JSON 的自动一致性校验。
- GitHub Actions CI、实验产物哈希清单和发布检查流程。
- 测试覆盖环境、solver、训练器和 Windows agent 行为。

当前本地测试基线为 `147` 项通过，覆盖代码行为、Windows agent 逻辑、实验汇总、统计置信区间、报告资产、失败分析、证据校验、声明校验、文档校验、发布校验和产物清单。

## 局限性

- 最新 952 局桌面集成胜率为 `39.60%`，略低于 40%。
- 内部仿真和真实桌面之间存在分布差异。
- 当前模型结构对全局剩余雷数组合推理的表达仍有限。
- checkpoint 集成提升了稳定性，但增加了推理成本。
- hard-loss refine 还没有形成稳定增益，需要更谨慎的离线数据混合与 RL 约束。

## 后续计划

1. **残局课程学习**：按 `safe_left <= 100 / 60 / 30 / 10` 分段训练，逐步强化全局计数。
2. **错旗惩罚重构**：提高错旗长期代价，降低末盘 remaining-mine 误导。
3. **分布保持微调**：hard-state replay 与原始成功轨迹混合，避免策略坍缩。
4. **更强全局模型**：引入轻量 attention 或全局组件摘要特征，提高长程约束表达。
5. **桌面/仿真对齐**：记录真实桌面状态序列并回放进仿真评估，定位分布偏差。
6. **报告自动化**：将 `summarize_windows_games.py` 输出接入固定实验表格。

## 简历表述建议

可写为：

> 独立实现高级扫雷强化学习智能体，覆盖环境建模、Actor-Critic 策略网络、solver 辅助训练、真实 Windows 执行层和实验日志系统；在 `16 x 30 / 99` 高级图上实现纯 RL 决策，内部 1000 局达到 `43.0%` 胜率，真实 Windows 扫雷完成 `10` 连胜，单局平均约 `27` 秒。

更工程化版本：

> 构建可复现扫雷 RL 研究平台：实现 1920 维动作空间策略网络、可见信息约束 solver、DAgger/risk 辅助训练、TTA checkpoint ensemble 与 Windows GUI 自动化；通过逐局 JSON 日志验证点击/读盘稳定性，完成 952 局桌面评估和 10 连胜复盘。

更研究化版本：

> 研究高风险稀疏奖励博弈中的 RL 决策与逻辑先验融合，在扫雷高级图中使用 solver 作为训练期 teacher 和诊断器，比较纯 RL、solver baseline、模型集成和 hard-loss refinement，分析错旗导致的全局剩余雷数偏差与残局组合推理不足。

## 复现入口

- 项目入口：[README.md](README.md)
- 架构总览：[ARCHITECTURE.md](ARCHITECTURE.md)
- 演示指南：[DEMO_GUIDE.md](DEMO_GUIDE.md)
- 消融与对照研究：[ABLATION_STUDY.md](ABLATION_STUDY.md)
- 评估协议：[EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)
- 简历项目卡：[RESUME_PROJECT_CARD.md](RESUME_PROJECT_CARD.md)
- Windows 结果记录：[WINDOWS_AGENT_RESULTS.md](WINDOWS_AGENT_RESULTS.md)
- 内部集成评估：`scripts/evaluate_ensemble.py`
- Windows 逐局汇总：`scripts/summarize_windows_games.py`
- 桌面执行器：`scripts/windows_minesweeper_agent.py`
