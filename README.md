# Minesweeper RL

面向经典高级扫雷 `16 x 30 / 99` 的强化学习项目。目标是在内部仿真环境和 Windows 扫雷桌面程序上训练并验证一个可复现的智能体，使最终动作决策来自神经网络策略，而不是由 solver 直接代打。

系统分层和模块职责见 [ARCHITECTURE.md](ARCHITECTURE.md)。它按仿真环境、状态编码、RL 模型、solver 训练/诊断、Windows 执行层和证据链解释整个项目。

## 项目亮点

- **完整环境建模**：实现高级扫雷环境、延迟布雷、首点安全、打开、插旗、撤旗和 chord 操作。
- **纯可见状态输入**：模型只接收玩家可见棋盘、局部邻域统计、坐标先验和全局剩余雷数估计。
- **Actor-Critic 策略网络**：全卷积残差网络输出 `open / flag / unflag / chord` 四类动作，带 value head 和可选 risk head。
- **solver 作为训练信号和基线**：solver 可提供模仿、风险排序、hard-state 诊断和对照评估；最终 `rl` 推理路径不允许 solver 决策。
- **Windows 执行层**：支持真实 Windows 扫雷窗口读盘、点击、快速单格读数、终局检测和逐步 JSON 日志。
- **可复现实验记录**：已记录桌面 10 连胜、500 局 40%+ 通过结果、1000 局内部模型集成评估。

## 当前结果

| 场景 | 策略 | 局数 | 胜率 | 速度/备注 |
| --- | --- | ---: | ---: | --- |
| 内部仿真 | `full_rlmix_20 + full_rlmix_100_best` 概率集成 | 1000 | 43.0% | 纯 RL 决策，最长 9 连胜 |
| Windows 桌面 | `full_rlmix_20` 单模型 | 495 完成局 | 40.40% | 平均 55.83 秒/局，目标通过 |
| Windows 桌面 | `full_rlmix_20 + full_rlmix_100_best` 集成 | 952 完成局 | 39.60% | 平均 27.21 秒/局，出现 10 连胜 |

最新十连胜出现在：

```text
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json
through
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_756.json
```

十局均为 `won: true`、`done: true`，无重复点击、无未确认打开、无读盘恢复。

## 快速开始

安装依赖：

```powershell
pip install -e .[dev]
```

内部仿真评估单模型：

```powershell
python -m minesweeper_rl.cli evaluate `
  --checkpoint artifacts/full_rlmix_20.pt `
  --mode rl `
  --games 200 `
  --batch-size 64 `
  --device cuda `
  --decision-actions full `
  --inference-flips `
  --inference-ensemble probs
```

内部仿真评估 RL 集成：

```powershell
python scripts/evaluate_ensemble.py `
  --checkpoint artifacts/full_rlmix_20.pt `
  --ensemble-checkpoint artifacts/full_rlmix_100_best.pt `
  --games 1000 `
  --batch-size 64 `
  --device cuda `
  --output artifacts/ensemble_20_100best_eval_1000_seed0.json
```

复盘 Windows 逐局日志：

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json `
  --streak-report-output artifacts/report_assets/ten_streak_review.md
```

生成报告表格和 SVG 图表：

```powershell
python scripts/generate_report_assets.py `
  --output-dir artifacts/report_assets
```

生成 Wilson 95% 置信区间统计表：

```powershell
python scripts/generate_statistical_report.py
```

校验 README、报告和简历卡中的结果声明是否过期或过度声称：

```powershell
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
```

生成输局与 solver 对照失败分析附录：

```powershell
python scripts/summarize_failure_analysis.py
```

校验报告数字和本地实验 JSON 是否一致：

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json
```

校验核心文档入口和本地链接：

```powershell
python scripts/validate_documentation.py --expected-tests 147
```

生成可提交的实验产物哈希清单：

```powershell
python scripts/build_artifact_manifest.py
```

发布前总校验：

```powershell
python scripts/validate_release.py --expected-tests 147
```

## Windows 扫雷运行命令

```powershell
cd D:\python\mineRL

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py clear-stop

C:\Users\88911\anaconda3\python.exe scripts\windows_minesweeper_agent.py `
  --checkpoint artifacts\full_rlmix_20.pt `
  --ensemble-checkpoint artifacts\full_rlmix_100_best.pt `
  --device cuda `
  --capture-backend auto `
  --read-mode fast `
  --click-method mouse_event `
  --solver-assist none `
  --solver-safety-filter none `
  --quick-number-read `
  --center-first-open `
  --start-mode new `
  --flag-mode memory `
  --speed-profile custom `
  --inference-flips `
  --inference-ensemble probs `
  --action-delay 0 `
  --capture-delay 0.0005 `
  --stable-reads 1 `
  --stable-read-delay 0.0005 `
  --click-hold 0.02 `
  --cursor-settle 0.004 `
  --post-click-settle 0.014 `
  --click-confirm-retries 2 `
  --no-progress-reclicks 0 `
  --max-steps 600 `
  --stall-limit 20 `
  --record-frames final `
  --no-final-images `
  --output-dir artifacts\windows_agent\pure_rl_ensemble_20_100best_1000 `
  benchmark --games 1000
```

停止正在运行的桌面 agent：

```powershell
python scripts/windows_minesweeper_agent.py stop
```

## 代码结构

- `src/minesweeper_rl/game.py`：扫雷环境、奖励和胜负逻辑。
- `src/minesweeper_rl/features.py`：可见状态编码、合法动作掩码和全局特征。
- `src/minesweeper_rl/model.py`：残差 Actor-Critic 网络。
- `src/minesweeper_rl/trainer.py`：训练、评估、solver 辅助标签、checkpoint。
- `src/minesweeper_rl/solver.py`：可见信息 solver、约束枚举、全局剩余雷数概率。
- `scripts/windows_minesweeper_agent.py`：真实 Windows 扫雷读盘与执行层。
- `scripts/evaluate_ensemble.py`：纯 RL checkpoint 概率集成评估。
- `scripts/summarize_windows_games.py`：逐局 Windows JSON 结果汇总。
- `scripts/generate_report_assets.py`：从实验 JSON 生成报告表格和 SVG 图表。
- `scripts/generate_statistical_report.py`：从实验 JSON 生成 Wilson 95% 置信区间统计表。
- `scripts/validate_claims.py`：校验报告和简历卡中的结果声明是否过期或过度声称。
- `scripts/summarize_failure_analysis.py`：从输局/solver 对照 JSON 生成失败分析附录。
- `scripts/validate_project_evidence.py`：校验报告数字、十连胜区间和执行层异常。
- `scripts/validate_documentation.py`：校验核心 Markdown 入口、本地链接和测试数量。
- `scripts/validate_release.py`：发布前一键运行脚本编译、测试、文档、证据和产物清单校验。
- `scripts/build_artifact_manifest.py`：生成关键实验产物的 SHA-256 清单。
- `scripts/hard_loss_refine.py`：输局/残局 hard-state 挖掘与微调实验。
- `WINDOWS_AGENT_RESULTS.md`：桌面实验结果记录。
- `ARCHITECTURE.md`：系统分层、训练/执行链路、solver 边界和证据链总览。
- `DEMO_GUIDE.md`：面试、答辩或 GitHub 展示时的演示路线、证据命令和回答边界。
- `ABLATION_STUDY.md`：单模型、checkpoint 集成、Windows 执行层、失败诊断和 exact solver 深度的对照证据。
- `EVALUATION_PROTOCOL.md`：内部仿真、Windows 桌面、十连胜和报告口径的统一评估协议。
- `PROJECT_REPORT.md`：科研项目报告。
- `MODEL_CARD.md`：最终模型/集成策略的输入、输出、训练信号、评估结果和局限性。
- `ARTIFACTS.md`：checkpoint、实验产物、哈希清单和归档策略说明。
- `FAILURE_ANALYSIS.md`：输局、错旗、残局和 solver 对照分析附录。
- `REPRODUCIBILITY.md`：复现实验、报告资产和证据校验流程。
- `EXPERIMENT_MANIFEST.md`：可提交的实验产物指标和哈希索引。
- `RELEASE_CHECKLIST.md`：GitHub 发布、归档和简历展示前检查清单。
- `.github/workflows/ci.yml`：Windows + Python 3.12 自动测试流程。

## 报告与简历

一页式项目总览见 [PROJECT_ONE_PAGER.md](PROJECT_ONE_PAGER.md)，适合 GitHub 首页浏览、面试开场或投递材料快速预览。

完整项目报告见 [PROJECT_REPORT.md](PROJECT_REPORT.md)。报告包含问题定义、方法、训练信号、实验设置、十连胜复盘、局限性和简历表述建议。

架构总览见 [ARCHITECTURE.md](ARCHITECTURE.md)，用于快速理解环境、模型、solver、Windows 执行层和证据链如何连接。

演示指南见 [DEMO_GUIDE.md](DEMO_GUIDE.md)，包含 30 秒开场、5 分钟演示路线、可运行证据命令和常见追问回答。

消融与对照研究见 [ABLATION_STUDY.md](ABLATION_STUDY.md)，用于回答“哪些设计有效、证据强弱如何、下一步为什么做残局训练”。

评估协议见 [EVALUATION_PROTOCOL.md](EVALUATION_PROTOCOL.md)，定义胜率、连胜、速度、执行异常、样本量和报告边界。

模型卡见 [MODEL_CARD.md](MODEL_CARD.md)，说明最终 RL 集成策略、输入输出、训练信号、评估结果和限制。

一页式简历项目卡见 [RESUME_PROJECT_CARD.md](RESUME_PROJECT_CARD.md)，包含可直接放进简历的 bullet、面试讲述结构和常见追问回答。

报告图表资产位于 `artifacts/report_assets/`，包括 `experiment_summary.md`、`statistical_summary.md`、`claim_audit.json`、`win_rate_comparison.svg`、`longest_streak_comparison.svg`、`ten_streak_times.svg` 和 `ten_streak_review.md`。

checkpoint 和大体积实验产物说明见 [ARTIFACTS.md](ARTIFACTS.md)。

复现实验和证据校验流程见 [REPRODUCIBILITY.md](REPRODUCIBILITY.md)。

实验产物哈希清单见 [EXPERIMENT_MANIFEST.md](EXPERIMENT_MANIFEST.md)。

提交 GitHub 或用于简历展示前，可按 [RELEASE_CHECKLIST.md](RELEASE_CHECKLIST.md) 做最后检查。

## 测试

```powershell
python -m pytest -q
```

当前 `147` 项测试覆盖环境、训练器、solver、Windows agent 参数与执行逻辑、实验汇总、统计置信区间、报告资产、失败分析、证据校验、声明校验、文档校验、发布校验和产物清单。
