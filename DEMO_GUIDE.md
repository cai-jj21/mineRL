# 项目演示指南

本文档用于面试、答辩或 GitHub 项目展示时快速组织讲述路线。它不替代 `PROJECT_REPORT.md`，而是把报告、架构、评估协议、复现命令和证据文件串成一套可现场演示的流程。胜率、连胜和执行异常的统一口径见 `EVALUATION_PROTOCOL.md`。

## 30 秒开场

可以这样开场：

> 这个项目是一个高级扫雷 `16 x 30 / 99` 强化学习智能体。我自己实现了仿真环境、可见状态编码、Actor-Critic 策略网络、solver-guided 训练信号、Windows 桌面执行层和逐局日志系统。最终验证时 solver 不参与动作选择，动作来自 RL checkpoint 或多个 RL checkpoint 的概率集成；内部 1000 局达到 `43.0%` 胜率，真实 Windows 扫雷记录到完整 `10` 连胜。

## 5 分钟演示路线

建议按这个顺序讲：

1. 打开 `README.md`，先看目标、当前结果和十连胜路径。
2. 打开 `ARCHITECTURE.md`，用架构图解释五层系统：环境、特征、模型、solver、Windows 执行与证据。
3. 打开 `EVALUATION_PROTOCOL.md`，解释内部仿真、Windows 桌面和十连胜的评估口径。
4. 打开 `MODEL_CARD.md`，说明最终策略输入输出、checkpoint、训练信号和限制。
5. 打开 `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`，展示 `longest_streak = 10`、区间 `747-756`、执行异常为 0。
6. 运行证据校验命令，证明报告数字来自本地 JSON。
7. 最后打开 `RESUME_PROJECT_CARD.md`，展示可以放进简历的 bullet 和面试回答。

这条路线的好处是先给结果，再解释系统，再给证据，最后回到简历表达。

## 快速证据命令

这些命令适合现场演示，不需要重新训练模型：

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json
```

预期重点：

- `completed_games = 952`
- `wins = 377`
- `win_rate_completed = 0.3960084033613445`
- `longest_streak = 10`
- `longest_streak_start = 747`
- `longest_streak_end = 756`
- `total_reclicks = 0`
- `total_unconfirmed_open_actions = 0`
- `total_read_recoveries = 0`

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json
```

预期重点：

- `ok = true`
- `25 / 25` 项通过
- 内部集成、Windows 汇总、十连胜区间和报告资产一致。

```powershell
python scripts/validate_documentation.py --expected-tests 147 --check-artifacts
```

预期重点：

- `ok = true`
- 核心文档存在。
- 本地链接有效。
- 文档中的测试数量没有过期。

发布前也可以直接跑总校验：

```powershell
python scripts/validate_release.py --expected-tests 147
```

## 可选现场运行

如果现场有 Windows 扫雷窗口、CUDA 环境和 checkpoint，可以跑小批量桌面验证：

```powershell
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
  --output-dir artifacts\windows_agent\demo_run `
  benchmark --games 10
```

现场运行有随机性，不建议把 10 局结果当作胜率结论。它的用途是展示真实 GUI 执行层、读盘、点击、终局检测和 JSON 日志。

停止命令：

```powershell
python scripts/windows_minesweeper_agent.py stop
```

## 纯 RL 边界怎么证明

演示时重点指出三处证据：

- 运行命令中使用 `--solver-assist none --solver-safety-filter none`。
- `manifest.json` 中记录 `final_decision_mode = rl`、`solver_allowed_during_final_decision = false`。
- `game_*.json` 汇总中 `solver_assist_actions = 0`，说明十连胜动作不是 solver 托管。

可以这样回答：

> solver 是训练 teacher、基线和失败诊断器，不是最终动作执行器。最终 Windows 验证里的动作来自策略网络和 checkpoint 概率集成。

## 十连胜怎么复盘

十连胜证据入口：

```text
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json
...
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_756.json
```

每局 JSON 记录：

- 每一步 `open / flag` 动作。
- 目标行列和屏幕坐标。
- 点击前后的目标格状态。
- 快速单格读数或全盘读盘耗时。
- 是否重复点击、是否未确认打开、是否读盘恢复。
- 终局弹窗和最终棋盘数组。

复盘时可以先讲汇总，再抽一局，例如 `game_747.json`，展示首开、最大展开、末步胜利弹窗和执行层异常为 0。

## 常见追问

**Q: 为什么 Windows 集成 952 局是 `39.60%`，不是稳定超过 40%？**
A: 内部 1000 局集成是 `43.0%`，Windows 桌面运行存在识图、窗口、随机局面和执行分布差异。最新高速桌面批次略低于 40%，但可验证单模型日志 495 完成局达到 `40.40%`，十连胜和执行异常为 0 说明主要瓶颈已经转向模型策略，而不是点击层。

**Q: 这是不是规则 solver 项目？**
A: 不是。solver 用于训练期标注、风险监督、基线和失败分析。最终验证路径使用纯 RL 决策。

**Q: 最大技术难点是什么？**
A: 研究上是让模型吸收局部逻辑和全局剩余雷数约束；工程上是真实 Windows 执行层的读盘、点击和逐步可审计日志；实验上是把内部评估、桌面运行和报告证据做成闭环。

**Q: 失败主要来自哪里？**
A: 失败分析显示错旗会污染剩余雷数估计，残局全局计数弱于 exact solver。下一步应该做残局课程学习、错旗长期惩罚和更强全局上下文建模。

## 不要过度声称

建议避免以下说法：

- 不说“稳定 40%+ Windows 胜率”，因为最新 952 完成局是 `39.60%`。
- 不说“完全超越 solver”，solver baseline 仍是重要上界和诊断参照。
- 不说“十连胜代表真实胜率 100%”，十连胜是连续胜利证据，不是总体胜率。
- 不说“checkpoint 可以直接公开复现”，除非 checkpoint 和 `artifacts/` 已随仓库或 Release 一起归档。

更稳妥的说法：

> 当前项目已形成可复现研究闭环：内部纯 RL 集成达到 `43.0%`，真实 Windows 桌面有 952 完成局、`39.60%` 胜率和完整十连胜日志；执行层异常为 0，后续主要优化方向是模型策略和残局全局推理。
