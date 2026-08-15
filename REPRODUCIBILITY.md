# 复现实验与证据校验

本文档用于复现项目报告中的核心数字，并校验报告资产是否与本地实验 JSON 一致。`artifacts/` 默认不进入 git，因此这些命令面向本地实验目录或单独归档的实验产物。

系统分层和模块职责见 `ARCHITECTURE.md`。评估口径和通过门槛见 `EVALUATION_PROTOCOL.md`。面试、答辩或 GitHub 展示路线见 `DEMO_GUIDE.md`。模型身份、输入输出和局限性见 `MODEL_CARD.md`。checkpoint 与大体积产物的归档策略见 `ARTIFACTS.md`。

## 环境准备

```powershell
cd D:\python\mineRL
pip install -e .[dev]
python -m pytest -q
```

当前代码测试基线：

```text
147 passed
```

## 内部仿真评估

单模型评估：

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

集成模型评估：

```powershell
python scripts/evaluate_ensemble.py `
  --checkpoint artifacts/full_rlmix_20.pt `
  --ensemble-checkpoint artifacts/full_rlmix_100_best.pt `
  --games 1000 `
  --batch-size 64 `
  --device cuda `
  --output artifacts/ensemble_20_100best_eval_1000_seed0.json
```

当前报告采用的内部集成证据：

```text
artifacts/ensemble_20_100best_eval_1000_seed0.json
games: 1000
wins: 430
win_rate: 43.00%
longest_streak: 9
```

## Windows 桌面评估

运行前需要打开 Windows 扫雷高级局面，并保证窗口在前台可被截图和点击。先清理停止标记：

```powershell
python scripts/windows_minesweeper_agent.py clear-stop
```

当前高速纯 RL 集成评估命令：

```powershell
python scripts/windows_minesweeper_agent.py `
  --checkpoint artifacts/full_rlmix_20.pt `
  --ensemble-checkpoint artifacts/full_rlmix_100_best.pt `
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

## 汇总逐局日志

```powershell
python scripts/summarize_windows_games.py `
  artifacts/windows_agent/pure_rl_ensemble_20_100best_1000 `
  --streak-start 747 `
  --streak-end 756 `
  --output artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json
```

当前桌面证据：

```text
terminal_games: 952
wins: 377
win_rate_completed: 39.60%
longest_streak: 10
longest_streak_start: 747
longest_streak_end: 756
avg_elapsed_seconds: 27.21
total_reclicks: 0
total_unconfirmed_open_actions: 0
total_read_recoveries: 0
```

## 生成报告资产

```powershell
python scripts/generate_report_assets.py `
  --output-dir artifacts/report_assets
```

生成 Wilson 95% 置信区间统计表：

```powershell
python scripts/generate_statistical_report.py
```

校验报告和简历卡中的结果声明：

```powershell
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
```

输出：

```text
artifacts/report_assets/experiment_summary.md
artifacts/report_assets/experiment_summary.json
artifacts/report_assets/win_rate_comparison.svg
artifacts/report_assets/longest_streak_comparison.svg
artifacts/report_assets/ten_streak_times.svg
```

## 生成失败分析附录

```powershell
python scripts/summarize_failure_analysis.py
```

输出：

```text
FAILURE_ANALYSIS.md
artifacts/report_assets/failure_analysis_summary.json
```

当前失败分析基于 `500` 局内部输局 solver 对照，记录 `291` 个输局；其中 `23.02%` 的输局存在错旗，`31.62%` 的终局仍有 solver 可见 forced move。

## 校验证据链

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json
```

该脚本会检查：

- 单模型内部评估是否达到 `200` 局和 `40%+`。
- 内部集成是否达到 `1000` 局、`430` 胜、`43.0%` 和 `9` 连胜。
- Windows 桌面汇总是否达到 `900+` 终局、`39%+`、`10` 连胜和 `35` 秒以内平均耗时。
- 十连胜区间是否为 `game_747` 到 `game_756`，且十局均胜利。
- 桌面执行异常是否为 `0`。
- `artifacts/report_assets/experiment_summary.json` 是否与源 JSON 指标一致。

当前本地校验结果：

```text
checks: 25
passed: 25
failed: 0
```

## 校验文档

```powershell
python scripts/validate_documentation.py --expected-tests 147
```

该脚本检查核心 Markdown 是否存在、本地链接是否有效、关键证据入口是否被 README/报告/简历卡引用，以及测试数量声明是否过期。

## 发布级总校验

发布、归档或面试展示前，可以用一条命令运行脚本编译、全量测试、文档校验、证据校验和产物清单：

```powershell
python scripts/validate_release.py --expected-tests 147
```

如果只想在没有本地 `artifacts/` 的环境中快速检查代码和文档，可以跳过本地实验产物：

```powershell
python scripts/validate_release.py `
  --expected-tests 147 `
  --skip-claims `
  --skip-evidence `
  --skip-manifest `
  --skip-artifact-links
```

## 生成产物清单

因为 `artifacts/` 默认被 git 忽略，建议生成一份可提交的实验产物清单，记录关键 JSON、SVG 和十连胜逐局文件的 SHA-256：

```powershell
python scripts/build_artifact_manifest.py
```

输出：

```text
EXPERIMENT_MANIFEST.md
artifacts/report_assets/artifact_manifest.json
```

当前清单覆盖 `26` 个关键产物，包括内部评估 JSON、Windows 汇总 JSON、报告资产、失败分析、证据校验结果，以及 `game_747.json` 到 `game_756.json` 十个逐局日志。

## CI 与发布检查

GitHub Actions 工作流位于：

```text
.github/workflows/ci.yml
```

该工作流在 Windows + Python 3.12 上安装 `.[dev]`，编译关键脚本，并执行：

```powershell
python -m pytest -q
```

提交或展示前的检查清单见：

```text
RELEASE_CHECKLIST.md
```

## 归档建议

用于简历或面试展示时，建议保留以下证据：

- `PROJECT_REPORT.md`
- `ARCHITECTURE.md`
- `DEMO_GUIDE.md`
- `EVALUATION_PROTOCOL.md`
- `MODEL_CARD.md`
- `ARTIFACTS.md`
- `FAILURE_ANALYSIS.md`
- `RESUME_PROJECT_CARD.md`
- `WINDOWS_AGENT_RESULTS.md`
- `REPRODUCIBILITY.md`
- `RELEASE_CHECKLIST.md`
- `EXPERIMENT_MANIFEST.md`
- `artifacts/report_assets/`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json` 到 `game_756.json`
