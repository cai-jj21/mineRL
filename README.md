# Minesweeper RL

面向经典高级扫雷 `16 x 30 / 99` 的强化学习项目。训练阶段可以借助 `solver` 做 teacher、诊断和风险标注，但最终评估与 Windows 桌面执行都走纯 RL 决策链。

## 结果快照

| 场景 | 策略 | 局数 | 胜率 | 备注 |
| --- | --- | ---: | ---: | --- |
| 内部仿真 | `full_rlmix_20 + full_rlmix_100_best` | 1000 | 43.0% | `artifacts/ensemble_20_100best_eval_1000_seed0.json` |
| Windows 桌面 | `full_rlmix_20` | 495 完成局 | 40.40% | `artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json` |
| Windows 桌面 | `full_rlmix_20 + full_rlmix_100_best` | 952 完成局 | 39.60% | `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json` |
| Windows 十连胜 | `full_rlmix_20 + full_rlmix_100_best` | `game_747` - `game_756` | 10 / 10 | `artifacts/report_assets/ten_streak_review.md` |

## 快速开始

```powershell
cd D:\python\mineRL
pip install -e .[dev]
python -m pytest -q
```

文档校验基线：

```powershell
python scripts/validate_documentation.py --expected-tests 160
python scripts/validate_release.py --expected-tests 160
```

安装后可直接使用：

```powershell
minesweeper-rl evaluate --checkpoint artifacts/full_rlmix_20.pt --mode rl --games 200 --batch-size 64 --device cuda
minesweeper-rl watch --checkpoint artifacts/full_rlmix_20.pt --mode rl --speed-ms 250
```

源码模式也可以：

```powershell
python -m minesweeper_rl.cli evaluate --checkpoint artifacts/full_rlmix_20.pt --mode rl --games 200 --batch-size 64 --device cuda
```

Windows 桌面执行见：

```powershell
python scripts/windows_minesweeper_agent.py benchmark --games 10 --output-dir artifacts/windows_agent/demo
```

## 仓库结构

- `src/minesweeper_rl/`：环境、特征、模型、训练器、solver、replay、Windows 复盘。
- `scripts/`：评估、报表、数据仓库、训练反馈、Windows 自动化脚本。
- `artifacts/`：checkpoint、实验结果、SQLite 仓库、图表、逐局 JSON。
- `sql/warehouse_analysis.sql`：数据仓库分析视图与指标模板。
- `tests/`：核心模块、脚本和文档校验测试。

## 数据回路

项目不只是做一个扫雷 agent，也把“数据开发 -> 训练反馈 -> 再训练”串起来了：

1. 收集逐局 `game_*.json`、截图和评估结果。
2. 生成 `artifacts/report_assets/minesweeper_experiments.sqlite`。
3. 通过 `scripts/analyze_experiment_database.py` 和 `sql/warehouse_analysis.sql` 做分层分析。
4. 用 `scripts/generate_training_feedback_plan.py` 提炼 hard-loss、边角局、极端局和猜测局。
5. 用 `scripts/export_training_feedback_dataset.py` 导出定向训练样本，反哺模型。

## 关键文档

### 总览

- `PROJECT_ONE_PAGER.md`
- `PROJECT_REPORT.md`
- `ARCHITECTURE.md`

### 评估与复现

- `EVALUATION_PROTOCOL.md`
- `REPRODUCIBILITY.md`
- `MODEL_CARD.md`
- `ARTIFACTS.md`
- `EXPERIMENT_MANIFEST.md`

### Windows 执行

- `DEMO_GUIDE.md`
- `WINDOWS_AGENT_RESULTS.md`
- `FAILURE_ANALYSIS.md`

### 数据开发

- `DATA_DEVELOPMENT_CASE.md`
- `DATA_ASSET_TRAINING_FEEDBACK.md`
- `PROJECT_COMPLETION_AUDIT.md`
- `RESUME_PROJECT_CARD.md`
- `INTERVIEW_QA.md`

### 消融与发布

- `ABLATION_STUDY.md`
- `RELEASE_CHECKLIST.md`

## 常用脚本

### 报表与校验

- `scripts/generate_report_assets.py`
- `scripts/generate_statistical_report.py`
- `scripts/validate_claims.py`
- `scripts/validate_project_evidence.py`
- `scripts/validate_documentation.py`
- `scripts/validate_release.py`

### 数据仓库

- `scripts/build_experiment_database.py`
- `scripts/analyze_experiment_database.py`
- `scripts/generate_training_feedback_plan.py`
- `scripts/export_training_feedback_dataset.py`
- `sql/warehouse_analysis.sql`

## 说明

- 最终评估路径是 `--solver-assist none --solver-safety-filter none`。
- `solver` 主要用于训练辅助、诊断和失败复盘，不参与最终动作选择。
- 如果你只想先看项目结果，直接打开 `PROJECT_REPORT.md` 和 `WINDOWS_AGENT_RESULTS.md`。
