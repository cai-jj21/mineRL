# 复现实验与证据校验

本文档用于复现项目报告中的核心数字，并校验报告资产是否与本地实验 JSON 一致。系统分层见 `ARCHITECTURE.md`，数据开发链路见 `DATA_DEVELOPMENT_CASE.md`，演示流程见 `DEMO_GUIDE.md`，评估口径见 `EVALUATION_PROTOCOL.md`，发布检查见 `RELEASE_CHECKLIST.md`，产物归档策略见 `ARTIFACTS.md`。

## 环境准备

```powershell
cd D:\python\mineRL
pip install -e .[dev]
python -m pytest -q
```

当前测试基线为 `160` 项测试，预期输出包含 `160 passed`。

## 核心复现命令

```powershell
python scripts/validate_project_evidence.py --output artifacts/report_assets/evidence_validation.json
python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json
python scripts/generate_statistical_report.py
python scripts/summarize_failure_analysis.py
python scripts/build_artifact_manifest.py
python scripts/validate_release.py --expected-tests 160
```

内部仿真和 Windows 桌面运行命令会读取本地 `artifacts/` checkpoint 与日志。`.github/workflows/ci.yml` 负责在 GitHub Actions 上执行基础 CI。

## 数据仓库与训练反馈

```powershell
python scripts/build_experiment_database.py
python scripts/analyze_experiment_database.py
python scripts/generate_training_feedback_plan.py
python scripts/export_training_feedback_dataset.py
```

数据库产物为 `artifacts/report_assets/minesweeper_experiments.sqlite`，分析 SQL 位于 `sql/warehouse_analysis.sql`。证据校验目标为 `25 / 25` 通过，当前 artifact manifest 清单覆盖 `27` 个关键产物。
