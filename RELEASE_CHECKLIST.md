# 发布检查清单

这个清单用于提交 GitHub、写简历或做面试展示前的最后检查。

## 代码质量

- [ ] `python -m pytest -q` 通过，当前基线为 `160`。
- [ ] `python scripts/validate_documentation.py --expected-tests 160 --check-artifacts` 通过。
- [ ] `python scripts/validate_release.py --expected-tests 160` 通过。
- [ ] `scripts/validate_claims.py`、`scripts/build_experiment_database.py`、`scripts/analyze_experiment_database.py`、`scripts/generate_training_feedback_plan.py`、`scripts/export_training_feedback_dataset.py`、`sql/warehouse_analysis.sql` 与报告口径一致。

## 文档入口

- [ ] `ARCHITECTURE.md`
- [ ] `ABLATION_STUDY.md`
- [ ] `PROJECT_COMPLETION_AUDIT.md`
- [ ] `DEMO_GUIDE.md`
- [ ] `INTERVIEW_QA.md`
- [ ] `DATA_DEVELOPMENT_CASE.md`
- [ ] `EVALUATION_PROTOCOL.md`
- [ ] `MODEL_CARD.md`
- [ ] `ARTIFACTS.md`
- [ ] `EXPERIMENT_MANIFEST.md`
- [ ] `FAILURE_ANALYSIS.md`
- [ ] `PROJECT_REPORT.md`
- [ ] `REPRODUCIBILITY.md`
- [ ] `RESUME_PROJECT_CARD.md`

## 提交边界

- [ ] 提交代码、文档、`src/`、`scripts/`、`tests/`、`sql/` 和 `.github/workflows/ci.yml`。
- [ ] 不提交 `artifacts/`、`checkpoints/`、`runs/`、模型权重和个人配置。
