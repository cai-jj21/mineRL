# 发布检查清单

这个清单用于把项目提交到 GitHub、写进简历或做面试展示前的最后检查。

## 代码质量

- [ ] `python -m pytest -q` 通过。
- [ ] `python -m py_compile scripts/build_artifact_manifest.py scripts/validate_project_evidence.py scripts/validate_documentation.py scripts/validate_release.py scripts/generate_report_assets.py scripts/generate_statistical_report.py scripts/summarize_failure_analysis.py scripts/summarize_windows_games.py scripts/validate_claims.py scripts/evaluate_ensemble.py scripts/hard_loss_refine.py scripts/windows_minesweeper_agent.py` 通过。
- [ ] `python scripts/validate_documentation.py --expected-tests 147` 通过。
- [ ] `python scripts/validate_release.py --expected-tests 147` 通过。
- [ ] GitHub Actions `CI` 在 Windows runner 上通过。
- [ ] `git status --short --untracked-files=all` 中没有误提交的个人文件，例如 `.vscode/settings.json`。

## 实验证据

- [ ] `python scripts/generate_report_assets.py --output-dir artifacts\report_assets` 已重新生成报告资产。
- [ ] `python scripts/generate_statistical_report.py` 已重新生成 Wilson 95% 置信区间统计表。
- [ ] `python scripts/validate_claims.py --output artifacts\report_assets\claim_audit.json` 输出 `ok: true`。
- [ ] `python scripts/summarize_failure_analysis.py` 已重新生成失败分析附录。
- [ ] `python scripts/validate_project_evidence.py --output artifacts\report_assets\evidence_validation.json` 输出 `ok: true`。
- [ ] `python scripts/build_artifact_manifest.py` 输出 `missing: []`。
- [ ] `EXPERIMENT_MANIFEST.md` 记录了内部评估、Windows 汇总、报告图表、失败分析和 `game_747.json` 到 `game_756.json` 的 SHA-256。
- [ ] `PROJECT_REPORT.md`、`RESUME_PROJECT_CARD.md`、`WINDOWS_AGENT_RESULTS.md` 中的数字与 `artifacts/report_assets/experiment_summary.md` 一致。

## 可复现性

- [ ] `REPRODUCIBILITY.md` 中的内部评估、Windows 运行、日志汇总、图表生成和证据校验命令仍然可执行。
- [ ] `artifacts/` 因为体积原因不提交到 git，但关键产物已单独归档或保留在本机。
- [ ] `artifacts/full_rlmix_20.pt` 和 `artifacts/full_rlmix_100_best.pt` 已归档；如果公开仓库不包含 checkpoint，需要在 README 中说明获取方式。

## 简历表达

- [ ] README 首屏能直接看到项目目标、核心方法和当前结果。
- [ ] `ARCHITECTURE.md` 能说明环境、模型、solver、Windows 执行层和证据链如何连接。
- [ ] `DEMO_GUIDE.md` 能提供 30 秒开场、5 分钟演示路线、证据命令和回答边界。
- [ ] `EVALUATION_PROTOCOL.md` 能定义内部仿真、Windows 桌面、十连胜和报告口径。
- [ ] `MODEL_CARD.md` 能说明最终模型、输入输出、solver 角色和局限性。
- [ ] `ARTIFACTS.md` 能说明 checkpoint、`artifacts/`、哈希清单和归档策略。
- [ ] `PROJECT_REPORT.md` 能解释 solver 只用于训练/诊断，最终验证为纯 RL 决策。
- [ ] `RESUME_PROJECT_CARD.md` 的 bullet 可以直接复制到简历。
- [ ] 面试时可用 `EXPERIMENT_MANIFEST.md` 和 `artifacts/report_assets/evidence_validation.json` 说明十连胜与胜率数字如何验证。

## 推荐提交内容

建议提交：

- `.github/workflows/ci.yml`
- `README.md`
- `ARCHITECTURE.md`
- `DEMO_GUIDE.md`
- `EVALUATION_PROTOCOL.md`
- `PROJECT_REPORT.md`
- `MODEL_CARD.md`
- `ARTIFACTS.md`
- `REPRODUCIBILITY.md`
- `RESUME_PROJECT_CARD.md`
- `EXPERIMENT_MANIFEST.md`
- `FAILURE_ANALYSIS.md`
- `WINDOWS_AGENT_RESULTS.md`
- `pyproject.toml`
- `src/`
- `scripts/`
- `tests/`

默认不提交：

- `artifacts/`
- `checkpoints/`
- `runs/`
- `.vscode/settings.json`
