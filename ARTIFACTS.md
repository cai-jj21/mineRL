# 产物与 checkpoint 说明

`artifacts/`、`checkpoints/`、`runs/` 和模型权重文件默认被 `.gitignore` 排除。原因是这些文件体积较大，且包含本地桌面运行日志、checkpoint、截图或临时实验输出。

## 必要 checkpoint

当前报告中的主结果依赖以下模型文件：

```text
artifacts/full_rlmix_20.pt
artifacts/full_rlmix_100_best.pt
```

如果公开仓库不包含 checkpoint，需要单独归档或通过 GitHub Release、网盘、私有备份等方式保存。没有这些 checkpoint 时，代码、报告和测试仍可阅读和运行，但不能直接复现最终 43.00% 内部集成和 Windows 十连胜结果。

## 关键实验产物

报告和简历卡引用的核心 JSON/日志包括：

```text
artifacts/candidate_eval_200_seed0.json
artifacts/ensemble_20_100best_eval_1000_seed0.json
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json
artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json
...
artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_756.json
artifacts/report_assets/experiment_summary.json
artifacts/report_assets/statistical_summary.json
artifacts/report_assets/statistical_summary.md
artifacts/report_assets/claim_audit.json
artifacts/report_assets/evidence_validation.json
artifacts/report_assets/failure_analysis_summary.json
artifacts/report_assets/ten_streak_review.md
artifacts/report_assets/artifact_manifest.json
```

可提交的索引文件是：

```text
EXPERIMENT_MANIFEST.md
```

它记录关键本地产物的路径、大小和 SHA-256，用于证明报告数字对应的是哪一批实验文件。

## 重新生成报告资产

```powershell
python scripts/generate_report_assets.py `
  --output-dir artifacts/report_assets

python scripts/generate_statistical_report.py

python scripts/validate_claims.py --output artifacts/report_assets/claim_audit.json

python scripts/summarize_failure_analysis.py

python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json

python scripts/build_artifact_manifest.py
```

预期结果：

```text
validate_project_evidence: 25 / 25 passed
build_artifact_manifest: 26 artifacts, missing []
```

## 归档建议

建议把以下内容作为“实验包”单独保存：

- `artifacts/full_rlmix_20.pt`
- `artifacts/full_rlmix_100_best.pt`
- `artifacts/report_assets/`
- `artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json` 到 `game_756.json`
- `EXPERIMENT_MANIFEST.md`
- `FAILURE_ANALYSIS.md`

如果需要压缩归档，归档后应重新保存压缩包哈希；仓库内的单文件 SHA-256 仍用于核对原始产物。

## 不建议提交

默认不要提交：

- `artifacts/`
- `checkpoints/`
- `runs/`
- `*.pt`
- `.vscode/settings.json`

这些文件要么体积较大，要么是个人环境配置。公开仓库保留代码、报告、复现命令、清单和 CI 即可。
