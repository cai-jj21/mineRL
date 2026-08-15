# 实验产物清单

该清单记录项目报告所依赖的本地实验产物、核心指标和 SHA-256 校验值。`artifacts/` 默认被 git 忽略，因此此文件用于保留可提交的证据索引。

## 核心指标

| 证据 | 指标 |
| --- | --- |
| 单模型内部评估 | 200 局，胜率 41.50%，最长 7 连胜 |
| 内部 RL 集成 | 1000 局，430 胜，胜率 43.00%，最长 9 连胜 |
| Windows 桌面集成 | 952 完成局，377 胜，胜率 39.60%，平均 27.21 秒 |
| 十连胜区间 | game_747 到 game_756，10 / 10 胜 |
| 证据校验 | 25 / 25 通过，失败 0 项 |
| 失败分析 | 500 局对照，291 输局，错旗输局 23.02%，终局 forced move 31.62% |

## 产物哈希

| Path | Size | SHA-256 |
| --- | ---: | --- |
| `artifacts/candidate_eval_200_seed0.json` | 8383 | `74471aff6ffccce0bbe6a9e59a337bee6375b77cda99ec2250ba6da95689b01a` |
| `artifacts/ensemble_20_100best_eval_1000_seed0.json` | 469 | `aa24017e851123948412485483d4cb2313204e2858cad2dfa463298c152fedf7` |
| `artifacts/windows_agent/pure_rl_fast2_500/per_game_summary.json` | 794 | `f93cc866c617b64aa0ea4be85310270ac0d12532f1307ce44a08e4791e4b2ea6` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json` | 13956 | `8e754ba0aeadbd6c910d2198aedc315fdb06b0ce8e8d0af554c4517bb969ffab` |
| `artifacts/report_assets/experiment_summary.md` | 695 | `29a955542688837a0bae3ecd17d8b1933e2f068bb94978b3e7dd0ae3023b3da3` |
| `artifacts/report_assets/experiment_summary.json` | 12690 | `d3a05549a0c0940da1fb93d01a3ea88b22f1f1e1b143b70f6bd5cef6ee025ffa` |
| `artifacts/report_assets/evidence_validation.json` | 6900 | `32982dff06b300be7a5c909b9204c7227a1d46f5edea60075cf031d758cc21a3` |
| `artifacts/report_assets/win_rate_comparison.svg` | 2235 | `0818335c7a3966b7f10def53a7a661c0362e9c0dd0b12ad8073fc58548fc7b48` |
| `artifacts/report_assets/longest_streak_comparison.svg` | 2219 | `ccd57a755e67a6bcd8429348c78f4741a0ef353f354044df0277b56bc9d04692` |
| `artifacts/report_assets/ten_streak_times.svg` | 3519 | `25d4aa5964d6f449383bc8a984dbbb7cd981136159ff9bfbe834499aa4e45c36` |
| `artifacts/report_assets/ten_streak_review.md` | 2086 | `29f9bc01c12d5dc0146c9d51ec01a976cfb8bf8b525fd319d945f9fc654d25a2` |
| `artifacts/report_assets/statistical_summary.json` | 2313 | `6467a10b1271ba1d3001d6b3d4ded908b4ebfaa14ca2801c3d3aba1a8a8d0284` |
| `artifacts/report_assets/statistical_summary.md` | 1213 | `8ce68c21b23957e2fb594fdf854106e2a7ac2eb6a35fc22142cffe9f7e0d634c` |
| `artifacts/report_assets/claim_audit.json` | 19149 | `053ea06460dbce22d2f8e24e72fed2a37d23cecc0421f8826190f1d30d64741d` |
| `artifacts/report_assets/failure_analysis_summary.json` | 7603 | `9ad7adc44114117e495bf68e10b6122675d15ef021ed378a4f867c81934929ab` |
| `FAILURE_ANALYSIS.md` | 3446 | `6ee06eee323829d0bd602b11d79de8ba0c292445635d738fcba7ad4d5d5544a4` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json` | 775234 | `81cd79343bd69ef93b7d98aaa8db2dc02fe03895595e1caeaac233d5700ef415` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_748.json` | 706223 | `2621796252967b91de9eae29ff75ebdb6bd7b1d64832185ca411402be31aab37` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_749.json` | 675891 | `16daa85824304ffef926a4cf56b7eedc6110649dc21b1036300649d367c007be` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_750.json` | 819278 | `2e597634d83d4b74377fb5a42a9c63e99471cb8770b7f64ccf995a34e4781852` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_751.json` | 696327 | `ee582ef8657edb57460c0a0217c6dfef5c727423e07b2a975bc8482028b701a5` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_752.json` | 697149 | `1822d8dd648e70a1be675a753c6f84f0a8ccd9719dd51dccefc88a675cedf623` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_753.json` | 747133 | `198b1ef09554825f395c82ab130d5b61138a6d3a01c252fa3b5b4af0fcd61288` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_754.json` | 807410 | `8060d190468ee6c75b6197c4823c9846726bb38cd38f98cfeee2da40b53e93f2` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_755.json` | 768453 | `f135299c7b357c11405c63d4b036d601f7943021006a10ff9635e2abb2ed64bd` |
| `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_756.json` | 642078 | `bc2c095537d23a80f1d3e25cb4aad2fbfcbd2c25432bbd36d187e28623e9c460` |

重新生成：

```powershell
python scripts/build_artifact_manifest.py
```
