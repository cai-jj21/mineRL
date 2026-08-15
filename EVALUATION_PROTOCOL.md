# 评估协议

本文档定义项目中的评估口径、指标、通过门槛和结果解释方式。复现命令见 `REPRODUCIBILITY.md`，模型输入输出见 `MODEL_CARD.md`，演示路线见 `DEMO_GUIDE.md`。

## 评估对象

核心任务固定为经典高级扫雷：

- 棋盘：`16 x 30`
- 雷数：`99`
- 安全格：`381`
- 动作空间：`4 x 16 x 30 = 1920`
- 可见状态：只允许玩家可见棋盘和由可见棋盘派生的特征。

最终评估不读取真实雷图，不读取 Windows 扫雷内部内存，不允许 solver 直接代打。

## 决策模式

项目中有三类容易混淆的模式：

| 模式 | 用途 | 是否用于最终结果 |
| --- | --- | --- |
| `rl` | 神经网络策略直接选择动作 | 是 |
| checkpoint ensemble | 多个 RL checkpoint 的 masked policy probability 平均 | 是 |
| solver baseline / solver assist | 训练标签、诊断、上界参照、局部辅助实验 | 否 |

最终 Windows 命令必须包含：

```text
--solver-assist none --solver-safety-filter none
```

十连胜和桌面胜率报告中，solver 不参与动作选择。

## 指标定义

主要指标：

- `games`：请求或内部仿真完成的局数。
- `terminal_games`：Windows 桌面中真正出现胜负结算的局数。
- `completed_games`：汇总脚本认定已完成且可计入胜率的局数。
- `wins` / `losses`：胜利和失败局数。
- `win_rate_completed`：`wins / completed_games`。
- `longest_streak`：按逐局日志顺序重算的最长连续胜利数。
- `avg_elapsed_seconds`：每个完成局的平均用时。
- `actions_per_second`：桌面执行层每秒动作数。
- `execution anomalies`：重复点击、未确认打开、读盘恢复、目标偏移等执行层异常。

报告胜率时优先使用 `completed_games` 或 `terminal_games`，不把未结算局混入胜率分母。

## 通过门槛

当前项目的展示门槛是：

| 场景 | 门槛 |
| --- | --- |
| 内部单模型 | 至少 `200` 局，胜率 `40%+`，最长连胜至少 `7` |
| 内部 RL 集成 | 至少 `1000` 局，胜率 `43.0%` 附近，最长连胜至少 `9` |
| Windows 桌面 | 至少 `900` 个终局，胜率 `39%+`，最长连胜至少 `10` |
| Windows 执行层 | 十连胜区间重复点击、未确认打开、读盘恢复为 `0` |
| 速度 | Windows 平均耗时不超过 `35` 秒/局 |
| 证据链 | `scripts/validate_project_evidence.py` 输出 `25 / 25` 通过 |

这里把 Windows 桌面门槛写成 `39%+`，是为了诚实反映最新 952 完成局为 `39.60%`。内部仿真和历史桌面单模型证明模型具备 40% 量级能力，但最新高速桌面集成不应被表述成“稳定 40%+”。

## 当前主结果

| 实验 | 胜负 | 胜率 | Wilson 95% 区间 | 说明 |
| --- | ---: | ---: | ---: | --- |
| Internal RL ensemble | `430 / 1000` | `43.00%` | `39.96% - 46.09%` | 纯 RL checkpoint 概率集成 |
| Windows desktop ensemble | `377 / 952` | `39.60%` | `36.54% - 42.74%` | 高速真实桌面运行，含 10 连胜 |
| Internal single RL | `83 / 200` | `41.50%` | `34.89% - 48.43%` | 单模型候选评估 |
| Windows single model | `200 / 495` | `40.40%` | `36.17% - 44.78%` | 可验证逐局 JSON 桌面批次 |

Wilson 区间由 `scripts/generate_statistical_report.py` 生成，用于提醒读者：胜率是随机变量，单批次结果不能被解读成精确常数。项目报告保留单点结果，同时用样本量、逐局日志和证据校验降低偶然性。

## 内部仿真协议

内部评估用于模型选择和策略比较：

1. 固定高级图 `16 x 30 / 99`。
2. 使用 `mode=rl` 或 checkpoint ensemble。
3. 关闭 solver 决策，只允许模型策略输出动作。
4. 首选 `1000` 局做最终集成评估；小样本 `200` 局只能作为候选筛选。
5. 记录 `games`、`wins`、`win_rate`、`longest_streak`。
6. 将最终 JSON 保存到 `artifacts/`，并纳入报告资产生成和证据校验。

内部仿真不包含屏幕识别、窗口焦点、鼠标点击和真实 GUI 结算，因此不能单独证明桌面可用性。

## Windows 桌面协议

Windows 桌面评估用于验证真实执行：

1. 打开 Windows 扫雷高级局面，保证窗口前台可截图和点击。
2. 使用 `scripts/windows_minesweeper_agent.py`。
3. 最终结果命令包含 `--solver-assist none --solver-safety-filter none`。
4. 每局保存 `game_*.json`，记录动作、点击、读盘和终局。
5. 使用 `scripts/summarize_windows_games.py` 重算胜率、最长连胜和异常信号。
6. 十连胜区间必须能从逐局 JSON 重算，而不是只看终端输出。

Windows 桌面结果需要同时看胜率、速度和执行异常。若胜率下降但异常为 0，优先怀疑模型策略或分布差异，而不是点击层。

## 十连胜协议

十连胜必须满足：

- 连续 `10` 个逐局 JSON 均为 `won: true`。
- 每局均为 `done: true` 或有胜利终局弹窗。
- 汇总脚本输出 `longest_streak = 10`。
- 区间起止与报告一致：`game_747.json` 到 `game_756.json`。
- 十连胜区间 `reclicks = 0`、`unconfirmed_open_actions = 0`、`read_recoveries = 0`。

十连胜证明模型和执行层能连续完成高质量桌面局面，但不等价于总体胜率 100%。

## 报告口径

推荐表述：

- “内部纯 RL 集成 1000 局达到 `43.0%`。”
- “真实 Windows 桌面最新高速集成 952 完成局达到 `39.60%`，并记录完整十连胜。”
- “可验证 Windows 单模型日志 495 完成局达到 `40.40%`。”
- “最终验证路径中 solver 不参与动作选择。”
- “执行层异常在十连胜区间和汇总中为 0，后续主要优化方向是模型策略和残局全局推理。”

避免表述：

- “稳定超过 40% Windows 胜率。”
- “已经超越 solver。”
- “十连胜证明胜率 100%。”
- “没有 checkpoint 也能完整复现最终桌面结果。”

## 证据入口

- `artifacts/ensemble_20_100best_eval_1000_seed0.json`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/per_game_summary.json`
- `artifacts/windows_agent/pure_rl_ensemble_20_100best_1000/game_747.json` 到 `game_756.json`
- `artifacts/report_assets/experiment_summary.json`
- `artifacts/report_assets/evidence_validation.json`
- `EXPERIMENT_MANIFEST.md`

对应校验命令：

```powershell
python scripts/validate_project_evidence.py `
  --output artifacts/report_assets/evidence_validation.json

python scripts/validate_documentation.py --expected-tests 147 --check-artifacts

python scripts/build_artifact_manifest.py

python scripts/validate_release.py --expected-tests 147
```
