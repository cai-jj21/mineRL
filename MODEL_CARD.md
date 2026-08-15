# 模型卡

## 模型身份

本项目当前主力策略是高级扫雷 `16 x 30 / 99` 上的纯 RL checkpoint 集成：

- 主 checkpoint：`artifacts/full_rlmix_20.pt`
- 集成 checkpoint：`artifacts/full_rlmix_100_best.pt`
- 推理方式：masked policy probability ensemble
- 最终决策：RL policy 输出动作；Windows 验证命令使用 `--solver-assist none --solver-safety-filter none`

## 任务

模型面对经典高级扫雷局面，基于玩家可见信息选择下一步动作。目标是在不读取隐藏雷分布、不让 solver 直接代打的前提下，提升胜率和真实桌面执行稳定性。

棋盘配置：

```text
rows: 16
cols: 30
mines: 99
safe_radius: 1
```

动作空间：

```text
4 x 16 x 30 = 1920
```

动作类型：

- `open`
- `flag`
- `unflag`
- `chord`

## 输入

模型只接收可见状态特征：

- hidden / flagged mask
- 已揭示数字 `0` 到 `8`
- frontier、邻近 flag、邻近 hidden、邻近 revealed
- chord-ready mask
- 行列坐标、边缘距离、中心先验
- 全局特征：步数比例、已揭示比例、flag 比例、hidden 比例、剩余雷数估计、covered 比例

不输入真实雷图，不读取 Windows 扫雷内部内存。

## 训练信号

训练过程使用 solver 作为 teacher 和诊断器，而不是最终代理：

- imitation / DAgger replay
- mine auxiliary loss
- risk supervision loss
- solver risk ranking
- hard-state / loss-state diagnosis

最终验证路径中，solver 不参与动作选择；solver 只用于训练期标签、基线评估和失败分析。

## 结果

| 场景 | 局数 | 胜率 | 最长连胜 | 备注 |
| --- | ---: | ---: | ---: | --- |
| 内部单模型 `full_rlmix_20.pt` | 200 | 41.50% | 7 | 主力单模型 |
| 内部 RL 集成 | 1000 | 43.00% | 9 | 纯 RL 概率集成 |
| Windows 桌面单模型 | 495 完成局 | 40.40% | 7 | 平均 55.83 秒/局 |
| Windows 桌面集成 | 952 完成局 | 39.60% | 10 | 平均 27.21 秒/局 |

Windows 十连胜区间：

```text
game_747.json ~ game_756.json
```

该区间十局均 `won: true`、`done: true`，重复点击、未确认打开和读盘恢复均为 `0`。

## 已知局限

- 可验证 Windows 桌面单模型日志为 `40.40%`，最新高速桌面集成胜率为 `39.60%`，略低于 40%。
- 模型会受到错旗影响，导致剩余雷数估计被污染。
- 残局中全局雷数约束和组件组合推理仍弱于 exact solver。
- checkpoint 集成提高稳定性，但增加推理成本。
- 真实 Windows 执行依赖窗口缩放、前台焦点和本地输入权限。

评估协议见 `EVALUATION_PROTOCOL.md`，失败分析见 `FAILURE_ANALYSIS.md`。

## 适用场景

- 强化学习项目展示
- 部分可观测高风险决策研究
- solver-guided RL 与真实 GUI 自动执行工程案例
- 简历/面试中的完整科研项目证据链

## 不适用场景

- 作为通用扫雷最优 solver
- 在未验证的扫雷客户端或不同 UI 缩放下直接运行
- 无 checkpoint 文件时复现最终 Windows 胜率

## 复现入口

- 项目报告：`PROJECT_REPORT.md`
- 评估协议：`EVALUATION_PROTOCOL.md`
- 复现实验：`REPRODUCIBILITY.md`
- 失败分析：`FAILURE_ANALYSIS.md`
- 产物说明：`ARTIFACTS.md`
- 产物哈希：`EXPERIMENT_MANIFEST.md`
