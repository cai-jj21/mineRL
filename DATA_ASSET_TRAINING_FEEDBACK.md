# 数据资产反哺训练记录

本文记录 2026-08-29 这一轮围绕“极端局面数据资产推动 RL 决策提升”的工作。目标不是简单堆局数，而是把模型容易输的固定短板局面抽出来，给候选 open 格做反事实估值，再用于定向训练和门控评估。

## 当前基线

当前仍建议保留的稳定配置：

- Base ensemble: `artifacts/full_rlmix_20.pt` + `artifacts/full_rlmix_100_best.pt`
- Specialist: `artifacts/full_rlmix_guess_specialist_cf1995_riskhead_strong_last.pt`
- Gate: `safe_left_threshold=60`, `specialist_weight=0.1`, `blend_scope=open-disagree-margin`, `specialist_margin=0.05`, `specialist_risk_head_weight=0.05`
- 已记录结果：`artifacts/report_assets/gated_cf1995_riskhead_strong_20_100best_60_w01_risk005_200.json`
- 200 局 seed0：89/200，44.5%

## 新增数据资产

本轮重点构建了 current-best failure dataset，即用当前最好策略自己跑局，专门从失败局的尾盘、边角、猜测状态里抽样。

| 数据集 | 记录数 | 候选标签 | 负候选率 | 平均行为 regret | 用途 |
| --- | ---: | ---: | ---: | ---: | --- |
| `model_failure_guess_modeltopk_cf_160.npz` | 88 | 1954 | 29.5% | 3.38 | 基础模型失败猜测样本 |
| `gated_best_failure_modeltopk_cf_256.npz` | 260 | 6438 | 28.3% | 3.60 | 当前最好策略失败样本 |
| `modeltopk_plus_gated_failure_cf_348.npz` | 348 | 8392 | 28.6% | 3.55 | 合并训练集 |
| `currentbest_plain_guess_cf_101.npz` | 101 | 3459 | 23.6% | 6.98 | 普通 guess 专项 |
| `gated_plain_guess_modeltopk_cf_1024.npz` | 142 | 5023 | 24.8% | 7.64 | 当前策略普通 guess 新增样本 |
| `plain_guess_modeltopk_cf_243.npz` | 243 | 8482 | 24.3% | 7.36 | 普通 guess 合并训练集 |
| `modeltopk_plus_gated_failure_cf348_riskaware.npz` | 348 | 8392 | 28.6% | 0.88 | solver-risk-first 重标注集 |
| `sim_plain_guess_solveronly_cf_32.npz` | 17 | 136 | 11.8% | 3.21 | solver 轨迹普通 guess 补充样本 |
| `gated_live_guess_margin05_64.npz` | 61 | 871 | 25.1% | 3.15 | current-best live hard guess 快采样 |

最重要的诊断信号：

- 在 `gated_best_failure_modeltopk_cf_256` 上，base ensemble 选中候选里约 42.7% 是雷。
- 普通 `guess` family 的平均 regret 很高，说明模型不是只在边角/尾盘弱，非尾盘内部猜测也有明显短板。
- 边角/尾盘 value head 离线更容易学，普通 guess 的最优候选排序更难。
- 只从失败尾部抽样会漏掉胜局里的猜对状态。本轮新增 live guess 采样，能在对局过程中直接保存“solver 无强制解、模型正在 open 猜”的状态。
- live guess 探针显示，真 guess 很稀疏：64 局里通常只得到 10-56 条中盘普通 guess；大多数候选状态其实仍有 forced move，因此采样端必须先过滤 forced 状态。
- 加入 `current-interior` 与 policy margin 过滤后，采样更精准但记录更少：`policy_margin<=0.02` 的 64 局探针得到 10 条，`policy_margin<=0.005` 只得到 3 条，说明普通 guess 数据需要更专门的局面生成器，而不是只靠自然对局低频撞出来。
- solver 轨迹采样经过优化后，不再为 forced 状态构造再丢弃 transition；32 局 smoke 得到 17 条纯 `guess` 记录，用时约 53.5 秒。它证明定向采样链路可用，但规模还远远不够训练出稳定线上收益。
- `gated_live_guess_margin05_64` 进一步对齐 current-best 执行分布：64 局得到 61 条 selected guess-family 记录，其中 live 真实 guess 只有 11 条；`live_guess_forced_skips=2379`，说明大量“看起来在猜”的状态其实 solver 仍有 forced move。
- 采样脚本已经支持按最终筛选条件提前停止，后续大规模挖数据不会再因为达标后继续完整跑完而浪费时间。

## 代码能力

本轮新增/增强了这些可复用能力：

- `scripts/build_model_failure_extreme_dataset.py`
  - 支持 `--counterfactual-model-topk`
  - 支持 `--family-filter`
- `scripts/build_gated_failure_extreme_dataset.py`
  - 支持从 gated current-best policy 挖失败样本
  - 支持 `--family-filter`
  - 支持 `--risk-head-weight`、`--base-risk-head-weight`、`--specialist-risk-head-weight`，让数据采样与线上门控配置保持一致
  - 支持 `--collect-guess-states`，从胜负局过程中直接采集真实 guess 状态
  - 支持 `--collect-guess-safe-left-min/max`、`--collect-guess-max-per-game`
  - 支持 `--collect-guess-prefilter exact/basic/none`，用于降低 live 采样的 solver 标注成本
  - 支持 `--collect-guess-policy-margin-max` 和 `--collect-guess-region`，可以定向采集内部格/边缘格/高不确定性 guess
  - 支持按最终输出筛选条件统计 `selected_like_transitions` 并提前结束采样
- `scripts/build_sim_guess_training_dataset.py`
  - 支持 `--counterfactual-model-topk`，把当前策略 top-k 候选纳入反事实评估
  - 支持 `--policy-candidate-checkpoint`，可以用一个或多个模型共同提供候选 open
  - 支持 `--max-guesses-per-game`、`--collect-safe-left-min/max`、`--family-filter`，用于生成更高密度的专题 guess 数据
- `scripts/merge_extreme_datasets.py`
  - 支持 `--family-filter`，可以从大资产里切专题数据集
- `scripts/evaluate_gated_specialist.py`
  - 支持 `--counterfactual-value-region-gate`
  - 支持 `--counterfactual-value-min-safe-left`
  - 支持 `--counterfactual-value-policy-margin-max`，只在当前策略 OPEN top1/top2 分差较小时启用 value rerank
  - 记录 `counterfactual_value_changed_decisions`，区分“value head 被启用”和“最终点击真的被改变”
  - 记录 `candidate_changed_decisions`，区分“candidate calibrator 参与打分”和“最终 open 选择真的被改变”
  - 可以评估只在边角、内部格、指定 safe-left 区间启用 value rerank

这些改动让数据资产可以按 `guess`、`guess_tail`、`edge_guess_tail`、`corner_guess_tail` 等 family 定向生产和训练。

## 训练和评估结论

### Risk head 增量

训练产物：

- `artifacts/full_rlmix_currentbest_riskhead_pluscf_last.pt`

结果：

- 离线排序几乎没有明显改善。
- 同 seed 对照后，没有证据表明它优于当前稳定 risk-head specialist。
- 不建议采纳为最终策略。

### Current-best value head

训练产物：

- `artifacts/full_rlmix_currentbest_valuehead_cf348_last.pt`

离线效果：

- `modeltopk_plus_gated_failure_cf_348` 上 best-match 提高到 15.5%，near-best 30.7%。
- 但负候选率仍有 31.3%，避雷性不够稳定。

线上结果：

- `gated_currentbest_valuehead_cf348_w002_d050_k12_200.json`: 89/200，44.5%，与当前基线持平。
- 边角门控版本为 88/200，未提升。

结论：可保留为研究产物，但暂不替换当前基线。

### Plain guess value head

训练产物：

- `artifacts/full_rlmix_plain_guess_valuehead_last.pt`
- `artifacts/full_rlmix_plain_guess_valuehead_cf243_last.pt`

离线效果：

- 在 `currentbest_plain_guess_cf_101` 上，负候选率降到 17.8%，平均 regret 从约 6.98 降到 5.05。
- 但在边角/尾盘 family 上明显变差。
- 扩大到 `plain_guess_modeltopk_cf_243` 后，cf243 value head 在训练集上的 best-match 只有 2.5%，near-best 7.8%，平均 regret 5.36；虽然比原 specialist 的 6.52 regret 好，但仍不足以可靠改线上动作。

线上结果：

- 使用 `safe_left 141..381` + `both-interior` 门控，100 局 seed0/seed1000 都与对应基线持平。
- 50 局诊断中，value head 介入 408 次，但只改变最终动作 57 次，说明轻量 rerank 大量时候仍被 base policy 分数压住。
- cf243 value head 接入同样门控后，100 局 seed0 仍为 34/100，与当前同 seed baseline 持平。

结论：说明普通 guess 专项数据有信号，但当前 value-head rerank 还没有转成净胜率。

### Risk-aware value head

训练产物：

- `artifacts/full_rlmix_riskaware_valuehead_cf348_last.pt`

离线效果：

- `modeltopk_plus_gated_failure_cf_348` 被重标注为 risk-aware 目标：solver 可见风险优先，真实踩雷为硬负样本，展开收益只作为小权重 tie-breaker。
- 在 risk-aware 标签集上，当前 specialist 的 near-best 为 21.6%，chosen-negative 为 38.5%，平均 regret 为 0.765。
- 新 risk-aware value head 的 near-best 提高到 31.3%，chosen-negative 降到 23.3%，平均 regret 降到 0.486。
- 改善主要集中在 `corner_guess_tail` 和 `edge_guess_tail`；普通 `guess` 仍弱。

线上结果：

- 宽门控 `safe_left<=140`、`weight=0.10`：100 局 seed0 为 33/100，低于 baseline。
- 加 policy 不确定性门 `policy_margin_max=0.005`：100 局 seed0 为 34/100，持平。
- 再加边界门 `either-edge-or-corner`：100 局 seed0 为 34/100，持平。

结论：risk-aware 标签是更稳的离线数据资产，但当前 value-head 还不能默认进入最终策略；后续应把它用于训练更强的 candidate ranker 或 policy head。

### Risk-aware policy ranker

训练产物：

- `artifacts/full_rlmix_riskaware_policy_cf348_kl_last.pt`
- `artifacts/full_rlmix_riskaware_policy_cf348_tinykl_last.pt`

结果：

- 直接用 risk-aware 标签训练 policy head，即使冻结 backbone 并加 teacher KL，也很容易破坏整体策略。
- 较强版本：240 次 update 后，单模型 50 局从 28% 掉到 2%。
- 极轻版本：80 次 update、较小学习率和更强 teacher KL 后，单模型 50 局从 30% 掉到 26%。

结论：直接改 policy head 的收益/风险比不好，暂不接入线上；更合适的路线是训练独立 candidate-ranker，再通过严格门控给主策略小幅加分。

### Candidate calibrator

训练产物：

- `artifacts/candidate_riskaware_plus_sim_plain_guess.json`

训练数据：

- `modeltopk_plus_gated_failure_cf348_riskaware.npz`
- `sim_plain_guess_solveronly_cf_32.npz`

结果：

- 训练集共 365 个局面、8528 个候选标签；目标正样本率约 71.6%，负候选率约 28.4%。
- 50 局快筛中，`candidate_weight=0.02/0.05/0.10/0.20` 都没有超过当前同 seed 基线。
- `candidate_weight=0.05` 跑 100 局 seed0 为 33/100；candidate calibrator 介入 8360 次，但没有形成胜率收益。

结论：候选级数据链路已经打通，但线性 calibrator 表达力不够；下一步应改成小型神经网络 ranker，或者让它只在更窄的真实 guess 门控中介入。

### Guess-specialist softmax ranker

训练产物：

- `artifacts/candidate_guess_specialist_softmax.json`
- `artifacts/candidate_guess_specialist_softmax_live64.json`

训练数据：

- `currentbest_plain_guess_cf_101.npz`
- `sim_guess_counterfactual_128.npz`
- `sim_plain_guess_solveronly_cf_32.npz`

结果：

- 530 个 transition、14765 个候选标签。
- 离线 best_match 提高到 17.0%，near-best 到 36.6%，平均 regret 降到 2.53。
- 线上 50 局 seed0 仍停在 15/50，candidate_weight=0.05 时 candidate_changed_decisions=782。
- 把权重降到 `0.02` 后，50 局变成 14/50，candidate_changed_decisions=347。
- 再加 `candidate_policy_margin_max=0.005`，50 局仍是 15/50，candidate_changed_decisions=592。
- 用 `adjustment_gate=solver-guess` 做诊断时，`weight=0.05` 只有 7 次真实改动作，50 局仍是 15/50；强接管 `weight=1.0/topk=32` 改了 99 次动作，反而降到 14/50。
- 并入 `gated_live_guess_margin05_64` 后，softmax ranker 离线 best_match 反而从 17.0% 降到 12.9%，chosen-negative 升到 24.5%，说明新样本需要分 family/阶段加权，而不是直接混入。

结论：softmax ranker 的离线排序比回归版更像样，但还没转成胜率收益，说明真正瓶颈仍在数据分布和门控范围，而不只是损失函数。

### Headonly policy micro-refine

训练产物：

- `artifacts/full_rlmix_guess_headonly_cfpolicy_live64_last.pt`

结果：

- 从历史 500 局最高的 `full_rlmix_guess_specialist_headonly.pt` 出发，冻结 backbone，只训练 policy head。
- 使用 `gated_live_guess_margin05_64`、强 teacher KL、轻量 counterfactual policy loss 做 80 次更新。
- 脚本内单模型 20 局从 35% 降到 30%；接回 gated ensemble 后 50 局为 15/50，没有收益。

结论：直接微调 policy head 仍然容易破坏原策略，即使只用 61 条 hard guess 小数据和强 KL 也没有改善。

### Ensemble reduction 复核

结果：

- `headonly` 原配置当前代码复跑：200 局 seed0 为 87/200，43.5%，与历史 200 局一致。
- 将 base ensemble reduction 从 `mean` 改为 `geomean` 后，200 局 seed0 为 85/200，42.5%。

结论：当前 base/specialist 分数融合仍使用 `mean`；geomean 不采纳。

### Base 权重验证

验证目的：排除当前 0.5/0.5 base ensemble 不是最优权重的可能。

- `w20=1.0, w100=0.0`：200 局 seed0 为 83/200，41.5%。
- `w20=0.75, w100=0.25`：200 局 seed0 为 87/200，43.5%。
- `w20=0.60, w100=0.40`：200 局 seed0 为 87/200，43.5%。
- 当前 0.5/0.5 稳定配置：200 局 seed0 为 89/200，44.5%。

结论：当前 base ensemble 权重暂不调整。

## 当前判断

当前最可靠版本仍是 `cf1995_riskhead_strong` 配置，实测 200 局 44.5%，500 局历史最好约 44.4%。今天新增的数据资产和训练工具已经把短板定位得更清楚：

- 后期猜测、边角尾盘可以通过 value/risk 小头学到一部分。
- 真正拉开 45% 以上胜率的瓶颈更可能是普通 guess 排序，即 solver 没有强制安全格、但候选 open 的长期价值差别很大时，模型还不能稳定挑中高价值候选。
- 只训练小头不一定足够，下一步应该让 policy head 或一个更强的 candidate model 直接学习候选排序，而不是只把 value head 当很轻的线上扰动。
- 宽门控 value-head 会误伤大量并非真 guess 的局面；门控要么来自更好的模型不确定性特征，要么训练一个专门判断“当前是否该使用 guess-ranker”的 gate。
- 直接 fine-tune 主 policy head 目前不稳，容易把已有 40%+ 的策略能力冲坏。

## 下一步建议

1. 扩大 `guess` family 专项数据，目标 500-1000 条记录，而不是继续混合边角/尾盘。
2. 为普通 guess 建局面生成器或采样器，主动制造 `safe_left > 140`、内部 open、多候选低 margin 的局面，提高数据密度。
3. 训练独立 candidate-ranker 专项模型，直接优化候选 open 排序，但不直接覆盖主 policy。
4. 每个候选策略先跑两个 100 局 seed 快筛，再跑 200 局确认，只有超过 45% 才进入 500 局验证。
5. 当前生产/Windows 侧仍使用稳定基线，不把今天的 value-head 实验默认接入。

## 2026-09-02 all-open 候选排序复核

本轮进一步把 `guess` 状态的反事实标签扩展到所有合法 OPEN 候选，并训练了两个独立的 candidate ranker：

- 数据集：`artifacts/report_assets/extreme_training_dataset/sim_guess_currentstrong_allopen_320.npz`
- 248 条 transition，53,479 个候选标签
- 负反事实候选率 22.38%
- 平均行为 regret 3.06，说明当前策略在部分猜测状态确实存在可学习的候选差异
- MLP：`artifacts/candidate_currentstrong_allopen_mlp.json`
- Softmax：`artifacts/candidate_currentstrong_allopen_softmax.json`

线上同 seed 对照：

| 配置 | 胜率 | candidate 调整次数 | 真正改变 OPEN 选择 |
| --- | ---: | ---: | ---: |
| all-open MLP，原始全局门控 | 4/50 = 8% | 3,645 | 1,165 |
| all-open MLP，修复分数尺度后 | 13/50 = 26% | 6,457 | 1,839 |
| all-open MLP，`solver-guess` 门控 | 14/50 = 28% | 24 | 4 |
| all-open Softmax，`solver-guess` 门控 | 14/50 = 28% | 24 | 6 |

本轮还修复了两个线上接入问题：

1. candidate 概率原本是“OPEN 通道内归一化”，却直接和“全动作空间 policy 分数”混合，导致 MLP 大面积放大候选信号；现在先映射回 OPEN 通道的原始概率尺度。
2. candidate 原本复用 specialist 的 `safe-left` 门控，无法区分 forced move 和真实 guess；现在支持独立的 `--candidate-adjustment-gate solver-guess`，solver 只负责识别“没有 forced move”的状态，最终动作仍由 RL 分数选择。

代码验证：

- `tests/test_gated_specialist.py` 与 `tests/test_candidate_calibration.py` 共 23 项通过。
- `scripts/evaluate_gated_specialist.py` 已通过编译检查。

阶段性结论：

- all-open 数据资产已经能发现候选价值差异，但当前 ranker 尚不能稳定超过 `cf1995_riskhead_strong` 基线的 44.5%。
- 直接让 candidate 在所有低 margin 状态介入会破坏主策略；必须限制到真实 guess 状态。
- 当前 248 条 transition 仍不足以支撑 45% 目标，尤其是普通内部 guess 的覆盖不够，`corner_guess_tail`/`edge_guess_tail` 与普通 `guess` 的分布也不能简单混训。
- 下一轮应优先定向生成 500-1000 条真实 `guess` transition，按 `safe_left`、边界区域、候选数量和 policy margin 分层采样，再做 family-balanced candidate ranker。

## 2026-09-02 普通内部 guess 小批

为了验证“不要只堆局数，而是抽短板局面”的路线，本轮单独采样普通内部 `guess`，即：

- solver 没有 forced move
- 当前动作是 OPEN
- 非边缘、非角落
- `safe_left > 140`

新增数据集：

- `artifacts/report_assets/extreme_training_dataset/sim_guess_currentstrong_plain_96_allopen.npz`
- 请求 64 局，完成 56 局
- 扫到 170 个真实 guess 状态
- 最终选中普通内部 `guess` 41 条
- 17,674 个 all-open 候选标签
- 负候选率 21.52%
- 平均 best counterfactual value 7.47
- 平均 behavior counterfactual value 1.56
- 平均 behavior regret 5.90
- 普通内部 guess 命中率约 24.12%

这批数据的诊断价值很高：它的平均行为 regret 明显高于混合 all-open 数据集的 3.06，说明模型在普通内部猜测时确实存在“候选差异很大但没挑中”的短板。但它也暴露了采样效率问题：自然对局里普通内部 guess 很稀疏，64 局只拿到 41 条，后续必须用可续跑/分批落盘的采样管道扩大规模。

本轮为采样脚本补了数据工程能力：

- `scripts/build_sim_guess_training_dataset.py` 支持 `--save-every-records`
- 支持 `--partial-output`
- 中断时如果已有样本，会自动保存 partial dataset
- manifest 新增 `scanned_guess_states`、`selected_records`、`selected_to_scanned_guess_rate`

专项 ranker：

- `artifacts/candidate_currentstrong_plain_guess_allopen_softmax.json`
- 训练数据：原 all-open 数据中的 75 条 `guess` + 新增 41 条普通内部 `guess`
- 共 116 条 transition、48,708 个候选标签
- 离线 best_match 18.97%
- near-best 37.93%
- chosen-negative 18.10%
- avg_regret 4.31

线上 50 局 seed0，使用 `--candidate-adjustment-gate solver-guess`：

- 胜率 14/50 = 28%
- solver_guess_decisions 186
- candidate_adjusted_decisions 25
- candidate_changed_decisions 9

低温度版本：

- `artifacts/candidate_currentstrong_plain_guess_allopen_softmax_sharp.json`
- 离线 best_match 15.52%，near-best 36.21%，chosen-negative 13.79%
- 50 局 seed0 为 16/50 = 32%，同口径基线为 14/50
- 扩大到 200 局 seed0 后为 87/200 = 43.5%，低于当前基线 89/200 = 44.5%
- 200 局中 solver_guess_decisions 665，candidate_adjusted_decisions 92，candidate_changed_decisions 23

结论：

- 普通内部 guess 数据确实更“尖”，更适合反哺训练。
- 当前新增 41 条还太少，只能证明方向，不足以拉动线上胜率。
- `solver-guess` 门控是必要的：它把 ranker 影响限制在真实猜测状态，避免破坏 forced move。
- 下一步不应继续微调主 policy，也不应扩大 candidate 介入范围；应该持续分批采样普通内部 guess，目标至少 500 条，再按 `safe_left` 和候选数量分层训练。

## 2026-09-02 candidate gate 复核

这轮我把“是否采纳 candidate”的门控也单独做了出来，训练目标是：

- 只在 candidate 的 counterfactual 收益明显超过当前策略时才放行
- 让模型真正只做决策，不靠更强的门控去掩盖排序短板

实验结果很明确：

- 只用 348 条失败回放训练 gate，200 局线上等价仿真是 `72/200 = 36%`
- 再混入 all-open 和普通 guess 数据后，50 局结果直接掉到 `12/50 = 24%`
- 固定 guess benchmark 上，candidate 的单步 regret 确实下降了，但没转化成整局胜率提升

结论：

- candidate gate 不是当前瓶颈的解法
- 现在最该补的是普通内部 guess 的短板样本，尤其是：
  - `safe_left > 140`
  - 边角 / 边缘 / 内部三类分层
  - solver 没有 forced move 的真实 guess
  - 候选数多、policy margin 小、行为 regret 高的状态

下一步建议直接做一个“短板采样批次”：

1. 以 `--trajectory-mode policy` 为主，持续采普通内部 guess
2. 再补 edge/corner guess_tail，单独成类，不要和普通 guess 混得太厉害
3. 固定每类 cap，按 `safe_left` 和 `behavior_regret` 分层落盘
4. 先把普通 guess 规模拉到 500-1000 条，再重新训 candidate ranker

## 2026-09-02 生产策略 guess 采样与线上复核

这轮把采样对象从“旧策略/失败尾盘”进一步推到当前生产配置本身：`full_rlmix_20.pt` + `full_rlmix_100_best.pt` 作为 base ensemble，再接 `cf1995_riskhead_strong` specialist。目标不是再堆随机局数，而是直接抽当前策略里 solver 无 forced move、模型必须自己猜的普通内部 guess 状态。

新增数据集：

- `artifacts/report_assets/extreme_training_dataset/gated_live_plain_guess_96_allopen.npz`
- 请求 96 局，完成 96 局
- 当前生产策略胜负为 35/96
- live guess label attempts: 7710
- live guess transitions: 45
- 最终选中普通 guess/近似 guess 70 条
- all-open 候选标签 28,679 个
- 负候选率 21.38%
- 平均 best counterfactual value 8.24
- 平均 behavior counterfactual value 0.93
- 平均 behavior regret 7.32

这批数据比前面的普通 guess 小批更“尖”：平均 regret 从 5.90 进一步升到 7.32，说明当前主策略在真实线上 guess 状态里确实存在可学习的候选排序短板。

新 ranker：

- `artifacts/candidate_gated_live_plain_guess_context5x5_risk_softmax.json`
- 训练混合：旧普通 guess、policy trajectory guess、这轮 gated-live guess
- 共 252 条 transition，106,601 个候选标签
- 训练集 best_match 9.92%
- near_best 23.0%
- chosen_negative 9.52%
- avg_regret 3.76

固定局面 benchmark：

- `policy_blend=0.2` 时，平均 regret 从 6.98 降到 6.19
- improved 12.87%
- worsened 4.95%
- unchanged 82.18%

但整局线上 50 局快筛结果变差：

- `artifacts/report_assets/candidate_gated_live_plain_guess_internal50_blend020.json`
- 12/50 = 24%
- candidate_adjusted 107 次
- candidate_changed 69 次

阶段结论：

- 固定局面 counterfactual benchmark 能证明“这个状态上 candidate 有价值”，但还不能证明“接入整局后能提升胜率”。
- 当前 ranker 会把单步 regret 降下来，却会改变太多真实对局中的路径，导致后续局面分布漂移，整局胜率下降。
- 数据资产反哺训练的验证口径要分两层：先看局面级 regret，再看整局 seed 快筛；只有两者同时过线才进入 200/500 局确认。
- 下一步应扩大当前生产策略的普通 guess 样本到 500-1000 条，并按 `safe_left`、候选数量、policy margin、behavior regret 分层训练；在此之前，不把 candidate ranker 接入生产默认策略。

工程侧补强：

- `scripts/build_gated_failure_extreme_dataset.py` 已支持定时 progress 输出。
- 支持 `--save-every-records` 和 `--partial-output`。
- 异常中断时会尽量保存 `.partial.npz`，避免长跑采样白费。

## 2026-09-02 短板分层采样器完成

我又把 gated 采样器补成了更像“数据开发”的形态，不再只是收 guess：

- 新增 `--collect-guess-open-candidates-min / max`
- 新增 `--collect-guess-min-behavior-regret`
- 新增 `--collect-guess-behavior-mine-only`
- `collect-guess-region` 现在支持 `current-edge` 和 `current-corner`
- 长跑时会定时落 `.partial.npz`
- 异常退出也会尽量保留 partial

这意味着后面可以直接抽下面几类短板数据：

1. 候选格很多，但模型判断不稳的 guess
2. 边缘、角落、内部三类分别采
3. 只要行为 regret 高的 hard guess
4. 只要真实踩雷的 guess

我还做了一个 12 局 smoke：

- `artifacts/report_assets/extreme_training_dataset/smoke_gated_live_guess_stratified.npz`
- 2 条高质量 guess transition
- 911 个 counterfactual 标签
- 平均 behavior regret 8.09

这说明采样器已经能稳定把“尖锐局面”切出来了，接下来就不是再堆随机局数，而是批量按短板类型采样，然后只拿这些子集去反哺 candidate/ranker 训练。

## 2026-09-02 训练反馈数据仓库闭环

短板画像已经接入现有 SQLite 实验数仓：

- 数据库：`artifacts/report_assets/minesweeper_experiments.sqlite`
- 分析报告：`artifacts/report_assets/database_analysis.json`
- Markdown 报告：`artifacts/report_assets/database_analysis.md`
- schema version: `3`
- `training_dataset_profile`: 1
- `training_dataset_record`: 134
- source lineage 中新增训练反馈画像文件记录

新增查询对象：

- `v_ads_training_dataset_profile`：画像级总体指标
- `v_ads_training_feedback_slice`：按区域、safe-left、候选数量、regret 分层
- `v_ads_training_feedback_priority`：按 critical/high/medium/low 排序的优先回放样本

当前数仓给出的最高价值切片是：

- `interior`
- `safe_left=241_391`
- `open_candidate_count=201_480`
- `behavior_regret=gte_8`
- 57 条记录
- 平均 behavior regret 9.58
- 最高 behavior regret 11.78

这比“全局随机采样”更适合作为训练反馈入口：它直接指出当前生产策略最需要补的是开局到中前期、候选格很多、没有 forced move 时的内部 guess 排序。边角数据目前只有 13 条，且主要来自失败尾盘，不能和内部 live guess 等权混训。

最终决策：

- 当前稳定生产配置不变，仍以约 44.5% 的 base/specialist gated policy 为基线。
- 高 regret candidate ranker 暂不接入默认策略；50 局同 seed 只从 30% 到 32%，且耗时明显增加。
- 后续训练优先从数仓的 `v_ads_training_feedback_priority` 抽样，扩大内部 live guess 到 500-1000 条，再按切片单独训练和验证。

## 2026-09-03 live high-regret guess 训练复核

今天把采样对象进一步对齐到当前生产策略本身：`full_rlmix_20.pt` + `full_rlmix_100_best.pt` 作为 base ensemble，`cf1995_riskhead_strong` 作为原 specialist。新增数据集：

- `artifacts/report_assets/extreme_training_dataset/gated_live_plain_guess_highregret160_20260903.npz`
- 160 条全部为 `guess`
- `safe_left=141..391`
- `collect_guess_region=current-interior`
- all-open counterfactual 标签 71,019 个
- 平均 best counterfactual value: 8.27
- 平均 behavior counterfactual value: 0.33
- 平均 behavior regret: 7.94

随后合并出训练资产：

- `artifacts/report_assets/extreme_training_dataset/guess_counterfactual_policy_v3_livehighregret_merged.npz`
- 610 条纯 `guess`
- all-open counterfactual 标签 103,967 个
- 平均 behavior regret: 7.30

### 候选排序器复核

新增了 pairwise candidate ranker 训练路径，并补了 policy-topK shortlist、ranker margin gate 和匹配 risk-head 的 benchmark 参数。相关测试已覆盖：

- `tests/test_candidate_calibration.py`
- `tests/test_candidate_benchmark.py`
- `tests/test_gated_specialist.py`

结论很明确：pairwise ranker 在训练/旧 holdout 上能降低固定局面 regret，但在独立 live guess holdout 上不稳定，会把平均 regret 拉高。因此 candidate ranker 仍不接入默认生产策略。

### 直接 policy 微调复核

新增了 ensemble teacher KL 与 teacher-prior PPO 形式的离线 policy-improvement loss：

- 支持 `--teacher-ensemble-checkpoint`
- 支持 `--teacher-ensemble-weight`
- 支持 `--teacher-prior-policy-coef`
- 支持 `--teacher-prior-objective ppo`
- 支持 `--teacher-prior-policy-topk`

验证结果：

| 配置 | 口径 | 结果 | 结论 |
|---|---:|---:|---|
| v1 全盘 counterfactual policy | 50 局单模型 | 13/50 | 漂移，淘汰 |
| v2 强 teacher KL 小学习率 | 50 局单模型 | 13/50 | 仍漂移 |
| v3 teacher-prior PPO | 50 局 base ensemble | 15/50 | 持平 |
| v3 solver-guess 替换 OPEN | 50 局 | 16/50 | 小样本正信号 |
| v1 solver-guess 替换 OPEN | 200 局 | 83/200 = 41.5% | 低于 89/200 基线，淘汰 |
| v4 teacher topK16 PPO | 50 局 | 13/50 | 淘汰 |

旧模型横向 alignment 也做了复核：

- `full_rlmix_weighted_smoke.pt` 在 160 条 high-regret guess 上选雷率最低，约 0.6%，平均 regret 约 7.08。
- `full_rlmix_100best_feedback_refine.pt` 平均 regret 最低，约 6.48，但选雷率约 5.6%，线上 50 局 solver-guess 替换只有 13/50。
- `full_rlmix_extreme_weighted.pt` 线上 50 局 solver-guess 替换为 16/50，但没有强到值得直接拉 200 局。

### 当前判断

- 200 局 44.5% 基线不应被今天的新模型替换；1000 局历史纯 ensemble 为 43.0%，说明小样本超过 45% 有明显波动。
- 数据资产方向是有效的：它成功定位出当前模型在中前期内部真实 guess、候选格很多、behavior regret 很高时的短板。
- 但直接用这些标签推 policy 会产生路径分布漂移；下一轮训练必须限制到 teacher/topK 候选或训练专门的 guess expert，再用严格门控做最终动作。
- 当前最缺的是分层覆盖，不是继续反复训练同一批 160 条。下一批数据应补：
  - `current-edge-or-corner`
  - `safe_left <= 140` 的尾盘真实 guess
  - `open_candidate_count` 低/中/高三档
  - `behavior_mine_only` 的踩雷样本
  - 每类单独记录，不与内部 high-regret 等权混训

## 2026-09-03 边角残局资产与窄门控复核

为补齐上一轮几乎没有覆盖的边缘/角落残局，本轮用当前生产策略定向采样：

- 采样条件：`solver` 无 forced OPEN、`safe_left <= 140`、当前 OPEN 位于 edge 或 corner、行为 regret >= 1.0
- 数据集：`artifacts/report_assets/extreme_training_dataset/gated_live_edgecorner_tail_guess_64_20260903.npz`
- 64 条 guess transition，其中 10 条 `corner_guess_tail`、21 条 `edge_guess_tail`
- all-open counterfactual 标签 4,679 个
- 平均 behavior regret 2.69，平均 behavior counterfactual value -0.35

随后与原有 610 条内部 high-regret guess 合并：

- `artifacts/report_assets/extreme_training_dataset/guess_counterfactual_policy_v4_stratified_merged.npz`
- 674 条 transition、108,646 个候选标签
- family：`guess=616`、`guess_tail=27`、`edge_guess_tail=21`、`corner_guess_tail=10`
- region：`interior=643`、`edge=21`、`corner=10`
- 平均 behavior regret 6.87，负候选率 21.71%

### 训练试验

1. 直接用 v4 资产微调主 policy head：100 局 seed0 为 34/100，低于微调前 35/100，淘汰。
2. 只训练 counterfactual value head，并限制在 `solver-guess + current-edge-or-corner + safe_left<=140`：
   - 离线 corner best-match 80%，edge best-match 43%，说明边角子任务确实可学。
   - 全盘窄门控 sweep：weight `0.10/0.25/0.50` 分别为 `34/100`、`33/100`、`34/100`。
   - 最终动作只改变 1、6、14 次，未形成净胜率收益。

结论：边角 value head 可以作为研究资产保留，但当前不进入生产策略；它证明了“按短板建数据资产”能改善局面级排序，却还没有解决整局路径分布漂移问题。

### 数仓闭环更新

`artifacts/report_assets/minesweeper_experiments.sqlite` 已登记第 4 个训练画像：

- `training_dataset_profile=4`
- `training_dataset_record=1,578`
- 训练资产总候选标签 328,866 个
- 加权平均 behavior regret 7.11
- 其中 edge 33 条、corner 11 条

`scripts/generate_training_feedback_plan.py` 也已修正统计口径：反馈计划现在从底层 `training_dataset_profile` 和 `training_dataset_record` 读取训练资产，不再把 `guess` 资产误报为 0 条。当前相关测试 63 项全部通过。

当前生产基线仍保持不变：`gated_cf1995_riskhead_strong_20_100best_60_w01_risk005_200.json` 的 89/200，即 44.5%。

## 2026-09-03 behavior-mine-only 资产

针对“模型在残局剩余雷数判断上会直接踩雷”的问题，又单独采集了行为动作已知为雷的样本：

- 数据集：`artifacts/report_assets/extreme_training_dataset/gated_live_mineonly_edgecorner_tail_48_20260903.npz`
- 条件：`safe_left <= 100`、当前动作位于 edge/corner、solver 无 forced OPEN、`behavior_mine_only=true`
- 48 条 transition，all-open 标签 2,659 个
- 平均 behavior counterfactual value `-0.53`
- 平均 behavior regret `2.90`
- family：`corner_guess_tail=7`、`edge_guess_tail=23`，其余为低候选数 guess 尾盘

它已经合并为：

- `artifacts/report_assets/extreme_training_dataset/guess_counterfactual_policy_v5_stratified_mineonly_merged.npz`
- 722 条 transition、111,305 个候选标签
- region：`interior=660`、`edge=45`、`corner=17`

这类数据不应直接复制成大量 policy imitation，而应作为 hard negative / risk calibration 专题资产，单独控制采样比例，避免模型为了规避少量残局雷而破坏中盘策略。

当前 SQLite 数仓已更新为 `training_dataset_profile=5`、`training_dataset_record=2,300`，反馈计划已指向 v5 合并资产。
