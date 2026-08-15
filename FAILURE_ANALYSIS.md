# 失败分析附录

本附录汇总 `full_rlmix_20.pt` 在内部仿真输局上的 solver 对照分析，用于解释当前模型的主要瓶颈和后续优化方向。

## 数据来源

- Batched 输局分析：`artifacts\analysis_loss_solver_compare_500_batched.json`
- exact-limit 对照：`artifacts\analysis_loss_solver_compare_200.json`
- hard-loss refine 记录：`artifacts\full_rlmix_20_hard_refine_a.json`, `artifacts\full_rlmix_20_hard_refine_b.json`

## 核心结论

| 指标 | 数值 |
| --- | ---: |
| 评估局数 | 500 |
| 胜率 | 41.80% |
| 输局数 | 291 |
| 输局中存在错旗 | 67 (23.02%) |
| 终局仍有 solver forced move | 92 (31.62%) |
| 终局目标被 solver 判定为已知雷 | 7 (2.41%) |
| 终局目标被 solver 判定为已知安全 | 64 (21.99%) |
| 平均终局目标风险 | 0.287 |
| solver 最优可见猜测平均风险 | 0.226 |
| 平均风险差距 | 0.062 |

解释：模型并非只输在纯随机猜测；不少输局发生在 solver 仍能从可见约束推出 forced move，或模型在残局/错旗后对剩余雷数和风险排序处理不稳。

## 终局风险分布

| 风险区间 | 输局数 |
| --- | ---: |
| <0.20 | 121 |
| 0.20-0.33 | 37 |
| 0.33-0.50 | 33 |
| 0.50-0.75 | 89 |
| 0.75-1 | 4 |
| 1.00 | 7 |

## 残局切片

| 条件 | 数量 | 错旗 | forced available | known mine | target risk | best guess risk |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| safe left le 20 | 146 | 23 | 30 | 5 | 0.404 | 0.340 |
| safe left le 40 | 175 | 31 | 42 | 5 | 0.366 | 0.306 |
| safe left le 60 | 189 | 34 | 47 | 6 | 0.352 | 0.290 |
| safe left le 100 | 213 | 42 | 54 | 6 | 0.334 | 0.271 |

## exact-limit 对照

| exact limit | forced available | known mine | target risk | best guess risk |
| ---: | ---: | ---: | ---: | ---: |
| 8 | 35 | 1 | 0.289 | 0.258 |
| 24 | 41 | 2 | 0.279 | 0.221 |
| 32 | 45 | 2 | 0.278 | 0.217 |

## 典型输局样本

| seed | safe left | wrong flags | forced safe | target risk | best guess risk | solver known safe |
| ---: | ---: | ---: | ---: | ---: | ---: | --- |
| 60 | 1 | 1 | 0 | 0.500 | 0.500 | False |
| 61 | 8 | 2 | 2 | 0.000 | 0.000 | True |
| 3 | 1 | 1 | 1 | 0.000 | - | True |
| 82 | 1 | 1 | 1 | 0.000 | - | True |
| 190 | 1 | 1 | 1 | 0.000 | - | True |
| 319 | 7 | 1 | 3 | 0.000 | 0.000 | True |
| 2 | 20 | 1 | 5 | 0.000 | 0.083 | True |
| 445 | 37 | 1 | 2 | 0.000 | 0.085 | True |

## hard-loss refine 结果

| 文件 | round | games | win rate | longest streak | replay size |
| --- | ---: | ---: | ---: | ---: | ---: |
| `artifacts\full_rlmix_20_hard_refine_a.json` | 0 | 200 | 42.00% | 6 | 0 |
| `artifacts\full_rlmix_20_hard_refine_b.json` | 0 | 200 | 41.50% | 7 | 0 |

这些记录显示，小规模 hard-state refine 没有形成稳定增益；更合理的下一步是把 hard-state replay 与原始成功/中盘分布混合，并加强错旗后的长期惩罚。

## 后续优化方向

1. 增强错旗惩罚与 flag 校准，减少剩余雷数估计被污染。
2. 针对 `safe_left <= 40` 的残局做课程学习，引入更多全局雷数约束状态。
3. 训练时保留 solver 风险排序信号，但最终推理仍只使用 RL policy/ensemble。
4. hard-state refine 需要混合原始分布，避免只拟合输局尾部导致策略退化。

重新生成：

```powershell
python scripts/summarize_failure_analysis.py
```
