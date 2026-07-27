# Minesweeper RL

一个面向经典高级扫雷 `16 x 30 / 99` 的强化学习项目。

当前结构是：

- `rl/policy`：纯 RL，模型自己做所有动作决策
- `solver`：逻辑/概率基线，仅用于对照和验收
- `hybrid`：可选实验模式，给模型加风险先验

## 目标

- 高级地图：`16 x 30`，`99` 雷
- 连胜目标：`10` 连胜
- 胜率目标：`40%+`

## 训练

```powershell
python -m minesweeper_rl.cli train `
  --pretrain-episodes 5000 `
  --pretrain-imitation-coef 1.0 `
  --pretrain-epochs-per-episode 2 `
  --pretrain-batch-size 256 `
  --expert-replay-size 8192 `
  --dagger-updates-per-episode 1 `
  --dagger-batch-size 256 `
  --dagger-replay-size 8192 `
  --episodes 50000 `
  --lr 1e-3 `
  --eval-every 500 `
  --eval-games 100 `
  --solver-imitation-coef 0.2 `
  --mine-aux-coef 0.1 `
  --risk-supervision-coef 0.2 `
  --save-path checkpoints/rl.pt `
  --device cuda
```

如果没有 CUDA，把 `--device cuda` 改成 `--device cpu`。

训练时可以使用 solver 作为辅助教练：预训练阶段沿 solver 轨迹做行为克隆，RL 阶段也可保留 `solver_imitation_loss` 和 DAgger replay 辅助信号；`mine_aux_loss` 和 `risk_supervision_loss` 只使用训练期信息帮助模型学习风险与排序，不参与推理。
模型动作空间包含 `open`、`flag`、`unflag`、`chord`，撤旗也由模型自己决定。
评估和回放使用 `--mode rl` 时，决策仍然只来自神经网络；最终验收的 `target-check` 固定使用纯 RL 决策。

## 评估

纯 RL：

```powershell
python -m minesweeper_rl.cli evaluate --mode rl --checkpoint checkpoints/rl.pt --games 200
```

solver 基线：

```powershell
python -m minesweeper_rl.cli evaluate --mode solver --games 200
```

## 验收

纯 RL 验收：

```powershell
python -m minesweeper_rl.cli target-check --checkpoint checkpoints/rl.pt
```

solver 基线只用于对照评估，不作为最终验收：

```powershell
python -m minesweeper_rl.cli evaluate --mode solver --games 200
```

## 可视化

默认看纯 RL：

```powershell
python -m minesweeper_rl.cli watch --checkpoint checkpoints/rl.pt
```

看 solver 基线：

```powershell
python -m minesweeper_rl.cli watch --mode solver
```

保存回放：

```powershell
python -m minesweeper_rl.cli record --mode rl --checkpoint checkpoints/rl.pt --output replays/latest.json
```

## 代码结构

- `src/minesweeper_rl/game.py`：扫雷环境与奖励
- `src/minesweeper_rl/features.py`：纯可见状态编码
- `src/minesweeper_rl/model.py`：策略网络 + value head
- `src/minesweeper_rl/trainer.py`：RL 收集、更新、评估
- `src/minesweeper_rl/replay.py`：回放与保存
- `src/minesweeper_rl/visualizer.py`：可视化
- `src/minesweeper_rl/solver.py`：基线求解器

## 说明

`solver` 可以参与训练期辅助损失，也保留为基线；但最终验收固定走 `target-check` 的纯 `rl` 路径，`rl/policy` 推理路径不会让 solver 做任何动作决策。  
如果要继续冲更高胜率，可以在 `policy` 上继续做 A2C / PPO / curriculum / self-play 式训练增强。
