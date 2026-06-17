# Training Metrics

更新时间: 2026-06-17 14:42:55 CST

## 术语白话说明

- `step`: 参数更新次数；每走一步，模型根据一批训练样本更新一次权重。
- `train loss`: 训练集上的预测错误程度，越低通常表示模型正在学到 DNA 序列规律。
- `val loss`: 验证集上的预测错误程度；它比 train loss 更能反映模型是否泛化到没训练过的 assembly。
- `learning rate`: 学习率，控制每一步参数更新幅度；当前仍处于 warmup 上升阶段。
- `checkpoint`: 阶段性保存的模型权重文件；只记录路径和指标，不上传 GitHub。

## v1-backbone Stage_B 当前状态

- 训练日志: `training_server_transfer/logs/v1_backbone_stage_B_gpu2_20260614_230834.log`
- 当前训练状态: 正在运行
- 当前 GPU: `gpu10` 的 GPU 2，显存 `32561/40960 MiB`，利用率 `100%`
- 当前 step: `5060`
- 当前 train loss: `1.094867`
- 最近 100 step 平均 train loss: `1.111234`
- 最近 500 step 平均 train loss: `1.120303`
- 当前 learning rate: `0.0001012`
- 最近 validation: step `5000`，val loss `1.152421`
- 最新 checkpoint: `runs/Stage_B/checkpoints/step_00005000.pt`，大小约 `2.05 GB`
- 当前运行进程 checkpoint 间隔: 每 `5000` step
- 已更新配置文件 checkpoint 间隔: 每 `1000` step，需训练进程重启或从 checkpoint resume 后生效

![Stage_B training metrics](../assets/training_metrics/stage_B_loss.svg)

## Validation 记录

| step | val loss |
|---:|---:|
| 2500 | 1.228092 |
| 5000 | 1.152421 |

## Checkpoint 记录

| 时间 | step | checkpoint | 说明 |
|---|---:|---|---|
| 2026-06-17 14:38:35 CST | 5000 | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` | 首个 Stage_B checkpoint 已生成；文件本身不上传 GitHub。 |

## 阶段判断

- Stage_B 训练方向正常: train loss 从 step 10 的约 `1.98` 降到 step `5060` 的约 `1.095`。
- val loss 从 step 2500 的 `1.228092` 降到 step 5000 的 `1.152421`，说明截至首个 checkpoint，验证集也在改善。
- GPU 2 利用率为 `100%`，当前不是显存闲置导致的低效运行；不建议为“占满显存”而重启或扩大 batch。
- 下一次重要观察点是 step `10000` 附近；当前运行进程仍按旧启动参数每 5000 step 保存一次。

## 更新规则

- 每到 checkpoint 后，刷新本文件和 SVG 曲线。
- checkpoint 文件本身不上传 GitHub，只记录 step、loss、val loss、learning rate 和本地路径。
- 若训练配置、数据或 checkpoint 路径变化，新增记录，不覆盖历史结论。
