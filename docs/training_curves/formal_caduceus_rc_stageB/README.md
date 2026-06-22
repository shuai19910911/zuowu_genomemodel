# Formal CaduceusRC Stage_B training loss curves

更新时间: 2026-06-22 12:30 CST

本目录记录 formal CaduceusRC（正式反向互补一致性模型）Stage_B（第二阶段预训练）停止时的训练曲线。所有文件都是 GitHub（代码托管平台）可保存的轻量结果；不包含 checkpoint（模型存档点）、训练输入 shard（分片）或完整日志。

## 当前日志快照

| 项目 | 数值 |
|---|---:|
| train loss（训练损失）记录点 | 590 |
| validation loss（验证损失）记录点 | 6 |
| 训练状态 | stopped by user request（用户要求停止） |
| 最新 train step（训练步） | 8500 |
| 最新 train loss（训练损失） | 1.037332 |
| 最新 MLM loss（遮盖碱基预测损失） | 1.034064 |
| 最新 RC loss（反向互补一致性损失） | 0.108925 |
| 最新 learning rate（学习率） | 1.70e-04 |
| 最新 validation step（验证步） | 8000 |
| 最新 validation loss（验证损失） | 1.122568 |
| 最佳 validation step（验证步） | 5000 |
| 最佳 validation loss（验证损失） | 1.079774 |

## 图和 source data（源数据）

- Loss curve PNG（损失曲线位图预览）: [`figures/formal_caduceus_rc_stageB_loss_curve.png`](figures/formal_caduceus_rc_stageB_loss_curve.png)
- Loss curve PDF（损失曲线矢量图）: [`figures/formal_caduceus_rc_stageB_loss_curve.pdf`](figures/formal_caduceus_rc_stageB_loss_curve.pdf)
- Loss curve source data（损失曲线源数据）: [`source_data/formal_caduceus_rc_stageB_loss_curve.tsv`](source_data/formal_caduceus_rc_stageB_loss_curve.tsv)
- Summary source data（摘要源数据）: [`source_data/formal_caduceus_rc_stageB_loss_summary.tsv`](source_data/formal_caduceus_rc_stageB_loss_summary.tsv)
- Figure QA（图质量检查）: [`source_data/formal_caduceus_rc_stageB_loss_curve_qa.tsv`](source_data/formal_caduceus_rc_stageB_loss_curve_qa.tsv)
- Final evaluation（最终评价）: [`../../TRAINING_EVALUATION_STOP_20260622.md`](../../TRAINING_EVALUATION_STOP_20260622.md)

未生成 SVG（可缩放矢量图）。

## 解读边界

train loss（训练损失）明显下降，说明预训练主任务有效学习；但 validation loss（验证损失）最低点在 step5000，不是最新完整 checkpoint（模型存档点）step8000。下游能力仍以后续同口径 probe（探针评测）和正式 benchmark（基准评测）为准。
