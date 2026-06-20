# Formal CaduceusRC Stage_B training loss curves

更新时间: 2026-06-20 08:18 CST

本目录记录 formal CaduceusRC（正式反向互补一致性模型）Stage_B（第二阶段预训练）当前运行的训练曲线。所有文件都是 GitHub（代码托管平台）可保存的轻量结果；不包含 checkpoint（模型存档点）、训练输入 shard（分片）或完整日志。

## 当前日志快照

| 项目 | 数值 |
|---|---:|
| train loss（训练损失）记录点 | 466 |
| validation loss（验证损失）记录点 | 4 |
| 最新 train step（训练步） | 4660 |
| 最新 train loss（训练损失） | 1.133583 |
| 最新 MLM loss（遮盖碱基预测损失） | 1.131287 |
| 最新 RC loss（反向互补一致性损失） | 0.076531 |
| 最新 learning rate（学习率） | 9.32e-05 |
| 最新 validation step（验证步） | 4000 |
| 最新 validation loss（验证损失） | 1.157540 |

## 图和 source data（源数据）

- Loss curve PNG（损失曲线位图预览）: [`figures/formal_caduceus_rc_stageB_loss_curve.png`](figures/formal_caduceus_rc_stageB_loss_curve.png)
- Loss curve PDF（损失曲线矢量图）: [`figures/formal_caduceus_rc_stageB_loss_curve.pdf`](figures/formal_caduceus_rc_stageB_loss_curve.pdf)
- Loss curve source data（损失曲线源数据）: [`source_data/formal_caduceus_rc_stageB_loss_curve.tsv`](source_data/formal_caduceus_rc_stageB_loss_curve.tsv)
- Summary source data（摘要源数据）: [`source_data/formal_caduceus_rc_stageB_loss_summary.tsv`](source_data/formal_caduceus_rc_stageB_loss_summary.tsv)
- Figure QA（图质量检查）: [`source_data/formal_caduceus_rc_stageB_loss_curve_qa.tsv`](source_data/formal_caduceus_rc_stageB_loss_curve_qa.tsv)

未生成 SVG（可缩放矢量图）。

## 解读边界

train loss（训练损失）和 validation loss（验证损失）下降说明预训练主任务仍在学习；这不等价于正式 downstream benchmark（下游基准评测）已经稳定提升。下游能力仍以后续同口径 probe（探针评测）和正式 benchmark（基准评测）为准。
