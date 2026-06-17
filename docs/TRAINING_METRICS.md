# Training and downstream metrics

更新时间: 2026-06-17 22:46:50 CST

## 术语白话说明

- `step`: 参数更新次数；每走一步，模型根据一批训练样本更新一次权重。
- `train loss`: 训练集上的预测错误程度，越低通常表示模型正在学到 DNA 序列规律。
- `val loss`: 验证集上的预测错误程度；它比 train loss 更能反映模型是否泛化到没训练过的 assembly（基因组装版本）。
- `checkpoint`: 阶段性保存的模型权重文件；GitHub 只记录路径和指标，不上传文件本身。
- `downstream probe`: 下游探针评测；用少量有标签任务快速检查 checkpoint（模型存档点）是否已有可迁移表示。
- `benchmark`: 基准评测；比 probe 更正式，需要更大样本、固定 split（数据划分）、基线和重复实验。

## 1. 训练版本总览

| Version ID | 状态 | checkpoint（模型存档点） | 当前/最终 step | 说明 | 下游结果 |
|---|---|---|---:|---|---|
| `v1_backbone_stageB_step5000` | 已停止，作为上一版对照 | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` | 5000 | legacy HyenaLite（旧版长卷积序列模型）Stage_B（第二阶段预训练）；不是当前正式 CaduceusRC（反向互补一致性）版本 | [`docs/downstream/v1_backbone_stageB_step5000/`](downstream/v1_backbone_stageB_step5000/) |
| `formal_caduceus_rc_stageB_mb5` | 正在训练 | 尚未到第一个 checkpoint；配置为每 1000 step 保存 | 420 | 当前正式 CaduceusRC（反向互补一致性）Stage_B，micro-batch（单次显卡小批量）=5，GPU2（2号显卡）训练 | 待第一个 checkpoint 后评测 |

## 2. 当前正式 CaduceusRC Stage_B 状态

- 训练日志: `training_server_transfer/logs/formal_caduceus_rc_stage_B_gpu2_mb5_20260617_170004.log`
- 当前训练状态: 正在运行
- 当前 GPU: `gpu10` 的 GPU2（2号显卡）
- 当前 step: `420`
- 当前 train loss: `1.257612`
- 当前 MLM loss（masked language modeling，遮盖碱基预测损失）: `1.257399`
- 当前 RC loss（reverse-complement consistency，反向互补一致性损失）: `0.007106`
- 当前 learning rate（学习率）: `8.4e-06`
- 最近 validation（验证）: 尚未到 eval_every（验证间隔）=1000 step
- 最新 checkpoint（模型存档点）: 尚未到 save_every（保存间隔）=1000 step

## 3. 上一版 v1 checkpoint 下游 first-pass probe

任务: `region_bucket_classification`（功能区域桶分类）。

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | train n（训练样本数） | test n（测试样本数） |
|---|---:|---:|---:|---:|---:|
| CropGenome-FM v1 frozen embedding（冻结模型表示） | 0.232143 | 0.188456 | 0.232143 | 56 | 56 |
| 1-mer composition baseline（单碱基组成基线） | 0.196429 | 0.153571 | 0.196429 | 56 | 56 |
| Delta（模型 - 基线） | +0.035714 | +0.034885 | +0.035714 | - | - |

结论: v1 旧 checkpoint 的 frozen embedding（冻结表示）略高于 1-mer composition baseline（单碱基组成基线），但这是 CPU-bounded first-pass probe（CPU 限定第一轮探针），每类 test（测试）只有 8 个窗口，且只使用 512 bp（碱基对）序列片段；不能作为正式 benchmark（基准评测）结论。

详细结果、PNG/PDF 图和 source-data TSV（源数据表）见: [`docs/downstream/v1_backbone_stageB_step5000/`](downstream/v1_backbone_stageB_step5000/)

## 4. GitHub 记录规则

- GitHub 只保存轻量 README、TSV（表格源数据）、PNG/PDF 图和 QA（质量检查）表。
- 不上传 checkpoint（模型存档点）、embedding（向量表示）、逐样本大预测文件、训练输入 shard（分片）或日志大文件。
- 新图只导出 PNG（位图预览）和 PDF（矢量图）；不新增 SVG 图。
- 不同训练版本必须新增独立目录或独立表行，不能覆盖旧结果。
