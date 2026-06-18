# Training and downstream metrics

更新时间: 2026-06-18 09:34:36 CST

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
| `v1_backbone_stageB_step5000` | 已停止，作为上一版对照 | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` | 5000 | legacy HyenaLite（旧版长卷积序列模型）Stage_B（第二阶段预训练）；保留 512 bp（碱基对）结果，并补充 128 bp 公平对照 | [`docs/downstream/v1_backbone_stageB_step5000/`](downstream/v1_backbone_stageB_step5000/) |
| `formal_caduceus_rc_stageB_step1000` | checkpoint 已评估，训练继续 | `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00001000.pt` | 1000 | 当前正式 CaduceusRC（反向互补一致性）Stage_B 第一个 checkpoint | [`docs/downstream/formal_caduceus_rc_stageB_step1000/`](downstream/formal_caduceus_rc_stageB_step1000/) |
| `formal_caduceus_rc_stageB_mb5` | 正在训练 | 后续 checkpoint 待填 | 1220 | 当前正式 CaduceusRC（反向互补一致性）Stage_B，micro-batch（单次显卡小批量）=5，GPU2（2号显卡）训练 | step2000/step3000 待追加 |

## 2. 当前正式 CaduceusRC Stage_B 状态

- 训练日志: `training_server_transfer/logs/formal_caduceus_rc_stage_B_gpu2_mb5_20260617_170004.log`
- 当前训练状态: 正在运行
- 当前 GPU: `gpu10` 的 GPU2（2号显卡）
- 当前 step: `1220`
- 当前 train loss: `1.249460`
- 当前 MLM loss（masked language modeling，遮盖碱基预测损失）: `1.249364`
- 当前 RC loss（reverse-complement consistency，反向互补一致性损失）: `0.003212`
- 当前 learning rate（学习率）: `2.44e-05`
- 最近 validation（验证）: step `1000`，val loss `1.244846`，val MLM loss `1.244806`，val RC loss `0.001336`
- 最新 checkpoint（模型存档点）: `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00001000.pt`

## 3. 下游 first-pass probe 结果

任务: `region_bucket_classification`（功能区域桶分类）。

### 3.1 128 bp 公平对比

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | Delta Macro-F1 vs baseline（相对基线提升） |
|---|---:|---:|---:|---:|
| 1-mer composition baseline（单碱基组成基线） | 0.178571 | 0.147429 | 0.178571 | 0.000000 |
| v1 backbone step5000 | 0.196429 | 0.166405 | 0.196429 | +0.018976 |
| formal CaduceusRC step1000 | 0.214286 | 0.200834 | 0.214286 | +0.053405 |

结论: 在相同 128 bp（碱基对）CPU-bounded first-pass probe（CPU 限定第一轮探针）口径下，formal CaduceusRC step1000 的 Macro-F1（类别平均 F1）高于旧 v1 step5000 和 1-mer composition baseline（单碱基组成基线）。这说明正式 CaduceusRC 的早期 checkpoint 已经出现更强的区域表示信号；但该结论仍不能写成正式 benchmark（基准评测）胜利。

详细对比见: [`docs/downstream/comparisons/stageB_128bp_first_pass/`](downstream/comparisons/stageB_128bp_first_pass/)

### 3.2 v1 512 bp 历史结果

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） |
|---|---:|---:|---:|
| CropGenome-FM v1 frozen embedding（冻结模型表示） | 0.232143 | 0.188456 | 0.232143 |
| 1-mer composition baseline（单碱基组成基线） | 0.196429 | 0.153571 | 0.196429 |

v1 512 bp（碱基对）结果保留为历史记录，但不直接和 formal 128 bp（碱基对）结果混用；跨版本比较以 128 bp 公平对照为准。

## 4. GitHub 记录规则

- GitHub 只保存轻量 README、TSV（表格源数据）、PNG/PDF 图和 QA（质量检查）表。
- 不上传 checkpoint（模型存档点）、embedding（向量表示）、逐样本大预测文件、训练输入 shard（分片）或日志大文件。
- 新图只导出 PNG（位图预览）和 PDF（矢量图）；不新增 SVG 图。
- 不同训练版本必须新增独立目录或独立表行，不能覆盖旧结果。
