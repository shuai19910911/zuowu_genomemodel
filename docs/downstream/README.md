# Downstream evaluation index

本目录只保存可提交到 GitHub 的轻量结果：README、source-data TSV（源数据表）、PNG/PDF 图和简短 QA（质量检查）记录。不保存 checkpoint（模型存档点）、embedding（向量表示）、逐样本大文件或训练输入大目录。

## 训练版本说明

| Version ID | 状态 | checkpoint（模型存档点） | 说明 | 下游结果 |
|---|---|---|---|---|
| `v1_backbone_stageB_step5000` | 已完成 first-pass probe（第一轮探针评测） | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` | 上一版 legacy HyenaLite（旧版长卷积序列模型）Stage_B（第二阶段预训练）checkpoint；不是当前正在训练的 CaduceusRC（反向互补一致性正式版） | [`v1_backbone_stageB_step5000/`](v1_backbone_stageB_step5000/) |
| `formal_caduceus_rc_stageB_mb5` | 正在训练，尚未做下游评测 | 待第一个可用 checkpoint 后填写 | 当前正式 CaduceusRC（反向互补一致性）Stage_B，micro-batch（单次显卡小批量）=5，GPU2（2号显卡）训练 | 待填 |

## 当前已提交的下游任务

| 任务 | 数据 | 方法 | 当前结论 |
|---|---|---|---|
| region_bucket_classification（功能区域桶分类） | Stage_B（第二阶段）held-out test split（保留测试划分）窗口；7 类：coding（编码区）、splice（剪接区域）、promoter（启动子）、UTR（非翻译区）、TES（转录终止区域）、gene_body（基因主体）、background（背景区域） | frozen embedding nearest-centroid probe（冻结表示最近质心探针）对比 1-mer composition baseline（单碱基组成基线） | v1 旧 checkpoint 略高于组成基线，但样本很小、序列截断到 512 bp（碱基对），只作为 first-pass sanity probe（第一轮合理性探针），不能作为正式 benchmark（基准评测）结论 |

## 图和源数据规则

- 图只导出 PNG（位图预览）和 PDF（矢量图），不生成 SVG。
- 每张图必须有对应 source-data TSV（源数据表）。
- 所有结果必须标明训练版本，避免把旧 v1 checkpoint 和当前 CaduceusRC 正式训练混在一起。
