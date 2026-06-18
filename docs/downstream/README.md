# Downstream evaluation index

本目录只保存可提交到 GitHub 的轻量结果：README、source-data TSV（源数据表）、PNG/PDF 图和简短 QA（质量检查）记录。不保存 checkpoint（模型存档点）、embedding（向量表示）、逐样本大文件或训练输入大目录。

## 训练版本说明

| Version ID | 状态 | checkpoint（模型存档点） | 说明 | 下游结果 |
|---|---|---|---|---|
| `v1_backbone_stageB_step5000` | 已完成 first-pass probe（第一轮探针评测） | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` | 上一版 legacy HyenaLite（旧版长卷积序列模型）Stage_B（第二阶段预训练）checkpoint；保留 512 bp（碱基对）结果，并补充 128 bp 公平对照 | [`v1_backbone_stageB_step5000/`](v1_backbone_stageB_step5000/) |
| `formal_caduceus_rc_stageB_step1000` | 已完成 first-pass probe（第一轮探针评测） | `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00001000.pt` | 当前正式 CaduceusRC（反向互补一致性）Stage_B 第一个 checkpoint；CPU-bounded（CPU 限定）128 bp 公平对照 | [`formal_caduceus_rc_stageB_step1000/`](formal_caduceus_rc_stageB_step1000/) |
| `formal_caduceus_rc_stageB_mb5` | 继续训练中 | 后续 step2000/step3000 待填 | 当前正式 CaduceusRC（反向互补一致性）Stage_B，micro-batch（单次显卡小批量）=5，GPU2（2号显卡）训练 | 待后续 checkpoint 追加 |

## 当前已提交的下游任务

| 任务 | 数据 | 方法 | 当前结论 |
|---|---|---|---|
| region_bucket_classification（功能区域桶分类） | Stage_B（第二阶段）held-out test split（保留测试划分）窗口；7 类：coding（编码区）、splice（剪接区域）、promoter（启动子）、UTR（非翻译区）、TES（转录终止区域）、gene_body（基因主体）、background（背景区域） | frozen embedding nearest-centroid probe（冻结表示最近质心探针）对比 1-mer composition baseline（单碱基组成基线） | 128 bp（碱基对）公平口径下，formal CaduceusRC step1000 的 Macro-F1（类别平均 F1）为 0.200834，高于 v1 step5000 的 0.166405，也高于 1-mer baseline（单碱基组成基线）的 0.147429；仍属于 first-pass probe（第一轮探针），不是正式 benchmark（基准评测） |

## 跨版本对比

- 128 bp 公平对比图: [`comparisons/stageB_128bp_first_pass/figures/stageB_128bp_comparison.png`](comparisons/stageB_128bp_first_pass/figures/stageB_128bp_comparison.png)
- 128 bp 公平对比源数据: [`comparisons/stageB_128bp_first_pass/source_data/model_comparison_metrics.tsv`](comparisons/stageB_128bp_first_pass/source_data/model_comparison_metrics.tsv)

## 图和源数据规则

- 图只导出 PNG（位图预览）和 PDF（矢量图），不生成 SVG。
- 每张图必须有对应 source-data TSV（源数据表）。
- 所有结果必须标明训练版本，避免把旧 v1 checkpoint 和当前 CaduceusRC 正式训练混在一起。
