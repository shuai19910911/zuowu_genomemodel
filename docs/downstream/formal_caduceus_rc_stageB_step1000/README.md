# formal_caduceus_rc_stageB_step1000 downstream first-pass probe

更新时间: 2026-06-18 CST

## 1. 版本边界

本页记录当前正式 CaduceusRC（反向互补一致性）Stage_B（第二阶段预训练）第一个 checkpoint（模型存档点）的 first-pass downstream probe（第一轮下游探针评测）。

| 项目 | 内容 |
|---|---|
| Version ID | `formal_caduceus_rc_stageB_step1000` |
| checkpoint（模型存档点） | `training_server_transfer/runs/Stage_B_formal_caduceus_rc/checkpoints/step_00001000.pt` |
| checkpoint step（训练步） | 1000 |
| backbone（主干模型） | formal CaduceusRC（反向互补一致性正式版），32 layers（32 层），d_model（隐藏维度）=1024，local attention（局部注意力）+ HyenaLite（长卷积序列模块） |
| 评测任务 | `region_bucket_classification`（功能区域桶分类） |
| 评测口径 | CPU-bounded first-pass（CPU 限定第一轮）；每类 train/test（训练/测试）各 8 个窗口；每个窗口只用前 128 bp（碱基对） |
| split（数据划分） | 使用 Stage_B 窗口自带 train/test split（训练/测试划分） |

## 2. 方法

任务是把窗口分成 7 个 region bucket（功能区域桶）：coding（编码区）、splice（剪接区域）、promoter（启动子）、UTR（非翻译区）、TES（转录终止区域）、gene_body（基因主体）和 background（背景区域）。

比较两个方法：

1. `CropGenome-FM formal CaduceusRC frozen embedding`（正式 CaduceusRC 冻结表示）：加载 step 1000 checkpoint（模型存档点），不微调 backbone（主干模型），抽取 mean-pooled embedding（平均池化向量表示），用 nearest-centroid（最近质心）分类。
2. `1-mer composition baseline`（单碱基组成基线）：只使用 A/C/G/T/N 比例、GC（鸟嘌呤+胞嘧啶比例）和长度特征，同样用 nearest-centroid（最近质心）分类。

## 3. 总体结果

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | train n（训练样本数） | test n（测试样本数） |
|---|---:|---:|---:|---:|---:|
| CropGenome-FM formal CaduceusRC frozen embedding | 0.214286 | 0.200834 | 0.214286 | 56 | 56 |
| 1-mer composition baseline | 0.178571 | 0.147429 | 0.178571 | 56 | 56 |
| Delta（模型 - 基线） | +0.035715 | +0.053405 | +0.035715 | - | - |

结论：formal CaduceusRC step1000 的 frozen embedding（冻结表示）在 128 bp（碱基对）CPU-bounded probe（CPU 限定探针）中超过 1-mer composition baseline（单碱基组成基线）。相同 128 bp 口径下，formal CaduceusRC step1000 也高于上一版 v1 step5000 的 macro-F1（类别平均 F1）。但这仍是小样本 first-pass probe（第一轮探针），不是正式 benchmark（基准评测）。

## 4. 分类别 F1

| Class（类别） | Support（测试样本数） | Precision（精确率） | Recall（召回率） | F1 |
|---|---:|---:|---:|---:|
| coding（编码区） | 8 | 0.000000 | 0.000000 | 0.000000 |
| splice（剪接区域） | 8 | 0.090909 | 0.125000 | 0.105263 |
| promoter（启动子） | 8 | 0.500000 | 0.125000 | 0.200000 |
| UTR（非翻译区） | 8 | 0.500000 | 0.125000 | 0.200000 |
| TES（转录终止区域） | 8 | 0.666667 | 0.250000 | 0.363636 |
| gene_body（基因主体） | 8 | 0.172414 | 0.625000 | 0.270270 |
| background（背景区域） | 8 | 0.285714 | 0.250000 | 0.266667 |

## 5. 图和 source data（源数据）

- Figure PNG（位图预览）: [`figures/region_probe_128bp_overview.png`](figures/region_probe_128bp_overview.png)
- Figure PDF（矢量图）: [`figures/region_probe_128bp_overview.pdf`](figures/region_probe_128bp_overview.pdf)
- Metrics source data（指标源数据）: [`source_data/region_probe_128bp_metrics_summary.tsv`](source_data/region_probe_128bp_metrics_summary.tsv)
- Per-class source data（分类别源数据）: [`source_data/region_probe_128bp_per_class_metrics.tsv`](source_data/region_probe_128bp_per_class_metrics.tsv)
- Sample count source data（样本数源数据）: [`source_data/region_probe_128bp_sample_counts.tsv`](source_data/region_probe_128bp_sample_counts.tsv)
- Figure QA（图质量检查）: [`source_data/region_probe_128bp_figure_qa.tsv`](source_data/region_probe_128bp_figure_qa.tsv)

未生成 SVG（可缩放矢量图）文件，符合本次约束。

## 6. 限制和下一步

- 这是 CPU-bounded first-pass probe（CPU 限定第一轮探针），不是正式 benchmark（基准评测）。
- 每类 test（测试）只有 8 个窗口，统计方差会很大。
- 只使用前 128 bp（碱基对），没有评估完整 8K context（8192 bp 长上下文）。
- 512 bp（碱基对）formal CPU probe（CPU 探针）首个样本耗时过长，已停止，避免拖数小时和影响主线。
- 下一步建议等 formal CaduceusRC step2000/step3000 checkpoint（模型存档点）出来后，用同样 128 bp 口径追加曲线；如果趋势稳定上升，再安排 GPU 或更高效 batched（批量）8K probe。
