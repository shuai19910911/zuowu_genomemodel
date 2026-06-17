# v1_backbone_stageB_step5000 downstream first-pass probe

更新时间: 2026-06-17 CST

## 1. 版本边界

本页记录的是上一版 checkpoint（模型存档点）的 first-pass downstream probe（第一轮下游探针评测），不是当前正在 GPU2（2号显卡）训练的正式 CaduceusRC（反向互补一致性）版本。

| 项目 | 内容 |
|---|---|
| Version ID | `v1_backbone_stageB_step5000` |
| checkpoint（模型存档点） | `training_server_transfer/runs/Stage_B/checkpoints/step_00005000.pt` |
| checkpoint step（训练步） | 5000 |
| backbone（主干模型） | legacy HyenaLite（旧版长卷积序列模型），28 layers（28 层），d_model（隐藏维度）=1024 |
| 评测任务 | region_bucket_classification（功能区域桶分类） |
| 评测口径 | CPU-bounded first-pass（CPU 限定第一轮）；每类 train/test（训练/测试）各 8 个窗口；每个窗口只用前 512 bp（碱基对） |
| split（数据划分） | 使用 Stage_B 窗口自带 train/test split（训练/测试划分） |

## 2. 方法

任务是把窗口分成 7 个 region bucket（功能区域桶）：coding（编码区）、splice（剪接区域）、promoter（启动子）、UTR（非翻译区）、TES（转录终止区域）、gene_body（基因主体）和 background（背景区域）。

比较两个方法：

1. `CropGenome-FM v1 frozen embedding`（冻结模型表示）：加载 step 5000 checkpoint（模型存档点），不微调 backbone（主干模型），抽取 mean-pooled embedding（平均池化向量表示），用 nearest-centroid（最近质心）分类。
2. `1-mer composition baseline`（单碱基组成基线）：只使用 A/C/G/T/N 比例、GC（鸟嘌呤+胞嘧啶比例）和长度特征，同样用 nearest-centroid（最近质心）分类。

## 3. 总体结果

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | train n（训练样本数） | test n（测试样本数） |
|---|---:|---:|---:|---:|---:|
| CropGenome-FM v1 frozen embedding | 0.232143 | 0.188456 | 0.232143 | 56 | 56 |
| 1-mer composition baseline | 0.196429 | 0.153571 | 0.196429 | 56 | 56 |
| Delta（模型 - 基线） | +0.035714 | +0.034885 | +0.035714 | - | - |

结论：上一版 checkpoint 的 frozen embedding（冻结表示）在这个很小的 CPU-bounded probe（CPU 限定探针）中略高于 1-mer composition baseline（单碱基组成基线），说明表示里可能已有少量区域区分信号；但绝对分数仍低，不能作为正式 downstream benchmark（下游基准评测）结论。

## 4. 分类别 F1

| Class（类别） | Support（测试样本数） | Precision（精确率） | Recall（召回率） | F1 |
|---|---:|---:|---:|---:|
| coding（编码区） | 8 | 0.240000 | 0.750000 | 0.363636 |
| splice（剪接区域） | 8 | 0.000000 | 0.000000 | 0.000000 |
| promoter（启动子） | 8 | 0.000000 | 0.000000 | 0.000000 |
| UTR（非翻译区） | 8 | 0.000000 | 0.000000 | 0.000000 |
| TES（转录终止区域） | 8 | 0.200000 | 0.250000 | 0.222222 |
| gene_body（基因主体） | 8 | 0.500000 | 0.125000 | 0.200000 |
| background（背景区域） | 8 | 0.571429 | 0.500000 | 0.533333 |

## 5. 图和 source data（源数据）

- Figure PNG（位图预览）: [`figures/region_probe_overview.png`](figures/region_probe_overview.png)
- Figure PDF（矢量图）: [`figures/region_probe_overview.pdf`](figures/region_probe_overview.pdf)
- Metrics source data（指标源数据）: [`source_data/region_probe_metrics_summary.tsv`](source_data/region_probe_metrics_summary.tsv)
- Per-class source data（分类别源数据）: [`source_data/region_probe_per_class_metrics.tsv`](source_data/region_probe_per_class_metrics.tsv)
- Sample count source data（样本数源数据）: [`source_data/region_probe_sample_counts.tsv`](source_data/region_probe_sample_counts.tsv)
- Figure QA（图质量检查）: [`source_data/region_probe_figure_qa.tsv`](source_data/region_probe_figure_qa.tsv)

未生成 SVG（可缩放矢量图）文件，符合本次约束。

## 6. 限制和下一步

- 这是 first-pass sanity probe（第一轮合理性探针），不是正式 benchmark（基准评测）。
- 因 GPU2（2号显卡）正在跑正式训练，本次不抢占 GPU，只做 CPU-bounded（CPU 限定）评测。
- 每类 test（测试）只有 8 个窗口，统计方差会很大。
- 只使用前 512 bp（碱基对），没有评估完整 8K context（8192 bp 长上下文）。
- 当前 formal CaduceusRC（反向互补一致性正式版）产生 checkpoint 后，应按相同目录结构新增独立版本页，不能覆盖本结果。
