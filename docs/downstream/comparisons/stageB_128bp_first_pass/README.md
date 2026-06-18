# Stage_B 128 bp first-pass downstream comparison

更新时间: 2026-06-18 CST

## 1. 比较口径

本页比较两个 checkpoint（模型存档点）在同一个 CPU-bounded first-pass probe（CPU 限定第一轮探针评测）上的表现。

| 项目 | 内容 |
|---|---|
| 任务 | `region_bucket_classification`（功能区域桶分类） |
| 方法 | frozen embedding nearest-centroid probe（冻结表示最近质心探针） |
| 序列长度 | 128 bp（碱基对） |
| 采样规模 | 7 类，每类 train/test（训练/测试）各 8 个窗口；train n=56，test n=56 |
| baseline（基线） | 1-mer composition baseline（单碱基组成基线） |
| 图格式 | PNG（位图预览）+ PDF（矢量图），未生成 SVG |

## 2. 结果

| 方法 | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | Delta Macro-F1 vs baseline（相对基线提升） |
|---|---:|---:|---:|---:|
| 1-mer composition baseline | 0.178571 | 0.147429 | 0.178571 | 0.000000 |
| v1 backbone step5000 | 0.196429 | 0.166405 | 0.196429 | +0.018976 |
| formal CaduceusRC step1000 | 0.214286 | 0.200834 | 0.214286 | +0.053405 |

结论：在相同 128 bp（碱基对）first-pass probe（第一轮探针）口径下，formal CaduceusRC（反向互补一致性正式版）step1000 的 Macro-F1（类别平均 F1）最高，高于旧 v1 step5000，也高于 1-mer composition baseline（单碱基组成基线）。这说明正式 CaduceusRC 的早期 checkpoint 已经出现更强的区域表示信号；但该结论仍限于小样本 first-pass probe，不应写成正式 benchmark（基准评测）胜利。

## 3. 图和 source data（源数据）

- Comparison PNG（对比位图）: [`figures/stageB_128bp_comparison.png`](figures/stageB_128bp_comparison.png)
- Comparison PDF（对比矢量图）: [`figures/stageB_128bp_comparison.pdf`](figures/stageB_128bp_comparison.pdf)
- Metrics source data（指标源数据）: [`source_data/model_comparison_metrics.tsv`](source_data/model_comparison_metrics.tsv)
- Figure QA（图质量检查）: [`source_data/comparison_figure_qa.tsv`](source_data/comparison_figure_qa.tsv)

## 4. 限制

- 只评估 128 bp（碱基对），没有评估完整 8K context（8192 bp 长上下文）。
- 每类 test（测试）只有 8 个窗口，统计方差会很大。
- nearest-centroid（最近质心）只是轻量 probe，不是微调分类器。
- 后续应按 step2000/step3000/step5000 追加同口径曲线，再决定是否值得启动更完整 8K/GPU probe。
