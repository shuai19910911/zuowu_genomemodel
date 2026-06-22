# formal CaduceusRC 128 bp checkpoint trend

更新时间: 2026-06-22 10:45 CST

## 1. 对比口径

本页比较 formal CaduceusRC（正式反向互补一致性模型）Stage_B（第二阶段训练）多个 checkpoint（模型存档点）在同一 128 bp（128 个碱基对）first-pass probe（第一轮探针评测）上的表现。

“同一口径”表示每次只换 checkpoint（模型存档点），其余保持一致：同一个 `region_bucket_classification`（功能区域桶分类）任务、同一 7 类区域、每类 train/test（训练/测试）各 8 个窗口、同样只截取前 128 bp（碱基对）、同样的 nearest-centroid（最近质心）probe（探针评测）和同样的 1-mer composition baseline（单碱基组成基线）。

## 2. 趋势结果

| checkpoint（模型存档点） | Accuracy（准确率） | Macro-F1（类别平均 F1） | Balanced accuracy（类别平均召回） | Delta Macro-F1 vs baseline（相对基线） |
|---|---:|---:|---:|---:|
| step1000 | 0.214286 | 0.200834 | 0.214286 | +0.053405 |
| step2000 | 0.125000 | 0.088710 | 0.125000 | -0.058719 |
| step3000 | 0.214286 | 0.176644 | 0.214286 | +0.029215 |
| step5000 | 0.267857 | 0.242750 | 0.267857 | +0.095321 |
| step8000 | 0.232143 | 0.208736 | 0.232143 | +0.061307 |

1-mer composition baseline（单碱基组成基线）在同一批测试中的 Macro-F1（类别平均 F1）为 0.147429。

## 3. 公正解释

- step5000 是当前小样本 probe（探针评测）里的最高点，Macro-F1（类别平均 F1）为 0.242750。
- step8000 仍高于 baseline（基线）和 step3000，但低于 step5000，说明下游小测仍有波动。
- validation loss（验证损失）到 step8000 仍在下降，说明预训练主任务仍在学习。
- 因此当前结论是：模型有持续区域表示信号，step5000/8000 均超过 baseline（基线）；但 128 bp 小样本 probe（探针评测）不能当作正式 benchmark（基准评测）结论。

## 4. 图和 source data（源数据）

- Trend PNG（趋势位图）: [`figures/formal_caduceus_rc_128bp_step_trend.png`](figures/formal_caduceus_rc_128bp_step_trend.png)
- Trend PDF（趋势矢量图）: [`figures/formal_caduceus_rc_128bp_step_trend.pdf`](figures/formal_caduceus_rc_128bp_step_trend.pdf)
- Trend source data（趋势源数据）: [`source_data/step_trend_metrics.tsv`](source_data/step_trend_metrics.tsv)
- Figure QA（图质量检查）: [`figures/formal_caduceus_rc_128bp_step_trend_qa.tsv`](figures/formal_caduceus_rc_128bp_step_trend_qa.tsv)

未生成 SVG（可缩放矢量图）。
